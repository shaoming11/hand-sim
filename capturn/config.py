"""Env and PPO configs.

Values follow `PRD.md` except where a measured milestone overruled it; each of those is marked
`# M<n>:` below and written up in `NOTES.md`.

Everything the DR phase needs exists here from day one and is switched off rather than absent:
`noise_config.level = 0.0`, `action_delay.enabled = False`, `dr_params` zeroed. The DR phase
loads the nominal checkpoint, so the observation layout has to be byte-identical between the
two phases -- a noise scale that only appears at M7 would change the layout and make the
checkpoint unloadable.
"""

from __future__ import annotations

import numpy as np
from ml_collections import config_dict

# M2/M3: measured by `scripts/check_parity.py --probe` with 2x headroom. `naconmax` is a total
# across worlds, so it scales with `num_envs`; `njmax` is per world.
NACONMAX_PER_WORLD = 44
NJMAX = 150

# M3: the highest measured throughput, and it matches Playground's tuned LeapCubeRotateZAxis
# config so the whole PPO block transfers unchanged. The old comment here said 8192 "is the
# largest that fits" -- that was wrong. 16384 allocates fine on the H200 in a fresh process
# (the H100 failure was the documented single-process JAX/Warp starvation mode), it is just
# 4.6% *slower*: 30,856 vs 32,358 control-steps/s. 8192 wins on merit. NOTES.md, M3.
NUM_ENVS = 8192


def default_config() -> config_dict.ConfigDict:
    """Nominal-phase env config."""
    return config_dict.create(
        ctrl_dt=0.05,  # 20 Hz
        # M2: the PRD's 0.005 is not converged -- the contact solref timeconst is exactly one
        # step long there, which inflates cap rotation by 48%. 40 substeps, 2.31x the cost.
        sim_dt=0.00125,
        episode_length=400,  # 20 s
        action_repeat=1,
        # Delta position targets: target <- clip(target + action_scale * a, ctrlrange).
        action_scale=0.1,  # rad per control step
        history_len=3,
        early_termination=True,
        # Terminate when all three gaiting fingertips are this far from the cap surface.
        drop_distance=0.10,
        # M5: a second, much tighter termination -- both grasp surfaces must still be on the
        # cap. `drop_distance` only fires once the cap is gone entirely, and the flick policy
        # sat well inside it. 20 mm is read off the recordings: the M1 gait's thumb sits at
        # 2.2 mm and its index at 6.4 mm, and its release phase backs off 8 mm, so it clears
        # this comfortably; the flick policy's thumb sits at 20-22 mm.
        contact_distance=0.020,
        # M5: how many consecutive control steps the grasp may be lost before the episode ends.
        # A gait must release to regrip, so this cannot be instantaneous -- at 1 step it would
        # terminate the M1 scripted gait on 15% of its own steps. Measured on the recordings:
        # the gait's longest release is 3 steps, while the flick policy hovers for 18 at a
        # time and does it 11 times an episode. 8 sits between them with room either side.
        grasp_grace_steps=8,
        # Success: cumulative rotation within the episode, i.e. two full turns.
        success_rotation=float(4.0 * np.pi),
        reset_noise=config_dict.create(
            hand_qpos=0.05,  # rad, uniform, clipped to the joint range
            cap_angle=float(2.0 * np.pi),  # the cap starts at a uniformly random angle
        ),
        # Env-level DR hooks. `level` scales every entry in `scales`; 0.0 is the nominal phase.
        noise_config=config_dict.create(
            level=0.0,
            scales=config_dict.create(
                joint_pos=0.02,  # rad
                cap_pos=0.005,  # m
            ),
        ),
        # The buffer is `max_steps + 1` long whether or not delay is enabled, so the shape is
        # fixed from day one. Nominal uses index 0, i.e. no delay.
        action_delay=config_dict.create(enabled=False, max_steps=2),
        reward_config=config_dict.create(
            # M5: the PRD's `reach` was exp(-20 * mean gap of three fingertip sites). It could
            # not see the grasp the first policy used -- the side of the index finger -- and
            # averaging let one fingertip pay for two curled away. `grasp` replaces it: both
            # the thumb and the index finger on the cap, anywhere along their length, and
            # opposed. 100 is set off the recordings, where the M1 gait's worse surface sits at
            # 6.4 mm (exp -> 0.53) and the flick's thumb at 20 mm (exp -> 0.14, then zeroed by
            # the opposition factor anyway).
            grasp_sharpness=100.0,
            # M5: was 2.0 rad/s (115 deg/s). Success needs 4*pi in 20 s, i.e. 0.63 rad/s, so 2.0
            # paid for spinning three times faster than the task asks; the flick policy
            # saturated this clip every step at 4.6 rad/s. 1.0 still leaves 1.6x headroom over
            # the success rate and the M1 gait only reaches 0.11 rad/s, so nothing legitimate
            # touches it.
            rotate_clip=1.0,
            # PRD weights, with the sign of each cost moved from its definition into the weight
            # so the logged `reward/*` metrics read as signed contributions.
            scales=config_dict.create(
                rotate=1.0,
                # M5: 0.5 -> 0.15. At the pregrasp pose the old `reach` was 99.9% of the
                # signal and parking for a whole episode returned 8.93 against ~21 for
                # succeeding. With the sharper, opposition-gated term and this weight, parking
                # returns ~1.2 against ~14 for succeeding.
                grasp=0.15,
                # M5: the cost that makes the pinch two-fingered. `grasp` says what must touch
                # the cap; without this nothing says what must not, and `rotate` quietly pays
                # the middle finger to help turn it. Scaled so that all three idle tips in full
                # contact (3 x 1.0) exactly cancels a perfect grasp (1.0 x 0.15), which makes
                # recruiting a fourth finger worthless rather than merely discouraged. A first
                # setting, not a tuned one.
                idle_contact=-0.05,
                action_rate=-0.01,
                torques=-1e-3,
                joint_vel=-1e-4,
                success=10.0,
            ),
        ),
        # M2: `impl='jax'` cannot load this scene at all (no cylinder/mesh collisions), so warp
        # is the only backend, on the Mac as well as on the GPU box.
        impl="warp",
        naconmax=NACONMAX_PER_WORLD * NUM_ENVS,
        njmax=NJMAX,
    )


def ppo_config(env_config: config_dict.ConfigDict | None = None) -> config_dict.ConfigDict:
    """Brax PPO config, from Playground's tuned `LeapCubeRotateZAxis` settings.

    The PRD's fallback table matches the installed `config/manipulation_params.py` except for
    `discounting`, where Playground uses 0.97 and the PRD says 0.99. Playground's value wins:
    the installed source is the authority, and 0.97 over a 400-step episode still reaches far
    enough (a 33-step horizon at 20 Hz is 1.7 s, several gait cycles).
    """
    env_config = env_config if env_config is not None else default_config()
    return config_dict.create(
        num_timesteps=150_000_000,
        # M5: Brax writes one checkpoint per eval, so `num_evals` *is* the checkpoint density,
        # and the PRD wants one every few minutes because the box is preemptible. 150M steps at
        # the M3 rate of 31,449 control-steps/s is about 80 minutes of simulation, so 24 evals
        # is a checkpoint every ~3.5 min. Leap uses 10; the extra 14 cost one eval rollout each
        # (1024 envs x 400 steps, ~30 s at the M3 rate), which buys back more than it spends the
        # first time the machine is killed. `train.py --checkpoint-every MIN` re-derives it.
        num_evals=24,
        num_envs=NUM_ENVS,
        num_eval_envs=1024,  # PRD M5 judges success over 1000 eval episodes
        episode_length=env_config.episode_length,
        action_repeat=env_config.action_repeat,
        normalize_observations=True,
        reward_scaling=1.0,
        unroll_length=40,
        num_minibatches=32,
        num_updates_per_batch=4,
        batch_size=256,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        num_resets_per_eval=1,
        # M5: Brax defaults this to False, and so does Playground, which makes
        # `eval/episode_*` a sample from the stochastic policy. The M5 criterion is about the
        # policy that would be deployed, and `eval.py` reports the deterministic one, so the
        # two numbers should be measuring the same thing.
        deterministic_eval=True,
        # M4: the auto-reset wrapper restores `data` but not `info` at an episode boundary, and
        # with `full_reset=False` every env also restarts from the single start state cached at
        # the first reset -- 8192 start states for a 150M-step run. `env.py` is written to be
        # correct either way, but the reset randomisation the PRD asks for only actually happens
        # with this on. See NOTES.md, M4.
        full_reset=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
    )
