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

# M3: 8192 is the largest that fits. 16384 fails to allocate warp's multiccd_polygon buffer.
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
            reach_sharpness=20.0,
            # PRD weights, with the sign of each cost moved from its definition into the weight
            # so the logged `reward/*` metrics read as signed contributions.
            scales=config_dict.create(
                rotate=1.0,
                reach=0.5,
                action_rate=-0.01,
                torques=-1e-3,
                joint_vel=-1e-4,
                success=10.0,
            ),
        ),
        # M2: `impl='jax'` cannot load this scene at all (no cylinder/mesh collisions), so warp
        # is the only backend, on the Mac as well as on the H100.
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
        # (1024 envs x 400 steps, ~30 s on the H100), which buys back more than it spends the
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
