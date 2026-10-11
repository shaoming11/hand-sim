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
#
# M6: 44 -> 64. The M2 figure was measured on the *tripod* pregrasp, and re-running the probe on
# the re-fitted two-finger pose asks for **46** per world (`ncollision` 23, nacon 4, nefc 41) --
# so the configured 44 was below the requirement, which would have dropped contacts silently
# during the run. 64 rather than 46 because the scripted gait is the gentlest loading case in the
# project and an untrained policy is the harshest: M4 measured -4.5 mm of penetration and an
# EPA_HORIZON flag under a random policy against the gait's -0.288 mm.
#
# M6, after three OOMs: 64 -> 48. The extra buffer does *not* cost "about a gigabyte of the 96
# available" as first written -- warp's convex narrowphase allocates scratch sized off these
# buffers on every call (`collision_convex.convex_narrowphase` -> `warp.empty`), and at 64 per
# world that allocation starts failing mid-run: three runs died in warp with
# "Failed to allocate 1610612736 bytes" (twice) and "402653184 bytes" (once). M5 ran 91.75M
# steps at 44 per world without one. 48 is the compromise: above the 46 the re-fitted pose
# measures, and a total of 393,216 against M5's proven 360,448. `njmax` 150 still clears its
# measured 82.
#
# If 48 still OOMs, the next lever is NOT `num_envs` -- that is a PPO hyperparameter and moving
# it breaks comparability with the seeds already run -- but warp's mempool release threshold.
NACONMAX_PER_WORLD = 48
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
        # M5: 400 steps (20 s) demanded 36 deg/s to clear 4*pi, which is 5.8x the M1 scripted
        # gait's 6.2 deg/s -- the hand-authored reference could not have passed its own task, and
        # every trained policy answered by spinning the cap at 150-290 deg/s, far past
        # `rotate_clip` where the reward stops paying. 1200 steps (60 s) asks 12 deg/s, about 2x
        # the scripted gait, which leaves room for a deliberate regripping gait to succeed. The
        # step budget per run is unchanged, so this costs no wall-clock; it buys fewer episodes
        # per unit of experience.
        episode_length=1200,  # 60 s
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
            # M5: the idle-finger cost needs a much tighter falloff than the grasp term. At
            # `grasp_sharpness` (100) it reads 0.368 at 10 mm and 0.135 at 20 mm, so a middle
            # finger parked clear of the cap still paid, and since the index and middle are
            # adjacent the only way to stop paying was to lift the whole hand off -- which both
            # seeds duly did, one of them failing the task outright. At 500 the cost is 0.082 at
            # 5 mm and 0.007 at 10 mm: a finger that is merely near the cap is free, and only
            # one actually on it pays. Checked against the M1 scripted gait, which this term
            # must not punish: -5.3 of return at 100, -0.4 at 500.
            idle_sharpness=500.0,
            # M7: `stroke_budget` and the grip state machine are gone, and the reason is that
            # the hardware now does their job. They existed to force a release-and-regrip cycle
            # out of a hand that could turn the cap indefinitely without one. With `rh_WRJ0`
            # limited to 100 deg of travel, one stroke is about 90 deg of cap rotation and 4*pi
            # needs roughly ten of them -- enforced by geometry, which no reward term can be
            # gamed around. The `strokes` metric goes with them, which also disposes of the
            # ~40% undercount NOTES.md M6 records rather than having to fix it.
            #
            # What replaces the budget is `no_slip_ratio`: the cap is only paid for turning to
            # the extent the *forearm* turned it. Measured on the scene, an 80 deg roll moves
            # the cap +74 deg net of the grip-ramp baseline -- a ratio of 0.93, which is what a
            # non-slipping grip should give. 1.0 is used rather than 0.93 so the term never
            # charges a *better*-than-measured grip: the cap may track the roll one-for-one, and
            # anything beyond that is the fingers spinning it, which is the M6 behaviour this
            # milestone exists to stop paying for.
            no_slip_ratio=1.0,
            # M6: where the opposition factor saturates, in degrees of azimuth between the two
            # grasp surfaces. 180 is wrong and the pregrasp fit measured why: at 180 one finger
            # has to reach across to the far side of the cap, where the bottle neck is, and
            # every grip/twist/release combination tried there hits the bottle (NOTES.md, M5).
            # The fit settled on 150 and the gait it produces sits at 150.2 deg at the median,
            # so the reward stops paying for more than the pose can deliver.
            opposition_target_deg=150.0,
            # M6: how close a pinch surface may come to the bottle before it is costed, in
            # metres. The reference gait's closest approach over 192 steps is 3.41 mm (p1
            # 3.75), so 2 mm is clear of it and the term is exactly zero on the trajectory it
            # must not punish. Inside the margin it ramps linearly and keeps going past
            # contact, since `cylinder_sdf` is signed: penetrating costs more than touching.
            # Bounded at 5 so a bad early step cannot dominate the return.
            bottle_margin=0.002,
            # M6: the one remaining clip, and it only bounds a *cost*. Backward rotation is
            # charged at the rate it happens so that losing ground is strictly worse than
            # idling; this stops a numerical lurch from dominating an episode. Far past
            # anything physical -- the reference gait's fastest forward step is 3.811 rad/s.
            backdrive_clip=5.0,
            # PRD weights, with the sign of each cost moved from its definition into the weight
            # so the logged `reward/*` metrics read as signed contributions.
            scales=config_dict.create(
                # M6: replaces `rotate`, which paid for cap rotation however it was produced.
                # `rotate` and `grasp` were *added*, so turning the cap badly and holding it
                # well were alternatives the policy could choose between, and the arithmetic
                # said to turn it badly: at 1200 steps an exploit grinding past the old clip
                # returned 60.0 against 54.9 for the scripted gait, and NOTES.md M5 records
                # every seed duly finding the exploit. `twist` is the product instead.
                #
                # M7: the second factor is no longer "rotation up to the grip's budget" but
                # "rotation the forearm roll accounts for". Both conditions now have to hold at
                # once -- the pinch must be closed *and* the hand must be the thing turning --
                # which is the whole of the human motion in one term. A policy that grips
                # perfectly and spins the cap with its fingers is paid nothing by it.
                twist=1.0,
                # M6: the pinch is the gate on `twist`, and it stays a dense term of its own so
                # that a policy starting from the pregrasp pose has a gradient to follow before
                # it has ever turned the cap -- which is the one real risk of gating rotation.
                # 0.5 -> 0.15 at M5: at the pregrasp pose the old `reach` was 99.9% of the
                # signal and parking for a whole episode returned 8.93 against ~21 for
                # succeeding. Parking now returns ~9 against ~35 for gaiting.
                grasp=0.15,
                # M6: rotation the pinch did not produce. `(1 - pinch) * |d_theta|`, so a cap
                # spun by a knuckle or flicked loose is charged for the whole of it while a
                # clean pinch is charged for almost none. With `twist` at 1.0 the two cancel at
                # `pinch = 0.2`: below that, turning the cap is worth less than leaving it
                # alone.
                slip=-0.25,
                # M7: rotation the *roll* did not produce -- a firm pinch that turns the cap by
                # moving the fingers rather than the forearm. `slip` cannot see this: the grip
                # is good, so `(1 - pinch)` is near zero. This is the M6 policy's whole strategy
                # (132 finger-driven regrips an episode at 273 deg/s) and without a term for it
                # the new joint can simply be ignored. Charged at twice `slip` because it is the
                # specific behaviour being designed out, and gated on `pinch` so that a cap
                # free-spinning after release is charged once, by `slip`, not twice.
                freewheel=-0.5,
                # M6: losing ground, charged at the rate it happens and unclipped up to
                # `backdrive_clip`. The old `rotate` term clipped its own downside at -0.5 while
                # paying up to +1.0, which made a cap shaken symmetrically back and forth worth
                # +0.5 a cycle. Nothing was ever observed doing it -- but the reward paid for it.
                backdrive=-1.0,
                # M6: a pinch surface inside the bottle's margin. Nothing costed this before,
                # and all three of the high-rotation recorded policies drive their grasp geoms
                # into the bottle. The 150 deg opposition target is the other half of the fix.
                bottle=-0.05,
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
                # M6: 10.0 -> 100.0. At 10.0 the one-time bonus was worth 0.5 of return
                # (10.0 * ctrl_dt) against ~31 for the dense term over a 1200-step episode --
                # 1.6%, which shapes nothing, and NOTES.md listed it as an open question. At
                # 100.0 it is 5.0, i.e. the last 60 deg before 4*pi is worth about two extra
                # strokes. Still a tie-breaker rather than the objective.
                success=100.0,
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
