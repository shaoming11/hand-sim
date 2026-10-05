"""M4 acceptance checks for `capturn/env.py`.

PRD M4: "reset and step are jittable, obs shapes are fixed, reward terms are finite, random
actions run for 1000 steps without NaN."

`impl='jax'` cannot load this scene (no cylinder/mesh collisions), so every check that actually
steps physics runs on warp-cpu at roughly 0.2 s per control step. Those are opt-in:

    HAND_SIM_SLOW=1 python3 -m pytest tests/test_env.py -q

The default suite still covers the layout, the config and the geometry, because those are what
a later scene change would silently break.
"""

from __future__ import annotations

import os
import pathlib
import sys

import jax
import jax.numpy as jp
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from capturn import config as capturn_config  # noqa: E402
from capturn.env import (  # noqa: E402
    DR_PARAM_SLOTS, FRAME_SIZE, N_ACTUATORS, N_HAND_JOINTS, PRIVILEGED_EXTRA, CapTurn,
)
from capturn.scene import ALL_TIPS, GAIT_TIPS, cap_dimensions, surface_distance  # noqa: E402

slow = pytest.mark.skipif(
    not os.environ.get("HAND_SIM_SLOW"),
    reason="set HAND_SIM_SLOW=1 to step physics on warp-cpu",
)


def make_env(num_envs: int = 1, **overrides) -> CapTurn:
    """An env with its contact buffer sized for `num_envs` instead of the H100's 8192."""
    config = capturn_config.default_config()
    config.naconmax = capturn_config.NACONMAX_PER_WORLD * num_envs
    for key, value in overrides.items():
        if isinstance(value, dict):
            getattr(config, key).update(value)
        else:
            setattr(config, key, value)
    return CapTurn(config)


@pytest.fixture(scope="module")
def env():
    return make_env()


# --------------------------------------------------------------------------------------
# layout -- traced, not executed
# --------------------------------------------------------------------------------------
def test_action_size_is_the_actuator_count(env):
    assert env.action_size == N_ACTUATORS == env.mj_model.nu


def test_substeps_follow_the_measured_timestep(env):
    """M2 put sim_dt at 0.00125. If the config drifts back to the PRD's 0.005 the sim is no
    longer converged and every M3 throughput number is wrong too."""
    assert env.sim_dt == 0.00125
    assert env.n_substeps == 40
    assert env.mj_model.opt.timestep == env.sim_dt


def test_observation_shapes_are_fixed_and_match_the_documented_layout(env):
    sizes = env.observation_size
    assert sizes == {
        "state": (env._config.history_len * FRAME_SIZE,),
        "privileged_state": (env._config.history_len * FRAME_SIZE + PRIVILEGED_EXTRA,),
    }
    # the frame is the PRD's actor observation, nothing more
    assert FRAME_SIZE == N_HAND_JOINTS + 2 * N_ACTUATORS + 3 + 3
    assert sizes["state"] == (210,) and sizes["privileged_state"] == (283,)


def test_privileged_state_contains_the_actor_history_verbatim(env):
    """The DR phase fine-tunes the nominal checkpoint, so both heads' input layouts are frozen
    from day one. The critic's prefix being exactly the actor observation is what makes the
    layout checkable at all."""
    state = jax.eval_shape(env.reset, jax.random.PRNGKey(0))
    actor = state.obs["state"].shape[0]
    assert state.obs["privileged_state"].shape[0] - actor == PRIVILEGED_EXTRA


def test_metrics_cover_every_reward_term_and_the_task_measures(env):
    """The PRD's metric list -- cumulative rotation, mean angular velocity, success flag, each
    reward term -- as names that mean the right thing after Brax's eval aggregation sums them
    over the episode. M5 found that a metric defined as a *level* logs as the area under its
    own curve; see NOTES.md, M5."""
    state = jax.eval_shape(env.reset, jax.random.PRNGKey(0))
    scales = env._config.reward_config.scales
    assert {f"reward/{k}" for k in scales} <= set(state.metrics)
    assert {
        "cap_rotation",  # d_theta -> sums to the episode's net rotation
        "cap_rotation_abs",  # |d_theta| -> sums to the gross rotation
        "cap_angvel_per_step",  # Brax divides `_per_step` names by the episode length
        "success",  # the 4*pi crossing -> sums to the success flag
    } <= set(state.metrics)
    assert set(scales) == {
        "rotate", "reach", "action_rate", "torques", "joint_vel", "success"
    }, "the PRD's six reward terms"


def test_reset_and_step_are_jittable(env):
    """Traced, not run: `eval_shape` fails outright on anything not jittable."""
    state = jax.eval_shape(env.reset, jax.random.PRNGKey(0))
    stepped = jax.eval_shape(env.step, state, jp.zeros(env.action_size))
    assert stepped.reward.shape == () and stepped.done.shape == ()
    assert stepped.obs["state"].shape == state.obs["state"].shape


# --------------------------------------------------------------------------------------
# config: the DR hooks have to exist and be off
# --------------------------------------------------------------------------------------
def test_nominal_config_switches_every_dr_hook_off_without_removing_it(env):
    config = env._config
    assert config.noise_config.level == 0.0, "nominal phase is noiseless"
    assert config.noise_config.scales.joint_pos > 0 and config.noise_config.scales.cap_pos > 0
    assert not config.action_delay.enabled
    assert config.action_delay.max_steps == 2  # PRD: 0 to 2 control steps
    assert DR_PARAM_SLOTS > 0


def test_action_delay_buffer_is_sized_whether_or_not_delay_is_enabled():
    """The buffer shape is part of the observation-adjacent state the checkpoint sees, so it
    must not change between the nominal and DR phases."""
    off = jax.eval_shape(make_env().reset, jax.random.PRNGKey(0))
    on = jax.eval_shape(
        make_env(action_delay={"enabled": True}).reset, jax.random.PRNGKey(0)
    )
    assert off.info["action_buffer"].shape == on.info["action_buffer"].shape == (3, N_ACTUATORS)
    assert off.obs["state"].shape == on.obs["state"].shape


def test_buffer_sizing_matches_the_measured_counts():
    assert capturn_config.NACONMAX_PER_WORLD == 44  # M2, ncollision with 2x headroom
    assert capturn_config.NJMAX == 150
    assert capturn_config.NUM_ENVS == 8192  # M3: 16384 fails to allocate
    assert capturn_config.default_config().naconmax == 44 * 8192, "naconmax is a total"


def test_ppo_config_asks_for_asymmetric_actor_critic():
    ppo = capturn_config.ppo_config()
    assert ppo.network_factory.policy_obs_key == "state"
    assert ppo.network_factory.value_obs_key == "privileged_state"
    assert ppo.episode_length == 400
    assert ppo.full_reset, "see NOTES.md M4: info and reset randomisation both depend on this"


# --------------------------------------------------------------------------------------
# cap geometry
# --------------------------------------------------------------------------------------
def test_surface_distance_measures_the_cap_wall(env):
    radius, half_h = cap_dimensions(env.mj_model)
    assert radius == pytest.approx(0.016, abs=1e-4)
    assert half_h == pytest.approx(0.008, abs=1e-4)

    on_wall = surface_distance(np.array([radius, 0.0, 0.0]), radius, half_h)
    assert on_wall == pytest.approx(0.0, abs=1e-12)
    # inside the rim, only the radial gap counts
    assert surface_distance(np.array([radius + 0.005, 0.0, 0.004]), radius, half_h) == (
        pytest.approx(0.005)
    )
    # above the rim, both gaps count
    assert surface_distance(np.array([radius, 0.0, half_h + 0.003]), radius, half_h) == (
        pytest.approx(0.003)
    )
    assert surface_distance(np.zeros((4, 5, 3)), radius, half_h).shape == (4, 5)


def test_the_scene_has_a_sensor_for_every_fingertip(env):
    model = env.mj_model
    assert len(GAIT_TIPS) == 3 and len(ALL_TIPS) == 5
    for tip in ALL_TIPS:
        model.site(tip)  # raises if missing
        short = tip.removeprefix("rh_").removesuffix("_tip")
        assert model.sensor(f"{short}_tip_position").id >= 0


# --------------------------------------------------------------------------------------
# physics -- opt-in
# --------------------------------------------------------------------------------------
@slow
def test_reset_observes_real_sensor_values(env):
    """`mjx.make_data` leaves `sensordata` zeroed, so the first frame is only real if `reset`
    runs a forward pass. Without it the cap's position in the very first observation is 0,0,0."""
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    frame = state.obs["state"][:FRAME_SIZE]
    cap_pos = frame[N_HAND_JOINTS + 2 * N_ACTUATORS: N_HAND_JOINTS + 2 * N_ACTUATORS + 3]
    assert np.linalg.norm(cap_pos) > 1e-3, "cap position in the palm frame reads as the origin"
    assert jp.isfinite(state.obs["privileged_state"]).all()


@slow
def test_the_actor_cannot_see_the_cap_angle(env):
    """The cap is rotationally symmetric and the policy could not measure its angle on
    hardware, so turning the cap must not change the actor observation."""
    reset = jax.jit(env.reset)
    a = reset(jax.random.PRNGKey(0))
    b = reset(jax.random.PRNGKey(1))
    # same hand jitter would be ideal, but any two resets differ only in hand pose and cap
    # angle; compare the cap-derived slice, which must be identical up to the hand pose.
    assert float(abs(a.info["cap_angle_init"] - b.info["cap_angle_init"])) > 0.1

    from mujoco import mjx
    rotated = mjx.forward(
        env.mjx_model, a.data.replace(qpos=a.data.qpos.at[env._cap_qid].add(1.0))
    )
    base_frame = env._get_obs(a.data, dict(a.info), jp.zeros_like(a.obs["state"]))["state"]
    spun_frame = env._get_obs(rotated, dict(a.info), jp.zeros_like(a.obs["state"]))["state"]
    np.testing.assert_allclose(base_frame, spun_frame, atol=1e-6)


@slow
def test_random_actions_run_for_1000_steps_without_nan(env):
    """The PRD's M4 criterion, verbatim."""
    steps = 1000

    def rollout(rng):
        state = env.reset(rng)

        def one(carry, _):
            state, rng = carry
            rng, key = jax.random.split(rng)
            action = jax.random.uniform(key, (env.action_size,), minval=-1.0, maxval=1.0)
            state = env.step(state, action)
            terms = jp.stack([state.metrics[f"reward/{k}"]
                              for k in sorted(env._config.reward_config.scales)])
            return (state, rng), (state.reward, state.done, terms,
                                  state.metrics["cap_rotation"], state.data.ctrl)

        return jax.lax.scan(one, (state, rng), (), steps)[1]

    reward, done, terms, rotation, ctrl = jax.jit(rollout)(jax.random.PRNGKey(0))
    assert reward.shape == (steps,)
    assert jp.isfinite(reward).all(), "reward went non-finite"
    assert jp.isfinite(terms).all(), "a reward term went non-finite"
    assert jp.isfinite(rotation).all()
    assert jp.isfinite(done).all() and set(np.unique(np.asarray(done))) <= {0.0, 1.0}
    # the integrator must never leave the actuator range, however long the rollout
    lo, hi = env.mj_model.actuator_ctrlrange.T
    assert np.asarray(ctrl).min() >= lo.min() - 1e-5
    assert (np.asarray(ctrl) <= hi + 1e-5).all() and (np.asarray(ctrl) >= lo - 1e-5).all()


@slow
def test_vmap_over_envs(env):
    rngs = jax.random.split(jax.random.PRNGKey(0), 2)
    env2 = make_env(num_envs=2)
    states = jax.jit(jax.vmap(env2.reset))(rngs)
    actions = jp.zeros((2, env2.action_size))
    states = jax.jit(jax.vmap(env2.step))(states, actions)
    assert states.reward.shape == (2,)
    assert states.obs["state"].shape == (2, 210)
    assert jp.isfinite(states.obs["privileged_state"]).all()
    # different rngs must give different episodes
    assert float(abs(states.info["cap_angle_init"][0] - states.info["cap_angle_init"][1])) > 1e-3


@slow
def test_episode_boundary_does_not_leak_rotation_or_targets():
    """The regression test for the auto-reset design.

    `BraxAutoResetWrapper(full_reset=False)` restores `data` but leaves `info` untouched. Any
    episode-evolving quantity cached in `info` -- the integrator's target, a cumulative rotation
    counter, a success latch -- would survive into the next episode and corrupt it.

    Here nothing does, and the proof is strong: driven by a constant action, the second episode
    is a numerical repeat of the first. Cached state of any kind would show up as a divergence.
    """
    from mujoco_playground._src import wrapper

    episode_length = 3
    env = make_env(num_envs=2, episode_length=episode_length)
    wrapped = wrapper.wrap_for_brax_training(
        env, episode_length=episode_length, action_repeat=1, full_reset=False
    )
    step = jax.jit(wrapped.step)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 2))

    push = 0.5  # a steady squeeze, so the targets really do move away from the keyframe
    action = jp.full((2, env.action_size), push)
    rotations, dones, drift = [], [], []
    for _ in range(2 * episode_length):
        state = step(state, action)
        rotations.append(np.asarray(state.metrics["cap_rotation"]))
        dones.append(np.asarray(state.done))
        drift.append(np.abs(np.asarray(state.data.ctrl) - np.asarray(env._init_ctrl)).max())

    last = episode_length - 1  # the wrapper restores `data` on the step it reports done
    assert np.array(dones)[last].all() and not np.array(dones)[:last].any()

    # the integrator's target is back at the keyframe on the boundary step, and the next episode
    # integrates one step away from *that* rather than from where the last episode ended
    assert drift[last] == pytest.approx(0.0, abs=1e-6)
    assert drift[last - 1] > drift[0] > 0.0
    assert drift[last + 1] == pytest.approx(env._config.action_scale * push, abs=1e-5)

    # and the whole episode repeats
    np.testing.assert_allclose(rotations[:episode_length], rotations[episode_length:], atol=1e-9)


@slow
def test_full_reset_gives_each_episode_a_fresh_start_state():
    """Why `ppo_config()` asks for `full_reset=True`.

    With it off, `BraxAutoResetWrapper` restarts every env from the one state cached at the
    first reset -- so the PRD's reset randomisation (hand jitter, random cap angle) only ever
    produces `num_envs` start states for the whole run, and under a deterministic policy each
    env replays a single episode forever. That is what the test above measures as a feature; it
    is not what you want to train on.
    """
    from mujoco_playground._src import wrapper

    episode_length = 3
    seen = {}
    for full_reset in (False, True):
        env = make_env(num_envs=2, episode_length=episode_length)
        wrapped = wrapper.wrap_for_brax_training(
            env, episode_length=episode_length, action_repeat=1, full_reset=full_reset
        )
        step = jax.jit(wrapped.step)
        state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
        action = jp.zeros((2, env.action_size))
        angles = []
        for _ in range(2 * episode_length):
            state = step(state, action)
            angles.append(np.asarray(state.info["cap_angle_init"]))
        seen[full_reset] = np.array(angles)

    assert len(np.unique(seen[False])) == 2, "start angles frozen at the first reset"
    assert len(np.unique(seen[True])) > 2, "each episode should draw a new start angle"
