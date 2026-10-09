"""M5 acceptance checks for `scripts/train.py` and `scripts/eval.py`.

The milestone itself ("success rate above 80% over 1000 eval episodes") can only be met by a
real run on the H200. What is checkable here is everything that would make such a run wrong or
unresumable, and in this project those are mostly places where the installed Brax and Playground
do not behave the way their docstrings suggest:

* the PPO config is passed to `ppo.train` as keywords, so a stale key is a `TypeError` hours in
* `full_reset` has to be bound onto the wrapper, because `ppo.train` never passes it
* `restore_checkpoint_path` restores no step count, so `--resume` has to subtract one -- and
  Brax then names the next checkpoint *below* the one it resumed from, so the layout has to
  keep the global step recoverable or every later resume reloads the same stale weights
* `naconmax` is a total across worlds, so the eval env's cannot be the training env's
* Brax's eval metrics are episode *sums*, so the env's metrics have to be rates
* Brax cannot read back a checkpoint it just wrote, and JAX removed a symbol Brax calls

The end-to-end run (train a tiny policy, then evaluate it) is opt-in, because warp-cpu needs
minutes for it:

    HAND_SIM_SLOW=1 python3 -m pytest tests/test_train.py -q
"""

from __future__ import annotations

import inspect
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval as eval_script  # noqa: E402
import train as train_script  # noqa: E402
from capturn import checkpoints  # noqa: E402
from capturn import config as capturn_config  # noqa: E402

slow = pytest.mark.skipif(
    not os.environ.get("HAND_SIM_SLOW"),
    reason="set HAND_SIM_SLOW=1 to run a tiny training job on warp-cpu",
)


def args(*argv: str):
    return train_script.parse_args(list(argv))


# --------------------------------------------------------------------------------------
# the PPO config has to match the installed `ppo.train`
# --------------------------------------------------------------------------------------
def test_every_ppo_config_key_is_a_ppo_train_keyword():
    """`train.py` splats the config into `ppo.train`. A key the installed Brax does not take is
    a `TypeError` after the env has been built and the kernels compiled, which on the GPU box
    is minutes of a preemptible machine. Three keys are deliberately not keywords."""
    from brax.training.agents.ppo import train as ppo

    accepted = set(inspect.signature(ppo.train).parameters)
    handled_here = {"network_factory", "num_eval_envs", "full_reset"}
    keys = set(capturn_config.ppo_config())
    assert handled_here <= keys
    unknown = (keys - handled_here) - accepted
    assert not unknown, f"not `ppo.train` keywords: {sorted(unknown)}"
    # and the three it does handle itself are passed, not dropped
    assert {"network_factory", "num_eval_envs"} <= accepted


def test_ppo_train_ignores_full_reset_so_train_py_must_bind_it():
    """The reason `train.py` wraps the wrapper. If a later Brax starts forwarding `full_reset`,
    this fails and the binding can go away."""
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    assert "full_reset" in inspect.signature(wrapper.wrap_for_brax_training).parameters
    source = inspect.getsource(ppo._maybe_wrap_env)
    assert "full_reset" not in source, "Brax now forwards full_reset; drop the partial"


def test_bound_wrapper_actually_turns_full_reset_on():
    import functools

    from mujoco_playground import wrapper

    from capturn.env import CapTurn

    env = CapTurn(small_config(num_envs=1))
    wrap = functools.partial(wrapper.wrap_for_brax_training, full_reset=True)
    wrapped = wrap(env, episode_length=10, action_repeat=1, randomization_fn=None)
    assert wrapped._full_reset is True
    plain = wrapper.wrap_for_brax_training(env, episode_length=10, action_repeat=1)
    assert plain._full_reset is False, "the default is what Playground's train script gets"


def small_config(num_envs: int = 1):
    cfg = capturn_config.default_config()
    cfg.naconmax = capturn_config.NACONMAX_PER_WORLD * num_envs
    return cfg


# --------------------------------------------------------------------------------------
# the two dependency shims -- these tests are how the shims get deleted
# --------------------------------------------------------------------------------------
def test_jax_still_needs_the_device_put_replicated_shim():
    """If this fails, JAX or Brax has been fixed and `capturn/compat.py` can lose shim 1.

    Reads the deprecation registry rather than trying the attribute, so the result does not
    depend on whether some earlier test in this process already applied the shim. A `None`
    handler in that registry is JAX's marker for "raise", as opposed to "warn"."""
    import jax

    from brax.training.agents.ppo import train as ppo

    assert "jax.device_put_replicated(" in inspect.getsource(ppo.train), (
        "Brax no longer calls it; drop shim 1"
    )
    handler = jax._deprecations.get("device_put_replicated", ("", "not registered"))[1]
    assert handler is None, "JAX no longer raises on it; drop shim 1"


def test_the_shim_installs_the_symbol_and_is_idempotent():
    import jax

    from capturn import compat

    patched_now = compat.patch_jax_device_put_replicated()
    assert patched_now or callable(jax.device_put_replicated), "already applied this process"
    assert compat.patch_jax_device_put_replicated() is False, "idempotent"
    assert callable(jax.device_put_replicated)


def test_the_restored_symbol_does_what_brax_asks_of_it():
    import jax
    import jax.numpy as jp

    from capturn import compat

    compat.patch_jax_device_put_replicated()
    replicated = jax.device_put_replicated(jp.arange(3.0), jax.local_devices()[:1])
    assert replicated.shape == (1, 3), "a leading device axis, which is what pmap consumes"


def test_brax_still_cannot_read_back_its_own_none_initialiser():
    """If this fails, Brax's save/load asymmetry is fixed and shim 2 can go.

    `save` skips a `None` kernel initialiser; `load_config` looks it up anyway. Checked against
    the installed source rather than by writing a checkpoint, so it stays a fast test."""
    from brax.training import checkpoint as brax_checkpoint

    from capturn import compat

    save_src = inspect.getsource(brax_checkpoint.save)
    load_src = inspect.getsource(brax_checkpoint.load_config)
    assert "is None:\n      continue" in save_src, "save no longer skips None; recheck shim 2"
    assert "KERNEL_INITIALIZER[init_fn_name_]" in load_src
    assert "is None" not in load_src.split("KERNEL_INIT")[0].split("for init_fn_name")[-1], (
        "load_config now guards None; drop shim 2"
    )
    assert compat.patch_brax_kernel_initializer_none() is True
    assert compat.patch_brax_kernel_initializer_none() is False, "idempotent"


def test_the_default_network_factory_is_the_one_that_trips_shim_2():
    """The `None` comes from `mean_kernel_init_fn`, and the PPO config does not override it.
    Binding it to a registered initialiser would dodge shim 2 but change initialisation."""
    from brax.training.agents.ppo import networks as ppo_networks

    params = inspect.signature(ppo_networks.make_ppo_networks).parameters
    assert params["mean_kernel_init_fn"].default is None
    assert "mean_kernel_init_fn" not in capturn_config.ppo_config().network_factory


# --------------------------------------------------------------------------------------
# config assembly
# --------------------------------------------------------------------------------------
def test_naconmax_follows_each_env_count_separately():
    """`naconmax` is a total across worlds (M2). The eval env runs a different number of them,
    so sharing one config would either waste buffer or overflow it."""
    env_cfg, eval_cfg, ppo_cfg = train_script.build_configs(
        args("--num-envs", "2048", "--num-eval-envs", "256")
    )
    per_world = capturn_config.NACONMAX_PER_WORLD
    assert env_cfg.naconmax == per_world * 2048
    assert eval_cfg.naconmax == per_world * 256
    assert env_cfg.njmax == eval_cfg.njmax == capturn_config.NJMAX  # per world, does not scale


def test_nominal_phase_leaves_every_dr_hook_off():
    env_cfg, eval_cfg, _ = train_script.build_configs(args())
    for cfg in (env_cfg, eval_cfg):
        assert cfg.noise_config.level == 0.0
        assert not cfg.action_delay.enabled


def test_dr_phase_scales_the_env_level_hooks():
    env_cfg, _, _ = train_script.build_configs(args("--phase", "dr", "--dr-scale", "0.25"))
    assert env_cfg.noise_config.level == 0.25
    assert env_cfg.action_delay.enabled
    # the layout is unchanged, which is what makes the nominal checkpoint loadable
    assert env_cfg.action_delay.max_steps == capturn_config.default_config().action_delay.max_steps


def test_overrides_reach_nested_fields_and_reject_typos():
    env_cfg, _, ppo_cfg = train_script.build_configs(
        args("--env-override", '{"reward_config.scales.grasp": 0.15}',
             "--ppo-override", '{"entropy_cost": 0.02}')
    )
    assert env_cfg.reward_config.scales.grasp == 0.15
    assert ppo_cfg.entropy_cost == 0.02
    with pytest.raises(SystemExit):
        train_script.build_configs(args("--env-override", '{"reward_config.scales.rotat": 1.0}'))


def test_mismatched_episode_length_is_caught_before_training_starts():
    with pytest.raises(SystemExit, match="episode_length"):
        train_script.build_configs(args("--ppo-override", '{"episode_length": 200}'))


def test_num_envs_divisibility_is_caught_before_training_starts():
    """Brax asserts `batch_size * num_minibatches % num_envs == 0`. Bare asserts do not say
    which number to change."""
    with pytest.raises(SystemExit, match="divisible"):
        train_script.build_configs(args("--num-envs", "3000"))
    train_script.build_configs(args("--num-envs", "4096"))  # 256*32 = 8192, divisible


def test_checkpoint_cadence_is_a_few_minutes_at_the_measured_throughput():
    """PRD: 'Checkpoint every few minutes'. Brax writes one per eval, so this is `num_evals`."""
    ppo_cfg = capturn_config.ppo_config()
    gap_steps = ppo_cfg.num_timesteps / (ppo_cfg.num_evals - 1)
    minutes = gap_steps / train_script.MEASURED_CONTROL_STEPS_PER_S / 60
    assert 1.0 <= minutes <= 6.0, f"{minutes:.1f} min between checkpoints"

    assert train_script.derive_num_evals(150_000_000, 4.0) == pytest.approx(21, abs=2)
    assert train_script.derive_num_evals(1_000, 4.0) == 2  # never fewer than two
    with pytest.raises(SystemExit):
        train_script.derive_num_evals(1_000, 0.0)


def test_eval_is_deterministic_so_it_matches_eval_py():
    assert capturn_config.ppo_config().deterministic_eval is True


def test_run_name_is_deterministic_so_resume_can_find_the_run():
    assert train_script.run_name_for(args()) == "capturn-nominal-s1"
    assert train_script.run_name_for(args("--seed", "7")) == "capturn-nominal-s7"
    assert train_script.run_name_for(
        args("--phase", "dr", "--dr-scale", "0.5")
    ) == "capturn-dr-dr0.5-s1"
    assert train_script.run_name_for(args("--run-name", "mine")) == "mine"


def test_smoke_config_is_small_enough_to_be_a_cpu_pipeline_check():
    _, _, ppo_cfg = train_script.build_configs(args("--smoke"))
    assert ppo_cfg.num_timesteps <= 1_000 and ppo_cfg.num_envs <= 16
    assert ppo_cfg.batch_size * ppo_cfg.num_minibatches % ppo_cfg.num_envs == 0


# --------------------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------------------
def test_resolve_checkpoint_accepts_a_step_dir_an_attempt_dir_or_the_whole_tree(tmp_path):
    ckpts = tmp_path / "checkpoints"
    attempt = checkpoints.attempt_dir(ckpts, 0)
    for step in (327_680, 1_310_720):
        (attempt / f"{step:012d}").mkdir(parents=True)
    (attempt / "config.json").write_text("{}")

    path, step = train_script.resolve_checkpoint(str(ckpts))
    assert step == 1_310_720, "the latest, by step not by name order"
    assert path.name == "000001310720"
    assert train_script.resolve_checkpoint(str(attempt)) == (path, 1_310_720)
    assert train_script.resolve_checkpoint(str(path)) == (path, 1_310_720)
    with pytest.raises(SystemExit):
        train_script.resolve_checkpoint(str(tmp_path / "nope"))
    with pytest.raises(SystemExit, match="no step-named"):
        train_script.resolve_checkpoint(str(tmp_path))


def test_a_resumed_run_does_not_hide_its_newest_checkpoint_behind_a_stale_one(tmp_path):
    """The bug the attempt-directory layout exists to prevent.

    Brax names a checkpoint after `env_steps`, which `restore_checkpoint_path` does not restore
    -- so a run resumed at 6,553,600 steps writes its *next* checkpoint as `000000327680`. In
    one flat directory the stale 6.5M checkpoint stays the largest name forever, so every later
    `--resume` would reload it and silently throw away all the work since."""
    ckpts = tmp_path / "checkpoints"
    first = checkpoints.attempt_dir(ckpts, 0)
    (first / f"{6_553_600:012d}").mkdir(parents=True)
    resumed = checkpoints.attempt_dir(ckpts, 6_553_600)
    (resumed / f"{327_680:012d}").mkdir(parents=True)

    latest = checkpoints.find_latest(ckpts)
    assert latest.step == 6_881_280, "offset + name, not the largest name"
    assert latest.path.parent.name == resumed.name
    assert [c.step for c in checkpoints.find_all(ckpts)] == [6_553_600, 6_881_280]


def test_a_flat_checkpoint_layout_still_reads(tmp_path):
    """Checkpoints written before this layout existed, or by Playground's own train script."""
    (tmp_path / f"{327_680:012d}").mkdir()
    latest = checkpoints.find_latest(tmp_path)
    assert latest.step == 327_680 and latest.path.name == "000000327680"
    assert checkpoints.find_latest(tmp_path / "missing") is None


def test_checkpoints_sort_numerically_not_lexically(tmp_path):
    """Brax zero-pads to 12 digits so lexical order happens to work -- but only until a run
    passes 1e12 steps, and nothing here should depend on the padding."""
    for name in ("9", "10", "100"):
        (tmp_path / name).mkdir()
    assert [c.step for c in checkpoints.find_all(tmp_path)] == [9, 10, 100]


def test_eval_resolves_the_same_layout_and_the_same_global_step(tmp_path):
    """`eval.py` reports the step it evaluated and names its `.npz` files after it, so it has
    to agree with `train.py` about what step a resumed checkpoint is at."""
    ckpts = tmp_path / "checkpoints"
    (checkpoints.attempt_dir(ckpts, 6_553_600) / f"{327_680:012d}").mkdir(parents=True)
    ckpt, step = eval_script.resolve_checkpoint(eval_script.parse_args([str(ckpts)]))
    assert step == 6_881_280
    assert ckpt == train_script.resolve_checkpoint(str(ckpts)).path
    with pytest.raises(SystemExit, match="checkpoint path or --wandb-artifact"):
        eval_script.resolve_checkpoint(eval_script.parse_args([]))


def test_the_wandb_id_survives_the_default_dr_run_name():
    """The id is derived from the run name because `--resume` has to rejoin the run without
    being told an id. `capturn-dr-dr0.5-s1` has a dot in it."""
    assert train_script.wandb_id("capturn-nominal-s1") == "capturn-nominal-s1"
    assert train_script.wandb_id("capturn-dr-dr0.5-s1") == "capturn-dr-dr0.5-s1"
    assert train_script.wandb_id("a/b c:d") == "a-b-c-d"
    assert train_script.artifact_name("capturn-nominal-s1") == "ckpt-capturn-nominal-s1"


def test_restore_brings_back_no_step_count_which_is_why_resume_subtracts():
    """The fact that makes `--resume` nontrivial: Brax restores the normalizer and the policy
    and value params, and nothing else. `env_steps` restarts at zero."""
    from brax.training.agents.ppo import train as ppo

    source = inspect.getsource(ppo.train)
    restore = source[source.index("if restore_checkpoint_path is not None:"):][:400]
    assert "normalizer_params=params[0]" in restore
    assert "env_steps" not in restore, "Brax now restores env_steps; drop the subtraction"
    assert "optimizer_state" not in restore, "Brax now restores Adam; update the docstring"


# --------------------------------------------------------------------------------------
# eval: the metrics have to survive Brax's aggregation, and the exploit checks have to work
# --------------------------------------------------------------------------------------
def test_brax_sums_eval_metrics_so_the_env_reports_rates():
    """`EvalWrapper` accumulates `metric + metric * active` over the episode and only divides
    by the length for names ending in `_per_step`. A metric defined as a level logs as the area
    under its own curve, which is why `cap_rotation` is `d_theta` (NOTES.md, M5)."""
    import jax

    from brax.envs.wrappers import training as brax_training

    from capturn.env import CapTurn

    assert "a + b * state_metrics.active_episodes" in inspect.getsource(
        brax_training.EvalWrapper.step
    )

    env = CapTurn(small_config())
    metrics = jax.eval_shape(env.reset, jax.random.PRNGKey(0)).metrics
    assert "cap_rotation" in metrics and "cap_rotation_abs" in metrics
    assert "cap_angvel_per_step" in metrics, "Brax's suffix for 'divide by episode length'"
    assert "cap_angvel" not in metrics, "a bare name would log as a sum of velocities"
    assert "success" in metrics


def test_regrips_need_a_release_and_ignore_chatter_at_the_threshold():
    contact, release = 0.010, 0.020
    hold, off, edge = 0.004, 0.030, 0.0101  # `edge` is inside the hysteresis band

    def count(trace):
        gaps = np.array(trace, dtype=float).reshape(1, -1, 1)
        alive = np.ones((1, gaps.shape[1]))
        return int(eval_script.count_regrips(gaps, alive, contact, release)[0, 0])

    assert count([hold] * 10) == 0, "never let go"
    assert count([hold, hold, off, off, hold, hold]) == 1, "one release and regrip"
    assert count([hold, off, hold, off, hold]) == 2
    assert count([hold, edge] * 20) == 0, "chatter inside the band is not a regrip"
    assert count([off] * 5 + [hold] * 5) == 0, "arriving for the first time is not a regrip"


def test_regrips_stop_being_counted_after_the_episode_terminates():
    gaps = np.array([0.004, 0.030, 0.004, 0.030, 0.004], dtype=float).reshape(1, -1, 1)
    alive = np.array([[1.0, 1.0, 0.0, 0.0, 0.0]])
    assert int(eval_script.count_regrips(gaps, alive, 0.010, 0.020)[0, 0]) == 0


def test_success_counts_a_policy_that_overshoots_and_slips_back():
    """Success is 'ever reached 4*pi', the same upward crossing the reward bonus fires on, not
    'ended past 4*pi'. A cap that is turned twice and then slips back a few degrees succeeded."""
    threshold = float(capturn_config.default_config().success_rotation)
    steps = 20
    forward = np.full(steps, threshold / (steps - 2))
    slipped = forward.copy()
    slipped[-1] = -0.5  # ends below the threshold, having crossed it

    traj = {
        "d_theta": np.stack([forward, slipped]),
        "alive": np.ones((2, steps)),
        "angvel": np.zeros((2, steps)),
        "gaps": np.full((2, steps, 3), 0.005),
        "action": np.zeros((2, steps, 20)),
        "qpos": np.zeros((2, steps + 1, 25)),
    }
    from capturn.env import CapTurn

    env = CapTurn(small_config())
    summary = eval_script.summarize(env, traj, eval_script.parse_args([]))
    assert summary["success_rate"] == 1.0
    assert summary["net_over_gross"] > 0.9


def test_net_over_gross_separates_a_gait_from_shaking_the_cap():
    steps = 40
    turning = np.full(steps, 0.05)
    shaking = np.resize([0.05, -0.05], steps)
    traj = {
        "d_theta": np.stack([turning, shaking]),
        "alive": np.ones((2, steps)),
        "angvel": np.zeros((2, steps)),
        "gaps": np.full((2, steps, 3), 0.005),
        "action": np.zeros((2, steps, 20)),
        "qpos": np.zeros((2, steps + 1, 25)),
    }
    from capturn.env import CapTurn

    one = eval_script.summarize(
        CapTurn(small_config()), {k: v[:1] for k, v in traj.items()}, eval_script.parse_args([])
    )
    two = eval_script.summarize(
        CapTurn(small_config()), {k: v[1:] for k, v in traj.items()}, eval_script.parse_args([])
    )
    assert one["net_over_gross"] == pytest.approx(1.0)
    assert abs(two["net_over_gross"]) < 0.05


def test_overflow_flags_separate_a_busy_solver_from_a_corrupted_buffer():
    """Warp reports overflow as a bitmask on the Data instead of raising, and prints its
    warnings from inside kernels where Python cannot see them. A buffer overflow means the
    physics being evaluated is not the physics M2 verified; a solver flag only means it worked
    hard. `eval.py` has to tell them apart because the first invalidates the run."""
    from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType

    assert eval_script.overflow_flags(0) == []
    both = OverflowType.NEFC.value | OverflowType.ITERATIONS.value
    flags = eval_script.overflow_flags(both)
    assert set(flags) == {"NEFC", "ITERATIONS"}
    assert eval_script.BUFFER_OVERFLOWS.intersection(flags) == {"NEFC"}
    assert "ITERATIONS" not in eval_script.BUFFER_OVERFLOWS
    # EPA_HORIZON is a known, non-fatal warp limit (M2): a compile-time 24-entry horizon that
    # the model cannot configure. It must not be read as a buffer overflow.
    assert "EPA_HORIZON" not in eval_script.BUFFER_OVERFLOWS


def test_the_scene_has_the_sensors_eval_slices_by_hand():
    from capturn.env import CapTurn
    from capturn.scene import GAIT_TIP_SENSORS

    model = CapTurn(small_config()).mj_model
    for name in ("cap_angvel", "cap_angle", *GAIT_TIP_SENSORS):
        assert eval_script.sensor_slice(model, name).stop <= model.nsensordata
    with pytest.raises(SystemExit):
        eval_script.sensor_slice(model, "not_a_sensor")


# --------------------------------------------------------------------------------------
# end to end, on warp-cpu
# --------------------------------------------------------------------------------------
@slow
def test_smoke_train_then_eval_produces_a_loadable_policy(tmp_path):
    """The whole M5 pipeline on one CPU: train a few steps, checkpoint, resume from that
    checkpoint, then load the newest one back, roll it out and write a trajectory `render.py`
    can read. Nothing here learns anything -- it checks that each handoff works, and the
    handoffs are where this milestone's bugs were. About 6 minutes on an M1."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    train = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--smoke",
         "--logdir", str(tmp_path), "--no-wandb", "--run-name", "smoke"],
        capture_output=True, text=True, env=env, timeout=2400,
    )
    assert train.returncode == 0, train.stdout[-4000:] + train.stderr[-4000:]

    ckpts = tmp_path / "smoke" / "checkpoints"
    latest = checkpoints.find_latest(ckpts)
    assert latest is not None, f"no checkpoint written: {train.stdout[-2000:]}"
    assert latest.path.parent.name == checkpoints.attempt_dir(ckpts, 0).name
    assert (latest.path / "ppo_network_config.json").exists(), "eval.py rebuilds the net from this"
    assert json.loads((tmp_path / "smoke" / "env_config.json").read_text())["sim_dt"] == 0.00125

    # --- resume. The budget is already spent, so ask for twice it and check the second
    # attempt starts where the first stopped rather than from scratch or from a stale name.
    first_step = latest.step
    resumed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--smoke",
         "--num-timesteps", str(first_step * 2), "--resume",
         "--logdir", str(tmp_path), "--no-wandb", "--run-name", "smoke"],
        capture_output=True, text=True, env=env, timeout=2400,
    )
    assert resumed.returncode == 0, resumed.stdout[-4000:] + resumed.stderr[-4000:]
    assert "resuming from local checkpoint" in resumed.stdout, resumed.stdout[-2000:]
    assert f"budget: {first_step:,} steps from step {first_step:,}" in resumed.stdout, (
        "the restored step count was not subtracted from the budget\n"
        + resumed.stdout[-2000:]
    )

    after = checkpoints.find_latest(ckpts)
    assert after.step > first_step, (
        f"the newest checkpoint is still the one that was resumed from ({after.step})"
    )
    assert after.path.parent.name == checkpoints.attempt_dir(ckpts, first_step).name
    assert int(after.path.name) <= first_step, (
        "Brax should be naming the resumed attempt's checkpoints at or below the resume point, "
        "because it restores no step count -- if it is not, the attempt-directory layout may "
        "no longer be needed"
    )
    assert latest.path.name == after.path.name, (
        "the two checkpoints have the same directory name, so in one flat directory the "
        "resumed one would have overwritten the one it resumed from"
    )

    # --- eval the newest, which is only findable because of the layout
    out = tmp_path / "eval.json"
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval.py"), str(ckpts),
         "--episodes", "4", "--save", "1",
         "--save-prefix", str(tmp_path / "ep"), "--out", str(out)],
        capture_output=True, text=True, env=env, timeout=2400,
    )
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]

    summary = json.loads(out.read_text())
    assert summary["step"] == after.step, "eval evaluated a different checkpoint than the newest"
    assert summary["episodes"] == 4
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert np.isfinite(summary["net_rotation_deg"]["mean"])
    assert summary["worst_penetration_mm"] <= 0.0

    npz = np.load(tmp_path / "ep_ep00.npz")
    assert npz["qpos"].shape[1] == 25 and len(npz["qpos"]) >= 2
    assert float(npz["ctrl_dt"]) == 0.05, "render.py reads the playback rate from here"
    assert len(npz["cap_angle"]) == len(npz["qpos"])
