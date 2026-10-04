"""M3 checks for `scripts/benchmark.py`.

The throughput numbers themselves need a GPU, so what is testable here is that the harness
reports honest numbers: buffers sized per `num_envs`, compilation excluded from the timing, and
overflow surfaced rather than swallowed.
"""

from __future__ import annotations

import os
import pathlib
import sys

import mujoco
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.benchmark import (  # noqa: E402
    BUFFER_OVERFLOWS, CTRL_DT, NACONMAX_PER_WORLD, NJMAX, fmt_x, load_actions,
)
from scripts.check_parity import build_model  # noqa: E402

slow = pytest.mark.skipif(
    not os.environ.get("HAND_SIM_SLOW"), reason="set HAND_SIM_SLOW=1 to run an MJX rollout"
)


@pytest.fixture(scope="module")
def model():
    return build_model(None, None, None)


def test_buffer_defaults_cover_the_measured_counts(model):
    """These came from check_parity.py --probe at M2 with 2x headroom. If the scene changes
    enough to move them, the benchmark would silently overflow."""
    assert NACONMAX_PER_WORLD >= 44, "naconmax must cover warp's ncollision (22), not ncon"
    assert NJMAX >= 150


def test_actions_match_the_actuator_count(model):
    for kind in ("gait", "random"):
        actions = load_actions(kind, model, 12, seed=0)
        assert actions.shape == (12, model.nu)
        assert np.all(np.isfinite(actions))


def test_random_actions_stay_in_ctrlrange(model):
    actions = load_actions("random", model, 64, seed=3)
    lo, hi = model.actuator_ctrlrange.T
    assert np.all(actions >= lo - 1e-9) and np.all(actions <= hi + 1e-9)


def test_realtime_formatting_does_not_round_away_slow_runs():
    """A sub-realtime run must not print as "0x"."""
    assert fmt_x(0.41) == "0.41"
    assert fmt_x(9.9) == "9.90"
    assert fmt_x(12345.0) == "12,345"


def test_buffer_overflow_set_excludes_solver_limits():
    """Solver and linesearch limits degrade accuracy but do not corrupt buffers, so they must not
    mark a throughput measurement invalid. EPA_HORIZON likewise."""
    assert "ITERATIONS" not in BUFFER_OVERFLOWS
    assert "LS_ITERATIONS" not in BUFFER_OVERFLOWS
    assert "EPA_HORIZON" not in BUFFER_OVERFLOWS
    assert {"BROADPHASE", "NARROWPHASE", "NEFC"} <= BUFFER_OVERFLOWS


def test_substeps_follow_the_scene_timestep(model):
    assert int(round(CTRL_DT / model.opt.timestep)) == 40, (
        "M2 set sim_dt=0.00125; update the M3 numbers in NOTES.md if this changes"
    )


@slow
def test_benchmark_runs_and_scales_buffers(model):
    from scripts.benchmark import run_one

    actions = load_actions("gait", model, 3, seed=0)
    n_substeps = int(round(CTRL_DT / model.opt.timestep))
    res = run_one(model, 2, actions, n_substeps, NACONMAX_PER_WORLD, NJMAX, warmup=1)
    assert res["naconmax"] == NACONMAX_PER_WORLD * 2, "naconmax is a total across worlds"
    assert res["finite"] and not res["buffer_overflow"]
    assert res["control_steps_per_s"] > 0 and res["compile_seconds"] > 0
    assert res["sim_steps_per_s"] == pytest.approx(
        res["control_steps_per_s"] * n_substeps, rel=1e-6
    )
