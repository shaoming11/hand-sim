"""M2 acceptance checks: is the simulation the same one on CPU and on MJX-Warp?

The MJX-Warp rollout is far too slow on a CPU-only Mac to live in the default suite (minutes,
not seconds), so it is opt-in:

    HAND_SIM_SLOW=1 python3 -m pytest tests/test_parity.py -q

The CPU-side checks run always, and they are the ones that guard the M2 finding that actually
changed the scene: `sim_dt` had to drop from the PRD's 0.005 to 0.00125 to be converged.
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
from scripts.check_parity import build_model, cpu_rollout  # noqa: E402
from scripts.squeeze_twist import CTRL_DT, OpenLoopGait  # noqa: E402

TOLERANCE = 0.10  # PRD: "about 10%"
TRAJECTORY = ROOT / "artifacts" / "squeeze_twist.npz"

slow = pytest.mark.skipif(
    not os.environ.get("HAND_SIM_SLOW"),
    reason="set HAND_SIM_SLOW=1 to run the MJX-Warp rollout",
)


@pytest.fixture(scope="module")
def trajectory():
    if not TRAJECTORY.exists():
        pytest.skip(f"{TRAJECTORY} missing -- run scripts/squeeze_twist.py")
    npz = np.load(TRAJECTORY)
    return npz["ctrl"], int(npz["n_substeps"])


@pytest.fixture(scope="module")
def reference(trajectory):
    ctrl, n_substeps = trajectory
    return cpu_rollout(build_model(None, None, None), ctrl, n_substeps)


def test_trajectory_substeps_match_the_scene(trajectory):
    """The saved trajectory must have been generated at the scene's current timestep."""
    _, n_substeps = trajectory
    model = build_model(None, None, None)
    expected = int(round(CTRL_DT / model.opt.timestep))
    assert n_substeps == expected, (
        f"{TRAJECTORY.name} was generated with {n_substeps} substeps but the scene now needs "
        f"{expected} -- regenerate it with scripts/squeeze_twist.py"
    )


def test_timestep_is_converged(trajectory, reference):
    """PRD M2: halving sim_dt must not change the result materially.

    This is the check that moved sim_dt off the PRD's 0.005, where halving changed the cap
    rotation by 24% because the contact solref timeconst was exactly one step long.
    """
    ctrl, n_substeps = trajectory
    fine = build_model(None, None, None)
    fine.opt.timestep /= 2
    halved = cpu_rollout(fine, ctrl, n_substeps * 2)
    coarse_deg = np.degrees(reference["angle"][-1])
    fine_deg = np.degrees(halved["angle"][-1])
    rel = abs(fine_deg - coarse_deg) / max(abs(coarse_deg), 1e-9)
    assert rel <= TOLERANCE, (
        f"halving sim_dt moved the cap from {coarse_deg:.2f} to {fine_deg:.2f} deg ({rel*100:.1f}%)"
    )


def test_dynamics_are_not_chaotic(trajectory, reference):
    """Interpreting the timestep result as discretisation error only holds if the scene is not
    chaotic. A 1e-5 rad nudge to the starting hand pose must barely move the outcome."""
    ctrl, n_substeps = trajectory
    model = build_model(None, None, None)
    key = model.keyframe("pregrasp")
    cap = int(model.joint("cap_hinge").qposadr[0])
    rng = np.random.default_rng(0)

    data = mujoco.MjData(model)
    data.qpos[:] = np.array(key.qpos)
    data.qpos[:24] += 1e-5 * rng.standard_normal(24)
    data.ctrl[:] = key.ctrl
    for u in ctrl:
        for _ in range(n_substeps):
            data.ctrl[:] = u
            mujoco.mj_step(model, data)

    base = np.degrees(reference["angle"][-1])
    nudged = np.degrees(data.qpos[cap])
    rel = abs(nudged - base) / max(abs(base), 1e-9)
    assert rel < 0.05, f"a 1e-5 rad nudge moved the cap {rel*100:.1f}% ({base:.2f} -> {nudged:.2f})"


def test_pregrasp_leaves_room_to_squeeze():
    """Guards the minimum-standoff floor in fit_pregrasp.

    A re-fit once put the thumb 0.7 mm off the cap wall. Every other criterion looked fine and
    the gait collapsed from 88 to 5 degrees, because a pre-loaded fingertip cannot squeeze.
    """
    from scripts.view_scene import GAIT_TIPS, cap_surface_distance, load

    model, data = load("pregrasp")
    gaps = {t: cap_surface_distance(model, data, t) for t in GAIT_TIPS}
    assert min(gaps.values()) >= 0.0015, (
        "a gaiting fingertip starts loaded against the cap: "
        + ", ".join(f"{k}={v*1e3:.1f} mm" for k, v in gaps.items())
    )


@slow
def test_mjx_warp_matches_cpu(trajectory, reference):
    from scripts.check_parity import mjx_rollout

    ctrl, n_substeps = trajectory
    model = build_model(None, None, None)
    warp = mjx_rollout(model, ctrl, n_substeps, "warp", naconmax=1024, njmax=1024)

    cpu_deg = np.degrees(reference["angle"][-1])
    warp_deg = np.degrees(warp["angle"][-1])
    rel = abs(warp_deg - cpu_deg) / max(abs(cpu_deg), 1e-9)
    assert rel <= TOLERANCE, f"CPU {cpu_deg:.2f} deg vs warp {warp_deg:.2f} deg ({rel*100:.1f}%)"

    buffers = {"NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "NVMAX"}
    assert not buffers.intersection(warp["overflow_flags"]), warp["overflow_flags"]
    assert not {"ITERATIONS", "LS_ITERATIONS"}.intersection(warp["overflow_flags"]), (
        f"warp's solver did not converge: {warp['overflow_flags']}"
    )
    assert "EPA_HORIZON" not in warp["overflow_flags"], (
        "EPA horizon overflow is back -- the cap must stay a convex mesh, not a cylinder"
    )


@slow
def test_warp_buffer_sizing_is_driven_by_ncollision(trajectory, reference):
    """naconmax sizes warp's broadphase buffer, which counts candidate pairs, not contacts.
    Sizing it from the CPU contact count overflows on every step."""
    from scripts.check_parity import mjx_rollout

    ctrl, n_substeps = trajectory
    warp = mjx_rollout(build_model(None, None, None), ctrl, n_substeps, "warp", 1024, 1024)
    assert warp["ncollision_max"] > reference["ncon_max"], (
        "if ncollision ever drops to ncon, re-check the sizing advice in check_parity.py"
    )
