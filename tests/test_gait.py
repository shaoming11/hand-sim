"""M1 acceptance checks: can this hand actually turn the cap?

The milestone's bar is that a crude open-loop squeeze-and-twist moves the cap a few degrees
against `frictionloss=0.05`. These tests also pin the two findings that made it work, so a later
change to the scene cannot quietly undo them: grip must come from flexion rather than abduction,
and the convex-collision settings must keep the fingertips out of the cap.
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.squeeze_twist import OpenLoopGait, run  # noqa: E402

THREAD_FRICTION = 0.05  # N*m, the cap hinge's frictionloss


@pytest.fixture(scope="module")
def three_finger():
    return run(["th", "ff", "mf"], 0.25, 0.9, 0.008, 12, False)


def test_two_finger_gait_turns_the_cap(three_finger):
    """The PRD's literal ask: thumb and first finger only."""
    _, _, res = run(["th", "ff"], 0.25, 0.9, 0.008, 12, False)
    assert res["finite"]
    assert res["total_deg"] > 3.0, f"only {res['total_deg']:.2f} deg with thumb+first"


def test_three_finger_gait_turns_the_cap_well(three_finger):
    _, _, res = three_finger
    assert res["finite"]
    assert res["total_deg"] > 45.0, f"only {res['total_deg']:.1f} deg with thumb+first+middle"


def test_rotation_is_monotonic(three_finger):
    """Each cycle must add rotation, and the cap must hold its angle through the release phase.

    A gait that wound the cap forwards and let it spring back would average out to nothing; the
    thread friction is what makes the progress stick.
    """
    _, _, res = three_finger
    per_cycle = np.degrees(res["angle"][15::16])
    gains = np.diff(np.concatenate([[0.0], per_cycle]))
    assert (gains > 0.5).all(), f"per-cycle gains {gains.round(2)}"


def test_gait_does_not_touch_the_bottle(three_finger):
    _, _, res = three_finger
    assert res["bottle_contact_steps"] == 0, (
        f"fingers hit the bottle on {res['bottle_contact_steps']} control steps"
    )


def test_fingertips_stay_out_of_the_cap(three_finger):
    """Guards ccd_iterations/ccd_tolerance in scene.xml. Every fingertip here is a mesh against a
    cylinder, and at Playground's Leap setting of ccd_iterations=10 the tips sink 7.4 mm in."""
    _, _, res = three_finger
    assert res["worst_penetration_mm"] > -1.5, (
        f"deepest penetration {res['worst_penetration_mm']:.2f} mm"
    )


def test_grip_uses_flexion_not_abduction():
    """The finger abduction joints have a 0.70 rad ctrlrange and weak actuators. Spending grip on
    them wedges the fingertip sideways into the cap and blew the contact up to 268 N."""
    gait = OpenLoopGait(["th", "ff", "mf"])
    offset = gait.grip_offset(0.25)
    model = gait.model
    for a in range(model.nu):
        name = model.actuator(a).name
        if name.endswith(("FFJ4", "MFJ4", "RFJ4", "LFJ4", "LFJ5")):
            assert offset[a] == 0.0, f"grip is being spent on abduction joint {name}"
    span = model.actuator_ctrlrange[:, 1] - model.actuator_ctrlrange[:, 0]
    assert np.all(np.abs(offset) <= 0.5 * span + 1e-9), "a grip offset exceeds half its travel"


def test_hand_can_out_torque_the_thread_friction():
    """Headroom check. Grip force caps the achievable cap torque at mu * r * sum(Fn); if that
    ceiling sat at the thread friction the task would be a coin flip for any policy."""
    gait = OpenLoopGait(["th", "ff", "mf"])
    model = gait.model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("pregrasp").id)
    ctrl = data.ctrl + gait.grip_offset(0.35)
    for _ in range(600):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    force = np.zeros(6)
    normal = 0.0
    for i in range(data.ncon):
        con = data.contact[i]
        if gait.cap_geom in (con.geom1, con.geom2):
            mujoco.mj_contactForce(model, data, i, force)
            normal += abs(force[0])

    mu = model.geom_friction[gait.cap_geom, 0]
    ceiling = mu * gait.cap_radius * normal
    assert ceiling > 2.0 * THREAD_FRICTION, (
        f"grip {normal:.2f} N gives a torque ceiling of {ceiling:.4f} N*m against "
        f"{THREAD_FRICTION} N*m of thread friction"
    )
