"""M0 acceptance checks for `assets/scene.xml`, on CPU MuJoCo.

These encode the milestone's done-criteria: the scene loads, the cap hinge turns under a
realistic fingertip torque (and holds below the thread friction), the hand does not penetrate
the bottle at the pregrasp pose, and the three gaiting fingertips reach the cap rim.
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.view_scene import GAIT_TIPS, cap_surface_distance, load  # noqa: E402

REACH_TOLERANCE = 0.010  # PRD M0: gaiting fingertips within about 1 cm of the cap rim


@pytest.fixture(scope="module")
def scene():
    return load("pregrasp")


def test_scene_loads_with_expected_shape(scene):
    model, _ = scene
    assert model.nu == 20, "20 position actuators"
    assert model.nq == 25, "24 hand joints + 1 cap hinge"
    assert model.njnt == 25
    assert model.body("rh_forearm").parentid[0] == 0, "forearm hangs off the world"
    assert model.body("rh_forearm").jntnum[0] == 0, "forearm has no joint, so the base is fixed"
    assert model.body("bottle").jntnum[0] == 0, "bottle is fixed in place"
    assert model.joint("cap_hinge").type[0] == mujoco.mjtJoint.mjJNT_HINGE


def test_cap_is_the_only_moving_object(scene):
    model, _ = scene
    cap_dofs = [model.jnt_dofadr[model.joint("cap_hinge").id]]
    hand_dofs = [
        model.jnt_dofadr[j] for j in range(model.njnt) if model.joint(j).name.startswith("rh_")
    ]
    assert len(hand_dofs) == 24
    assert sorted(hand_dofs + cap_dofs) == list(range(model.nv))


def test_cap_bottle_pair_is_excluded(scene):
    """The cap sleeves over the neck on purpose; filterparent does not remove that pair."""
    model, data = scene
    cap_geom = model.geom("cap").id
    bottle_geoms = {model.geom("bottle_body").id, model.geom("bottle_neck").id}
    for i in range(data.ncon):
        pair = {data.contact[i].geom1, data.contact[i].geom2}
        assert not (cap_geom in pair and pair & bottle_geoms), "cap is colliding with the bottle"


def test_hand_does_not_penetrate_the_bottle(scene):
    """PRD M0. Touching the cap at pregrasp is fine and expected; touching the bottle is not."""
    model, data = scene
    bottle = {model.geom("bottle_body").id, model.geom("bottle_neck").id}
    offenders = []
    for i in range(data.ncon):
        con = data.contact[i]
        if {con.geom1, con.geom2} & bottle:
            offenders.append((con.dist, model.geom(con.geom1).name, model.geom(con.geom2).name))
    assert not offenders, f"hand is in contact with the bottle at pregrasp: {offenders}"


def test_no_gross_penetration_at_pregrasp(scene):
    """Light contact settles a fraction of a mm into the soft constraint; overlap beyond that is a
    modelling error, not solver softness."""
    model, data = scene
    worst = min((data.contact[i].dist for i in range(data.ncon)), default=0.0)
    assert worst > -1e-3, f"deepest penetration at pregrasp is {worst*1e3:.3f} mm"


def test_gaiting_fingertips_reach_the_cap(scene):
    model, data = scene
    gaps = {tip: cap_surface_distance(model, data, tip) for tip in GAIT_TIPS}
    assert all(g < REACH_TOLERANCE for g in gaps.values()), (
        "thumb/first/middle must start within 1 cm of the cap surface: "
        + ", ".join(f"{k}={v*1e3:.1f} mm" for k, v in gaps.items())
    )


def test_gaiting_fingertips_are_spread_around_the_cap(scene):
    """Three tips bunched on one side cannot produce a twisting couple."""
    model, data = scene
    cap_mat = data.site("cap_site").xmat.reshape(3, 3)
    az = []
    for tip in GAIT_TIPS:
        local = cap_mat.T @ (data.site(tip).xpos - data.site("cap_site").xpos)
        az.append(np.arctan2(local[1], local[0]))
    seps = [
        abs(np.arctan2(np.sin(az[i] - az[j]), np.cos(az[i] - az[j])))
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    assert min(seps) > np.radians(80.0), f"tip separations {np.degrees(seps).round(1)} deg"


def test_pregrasp_is_an_equilibrium(scene):
    """Holding the keyframe's own ctrl must not make the hand drift or nudge the cap."""
    model, _ = scene
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("pregrasp").id)
    ctrl = data.ctrl.copy()
    cap_adr = int(model.joint("cap_hinge").qposadr[0])
    for _ in range(int(round(2.0 / model.opt.timestep))):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
    assert np.abs(data.qvel).max() < 0.05, f"max|qvel| after 2 s = {np.abs(data.qvel).max():.4f}"
    assert abs(data.qpos[cap_adr]) < np.radians(2.0), "cap drifted while merely holding pregrasp"


def _drive_cap(model, torque, seconds=1.0):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("pregrasp").id)
    dof = int(model.jnt_dofadr[model.joint("cap_hinge").id])
    adr = int(model.joint("cap_hinge").qposadr[0])
    ctrl = data.ctrl.copy()
    for _ in range(int(round(seconds / model.opt.timestep))):
        data.ctrl[:] = ctrl
        data.qfrc_applied[dof] = torque
        mujoco.mj_step(model, data)
    return data.qpos[adr]


@pytest.mark.parametrize(
    "torque, should_turn",
    [(0.03, False), (0.10, True)],
    ids=["below-thread-friction", "above-thread-friction"],
)
def test_cap_hinge_turns_above_thread_friction(scene, torque, should_turn):
    """The viewer-drag check, done numerically.

    0.1 N*m at the 16 mm cap radius is about 6 N of total tangential fingertip force, which the
    hand can produce; the cap must turn there and stay put below its 0.05 N*m frictionloss.
    """
    model, _ = scene
    turned = abs(_drive_cap(model, torque))
    if should_turn:
        assert turned > 0.1, f"{torque} N*m only turned the cap {np.degrees(turned):.2f} deg"
    else:
        assert turned < np.radians(5.0), (
            f"{torque} N*m is below the 0.05 N*m frictionloss but turned the cap "
            f"{np.degrees(turned):.2f} deg"
        )


def test_cap_cannot_be_ratcheted_below_thread_friction(scene):
    """The exploit this scene must not permit: shaking the cap with sub-threshold torque and
    accumulating rotation for free. Guards the solreffriction/solimpfriction choice on the hinge."""
    model, _ = scene
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("pregrasp").id)
    dof = int(model.jnt_dofadr[model.joint("cap_hinge").id])
    adr = int(model.joint("cap_hinge").qposadr[0])
    ctrl = data.ctrl.copy()
    n = int(round(5.0 / model.opt.timestep))
    for i in range(n):
        data.ctrl[:] = ctrl
        data.qfrc_applied[dof] = 0.04 * np.sin(2 * np.pi * 2.0 * i * model.opt.timestep)
        mujoco.mj_step(model, data)
    assert abs(data.qpos[adr]) < np.radians(5.0), (
        f"+-0.04 N*m at 2 Hz ratcheted the cap {np.degrees(data.qpos[adr]):.2f} deg in 5 s"
    )


def test_breakaway_speed_matches_the_damping_model(scene):
    """Above threshold the hinge must behave like dry friction plus viscous damping, not like a
    soft constraint: steady speed = (tau - frictionloss) / damping."""
    model, _ = scene
    dof = int(model.jnt_dofadr[model.joint("cap_hinge").id])
    fl, damping = model.dof_frictionloss[dof], model.dof_damping[dof]
    torque = 0.12
    turned = _drive_cap(model, torque, seconds=2.0)
    expected = (torque - fl) / damping
    measured = turned / 2.0
    assert measured == pytest.approx(expected, rel=0.25), (
        f"mean speed {measured:.2f} rad/s vs analytic {expected:.2f} rad/s"
    )
