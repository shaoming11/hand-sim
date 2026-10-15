"""PRD M7: model-level domain randomisation, the ranges from the PRD's table.

    from capturn import randomize
    model, in_axes = randomize.domain_randomize(mjx_model, rng, scale=1.0)

`scale` is the PRD's `--dr_scale`: 0.0 is the nominal model exactly, 1.0 is the full table, and
anything between interpolates each bound toward its nominal value. The nominal value is read off
the model rather than hardcoded, so a scene change moves the ranges with it -- the cap's thread
friction is the one absolute range in the table (U(0.02, 0.15) N*m against a nominal 0.05), and
reading it back is what keeps `scale=0` a no-op instead of a silent 0.085.

Follows Playground's pattern exactly (`_src/locomotion/g1/randomize.py`): a `jax.vmap`-ed sampler
returning one tuple of fields per env, then `tree_replace` on the model and on an `in_axes` tree
of the same shape. Brax binds the rng itself -- `ppo.train` calls
`functools.partial(randomization_fn, rng=...)` -- so `scale` is the only argument a caller binds.

Which geoms get the friction randomisation: the ten collision geoms on the five distal bodies
plus the cap. They are unnamed in Menagerie, so they are resolved by body at import rather than
written down as ids, for the same reason everything else here resolves by name -- M8 moved the
mount and every hardcoded index in the project would have gone stale silently.

The PRD asks that every randomized field be read back and asserted to vary across envs. That is
`assert_varies(batched_model)`, exercised by `tests/test_randomize.py`.

One deliberate omission, carried from NOTES (M4, "three things that would have corrupted M5
silently"): `info["dr_params"]` stays zeroed. The 12-slot privileged block exists and this module
does not fill it, so the critic does not see the sampled parameters. Filling it means the env
reading its own per-env model inside `reset`, which is a change to the observation's meaning and
belongs with the DR run that first needs it, not with the sampler.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from capturn import scene

# The five distal bodies carry the pads that touch the cap; their geoms are unnamed.
DISTAL_BODIES = ("rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal")
CAP_GEOM = "cap"
BOTTLE_BODY = "bottle"

# PRD table, full scale.
FRICTION_SCALE = (0.6, 1.4)       # fingertip and cap geom friction, multiplicative
CAP_FRICTIONLOSS = (0.02, 0.15)   # cap thread friction, absolute N*m
CAP_DAMPING_SCALE = (0.5, 2.0)
ACTUATOR_GAIN_SCALE = (0.8, 1.2)
HAND_DAMPING_SCALE = (0.7, 1.3)
HAND_FRICTIONLOSS_SCALE = (0.5, 2.0)
HAND_ARMATURE_SCALE = (1.0, 1.2)
QPOS0_JITTER = 0.03               # rad, additive
BOTTLE_POS_JITTER = 0.015         # m, additive
BOTTLE_TILT = jp.deg2rad(5.0)     # rad, additive

RANDOMIZED_FIELDS = (
    "geom_friction", "dof_frictionloss", "dof_damping", "dof_armature",
    "actuator_gainprm", "actuator_biasprm", "qpos0", "body_pos", "body_quat",
)


def _ids():
    """Resolve every index this module touches, by name, from the compiled scene."""
    m = mujoco.MjModel.from_xml_path(scene.SCENE.as_posix())
    tip_geoms = [
        g for g in range(m.ngeom)
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) in DISTAL_BODIES
    ]
    cap_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CAP_GEOM)
    cap_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, scene.CAP_JOINT)
    cap_dof = int(m.jnt_dofadr[cap_jid])
    cap_qid = int(m.jnt_qposadr[cap_jid])
    hand_dofs = [d for d in range(m.nv) if d != cap_dof]
    hand_qids = [q for q in range(m.nq) if q != cap_qid]
    bottle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BOTTLE_BODY)
    return {
        "friction_geoms": jp.array(tip_geoms + [cap_geom]),
        "cap_dof": cap_dof,
        "hand_dofs": jp.array(hand_dofs),
        "hand_qids": jp.array(hand_qids),
        "bottle": bottle,
    }


IDS = _ids()


def _lerp(nominal, bound, scale):
    """The bound at `scale`: nominal at 0, the PRD's bound at 1."""
    return nominal + (bound - nominal) * scale


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def domain_randomize(model: mjx.Model, rng: jax.Array, scale: float = 1.0):
    """`(model, rng) -> (batched_model, in_axes)`, with `scale` bound by the caller."""
    friction_geoms = IDS["friction_geoms"]
    cap_dof = IDS["cap_dof"]
    hand_dofs = IDS["hand_dofs"]
    hand_qids = IDS["hand_qids"]
    bottle = IDS["bottle"]

    @jax.vmap
    def rand(rng):
        # Fingertip and cap sliding friction: *U(0.6, 1.4).
        rng, key = jax.random.split(rng)
        lo, hi = _lerp(1.0, FRICTION_SCALE[0], scale), _lerp(1.0, FRICTION_SCALE[1], scale)
        f = model.geom_friction[friction_geoms, 0] * jax.random.uniform(
            key, shape=(friction_geoms.shape[0],), minval=lo, maxval=hi)
        geom_friction = model.geom_friction.at[friction_geoms, 0].set(f)

        # Cap thread friction: U(0.02, 0.15) N*m, the one absolute range.
        rng, key = jax.random.split(rng)
        nominal = model.dof_frictionloss[cap_dof]
        lo = _lerp(nominal, CAP_FRICTIONLOSS[0], scale)
        hi = _lerp(nominal, CAP_FRICTIONLOSS[1], scale)
        cap_fl = jax.random.uniform(key, minval=lo, maxval=hi)

        # Hand joint friction: *U(0.5, 2.0).
        rng, key = jax.random.split(rng)
        lo = _lerp(1.0, HAND_FRICTIONLOSS_SCALE[0], scale)
        hi = _lerp(1.0, HAND_FRICTIONLOSS_SCALE[1], scale)
        hand_fl = model.dof_frictionloss[hand_dofs] * jax.random.uniform(
            key, shape=(hand_dofs.shape[0],), minval=lo, maxval=hi)
        dof_frictionloss = model.dof_frictionloss.at[cap_dof].set(cap_fl)
        dof_frictionloss = dof_frictionloss.at[hand_dofs].set(hand_fl)

        # Cap damping: *U(0.5, 2.0).  Hand joint damping: *U(0.7, 1.3).
        rng, key = jax.random.split(rng)
        lo = _lerp(1.0, CAP_DAMPING_SCALE[0], scale)
        hi = _lerp(1.0, CAP_DAMPING_SCALE[1], scale)
        cap_damp = model.dof_damping[cap_dof] * jax.random.uniform(key, minval=lo, maxval=hi)
        rng, key = jax.random.split(rng)
        lo = _lerp(1.0, HAND_DAMPING_SCALE[0], scale)
        hi = _lerp(1.0, HAND_DAMPING_SCALE[1], scale)
        hand_damp = model.dof_damping[hand_dofs] * jax.random.uniform(
            key, shape=(hand_dofs.shape[0],), minval=lo, maxval=hi)
        dof_damping = model.dof_damping.at[cap_dof].set(cap_damp)
        dof_damping = dof_damping.at[hand_dofs].set(hand_damp)

        # Hand armature: *U(1.0, 1.2).
        rng, key = jax.random.split(rng)
        lo = _lerp(1.0, HAND_ARMATURE_SCALE[0], scale)
        hi = _lerp(1.0, HAND_ARMATURE_SCALE[1], scale)
        arm = model.dof_armature[hand_dofs] * jax.random.uniform(
            key, shape=(hand_dofs.shape[0],), minval=lo, maxval=hi)
        dof_armature = model.dof_armature.at[hand_dofs].set(arm)

        # Actuator stiffness: *U(0.8, 1.2). These are position actuators, so the bias term is
        # -kp and has to move with the gain or the actuator stops being a position servo.
        rng, key = jax.random.split(rng)
        lo = _lerp(1.0, ACTUATOR_GAIN_SCALE[0], scale)
        hi = _lerp(1.0, ACTUATOR_GAIN_SCALE[1], scale)
        kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
            key, shape=(model.actuator_gainprm.shape[0],), minval=lo, maxval=hi)
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

        # Joint zero offset: +U(-0.03, 0.03) rad on the hand only.
        rng, key = jax.random.split(rng)
        j = QPOS0_JITTER * scale
        qpos0 = model.qpos0.at[hand_qids].set(
            model.qpos0[hand_qids]
            + jax.random.uniform(key, shape=(hand_qids.shape[0],), minval=-j, maxval=j))

        # Bottle placement: +-1.5 cm, and up to 5 deg of tilt about a random horizontal axis.
        rng, key = jax.random.split(rng)
        p = BOTTLE_POS_JITTER * scale
        body_pos = model.body_pos.at[bottle].set(
            model.body_pos[bottle] + jax.random.uniform(key, shape=(3,), minval=-p, maxval=p))
        rng, axis_key, angle_key = jax.random.split(rng, 3)
        phi = jax.random.uniform(axis_key, minval=0.0, maxval=2.0 * jp.pi)
        angle = jax.random.uniform(angle_key, minval=0.0, maxval=BOTTLE_TILT * scale)
        half = 0.5 * angle
        tilt = jp.array([jp.cos(half),
                         jp.sin(half) * jp.cos(phi),
                         jp.sin(half) * jp.sin(phi),
                         0.0])
        body_quat = model.body_quat.at[bottle].set(_quat_mul(tilt, model.body_quat[bottle]))

        return (geom_friction, dof_frictionloss, dof_damping, dof_armature,
                actuator_gainprm, actuator_biasprm, qpos0, body_pos, body_quat)

    sampled = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({f: 0 for f in RANDOMIZED_FIELDS})
    model = model.tree_replace(dict(zip(RANDOMIZED_FIELDS, sampled)))
    return model, in_axes


def assert_varies(batched_model: mjx.Model, atol: float = 0.0) -> dict[str, float]:
    """The PRD's check: every randomized field must actually differ across envs.

    Returns the per-field spread (max - min across envs, summed over the field) so a caller can
    print it; raises if any field is constant, which is what a mis-wired `in_axes` looks like.
    """
    spreads, dead = {}, []
    for field in RANDOMIZED_FIELDS:
        value = getattr(batched_model, field)
        spread = float(jp.sum(jp.max(value, axis=0) - jp.min(value, axis=0)))
        spreads[field] = spread
        if spread <= atol:
            dead.append(field)
    if dead:
        raise AssertionError(
            f"randomized fields that do not vary across envs: {', '.join(dead)}. "
            "Either in_axes is wrong for them or the sampled range collapsed."
        )
    return spreads
