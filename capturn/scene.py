"""The scene file and the cap geometry, shared by the env and the inspection scripts.

The cap is a 16-sided prism rather than a cylinder (NOTES.md, M2), and `geom_size` is all zeros
for a mesh geom, so its radius and half-height have to be measured from the mesh vertices.
Everything that reasons about the cap wall goes through here, so the geometry can change without
some caller silently reading a radius of zero.
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "assets" / "scene.xml"

# M5: the two fingers that form the pinch. A person unscrews a bottle cap with the thumb pad
# opposed to the side of the index finger, and that is the grasp this task is for. The middle
# finger was a gaiting finger through M4 and is not one now: on the M1 scripted gait it is in
# contact 93.8% of frames, *more* than the index at 64.6%, and the first policy trained with the
# opposition reward leaned on it hardest of all (71.3% of steps, against the thumb tip's 17.7%).
# Demoting it is what makes the two-finger pinch the only way to score. The ring and little
# fingers never reach the cap at all -- 36 and 42 mm at the closest.
GAIT_TIPS = ("rh_th_tip", "rh_ff_tip")
# The fingers that must stay *off* the cap, costed in the reward.
IDLE_TIPS = ("rh_mf_tip", "rh_rf_tip", "rh_lf_tip")
# Order is deliberately unchanged -- thumb, index, middle, ring, little -- so the privileged
# observation keeps the byte-identical layout M4 fixed. Only the partition moved.
ALL_TIPS = GAIT_TIPS + IDLE_TIPS

# `framepos` sensors on these, relative to `cap_site`, i.e. already in the cap frame.
GAIT_TIP_SENSORS = ("th_tip_position", "ff_tip_position")
IDLE_TIP_SENSORS = ("mf_tip_position", "rf_tip_position", "lf_tip_position")
ALL_TIP_SENSORS = GAIT_TIP_SENSORS + IDLE_TIP_SENSORS

CAP_JOINT = "cap_hinge"
# M7: the forearm roll -- pronation. Declared on `rh_forearm`, the root body, so it leads both
# qpos and ctrl; everything here resolves it by name anyway.
ROLL_JOINT = "rh_WRJ0"

# M6: the geoms a pinch must not reach. The re-fitted pregrasp opposes the two fingers at 150 deg
# rather than 180 precisely because the far hemisphere is where the neck is, and the reward has to
# agree with the pose: `reward_config.scales.bottle` costs a pinch surface for entering the
# bottle's margin. Both are z-aligned cylinders, read from the compiled model.
BOTTLE_GEOMS = ("bottle_body", "bottle_neck")

# M5: what a bottle-cap grasp actually is. A person opposes the thumb pad against the *side* of
# the index finger -- a lateral pinch -- not a tripod of fingertips. The first trained policy got
# half of this right by itself: it drove the cap with `rh_ffmiddle` and `rh_ffproximal`, the
# correct surfaces, but never brought the thumb in, so its contacts sat 66 deg apart instead of
# opposed and the grip was a one-sided push. Contact anywhere on these bodies counts.
# M5: the distal segment only -- the pad. It was `("rh_thdistal", "rh_thmiddle")`, contact
# anywhere along the thumb, and the policy took the loophole: it pressed with the middle phalanx
# and left the pad 12.15 mm off the cap, touching on 17.7% of steps against the M1 gait's 1.19 mm
# and 90.6%. The body-level gap read 1.87 mm and looked healthy while the thumb was not really
# gripping. The index stays full-length on purpose: a lateral pinch opposes the thumb pad to the
# *side* of the index finger, so its proximal and middle segments are legitimate contact.
THUMB_BODIES = ("rh_thdistal",)
INDEX_BODIES = ("rh_ffdistal", "rh_ffmiddle", "rh_ffproximal")
# Samples along each capsule's axis. The closest point on a finger segment is usually not an
# endpoint, and 5 is enough to find it to well under a millimetre on a 25 mm capsule.
SEGMENT_SAMPLES = 5


def segment_points(xpos, xmat, half_length, samples: int = SEGMENT_SAMPLES):
    """Points along each geom's local z axis: (n_geoms, samples, 3), in world coordinates.

    `xpos` is (n, 3), `xmat` is (n, 3, 3) and `half_length` is (n,) -- zero for anything that is
    not a capsule, which then contributes its centre alone. Plain operators, so this works under
    `jit` on jax arrays and on numpy in the CPU scripts.
    """
    # A numpy constant broadcasts against jax arrays, so this one expression serves both.
    t = np.linspace(-1.0, 1.0, samples)
    axis = xmat[:, :, 2]                                   # (n, 3) local z in world
    offsets = axis[:, None, :] * (half_length[:, None] * t[None, :])[..., None]
    return xpos[:, None, :] + offsets


def grasp_geometry(points, geom_radius, origin, mat, cap_radius: float, cap_half_h: float):
    """Closest approach of a group of geoms to the cap, and the direction it approaches from.

    `points` is (n_geoms, samples, 3) in world coordinates, `geom_radius` is (n_geoms,).
    Returns `(gap, unit_radial, contact, which)`: the smallest surface-to-surface gap over every
    sample; the horizontal unit vector in the cap frame pointing from the cap axis towards that
    closest sample; the closest sample itself, in *world* coordinates; and the row of `points` it
    came from, i.e. which geom of the group is the near one.

    The unit vector is what makes opposition measurable -- two grips are opposed when their
    vectors point opposite ways, which is a dot product rather than an angle, so there is no
    wrap-around to handle. M8: `contact` and `which` are what let a caller ask how fast that
    surface is sweeping around the cap, which needs the point in the world frame and the body
    that carries it.
    """
    local = (points - origin) @ mat                        # world -> cap frame
    gaps = surface_distance(local, cap_radius, cap_half_h) - geom_radius[:, None]
    flat = gaps.reshape(-1)
    best = flat.argmin()
    closest = local.reshape(-1, 3)[best]
    planar = closest[:2]
    norm = (planar[0] ** 2 + planar[1] ** 2) ** 0.5
    return (
        flat[best],
        planar / (norm + 1e-9),
        points.reshape(-1, 3)[best],
        best // points.shape[1],
    )


def triple(a, b, c):
    """Scalar triple product `a . (b x c)`, componentwise, for (..., 3) inputs.

    Written out rather than via `cross` because this module is shared between the env (jax
    arrays, under `jit`) and the CPU scripts (numpy), and the two libraries' `cross`/`stack` do
    not interchange.
    """
    return (
        a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1])
        + a[..., 1] * (b[..., 2] * c[..., 0] - b[..., 0] * c[..., 2])
        + a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0])
    )


def axial_sweep(point, velocity, origin, axis):
    """How fast `point` is going *around* the line `(origin, axis)`, in rad/s.

    M8: this is the quantity a non-slipping grip hands to the cap. A pad in contact with the cap
    wall and sweeping around the cap axis at omega turns the cap at omega -- that is the rolling
    constraint, and it is radius-independent, so it holds whether the pad sits at the cap's own
    16 mm or at the 24 mm a knuckle reaches from. Motion straight at the cap (pressing) or along
    its axis (sliding up the wall) contributes nothing, which is correct: neither turns it.

    `axis` must be a unit vector. `axis x planar` has magnitude `r` and points along the tangent,
    so `v . (axis x planar) / r**2` is `(v . tangent) / r` with no square roots and no division
    by a radius that could be zero at the axis itself.
    """
    rel = point - origin
    along = (rel * axis).sum(-1)
    planar = rel - along[..., None] * axis
    r2 = (planar * planar).sum(-1)
    return triple(velocity, axis, planar) / (r2 + 1e-12)


def cylinder_sdf(local, radius: float, half_h: float):
    """Signed distance from a point to a z-aligned cylinder, the point given in its frame.

    Negative inside, and exact. `surface_distance` below is the same quantity *outside* the
    cylinder and is kept for the cap, where it is all the callers need and where its numbers are
    on the record from M1 onwards; this one is for the bottle, where a finger sweeping over the
    shoulder really can end up inside and `surface_distance` would report it as comfortably
    clear (it squares the radial term, so inside reads positive).

    Plain operators, so this serves the env under `jit` and the CPU scripts alike.
    """
    radial = (local[..., 0] ** 2 + local[..., 1] ** 2) ** 0.5 - radius
    axial = abs(local[..., 2]) - half_h
    out_r, out_a = radial * (radial > 0), axial * (axial > 0)
    outside = (out_r ** 2 + out_a ** 2) ** 0.5
    deepest = radial * (radial > axial) + axial * (radial <= axial)  # max(radial, axial)
    return outside + deepest * (deepest < 0)


def cylinder_gaps(points, geom_radius, geom_pos, geom_mat, radius: float, half_h: float):
    """Surface-to-surface gaps from a group of geoms to one cylinder, same shape as `points[..., 0]`.

    `points` is (n_geoms, samples, 3) in world coordinates and `geom_radius` is (n_geoms,), as
    produced by `segment_points`; `geom_pos`/`geom_mat` place the cylinder.
    """
    local = (points - geom_pos) @ geom_mat
    return cylinder_sdf(local, radius, half_h) - geom_radius[:, None]


def cap_dimensions(model: mujoco.MjModel) -> tuple[float, float]:
    """The cap's outer radius and half-height, in metres.

    Reads the mesh vertices when the cap is the 16-sided prism and `geom_size` when it is a
    primitive, so the caller does not have to know which it is.
    """
    cap = model.geom("cap").id
    if model.geom_type[cap] != mujoco.mjtGeom.mjGEOM_MESH:
        return float(model.geom_size[cap, 0]), float(model.geom_size[cap, 1])
    mesh = model.geom_dataid[cap]
    start = model.mesh_vertadr[mesh]
    verts = np.array(model.mesh_vert[start:start + model.mesh_vertnum[mesh]], dtype=float)
    # The compiler re-frames mesh vertices and puts the correction on the geom, so the raw
    # vertices are not in the geom frame -- for this cap they come back with the prism axis
    # along x. Rotate them back before measuring, or the "radius" reads 17.9 mm.
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(model.geom_quat[cap], dtype=float))
    local = verts @ rot.reshape(3, 3).T + model.geom_pos[cap]
    return float(np.linalg.norm(local[:, :2], axis=1).max()), float(np.abs(local[:, 2]).max())


def surface_distance(local, radius: float, half_h: float):
    """Distance from a point to the cap's surface, with the point given in the cap frame.

    `local` is (..., 3) and the result is (...). Written with plain operators rather than
    `np.hypot`/`np.maximum` so the same function serves the env under `jit` (jax arrays, from
    the fingertip `framepos` sensors) and the CPU scripts (numpy, from `MjData` site positions).

    Radially outside the wall and axially within the rim this is the gap to the wall, which is
    where a gaiting fingertip belongs. Near the rim it rounds the corner off, so a tip sliding
    over the top edge sees the distance grow smoothly instead of jumping.
    """
    radial = (local[..., 0] ** 2 + local[..., 1] ** 2) ** 0.5 - radius
    axial = abs(local[..., 2]) - half_h
    axial = axial * (axial > 0)  # only the part of the offset that clears the rim counts
    return (radial ** 2 + axial ** 2) ** 0.5
