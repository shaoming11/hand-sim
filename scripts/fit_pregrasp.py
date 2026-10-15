"""Fit the hand mount transform and the `pregrasp` keyframe.

The hand placement is the most important design choice in this project, so it is solved
numerically rather than hand-tuned. Two artefacts come out of this script:

  * `rh_forearm` pos/quat in `assets/right_hand_mjx.xml`  -- where the fixed hand hangs.
  * the `pregrasp` key in `assets/scene.xml`              -- joint angles and ctrl targets.

Method
------
1. Seed from Menagerie's own `grasp sphere` keyframe, whose thumb/first/middle fingertips
   already sit on a ~14 mm-radius circle -- almost exactly our 16 mm cap. The plane of those
   three tips defines a natural "pinch frame"; we rotate the hand so that the plane normal
   becomes the cap axis (world +z) and translate so the tip centroid lands on the cap centre.
2. Refine with L-BFGS-B over the mount pose (3 translation + 3 rotation-vector) and the 20
   actuator targets, so that the three gaiting fingertips sit on a circle of
   `cap_radius + standoff`, at the cap's mid-height, spread around the circumference, with
   every geom pair clear of penetration. M5: "spread around the circumference" became
   "opposed", and the gaiting tips are the thumb and index finger only -- the middle finger
   joined the idle set and is pushed clear of the cap by `IDLE_RADIUS`.
3. Settle: run the real controller from the fitted pose until it stops moving, and use the
   settled state as the keyframe so the first simulated step is already an equilibrium.

Usage
-----
    python3 scripts/fit_pregrasp.py            # fit and report
    python3 scripts/fit_pregrasp.py --write    # also patch the two XML files in place
"""

from __future__ import annotations

import argparse
import pathlib
import re

import sys

import mujoco
import numpy as np
from scipy import optimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from view_scene import cap_dimensions  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "assets" / "scene.xml"
HAND = ROOT / "assets" / "right_hand_mjx.xml"
MENAGERIE_KEYFRAMES = ROOT / "assets" / "shadow_hand" / "keyframes.xml"

# M5: imported rather than redeclared, so the fitter and the env cannot drift apart. They had:
# this file still fitted a three-finger tripod after `capturn.scene` moved to a two-finger
# pinch, and the pose it produces is what decides whether the pinch is physically available.
from capturn.scene import (  # noqa: E402
    GAIT_TIPS,
    IDLE_TIPS,
    THUMB_BODIES,
    grasp_geometry,
    segment_points,
)

# M8: the surfaces the *reward* measures the pinch on. `pinch_quality` reads the closest point
# on a finger's collision geoms, not its fingertip site, and in the M7 pose the two disagreed by
# 5 mm: the index tip site read z = -4.3 mm on the cap while the geom actually touching it sat at
# -5.8 mm, in the bottom 2 mm of a cap that spans -8 to +8. The fit therefore looked centred and
# was gripping the lip. Fitting the same quantity the reward reads removes that gap.
#
# The thumb group is the reward's own; the index is narrowed to the distal pad on purpose. The
# reward lets a lateral pinch load the *side* of the index (`INDEX_BODIES` is all three segments)
# and that freedom is kept -- but a pose whose nearest index surface was the proximal phalanx
# would be a knuckle resting on the cap, not a pinch to open from.
PAD_BODIES = {"thumb": THUMB_BODIES, "index": ("rh_ffdistal",)}

# Seeding needs three non-collinear tips to define the hand's plane, which is a question about
# the hand's orientation and not about which fingers pinch. It stays a tripod even though the
# grasp does not.
SEED_TIPS = ("rh_th_tip", "rh_ff_tip", "rh_mf_tip")

STANDOFF = 0.003  # fingertip pad to cap wall at the pregrasp pose [m]
MIN_STANDOFF = 0.002  # ... and no gaiting tip may start closer than this [m]
CLEARANCE = 0.002  # every geom pair must stay at least this far apart [m]
BODY_CLEARANCE = 0.006  # ... and the hand must keep this far off the bottle itself [m]
AXIAL_BAND = 0.004  # keep gaiting tips within this much of the cap mid-plane [m]
IDLE_RADIUS = 0.030  # keep ring/little tips at least this far off the cap axis [m]
# M5: the two pinch tips want to be *opposed*, not merely separated. The old objective asked
# for >= 1.75 rad between three tips spread round the circumference, and the pose it produced
# could not oppose the thumb to the index at all: measured on it, the thumb is 1.54 mm from the
# cap at 20% flexion but only 158 deg from the index, reaches 178 deg at 50% flexion by which
# point it is 11.19 mm off the cap, and is 47 mm away by 65%. Its contact window and its
# opposition window never overlapped, so the thumb could never press -- 0.0334 N*m at best
# against the index finger's 0.1434, and nothing at high flexion. A pose search that *did*
# enforce opposition found thumb+index squeezes worth 0.1974 N*m from the same two fingers,
# 2.4x the fitted pregrasp. NOTES.md, M5.
TIP_OPPOSITION = 2.60  # target azimuthal separation between the two pinch tips [rad]
IK_MARGIN = 0.01  # temporary geom margin, so near-contacts are visible to the fitter [m]

# M7: the pregrasp belongs at the *top* of the usable roll range. The stroke runs 0 -> -45 deg,
# so a pose fitted at -45 would throw away half of it. Pinned rather than bounded, because the
# optimiser has every incentive to buy clearance by starting part-way through the stroke.
ROLL_NEUTRAL = 0.0
# M7: and the pinch is biased into the cap's *upper* half rather than centred on its mid-plane.
#
# The first pronation gait brushed the bottle on 91 of 220 steps, always `rh_ffdistal` against
# the neck. It is not a roll-range problem -- every static pose from -90 to 0 is clean at every
# grip -- and it is not fixed by a lighter grip. The cause is height: the neck is a sleeve
# running up *inside* the cap, with 24 mm of it exposed below the cap's bottom rim, and the M5
# pose puts the tips at cap-frame z = -3.0 and -4.3 mm, 3-5 mm above that rim. Rolling swings
# the finger down over the lip and onto the neck.
#
# A roll-sweep clearance penalty was tried first and is *not* kept: applying BODY_CLEARANCE at
# three extra angles outweighed the pinch terms, and the optimiser bought clearance by leaving
# the cap altogether (tips 30+ mm off the surface, 40 mm out of the axial band). It also
# constrained the wrong pose -- the collision happens when gripped, and the fitter only ever
# sees the ungripped pregrasp. Biasing the height attacks the measured cause instead.
AXIAL_TARGET = 0.003  # keep the pinch this far above the cap mid-plane [m]


# --------------------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------------------
def rodrigues(w: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def mat2quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(-1))
    return q


def quat2mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.ascontiguousarray(q, dtype=float))
    return R.reshape(3, 3)


def frame(e1: np.ndarray, e3: np.ndarray) -> np.ndarray:
    """Right-handed frame with third axis along e3 and first axis as close to e1 as possible."""
    e3 = e3 / np.linalg.norm(e3)
    e1 = e1 - (e1 @ e3) * e3
    e1 /= np.linalg.norm(e1)
    return np.column_stack([e1, np.cross(e3, e1), e3])


# --------------------------------------------------------------------------------------
# actuator target -> qpos
# --------------------------------------------------------------------------------------
def actuator_to_qpos_map(model: mujoco.MjModel) -> list[list[tuple[int, float]]]:
    """For each actuator, the (qpos address, share) pairs it drives.

    Position actuators on a `fixed` tendon (the coupled J2/J1 pairs of the four fingers) are
    split evenly, which is where the passive stiffness puts them anyway -- so the keyframe is
    an equilibrium of the coupling rather than a pose the tendon immediately pulls out of.
    """
    out: list[list[tuple[int, float]]] = []
    for a in range(model.nu):
        trntype, objid = model.actuator_trntype[a], model.actuator_trnid[a, 0]
        if trntype == mujoco.mjtTrn.mjTRN_JOINT:
            out.append([(int(model.jnt_qposadr[objid]), 1.0)])
        elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
            adr, num = model.tendon_adr[objid], model.tendon_num[objid]
            joints = [
                int(model.wrap_objid[adr + i])
                for i in range(num)
                if model.wrap_type[adr + i] == mujoco.mjtWrap.mjWRAP_JOINT
            ]
            out.append([(int(model.jnt_qposadr[j]), 1.0 / len(joints)) for j in joints])
        else:
            raise RuntimeError(f"actuator {model.actuator(a).name}: unsupported trntype {trntype}")
    return out


# --------------------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------------------
class PregraspFitter:
    def __init__(self, from_keyframe: bool = False) -> None:
        self.model = mujoco.MjModel.from_xml_path(SCENE.as_posix())
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.from_keyframe = from_keyframe

        self.forearm = m.body("rh_forearm").id
        self.cap_geom = m.geom("cap").id
        self.cap_radius, self.cap_half_h = cap_dimensions(m)
        self.cap_centre = np.array(m.body("cap").pos) + np.array(m.body("bottle").pos)
        self.cap_qadr = int(m.joint("cap_hinge").qposadr[0])
        self.bottle_geoms = {m.geom("bottle_body").id, m.geom("bottle_neck").id}
        # M7: the forearm roll, resolved by name. It is pinned at the pregrasp and swept for
        # clearance -- see ROLL_SWEEP.
        self.roll_act = int(m.actuator("rh_A_WRJ0").id)
        self.roll_qid = int(m.joint("rh_WRJ0").qposadr[0])
        self._hand_qadrs = [
            int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.joint(j).name.startswith("rh_")
        ]

        self.a2q = actuator_to_qpos_map(m)
        self.lo, self.hi = m.actuator_ctrlrange.T.copy()
        self.jnt_lo = m.jnt_range[:, 0].copy()
        self.jnt_hi = m.jnt_range[:, 1].copy()
        self.q_lo = np.full(m.nq, -np.inf)
        self.q_hi = np.full(m.nq, np.inf)
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                adr = int(m.jnt_qposadr[j])
                self.q_lo[adr], self.q_hi[adr] = self.jnt_lo[j], self.jnt_hi[j]

        # The fitter sees near-contacts, so it has a smooth field to push out of rather than a
        # penalty that only switches on once two geoms already overlap.
        self._true_margin = m.geom_margin.copy()
        m.geom_margin[:] = IK_MARGIN

        # M8: the pinch surfaces, resolved the way the env resolves them.
        capsule = mujoco.mjtGeom.mjGEOM_CAPSULE
        self.pad_groups = {}
        for name, bodies in PAD_BODIES.items():
            ids = [
                g for g in range(m.ngeom)
                if m.body(m.geom_bodyid[g]).name in bodies
                and (m.geom_contype[g] or m.geom_conaffinity[g])
            ]
            if not ids:
                raise ValueError(f"no collision geoms on {bodies}")
            self.pad_groups[name] = (
                np.array(ids),
                np.array([m.geom_size[g, 0] for g in ids], dtype=float),
                np.array([m.geom_size[g, 1] if m.geom_type[g] == capsule else 0.0 for g in ids],
                         dtype=float),
            )

        # M8: the mount and targets the scene already carries. Captured before `_seed()`, which
        # mutates both to build the Menagerie seed. Starting here is what `--from-keyframe` does:
        # the committed pose is a sound fit in every respect but the one being corrected -- the
        # pinch height -- so re-deriving the mount from Menagerie throws away work and lands the
        # optimiser somewhere else entirely. Measured: the Menagerie seed re-fits to a settled
        # cost of 320 with the tips 14-17 mm out of the axial band; this one starts at 41.
        self.keyframe_mount = (
            np.array(m.body_pos[self.forearm], dtype=float).copy(),
            quat2mat(np.array(m.body_quat[self.forearm], dtype=float).copy()),
        )
        self.keyframe_ctrl = np.array(m.keyframe("pregrasp").ctrl, dtype=float).copy()

        if from_keyframe:
            self.seed_pos, self.seed_mat = self.keyframe_mount
        else:
            self.seed_pos, self.seed_mat = self._seed()

    # -- seeding -----------------------------------------------------------------------
    def _seed(self) -> tuple[np.ndarray, np.ndarray]:
        m, d = self.model, self.data
        text = MENAGERIE_KEYFRAMES.read_text()
        q = np.array(
            [float(x) for x in re.search(r'name="grasp sphere" qpos="([^"]+)"', text).group(1).split()]
        )
        base_quat = m.body_quat[self.forearm].copy()

        m.body_pos[self.forearm] = 0.0
        d.qpos[:] = 0.0
        d.qpos[: q.size] = q
        mujoco.mj_forward(m, d)

        tips = np.array([d.site(t).xpos for t in SEED_TIPS])
        # Orient the tripod plane normal along the cap axis. This normal becomes world +z, so
        # pick the sign pointing from the tips back towards the palm -- that is what leaves the
        # palm above the cap, reaching down, rather than underneath it inside the bottle.
        normal = np.cross(tips[1] - tips[0], tips[2] - tips[0])
        normal /= np.linalg.norm(normal)
        if normal @ (tips.mean(0) - d.body("rh_palm").xpos) > 0:
            normal = -normal
        palm_dir = d.body("rh_palm").xpos - d.body("rh_forearm").xpos

        # Map (palm direction, tripod normal) onto (-x, +z): cap axis up, forearm out along +x.
        rot = frame(np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])) @ frame(palm_dir, normal).T
        mat = rot @ quat2mat(base_quat)

        m.body_quat[self.forearm] = mat2quat(mat)
        mujoco.mj_forward(m, d)
        pos = self.cap_centre - np.array([d.site(t).xpos for t in SEED_TIPS]).mean(0)
        return pos, mat

    def seed_params(self) -> np.ndarray:
        if self.from_keyframe:
            u = self.keyframe_ctrl.copy()
            u[self.roll_act] = ROLL_NEUTRAL
            return np.concatenate([self.seed_pos, np.zeros(3), np.clip(u, self.lo, self.hi)])
        text = MENAGERIE_KEYFRAMES.read_text()
        seed = np.array(
            [float(x) for x in re.search(r'name="grasp sphere" qpos="([^"]+)"', text).group(1).split()]
        )
        # M7: the seed keyframe is Menagerie's, for the hand *without* the forearm roll, so its
        # values are positional against a 24-joint model and this one has 25. Scatter them onto
        # the addresses they belong to rather than assuming the vectors line up -- adding a
        # joint at qpos 0 shifted every one of them, and indexing straight through silently read
        # the wrong joint before it ran off the end.
        q = np.zeros(self.model.nq)
        targets = [a for a in sorted(self._hand_qadrs) if a != self.roll_qid]
        if len(targets) != len(seed):
            raise ValueError(
                f"the Menagerie seed has {len(seed)} joints and this model has {len(targets)} "
                "besides the forearm roll; the seed keyframe needs revisiting"
            )
        q[targets] = seed
        q[self.roll_qid] = ROLL_NEUTRAL
        u = np.array([sum(s * q[adr] for adr, s in pairs) for pairs in self.a2q])
        return np.concatenate([self.seed_pos, np.zeros(3), np.clip(u, self.lo, self.hi)])

    def bounds(self) -> list[tuple[float, float]]:
        b = [(p - 0.08, p + 0.08) for p in self.seed_pos]
        b += [(-0.7, 0.7)] * 3
        limits = list(zip(self.lo, self.hi))
        # M7: the forearm roll is not a free parameter. The pregrasp is the start of a stroke.
        limits[self.roll_act] = (ROLL_NEUTRAL, ROLL_NEUTRAL)
        b += limits
        return b

    # -- forward kinematics ------------------------------------------------------------
    def qpos_of(self, u: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq)
        for a, pairs in enumerate(self.a2q):
            for adr, share in pairs:
                q[adr] = u[a] * share
        return np.clip(q, self.q_lo, self.q_hi)

    def place(self, x: np.ndarray, settle_steps: int = 0) -> None:
        m, d = self.model, self.data
        t, w, u = x[:3], x[3:6], x[6:]
        m.body_pos[self.forearm] = t
        m.body_quat[self.forearm] = mat2quat(rodrigues(w) @ self.seed_mat)
        d.qpos[:] = self.qpos_of(u)
        d.qvel[:] = 0.0
        d.ctrl[:] = u
        if settle_steps:
            # Evaluate the pose the keyframe will actually hold, not the one the targets imply:
            # gravity and contact sag move the fingertips by millimetres.
            margin = m.geom_margin.copy()
            m.geom_margin[:] = self._true_margin
            for _ in range(settle_steps):
                d.ctrl[:] = u
                mujoco.mj_step(m, d)
            m.geom_margin[:] = margin
        mujoco.mj_forward(m, d)

    # -- the pinch, as the reward sees it ----------------------------------------------
    def pad_geometry(self) -> dict[str, tuple[float, float, float]]:
        """`{pad: (surface gap, height above the cap mid-plane, azimuth)}` at the placed pose.

        The same walk `CapTurn.pinch_state` does, through the same `capturn.scene` helpers, so a
        pose that scores well here scores well under the reward. Call `place()` first.
        """
        d = self.data
        origin = np.array(d.site("cap_site").xpos)
        mat = np.array(d.site("cap_site").xmat).reshape(3, 3)
        out = {}
        for name, (ids, radius, half_length) in self.pad_groups.items():
            points = segment_points(
                d.geom_xpos[ids], d.geom_xmat[ids].reshape(-1, 3, 3), half_length
            )
            gap, direction, contact, _ = grasp_geometry(
                points, radius, origin, mat, self.cap_radius, self.cap_half_h
            )
            local = (contact - origin) @ mat
            out[name] = (float(gap), float(local[2]), float(np.arctan2(direction[1], direction[0])))
        return out

    # -- objective ---------------------------------------------------------------------
    def cost(self, x: np.ndarray, settle_steps: int = 0) -> float:
        self.place(x, settle_steps)
        d = self.data
        c = self.cap_centre
        total = 0.0

        # M8: measured on the collision geoms, which is where the reward measures it. Was the
        # two fingertip *sites*, radially against `R + STANDOFF`; the gap below is already
        # surface-to-surface, so the cap radius no longer enters.
        pads = self.pad_geometry()
        gaps = np.array([pads[n][0] for n in PAD_BODIES])
        heights = np.array([pads[n][1] for n in PAD_BODIES])
        total += 1e6 * np.sum((gaps - STANDOFF) ** 2)
        # Asymmetric floor on top of the symmetric target. A pad that starts already loaded
        # against the cap cannot squeeze -- a fit that left the thumb 0.7 mm off the wall scored
        # well on every other term and then gaited at 5 deg where a clean pose gets 88.
        # M5: 5e6 -> 5e8. With only two gaiting tips the optimiser has far fewer competing
        # radial terms and happily traded the floor away for opposition, landing the index
        # 0.18 mm off the cap -- pre-loaded, unable to squeeze, and failing
        # `test_pregrasp_leaves_room_to_squeeze`. The floor has to outrank the opposition term.
        total += 5e8 * np.sum(np.maximum(0.0, MIN_STANDOFF - gaps) ** 2)
        total += 1e6 * np.sum(
            np.maximum(0.0, np.abs(heights - AXIAL_TARGET) - AXIAL_BAND) ** 2
        )

        az = np.array([pads[n][2] for n in PAD_BODIES])
        sep = abs(np.arctan2(np.sin(az[0] - az[1]), np.cos(az[0] - az[1])))
        # Two-sided, unlike the old floor: a pinch is worse when the tips crowd one side *and*
        # when they splay past each other, and only 180 deg lets the squeeze become pure torque.
        total += 2e4 * (TIP_OPPOSITION - sep) ** 2

        idle = np.array([d.site(t).xpos for t in IDLE_TIPS]) - c  # tip sites: what `idle_contact` reads
        idle_r = np.linalg.norm(idle[:, :2], axis=1)
        total += 5e4 * np.sum(np.maximum(0.0, IDLE_RADIUS - idle_r) ** 2)

        for i in range(d.ncon):
            con = d.contact[i]
            touches_bottle = {con.geom1, con.geom2} & self.bottle_geoms
            want = BODY_CLEARANCE if touches_bottle else CLEARANCE
            gap = want - con.dist
            if gap > 0.0:
                total += 2e6 * gap * gap

        mid = 0.5 * (self.lo + self.hi)
        span = np.maximum(self.hi - self.lo, 1e-6)
        total += 0.05 * np.sum(((x[6:] - mid) / span) ** 2)
        return float(total)

    # -- reporting ---------------------------------------------------------------------
    def report(self, x: np.ndarray, settle_steps: int = 0) -> dict:
        self.place(x, settle_steps)
        d = self.data
        c = self.cap_centre
        tips = {t: np.array(d.site(t).xpos) for t in GAIT_TIPS + IDLE_TIPS}
        rel = {k: v - c for k, v in tips.items()}
        worst = []
        for i in range(d.ncon):
            con = d.contact[i]
            if con.dist < CLEARANCE:
                worst.append((float(con.dist), self._label(con.geom1), self._label(con.geom2)))
        worst.sort()
        az = {k: np.arctan2(v[1], v[0]) for k, v in rel.items()}
        seps = [
            float(abs(np.arctan2(np.sin(az[a] - az[b]), np.cos(az[a] - az[b]))))
            for i, a in enumerate(GAIT_TIPS) for b in GAIT_TIPS[i + 1:]
        ]
        pads = self.pad_geometry()
        pad_sep = abs(np.arctan2(
            np.sin(pads["thumb"][2] - pads["index"][2]),
            np.cos(pads["thumb"][2] - pads["index"][2]),
        ))
        return {
            "pads": pads,
            "pad_separation_deg": float(np.degrees(pad_sep)),
            "separations_deg": sorted(np.degrees(seps).round(1).tolist()),
            "radial": {k: float(np.linalg.norm(v[:2])) for k, v in rel.items()},
            "axial": {k: float(v[2]) for k, v in rel.items()},
            "azimuth_deg": {k: float(np.degrees(np.arctan2(v[1], v[0]))) for k, v in rel.items()},
            "surface_gap": {
                k: float(np.hypot(np.linalg.norm(v[:2]) - self.cap_radius,
                                  max(0.0, abs(v[2]) - self.cap_half_h)))
                for k, v in rel.items()
            },
            "violations": worst,
            "ncon": int(d.ncon),
        }

    def _label(self, gid: int) -> str:
        name = self.model.geom(gid).name
        body = self.model.body(self.model.geom_bodyid[gid]).name
        return f"{body}:{name}" if name else f"{body}:geom{gid}"

    # -- settling ----------------------------------------------------------------------
    def settle(self, x: np.ndarray, seconds: float = 2.0) -> tuple[np.ndarray, np.ndarray, dict]:
        """Hold the fitted ctrl targets until the hand stops moving; return the settled state."""
        m, d = self.model, self.data
        m.geom_margin[:] = self._true_margin
        self.place(x)
        u = x[6:].copy()
        n = int(round(seconds / m.opt.timestep))
        for _ in range(n):
            d.ctrl[:] = u
            mujoco.mj_step(m, d)
        qpos = d.qpos.copy()
        settled = {t: float(np.hypot(
            np.linalg.norm((d.site(t).xpos - self.cap_centre)[:2]) - self.cap_radius,
            max(0.0, abs((d.site(t).xpos - self.cap_centre)[2]) - self.cap_half_h)))
            for t in GAIT_TIPS}
        worst = min((d.contact[i].dist for i in range(d.ncon)), default=0.0)
        # The cap drifts a fraction of a degree while the hand settles against it. The env
        # randomises the cap angle on reset anyway, so park the keyframe at zero.
        qpos[self.cap_qadr] = 0.0
        stats = {
            "max_qvel": float(np.abs(d.qvel).max()),
            "cap_angle": float(d.qpos[self.cap_qadr]),
            "ncon": int(d.ncon),
            "gaps": settled,
            "worst_dist": float(worst),
        }
        m.geom_margin[:] = IK_MARGIN
        return qpos, u, stats


def fmt(a: np.ndarray, p: int = 6) -> str:
    return " ".join(f"{v:.{p}g}" for v in np.asarray(a).ravel())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="patch the XML files in place")
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-keyframe", action="store_true",
                    help="seed from the committed pregrasp rather than from Menagerie")
    ap.add_argument("--settled-iters", type=int, default=120,
                    help="L-BFGS-B iterations on the settled objective")
    ap.add_argument("--settled-candidates", type=int, default=5,
                    help="how many of the best kinematic fits to re-rank by settled cost")
    ap.add_argument("--settle-polish", type=float, default=0.6,
                    help="seconds of simulation inside the final objective (0 disables)")
    args = ap.parse_args()

    fit = PregraspFitter(from_keyframe=args.from_keyframe)
    bounds = fit.bounds()
    x0 = fit.seed_params()
    rng = np.random.default_rng(args.seed)

    candidates = []
    for k in range(args.restarts):
        start = x0.copy()
        if k:  # jitter the actuator targets only; the seeded mount pose is already sound
            start[6:] += rng.normal(0.0, 0.12, start.size - 6)
            start[3:6] += rng.normal(0.0, 0.05, 3)
            start = np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds])
        res = optimize.minimize(
            fit.cost, start, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 2000, "maxfun": 200000, "eps": 1e-5, "ftol": 1e-12, "gtol": 1e-10},
        )
        tag = "seed" if k == 0 else f"restart {k}"
        print(f"  {tag:10s} cost {res.fun:12.4f}")
        candidates.append((float(res.fun), res.x))

    candidates.sort(key=lambda c: c[0])
    best, best_cost = candidates[0][1], candidates[0][0]
    res = optimize.minimize(
        fit.cost, best, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 5000, "maxfun": 500000, "eps": 3e-6, "ftol": 1e-14, "gtol": 1e-12},
    )
    if res.fun < best_cost:
        candidates.insert(0, (float(res.fun), res.x))
        best, best_cost = res.x, float(res.fun)
    print(f"  {'polish':10s} cost {res.fun:12.4f}   (kinematic)")

    # The keyframe is the settled pose, so finish by optimising the settled pose itself. The mount
    # is frozen here: only the 20 actuator targets move, which keeps this stage cheap.
    steps = int(round(args.settle_polish / fit.model.opt.timestep))

    # M8: choose which candidate to polish by how it *settles*, not by how it scores
    # kinematically. The two come apart badly. The fit that prompted this scored 1.45 kinematic
    # -- all but perfect -- and 1015 settled: the hand sags onto the bottle shoulder the moment
    # gravity and contact are allowed, putting the index pad 20 mm below the cap. A candidate
    # scoring 4.3 kinematically settled an order of magnitude better. Kinematic cost is a cheap
    # proxy that the optimiser is free to overfit, and the keyframe is the settled pose, so rank
    # by the thing that is actually being chosen.
    if steps and len(candidates) > 1:
        print(f"  ranking the best {min(args.settled_candidates, len(candidates))} candidates by "
              "settled cost ...")
        ranked = []
        for kinematic, x in candidates[:args.settled_candidates]:
            settled = fit.cost(x, settle_steps=steps)
            print(f"    kinematic {kinematic:12.4f}  ->  settled {settled:12.4f}")
            ranked.append((settled, x))
        ranked.sort(key=lambda c: c[0])
        best = ranked[0][1]

    if steps:
        print(f"  settling {args.settle_polish:.2f} s inside the objective "
              f"({steps} steps/evaluation) ...")
        before = fit.cost(best, settle_steps=steps)
        # M8: optimise the *mount* here too, and with a gradient method.
        #
        # This stage used to run Powell over the 20 actuator targets with the mount frozen, and
        # that is the wrong thing to freeze. What goes wrong when the fitted pose is simulated is
        # sag: the hand settles onto the bottle shoulder and the pads slide off the cap. Where
        # the hand hangs is exactly what decides how much it sags, so holding it fixed left the
        # one stage that can see the problem unable to fix it -- measured, every one of eight
        # kinematic candidates settled to a cost above 976 while scoring 1.5 to 53 kinematically,
        # and the frozen-mount polish could only reach 470.
        res = optimize.minimize(
            lambda x: fit.cost(x, settle_steps=steps), best,
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": args.settled_iters, "maxfun": 200000, "eps": 1e-4,
                     "ftol": 1e-10, "gtol": 1e-8},
        )
        if res.fun < before:
            best = res.x
        print(f"  {'settled':10s} cost {min(float(res.fun), before):12.4f}   "
              f"(was {before:.4f} before this stage)")

    rep = fit.report(best, settle_steps=steps)
    # M8 first: this is what the *reward* measures, and what the objective now fits. The tip-site
    # table below it is the legacy view and the two genuinely disagree -- a pose can have its
    # fingertip sites several millimetres from the cap while the geom actually touching it is
    # where it should be, which is how the M7 pose came to be gripping the cap's bottom lip while
    # its tip-site numbers looked centred.
    print("\nthe pinch at the SETTLED pose, measured on the collision geoms as the reward does "
          f"(cap radius {fit.cap_radius*1e3:.0f} mm, half-height {fit.cap_half_h*1e3:.0f} mm):")
    for name, (gap, height, az) in rep["pads"].items():
        print(f"  {name:6s} surface gap {gap*1e3:6.2f} mm   height {height*1e3:+6.2f} mm "
              f"(cap spans {-fit.cap_half_h*1e3:+.0f} to {fit.cap_half_h*1e3:+.0f})   "
              f"azimuth {np.degrees(az):+7.1f} deg")
    print(f"  pad separation {rep['pad_separation_deg']:.1f} deg "
          f"(target {np.degrees(TIP_OPPOSITION):.1f})")
    print("\nfingertip geometry at the SETTLED pose, relative to the cap (radius "
          f"{fit.cap_radius*1e3:.0f} mm, half-height {fit.cap_half_h*1e3:.0f} mm):")
    print(f"  azimuthal separations: {rep['separations_deg']} deg")
    for t in GAIT_TIPS + IDLE_TIPS:
        print(f"  {t:10s} r={rep['radial'][t]*1e3:6.1f} mm  z={rep['axial'][t]*1e3:+6.1f} mm  "
              f"az={rep['azimuth_deg'][t]:+7.1f} deg  gap-to-surface={rep['surface_gap'][t]*1e3:5.1f} mm")
    print(f"near/penetrating pairs closer than {CLEARANCE*1e3:.1f} mm: {len(rep['violations'])}")
    for dist, g1, g2 in rep["violations"][:10]:
        print(f"  {dist*1e3:+7.2f} mm  {g1} / {g2}")

    qpos, ctrl, stats = fit.settle(best)
    print(f"\nafter settling 2 s under the fitted targets: max|qvel|={stats['max_qvel']:.4f} rad/s, "
          f"cap moved {np.degrees(stats['cap_angle']):.3f} deg, ncon={stats['ncon']}")
    print(f"  deepest contact at the settled pose: {stats['worst_dist']*1e3:+.3f} mm")
    for t in GAIT_TIPS:
        print(f"  {t:10s} gap-to-surface={stats['gaps'][t]*1e3:5.1f} mm")

    pos = best[:3]
    quat = mat2quat(rodrigues(best[3:6]) @ fit.seed_mat)
    print("\n--- assets/right_hand_mjx.xml ---")
    print(f'      pos="{fmt(pos)}" quat="{fmt(quat)}"')
    print("--- assets/scene.xml ---")
    print(f'      qpos="{fmt(qpos)}"')
    print(f'      ctrl="{fmt(ctrl)}"')

    if args.write:
        hand = HAND.read_text()
        new_hand, n = re.subn(r'(\n      )pos="[^"]*" quat="[^"]*"',
                              rf'\g<1>pos="{fmt(pos)}" quat="{fmt(quat)}"', hand, count=1)
        assert n == 1, "mount line not found in right_hand_mjx.xml"
        HAND.write_text(new_hand)

        scene = SCENE.read_text()
        new_scene, n = re.subn(r'qpos="[^"]*"', f'qpos="{fmt(qpos)}"', scene, count=1)
        assert n == 1
        new_scene, n = re.subn(r'ctrl="[^"]*"', f'ctrl="{fmt(ctrl)}"', new_scene, count=1)
        assert n == 1
        SCENE.write_text(new_scene)
        print("\npatched assets/right_hand_mjx.xml and assets/scene.xml")


if __name__ == "__main__":
    main()
