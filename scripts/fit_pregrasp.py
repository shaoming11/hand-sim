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
   every geom pair clear of penetration.
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

GAIT_TIPS = ("rh_th_tip", "rh_ff_tip", "rh_mf_tip")
IDLE_TIPS = ("rh_rf_tip", "rh_lf_tip")

STANDOFF = 0.003  # fingertip pad to cap wall at the pregrasp pose [m]
MIN_STANDOFF = 0.002  # ... and no gaiting tip may start closer than this [m]
CLEARANCE = 0.002  # every geom pair must stay at least this far apart [m]
BODY_CLEARANCE = 0.006  # ... and the hand must keep this far off the bottle itself [m]
AXIAL_BAND = 0.004  # keep gaiting tips within this much of the cap mid-plane [m]
IDLE_RADIUS = 0.030  # keep ring/little tips at least this far off the cap axis [m]
MIN_TIP_SEP = 1.75  # minimum azimuthal separation between gaiting tips [rad]
IK_MARGIN = 0.01  # temporary geom margin, so near-contacts are visible to the fitter [m]


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
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(SCENE.as_posix())
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.forearm = m.body("rh_forearm").id
        self.cap_geom = m.geom("cap").id
        self.cap_radius, self.cap_half_h = cap_dimensions(m)
        self.cap_centre = np.array(m.body("cap").pos) + np.array(m.body("bottle").pos)
        self.cap_qadr = int(m.joint("cap_hinge").qposadr[0])
        self.bottle_geoms = {m.geom("bottle_body").id, m.geom("bottle_neck").id}

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

        tips = np.array([d.site(t).xpos for t in GAIT_TIPS])
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
        pos = self.cap_centre - np.array([d.site(t).xpos for t in GAIT_TIPS]).mean(0)
        return pos, mat

    def seed_params(self) -> np.ndarray:
        text = MENAGERIE_KEYFRAMES.read_text()
        q = np.array(
            [float(x) for x in re.search(r'name="grasp sphere" qpos="([^"]+)"', text).group(1).split()]
        )
        u = np.array([sum(s * q[adr] for adr, s in pairs) for pairs in self.a2q])
        return np.concatenate([self.seed_pos, np.zeros(3), np.clip(u, self.lo, self.hi)])

    def bounds(self) -> list[tuple[float, float]]:
        b = [(p - 0.08, p + 0.08) for p in self.seed_pos]
        b += [(-0.7, 0.7)] * 3
        b += list(zip(self.lo, self.hi))
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

    # -- objective ---------------------------------------------------------------------
    def cost(self, x: np.ndarray, settle_steps: int = 0) -> float:
        self.place(x, settle_steps)
        d = self.data
        c, R, H = self.cap_centre, self.cap_radius, self.cap_half_h
        total = 0.0

        tips = np.array([d.site(t).xpos for t in GAIT_TIPS]) - c
        radial = np.linalg.norm(tips[:, :2], axis=1)
        total += 1e6 * np.sum((radial - (R + STANDOFF)) ** 2)
        # Asymmetric floor on top of the symmetric target. A tip that starts already loaded
        # against the cap cannot squeeze -- a fit that left the thumb 0.7 mm off the wall scored
        # well on every other term and then gaited at 5 deg where a clean pose gets 88.
        total += 5e6 * np.sum(np.maximum(0.0, (R + MIN_STANDOFF) - radial) ** 2)
        total += 1e6 * np.sum(np.maximum(0.0, np.abs(tips[:, 2]) - AXIAL_BAND) ** 2)

        az = np.arctan2(tips[:, 1], tips[:, 0])
        for i in range(3):
            for j in range(i + 1, 3):
                sep = abs(np.arctan2(np.sin(az[i] - az[j]), np.cos(az[i] - az[j])))
                total += 2e4 * max(0.0, MIN_TIP_SEP - sep) ** 2

        idle = np.array([d.site(t).xpos for t in IDLE_TIPS]) - c
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
        return {
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
    ap.add_argument("--settle-polish", type=float, default=0.6,
                    help="seconds of simulation inside the final objective (0 disables)")
    args = ap.parse_args()

    fit = PregraspFitter()
    bounds = fit.bounds()
    x0 = fit.seed_params()
    rng = np.random.default_rng(args.seed)

    best, best_cost = None, np.inf
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
        if res.fun < best_cost:
            best, best_cost = res.x, res.fun

    res = optimize.minimize(
        fit.cost, best, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 5000, "maxfun": 500000, "eps": 3e-6, "ftol": 1e-14, "gtol": 1e-12},
    )
    if res.fun < best_cost:
        best, best_cost = res.x, res.fun
    print(f"  {'polish':10s} cost {res.fun:12.4f}   (kinematic)")

    # The keyframe is the settled pose, so finish by optimising the settled pose itself. The mount
    # is frozen here: only the 20 actuator targets move, which keeps this stage cheap.
    steps = int(round(args.settle_polish / fit.model.opt.timestep))
    if steps:
        print(f"  settling {args.settle_polish:.2f} s inside the objective "
              f"({steps} steps/evaluation) ...")
        mount = best[:6].copy()
        res = optimize.minimize(
            lambda u: fit.cost(np.concatenate([mount, u]), settle_steps=steps),
            best[6:], method="Powell", bounds=bounds[6:],
            options={"maxiter": 40, "maxfev": 24000, "xtol": 1e-4, "ftol": 1e-6},
        )
        settled_cost = res.fun
        before = fit.cost(best, settle_steps=steps)
        if settled_cost < before:
            best = np.concatenate([mount, res.x])
        print(f"  {'settled':10s} cost {min(settled_cost, before):12.4f}   "
              f"(was {before:.4f} before this stage)")

    rep = fit.report(best, settle_steps=steps)
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
