"""M1 feasibility check: a crude open-loop squeeze-and-twist, on CPU MuJoCo.

Question this answers: with `frictionloss=0.05` on the cap hinge, can this hand turn the cap at
all? If a scripted gait cannot move it a few degrees, no amount of RL will, and the hand
placement or the contact settings need fixing first.

The Shadow Hand's base is fixed and it has no forearm roll (only `rh_WRJ1`/`rh_WRJ2`, which
flex and deviate), so the twist has to come from the fingers themselves. By default the wrist is
frozen, so this really is a finger gait rather than a wrist turn.

The gait is a four-phase cycle in the cap's cylindrical frame, applied to the chosen fingertips:

    squeeze   bring the tips onto the cap wall and wind in the grip
    twist     sweep them through `+twist` about the cap axis, gripping -- unscrew direction
    release   back the tips off to `cap_radius + release` and unwind the grip
    return    sweep back to the starting azimuth, clear of the cap

Where the fingers *go* and how hard they *squeeze* are solved separately. The IK tracks targets
on the cap wall itself, which keeps the postures well conditioned; grip force is then a constant
extra flexion added on top, each joint moving the way its own fingertip Jacobian says is
radially inward. Folding the squeeze into the IK target instead (aiming inside the cap wall)
makes the solver contort the hand at depth: tips slide off the bottom of the cap onto the neck,
and rotation *falls* as the squeeze deepens.

The actuator trajectory is solved *offline* by damped-least-squares IK against the fingertip
site Jacobians, then replayed as a fixed array of ctrl values. That keeps it genuinely
open-loop, so `check_parity.py` can replay the identical sequence on CPU MuJoCo and MJX-Warp
at M2.

    python3 scripts/squeeze_twist.py                     # default gait, report + save .npz
    python3 scripts/squeeze_twist.py --fingers th,ff     # the PRD's two-finger version
    python3 scripts/squeeze_twist.py --sweep             # scan grip strength x twist angle
"""

from __future__ import annotations

import argparse
import pathlib

import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fit_pregrasp import actuator_to_qpos_map  # noqa: E402
from view_scene import load  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "squeeze_twist.npz"

FINGER_TIPS = {
    "th": "rh_th_tip",
    "ff": "rh_ff_tip",
    "mf": "rh_mf_tip",
    "rf": "rh_rf_tip",
    "lf": "rh_lf_tip",
}
# Grip force comes from flexion. The finger abduction joints (J4, and LFJ5) have a 0.70 rad
# ctrlrange and weak actuators: letting the least-squares solver spend grip on them slams the
# joint to its stop and wedges the fingertip sideways into the cap, which blows the contact up to
# hundreds of newtons. The thumb keeps all of its joints -- THJ5 is its strongest (+-3 N*m) and is
# how a thumb opposes in the first place.
GRIP_EXCLUDE = ("rh_A_FFJ4", "rh_A_MFJ4", "rh_A_RFJ4", "rh_A_LFJ4", "rh_A_LFJ5", "rh_A_THJ3")

FINGER_ACTUATORS = {
    "th": "rh_A_TH",
    "ff": "rh_A_FF",
    "mf": "rh_A_MF",
    "rf": "rh_A_RF",
    "lf": "rh_A_LF",
    "wrist": "rh_A_WRJ",
}


def ease(t: np.ndarray) -> np.ndarray:
    """Smooth 0->1 ramp, so the open-loop targets do not step discontinuously."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))


class OpenLoopGait:
    def __init__(self, fingers: list[str], use_wrist: bool = False) -> None:
        self.model, self.data = load("pregrasp")
        m = self.model
        self.fingers = fingers
        self.tips = [FINGER_TIPS[f] for f in fingers]
        self.tip_ids = [m.site(t).id for t in self.tips]

        self.cap_geom = m.geom("cap").id
        self.cap_radius = float(m.geom_size[self.cap_geom, 0])
        self.cap_site = m.site("cap_site").id
        self.cap_qadr = int(m.joint("cap_hinge").qposadr[0])
        self.bottle_geoms = {m.geom("bottle_body").id, m.geom("bottle_neck").id}

        self.u0 = self.data.ctrl.copy()
        self.lo, self.hi = m.actuator_ctrlrange.T.copy()

        # actuator -> dof matrix, so the IK can work directly in the 20-dim ctrl space and the
        # coupled J2/J1 tendons come out right
        self.a2q = actuator_to_qpos_map(m)
        self.A = np.zeros((m.nv, m.nu))
        for a, pairs in enumerate(self.a2q):
            for adr, share in pairs:
                joint = int(np.flatnonzero(m.jnt_qposadr == adr)[0])
                self.A[m.jnt_dofadr[joint], a] = share

        allowed = [FINGER_ACTUATORS[f] for f in fingers] + (
            [FINGER_ACTUATORS["wrist"]] if use_wrist else []
        )
        self.mask = np.array(
            [any(m.actuator(a).name.startswith(p) for p in allowed) for a in range(m.nu)]
        )

        self.grip_mask = self.mask & np.array(
            [m.actuator(a).name not in GRIP_EXCLUDE for a in range(m.nu)]
        )
        self.span = self.hi - self.lo

        self.q_lo = np.full(m.nq, -np.inf)
        self.q_hi = np.full(m.nq, np.inf)
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                adr = int(m.jnt_qposadr[j])
                self.q_lo[adr], self.q_hi[adr] = m.jnt_range[j]

        # starting cylindrical coordinates of each tip, in the cap frame
        self.start = np.array([self._cyl(self.data.site(t).xpos) for t in self.tips])

    # -- cap-frame cylindrical coordinates ---------------------------------------------
    def _cap_frame(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site(self.cap_site).xpos.copy(), self.data.site(
            self.cap_site
        ).xmat.reshape(3, 3).copy()

    def _cyl(self, world: np.ndarray) -> np.ndarray:
        origin, mat = self._cap_frame()
        p = mat.T @ (world - origin)
        return np.array([np.linalg.norm(p[:2]), np.arctan2(p[1], p[0]), p[2]])

    def _world(self, cyl: np.ndarray) -> np.ndarray:
        origin, mat = self._cap_frame()
        r, th, z = cyl
        return origin + mat @ np.array([r * np.cos(th), r * np.sin(th), z])

    # -- the gait path -----------------------------------------------------------------
    def plan(self, twist: float, release: float, cycles: int, grip_z: float = 0.0,
             steps=(4, 6, 2, 4)) -> tuple[np.ndarray, np.ndarray]:
        """Tip targets (T, n_fingers, 3) in world coordinates, and the grip envelope (T,).

        Tips track the cap wall at `grip_z` above the cap mid-plane. Holding them at their
        pregrasp heights instead lets the first fingertip drift to the cap's bottom edge and slide
        onto the neck.
        """
        n_sq, n_tw, n_rl, n_rt = steps
        nf = len(self.tips)
        r_on, r_out = self.cap_radius, self.cap_radius + release
        path, grip = [], []

        def frame(radius, theta_of, height_of):
            return np.stack([np.array([radius, theta_of(i), height_of(i)]) for i in range(nf)])

        for _ in range(cycles):
            for k in range(n_sq):
                a = ease((k + 1) / n_sq)
                path.append(frame(self.start[:, 0].mean() + (r_on - self.start[:, 0].mean()) * a,
                                  lambda i: self.start[i, 1],
                                  lambda i: self.start[i, 2] + (grip_z - self.start[i, 2]) * a))
                grip.append(a)
            for k in range(n_tw):
                a = ease((k + 1) / n_tw)
                path.append(frame(r_on, lambda i, a=a: self.start[i, 1] + twist * a,
                                  lambda i: grip_z))
                grip.append(1.0)
            for k in range(n_rl):
                a = ease((k + 1) / n_rl)
                path.append(frame(r_on + (r_out - r_on) * a,
                                  lambda i: self.start[i, 1] + twist, lambda i: grip_z))
                grip.append(1.0 - a)
            for k in range(n_rt):
                a = ease((k + 1) / n_rt)
                path.append(frame(r_out, lambda i, a=a: self.start[i, 1] + twist * (1.0 - a),
                                  lambda i: grip_z))
                grip.append(0.0)
        world = np.array([[self._world(c) for c in f] for f in np.array(path)])
        return world, np.array(grip)

    def grip_offset(self, grip: float) -> np.ndarray:
        """Extra flexion, as a fraction of each flexion actuator's remaining travel.

        Each joint moves the way its own fingertip Jacobian says is radially inward, so the
        direction is read off the model rather than assumed from the sign of the ctrlrange. A
        least-squares solve against a radial target was tried first and is worse: it concentrates
        the whole grip on whichever joint has the best lever at the pregrasp pose, overshoots its
        travel, and wedges the fingertip into the cap.
        """
        m, d = self.model, self.data
        d.qpos[:] = self.qpos_of(self.u0)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        origin, mat = self._cap_frame()
        jac = np.zeros((3, m.nv))
        offset = np.zeros(m.nu)
        for finger, sid in zip(self.fingers, self.tip_ids):
            mujoco.mj_jacSite(m, d, jac, None, sid)
            columns = jac @ self.A
            p = mat.T @ (d.site(sid).xpos - origin)
            inward = -mat @ np.array([p[0], p[1], 0.0]) / np.linalg.norm(p[:2])
            for a in range(m.nu):
                if not self.grip_mask[a] or not m.actuator(a).name.startswith(
                    FINGER_ACTUATORS[finger]
                ):
                    continue
                toward = inward @ columns[:, a]
                if abs(toward) < 1e-4:
                    continue
                bound = self.hi[a] if toward > 0 else self.lo[a]
                offset[a] = grip * (bound - self.u0[a])
        return offset

    # -- offline IK --------------------------------------------------------------------
    def solve(self, path: np.ndarray, iters: int = 30, damping: float = 0.05) -> np.ndarray:
        """Damped-least-squares IK in actuator space, run without contact."""
        m, d = self.model, self.data
        u = self.u0.copy()
        out = np.zeros((len(path), m.nu))
        jac = np.zeros((3, m.nv))
        for t, targets in enumerate(path):
            for _ in range(iters):
                d.qpos[:] = self.qpos_of(u)
                d.qvel[:] = 0.0
                mujoco.mj_forward(m, d)
                rows, err = [], []
                for i, sid in enumerate(self.tip_ids):
                    mujoco.mj_jacSite(m, d, jac, None, sid)
                    rows.append((jac @ self.A)[:, self.mask])
                    err.append(targets[i] - d.site(sid).xpos)
                J = np.vstack(rows)
                dx = np.concatenate(err)
                du = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), dx)
                u[self.mask] = np.clip(u[self.mask] + du, self.lo[self.mask], self.hi[self.mask])
            out[t] = u
        return out

    def qpos_of(self, u: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq)
        for a, pairs in enumerate(self.a2q):
            for adr, share in pairs:
                q[adr] = u[a] * share
        return np.clip(q, self.q_lo, self.q_hi)

    # -- replay ------------------------------------------------------------------------
    def rollout(self, ctrl: np.ndarray, n_substeps: int = 10) -> dict:
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.keyframe("pregrasp").id)
        angle = np.zeros(len(ctrl))
        qpos = np.zeros((len(ctrl), m.nq))
        worst_pen, bottle_hits, cap_contacts = 0.0, 0, []
        for t, u in enumerate(ctrl):
            for _ in range(n_substeps):
                d.ctrl[:] = u
                mujoco.mj_step(m, d)
            angle[t] = d.qpos[self.cap_qadr]
            qpos[t] = d.qpos
            n_cap = 0
            for i in range(d.ncon):
                con = d.contact[i]
                worst_pen = min(worst_pen, con.dist)
                if {con.geom1, con.geom2} & self.bottle_geoms:
                    bottle_hits += 1
                if self.cap_geom in (con.geom1, con.geom2):
                    n_cap += 1
            cap_contacts.append(n_cap)
        return {
            "angle": angle,
            "qpos": qpos,
            "total_deg": float(np.degrees(angle[-1])),
            "worst_penetration_mm": worst_pen * 1e3,
            "bottle_contact_steps": bottle_hits,
            "mean_cap_contacts": float(np.mean(cap_contacts)),
            "finite": bool(np.all(np.isfinite(d.qpos))),
        }


def run(fingers, grip, twist, release, cycles, use_wrist, grip_z=0.0, steps=(4, 6, 2, 4)):
    gait = OpenLoopGait(fingers, use_wrist=use_wrist)
    path, envelope = gait.plan(twist, release, cycles, grip_z, steps)
    ctrl = gait.solve(path) + envelope[:, None] * gait.grip_offset(grip)[None, :]
    ctrl = np.clip(ctrl, gait.lo, gait.hi)
    return gait, ctrl, gait.rollout(ctrl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingers", default="th,ff,mf", help="comma-separated: th,ff,mf,rf,lf")
    ap.add_argument("--grip", type=float, default=0.25,
                    help="grip strength: fraction of each flexion joint's remaining travel")
    ap.add_argument("--twist", type=float, default=0.9, help="azimuth sweep per cycle [rad]")
    ap.add_argument("--release", type=float, default=0.008, help="back-off radius [m]")
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--phase-steps", default="4,6,2,4",
                    help="control steps for squeeze,twist,release,return")
    ap.add_argument("--grip-z", type=float, default=0.0,
                    help="tip height above the cap mid-plane while gripping [m]")
    ap.add_argument("--use-wrist", action="store_true", help="let the IK move the wrist too")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    fingers = [f.strip() for f in args.fingers.split(",") if f.strip()]
    phase_steps = tuple(int(x) for x in args.phase_steps.split(","))
    assert len(phase_steps) == 4, "--phase-steps needs four values"

    if args.sweep:
        twists = (0.3, 0.5, 0.7)
        print(f"cap rotation [deg] after {args.cycles} cycles "
              f"(deepest penetration in brackets)\n")
        for fset in (["th", "ff"], ["th", "ff", "mf"]):
            print(f"  fingers = {'+'.join(fset)}")
            print("    grip \\ twist  " + "".join(f"{np.degrees(t):>18.0f} deg" for t in twists))
            for grip in (0.15, 0.25, 0.35, 0.50):
                row = []
                for twist in twists:
                    _, _, res = run(fset, grip, twist, args.release, args.cycles,
                                    args.use_wrist, args.grip_z, phase_steps)
                    row.append(f"{res['total_deg']:10.1f} [{res['worst_penetration_mm']:5.2f}]")
                print(f"    {grip:8.2f}    " + "".join(row))
            print()
        return

    gait, ctrl, res = run(fingers, args.grip, args.twist, args.release, args.cycles,
                          args.use_wrist, args.grip_z, phase_steps)
    period = sum(phase_steps)
    per_cycle = np.degrees(res["angle"][period - 1::period])
    print(f"fingers={'+'.join(fingers)}  grip={args.grip:.2f}  "
          f"twist={np.degrees(args.twist):.0f} deg  release={args.release*1e3:.0f} mm  "
          f"cycles={args.cycles}x{period} steps  "
          f"wrist={'free' if args.use_wrist else 'frozen'}")
    secs = period * args.cycles * 0.05
    print(f"  cap rotation           {res['total_deg']:+.2f} deg over {secs:.1f} s "
          f"({res['total_deg']/secs:+.1f} deg/s)")
    print(f"  per cycle              {np.array2string(per_cycle, precision=1)}")
    print(f"  deepest penetration    {res['worst_penetration_mm']:+.3f} mm")
    print(f"  steps touching bottle  {res['bottle_contact_steps']}")
    print(f"  mean cap contacts      {res['mean_cap_contacts']:.2f}")
    print(f"  all finite             {res['finite']}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out, ctrl=ctrl, qpos=res["qpos"], cap_angle=res["angle"], fingers=np.array(fingers),
        ctrl_dt=0.05, sim_dt=gait.model.opt.timestep, n_substeps=10, keyframe="pregrasp",
        grip=args.grip, twist=args.twist, release=args.release, cycles=args.cycles,
        phase_steps=np.array(phase_steps), grip_z=args.grip_z,
    )
    print(f"\nsaved {out.relative_to(ROOT)}  ctrl{ctrl.shape} -- replayed by check_parity.py at M2")


if __name__ == "__main__":
    main()
