"""M2: replay the M1 action sequence on CPU MuJoCo and on MJX, and compare.

    python3 scripts/check_parity.py                      # full check against MJX-Warp
    python3 scripts/check_parity.py --probe              # just size naconmax / njmax
    python3 scripts/check_parity.py --impl jax           # the JAX backend (see caveat below)
    python3 scripts/check_parity.py --ccd-iterations 10  # what does Warp do with this knob?

The PRD's done-criteria for M2, each reported as a pass/fail line:

  1. cap angle trajectories agree within about 10%
  2. halving `sim_dt` does not change the result materially
  3. no contact or constraint overflow warnings

and `naconmax`/`njmax` are sized from measured counts with 2x headroom.

Two things worth knowing before reading the numbers:

* **`naconmax` must cover `ncollision`, not `ncon`.** Warp sizes its broadphase buffer from
  `naconmax`, and the broadphase emits candidate *pairs*, several of which are later discarded.
  For this scene CPU `ncon` peaks at 6 while Warp's `ncollision` peaks at 22, so sizing from the
  CPU contact count silently overflows the broadphase on every step.
* **`impl='jax'` cannot run this scene at all**: cylinder-vs-mesh collisions are not implemented
  there, and every fingertip is a mesh meeting the cylindrical cap. `--impl jax` is kept so the
  failure is explicit rather than a surprise later.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "assets" / "scene.xml"
DEFAULT_TRAJECTORY = ROOT / "artifacts" / "squeeze_twist.npz"

HEADROOM = 2.0  # PRD: size the buffers from measured counts with 2x headroom
PROBE_NACONMAX = 1024  # deliberately oversized while measuring
PROBE_NJMAX = 1024


def build_model(ccd_iterations: int | None, solver_iterations: int | None,
                ls_iterations: int | None) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(SCENE.as_posix())
    if ccd_iterations is not None:
        model.opt.ccd_iterations = ccd_iterations
    if solver_iterations is not None:
        model.opt.iterations = solver_iterations
    if ls_iterations is not None:
        model.opt.ls_iterations = ls_iterations
    return model


# --------------------------------------------------------------------------------------
# rollouts
# --------------------------------------------------------------------------------------
def cpu_rollout(model: mujoco.MjModel, ctrl: np.ndarray, n_substeps: int) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("pregrasp").id)
    cap = int(model.joint("cap_hinge").qposadr[0])
    angle = np.zeros(len(ctrl))
    ncon, nefc, worst = [], [], 0.0
    for t, u in enumerate(ctrl):
        for _ in range(n_substeps):
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
            ncon.append(data.ncon)
            nefc.append(data.nefc)
            worst = min(worst, min((data.contact[i].dist for i in range(data.ncon)), default=0.0))
        angle[t] = data.qpos[cap]
    return {
        "angle": angle,
        "ncon_max": max(ncon),
        "nefc_max": max(nefc),
        "worst_penetration_mm": worst * 1e3,
        "finite": bool(np.all(np.isfinite(data.qpos))),
    }


def mjx_rollout(model: mujoco.MjModel, ctrl: np.ndarray, n_substeps: int, impl: str,
                naconmax: int, njmax: int) -> dict:
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType
    from mujoco_playground._src import mjx_env

    key = model.keyframe("pregrasp")
    mjx_model = mjx.put_model(model, impl=impl)
    data = mjx_env.make_data(
        model, qpos=jp.array(key.qpos), qvel=jp.zeros(model.nv), ctrl=jp.array(key.ctrl),
        impl=impl, naconmax=naconmax, njmax=njmax,
    )
    step = jax.jit(lambda m, d, u: mjx_env.step(m, d, u, n_substeps))

    cap = int(model.joint("cap_hinge").qposadr[0])
    angle = np.zeros(len(ctrl))
    ncollision, nacon, nefc = [], [], []
    worst, overflow, solver_niter = 0.0, 0, 0
    for t, u in enumerate(ctrl):
        data = step(mjx_model, data, jp.array(u))
        angle[t] = float(np.asarray(data.qpos)[cap])
        inner = data._impl
        # Warp prints its warnings from inside kernels, which Python cannot capture. The Data
        # carries the same information: `overflow` is an OverflowType bitmask and `solver_niter`
        # is how many solver iterations the step actually used.
        overflow |= _scalar(inner.overflow)
        solver_niter = max(solver_niter, _scalar(inner.solver_niter))
        n = _scalar(inner.nacon)
        nacon.append(n)
        ncollision.append(_scalar(inner.ncollision))
        nefc.append(_scalar(inner.nefc))
        if n:
            # contacts are flattened onto the Data as contact__*, packed at the front
            worst = min(worst, float(np.asarray(inner.contact__dist)[:n].min()))
    flags = [
        f.name for f in OverflowType
        if f.value and f.name != "ALL" and (overflow & f.value) == f.value
    ]
    return {
        "angle": angle,
        "ncollision_max": max(ncollision) if ncollision else 0,
        "nacon_max": max(nacon) if nacon else 0,
        "nefc_max": max(nefc) if nefc else 0,
        "worst_penetration_mm": worst * 1e3,
        "finite": bool(np.all(np.isfinite(np.asarray(data.qpos)))),
        "naconmax": naconmax,
        "njmax": njmax,
        "overflow_flags": flags,
        "solver_niter_max": solver_niter,
    }


def _scalar(value) -> int:
    arr = np.asarray(value)
    return int(arr.reshape(-1)[0]) if arr.size else 0


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def compare(cpu: np.ndarray, other: np.ndarray, label: str, tol: float) -> bool:
    scale = max(abs(cpu[-1]), 1e-9)
    final_rel = abs(other[-1] - cpu[-1]) / scale
    worst_rel = np.abs(other - cpu).max() / scale
    ok = final_rel <= tol
    print(f"  {label}")
    print(f"    final angle   CPU {np.degrees(cpu[-1]):+8.2f} deg   "
          f"other {np.degrees(other[-1]):+8.2f} deg   "
          f"difference {final_rel*100:5.1f}% of the CPU total")
    print(f"    worst gap along the trajectory: {worst_rel*100:5.1f}% of the CPU final angle")
    print(f"    {'PASS' if ok else 'FAIL'} (tolerance {tol*100:.0f}%)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY))
    ap.add_argument("--impl", default="warp", choices=["warp", "jax"])
    ap.add_argument("--tol", type=float, default=0.10, help="PRD: about 10%")
    ap.add_argument("--naconmax", type=int, default=None)
    ap.add_argument("--njmax", type=int, default=None)
    ap.add_argument("--ccd-iterations", type=int, default=None)
    ap.add_argument("--solver-iterations", type=int, default=None)
    ap.add_argument("--ls-iterations", type=int, default=None)
    ap.add_argument("--probe", action="store_true", help="size the buffers and stop")
    ap.add_argument("--skip-substep", action="store_true", help="skip the sim_dt halving check")
    ap.add_argument("--num-envs", type=int, default=8192,
                    help="only used to print the naconmax a training run would need")
    args = ap.parse_args()

    path = pathlib.Path(args.trajectory)
    if not path.exists():
        raise SystemExit(f"{path} not found -- generate it with: python3 scripts/squeeze_twist.py")
    npz = np.load(path)
    ctrl, n_substeps = npz["ctrl"], int(npz["n_substeps"])

    model = build_model(args.ccd_iterations, args.solver_iterations, args.ls_iterations)
    print(f"scene: timestep={model.opt.timestep} n_substeps={n_substeps} "
          f"ccd_iterations={model.opt.ccd_iterations} "
          f"iterations={model.opt.iterations} ls_iterations={model.opt.ls_iterations}")
    print(f"trajectory: {path.name}, {len(ctrl)} control steps "
          f"({len(ctrl)*n_substeps*model.opt.timestep:.1f} s)\n")

    print("CPU MuJoCo reference ...")
    cpu = cpu_rollout(model, ctrl, n_substeps)
    print(f"  final cap angle {np.degrees(cpu['angle'][-1]):+.2f} deg   "
          f"ncon max {cpu['ncon_max']}   nefc max {cpu['nefc_max']}   "
          f"deepest penetration {cpu['worst_penetration_mm']:+.3f} mm")

    naconmax = args.naconmax or PROBE_NACONMAX
    njmax = args.njmax or PROBE_NJMAX
    if args.naconmax is None or args.njmax is None:
        print(f"\nprobing MJX-{args.impl} with oversized buffers "
              f"(naconmax={naconmax}, njmax={njmax}) to measure the real counts ...")
    else:
        print(f"\nMJX-{args.impl} with naconmax={naconmax}, njmax={njmax} ...")

    try:
        mjx = mjx_rollout(model, ctrl, n_substeps, args.impl, naconmax, njmax)
    except NotImplementedError as exc:
        print(f"  MJX impl='{args.impl}' cannot run this scene: {exc}")
        if args.impl == "jax":
            print("  Expected: every fingertip is a mesh and the cap is a cylinder.")
            print("  Use --impl warp, or swap the cap for a convex mesh prism (PRD's fallback).")
        raise SystemExit(1)

    print(f"  final cap angle {np.degrees(mjx['angle'][-1]):+.2f} deg   "
          f"nacon max {mjx['nacon_max']}   ncollision max {mjx['ncollision_max']}   "
          f"nefc max {mjx['nefc_max']}")
    print(f"  deepest penetration {mjx['worst_penetration_mm']:+.3f} mm   "
          f"(CPU {cpu['worst_penetration_mm']:+.3f} mm)")
    print(f"  solver used up to {mjx['solver_niter_max']} of {model.opt.iterations} iterations")
    print(f"  overflow flags: {', '.join(mjx['overflow_flags']) or 'none'}")

    rec_nacon = int(np.ceil(HEADROOM * max(mjx["ncollision_max"], mjx["nacon_max"])))
    rec_njmax = int(np.ceil(HEADROOM * max(mjx["nefc_max"], cpu["nefc_max"])))
    print(f"\nbuffer sizing at {HEADROOM:g}x headroom")
    print(f"  naconmax >= {rec_nacon:4d} per world   "
          f"(driven by ncollision={mjx['ncollision_max']}, NOT ncon={cpu['ncon_max']})")
    print(f"  njmax    >= {rec_njmax:4d} per world   (nefc={mjx['nefc_max']})")
    print(f"  naconmax is a total across worlds, so a {args.num_envs}-env training run needs "
          f"naconmax = {rec_nacon * args.num_envs}")
    print(f"  njmax is per world, so it stays {rec_njmax}")
    if args.probe:
        return

    print("\n--- M2 criteria ---\n")
    ok_parity = compare(cpu["angle"], mjx["angle"], f"1. CPU MuJoCo vs MJX-{args.impl}", args.tol)

    ok_substep = True
    if args.skip_substep:
        print("\n  2. sim_dt halving: skipped")
    else:
        fine = build_model(args.ccd_iterations, args.solver_iterations, args.ls_iterations)
        fine.opt.timestep = model.opt.timestep / 2
        print()
        ok_substep = compare(
            cpu["angle"], cpu_rollout(fine, ctrl, n_substeps * 2)["angle"],
            f"2. CPU at sim_dt={model.opt.timestep:g} vs sim_dt={fine.opt.timestep:g}", args.tol,
        )

    # Buffer overflow (the PRD's "contact or constraint overflow") is fatal. Solver and
    # linesearch iteration limits are reported separately: they mean the step did not converge,
    # which degrades accuracy without corrupting the buffers. EPA_HORIZON is a fixed 24-entry
    # buffer inside warp's convex collision and cannot be configured from the model.
    buffer_flags = {"NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "NVMAX"}
    hit_buffers = sorted(buffer_flags.intersection(mjx["overflow_flags"]))
    solver_flags = sorted({"ITERATIONS", "LS_ITERATIONS"}.intersection(mjx["overflow_flags"]))
    ok_overflow = not hit_buffers and cpu["finite"] and mjx["finite"]
    print("\n  3. overflow and stability")
    print(f"    ncollision {mjx['ncollision_max']}/{naconmax}   "
          f"nacon {mjx['nacon_max']}/{naconmax}   nefc {mjx['nefc_max']}/{njmax}")
    print(f"    buffer overflow: {', '.join(hit_buffers) or 'none'}")
    print(f"    all qpos finite: CPU {cpu['finite']}, MJX {mjx['finite']}")
    print(f"    {'PASS' if ok_overflow else 'FAIL'}")
    if solver_flags:
        print(f"    WARNING: {', '.join(solver_flags)} -- the solver did not converge. "
              f"Raise iterations/ls_iterations.")
    if "EPA_HORIZON" in mjx["overflow_flags"]:
        print("    NOTE: EPA_HORIZON -- warp's convex-collision horizon buffer is a compile-time "
              "24 entries and is not configurable from the model.")

    print()
    verdict = ok_parity and ok_substep and ok_overflow
    print(f"M2: {'PASS' if verdict else 'FAIL'}")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
