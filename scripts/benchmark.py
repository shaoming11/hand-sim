"""M3: MJX throughput against `num_envs`, and the cost of the M2 timestep decision.

    python3 scripts/benchmark.py                        # 1024 / 4096 / 8192 envs
    python3 scripts/benchmark.py --num-envs 2048,8192
    python3 scripts/benchmark.py --sim-dt 0.005,0.0025,0.00125   # what M2's fix actually costs
    python3 scripts/benchmark.py --actions random       # instead of replaying the M1 gait

Run this on the GPU box. It works on a CPU-only Mac too, but only for tiny `--num-envs`:
warp-cpu manages a few hundred sim-steps/s, so treat local runs as a correctness check on the
script, never as throughput data.

Before running on the H100:

    export XLA_PYTHON_CLIENT_PREALLOCATE=false   # or JAX takes the VRAM Warp needs
    export MUJOCO_GL=egl

Two things this measures that a naive timing loop would miss:

* **Buffer overflow at scale.** `naconmax` is a total across worlds, so it has to be scaled by
  `num_envs`. If it is too small the sim still runs and the numbers still look plausible, so
  every configuration is checked for overflow and the throughput is reported as invalid if any
  occurred.
* **Compilation is not throughput.** The first call compiles Warp kernels and traces XLA, which
  takes seconds. Warmup iterations are run and discarded, and reported separately.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "assets" / "scene.xml"
DEFAULT_TRAJECTORY = ROOT / "artifacts" / "squeeze_twist.npz"
CTRL_DT = 0.05

# Measured at M2 by check_parity.py --probe, with 2x headroom.
NACONMAX_PER_WORLD = 44
NJMAX = 150

BUFFER_OVERFLOWS = {"NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "NVMAX"}


def fmt_x(value: float) -> str:
    """Realtime factor: sub-1x values round to 0 with a plain integer format."""
    return f"{value:,.0f}" if value >= 10 else f"{value:.2f}"


def load_actions(kind: str, model: mujoco.MjModel, steps: int, seed: int) -> np.ndarray:
    """Control sequence to replay, shape (steps, nu)."""
    if kind == "gait" and DEFAULT_TRAJECTORY.exists():
        ctrl = np.load(DEFAULT_TRAJECTORY)["ctrl"]
        return np.resize(ctrl, (steps, model.nu))
    rng = np.random.default_rng(seed)
    lo, hi = model.actuator_ctrlrange.T
    base = np.array(model.keyframe("pregrasp").ctrl)
    noise = rng.uniform(-0.1, 0.1, size=(steps, model.nu)) * (hi - lo)
    return np.clip(base + np.cumsum(noise, axis=0) * 0.1, lo, hi)


def run_one(model: mujoco.MjModel, num_envs: int, actions: np.ndarray, n_substeps: int,
            naconmax_per_world: int, njmax: int, warmup: int) -> dict:
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType
    from mujoco_playground._src import mjx_env

    key = model.keyframe("pregrasp")
    mjx_model = mjx.put_model(model, impl="warp")
    naconmax = naconmax_per_world * num_envs

    def make(_):
        return mjx_env.make_data(
            model, qpos=jp.array(key.qpos), qvel=jp.zeros(model.nv), ctrl=jp.array(key.ctrl),
            impl="warp", naconmax=naconmax, njmax=njmax,
        )

    step = jax.jit(jax.vmap(lambda d, u: mjx_env.step(mjx_model, d, u, n_substeps)))

    compile_start = time.perf_counter()
    data = jax.vmap(make)(jp.arange(num_envs))
    batch = jp.tile(jp.array(actions[0]), (num_envs, 1))
    data = step(data, batch)
    jax.block_until_ready(data.qpos)
    compile_seconds = time.perf_counter() - compile_start

    for i in range(warmup):
        data = step(data, jp.tile(jp.array(actions[i % len(actions)]), (num_envs, 1)))
    jax.block_until_ready(data.qpos)

    overflow = 0
    start = time.perf_counter()
    for i, action in enumerate(actions):
        data = step(data, jp.tile(jp.array(action), (num_envs, 1)))
        if i % 8 == 0:
            overflow |= int(np.asarray(data._impl.overflow).reshape(-1)[0])
    jax.block_until_ready(data.qpos)
    seconds = time.perf_counter() - start
    overflow |= int(np.asarray(data._impl.overflow).reshape(-1)[0])

    flags = [
        f.name for f in OverflowType
        if f.value and f.name != "ALL" and (overflow & f.value) == f.value
    ]
    control_steps = len(actions) * num_envs
    sim_steps = control_steps * n_substeps
    return {
        "num_envs": num_envs,
        "sim_dt": float(model.opt.timestep),
        "n_substeps": n_substeps,
        "naconmax": naconmax,
        "njmax": njmax,
        "seconds": seconds,
        "compile_seconds": compile_seconds,
        "ms_per_control_step": seconds / len(actions) * 1e3,
        "control_steps_per_s": control_steps / seconds,
        "sim_steps_per_s": sim_steps / seconds,
        "realtime_factor": control_steps * CTRL_DT / seconds,
        "overflow_flags": flags,
        "buffer_overflow": sorted(BUFFER_OVERFLOWS.intersection(flags)),
        "finite": bool(np.all(np.isfinite(np.asarray(data.qpos)))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", default="1024,4096,8192")
    ap.add_argument("--sim-dt", default=None,
                    help="comma list; default is whatever assets/scene.xml uses")
    ap.add_argument("--steps", type=int, default=50, help="timed control steps per configuration")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--actions", default="gait", choices=["gait", "random"])
    ap.add_argument("--naconmax-per-world", type=int, default=NACONMAX_PER_WORLD)
    ap.add_argument("--njmax", type=int, default=NJMAX)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the results as JSON")
    args = ap.parse_args()

    import jax

    print(f"jax backend: {jax.default_backend()}   devices: {jax.devices()}")
    if jax.default_backend() != "gpu":
        print("WARNING: not running on a GPU. Treat these numbers as a smoke test, not data.")
    print()

    env_counts = [int(x) for x in args.num_envs.split(",")]
    base = mujoco.MjModel.from_xml_path(SCENE.as_posix())
    sim_dts = (
        [float(x) for x in args.sim_dt.split(",")] if args.sim_dt else [float(base.opt.timestep)]
    )

    results = []
    for sim_dt in sim_dts:
        model = mujoco.MjModel.from_xml_path(SCENE.as_posix())
        model.opt.timestep = sim_dt
        n_substeps = int(round(CTRL_DT / sim_dt))
        actions = load_actions(args.actions, model, args.steps, args.seed)
        print(f"sim_dt={sim_dt:g} ({n_substeps} substeps per {CTRL_DT:g} s control step), "
              f"actions={args.actions}, {args.steps} timed steps")
        print(f"  {'num_envs':>9}{'ms/step':>10}{'ctrl steps/s':>14}{'sim steps/s':>14}"
              f"{'x realtime':>12}{'compile s':>11}  status")
        for num_envs in env_counts:
            try:
                res = run_one(model, num_envs, actions, n_substeps,
                              args.naconmax_per_world, args.njmax, args.warmup)
            except Exception as exc:  # out of memory is the expected one at 8192
                print(f"  {num_envs:>9}  FAILED: {type(exc).__name__}: {str(exc)[:90]}")
                results.append({"num_envs": num_envs, "sim_dt": sim_dt, "error": str(exc)[:200]})
                continue
            if res["buffer_overflow"]:
                status = f"INVALID: {','.join(res['buffer_overflow'])}"
            elif not res["finite"]:
                status = "INVALID: non-finite qpos"
            elif res["overflow_flags"]:
                status = f"ok ({','.join(res['overflow_flags'])})"
            else:
                status = "ok"
            print(f"  {num_envs:>9}{res['ms_per_control_step']:>10.1f}"
                  f"{res['control_steps_per_s']:>14,.0f}{res['sim_steps_per_s']:>14,.0f}"
                  f"{fmt_x(res['realtime_factor']):>12}{res['compile_seconds']:>11.1f}  {status}")
            results.append(res)
        print()

    valid = [r for r in results if "error" not in r and not r["buffer_overflow"] and r["finite"]]
    if valid:
        best = max(valid, key=lambda r: r["sim_steps_per_s"])
        print(f"highest throughput: num_envs={best['num_envs']} at sim_dt={best['sim_dt']:g} "
              f"-> {best['sim_steps_per_s']:,.0f} sim-steps/s "
              f"({fmt_x(best['realtime_factor'])}x realtime)")
        if len(sim_dts) > 1:
            biggest = max(r["num_envs"] for r in valid)
            row = {r["sim_dt"]: r for r in valid if r["num_envs"] == biggest}
            coarsest = max(row)
            ref = row[coarsest]
            print(f"\ncost of the M2 timestep decision, at num_envs={biggest}")
            print("  (control steps/s is the training-relevant rate: it is policy experience "
                  "per second,")
            print("   and a smaller sim_dt buys accuracy by spending more substeps on each one.)")
            print(f"  {'sim_dt':>10}{'substeps':>10}{'ctrl steps/s':>14}"
                  f"{'vs ' + format(coarsest, 'g'):>12}")
            for dt in sorted(row, reverse=True):
                slowdown = ref["control_steps_per_s"] / row[dt]["control_steps_per_s"]
                print(f"  {dt:>10g}{row[dt]['n_substeps']:>10d}"
                      f"{row[dt]['control_steps_per_s']:>14,.0f}{slowdown:>11.2f}x")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
