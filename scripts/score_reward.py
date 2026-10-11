"""M6: score a saved trajectory with the live reward, on CPU.

The question this answers is the one M5 kept getting wrong: **does the reward prefer the
behaviour the project actually wants?** Through M5 that was only ever checked after a two-hour
GPU run, by reading `eval.py` and arguing about it, and twice the argument turned on a number
that had been computed differently from the reward it claimed to describe. Everything needed to
answer it beforehand was already on disk -- `artifacts/` holds the scripted reference gait and
five recorded policies, including the flick, the one-sided hover and both two-finger seeds.

So this replays those recordings through `CapTurn.evaluate`, which is the env's own per-step
reward, and prints what each one would earn. `tests/test_reward_ranking.py` then pins the
ordering, so a later change to a weight cannot silently invert it again.

    python3 scripts/score_reward.py                       # every trajectory in artifacts/
    python3 scripts/score_reward.py artifacts/foo.npz     # just this one
    python3 scripts/score_reward.py --per-step            # the step-by-step trace too

What it is and is not
---------------------
States are *injected*, not simulated: each recorded `qpos` is written into a `Data` and run
through `mjx.forward`, so the geometry, sensors and actuator forces are exactly those of the
recorded pose. The reward is then the live one, read from the live config. Two approximations
follow from what the recordings contain, and neither touches a term that decides the ranking:

* `qvel` is the control-rate difference of `qpos` (every joint here is a hinge, so this is a
  real velocity, just averaged over 50 ms rather than 1.25 ms). It feeds `joint_vel` only.
* an episode recorded by `eval.py` saves the policy's `action`, and the scripted gait saves
  `ctrl`; for the latter the equivalent action is recovered by inverting the delta integrator,
  `a = (ctrl_t - ctrl_{t-1}) / action_scale`, and the fraction of that which saturates [-1, 1]
  is reported. It feeds `action_rate` only.

Returns are reported per episode *as recorded*, and recordings differ in length (the scripted
gait is 192 steps, an eval episode up to 400), so the table also gives the per-second rate,
which is what to compare across rows.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capturn import config as capturn_config  # noqa: E402
from capturn.env import CapTurn  # noqa: E402

# The reference gait first: every other row is judged against it. The labels say what the
# recording *is*, because that is the thing the ranking is about -- NOTES.md, M5.
KNOWN = (
    ("squeeze_twist.npz", "scripted reference gait (two-finger, re-fitted pose)"),
    ("seed1_best_ep00.npz", "policy: best three-finger seed, opposition reshaped"),
    ("sharp_s1_ep00.npz", "policy: two-finger, sharpened idle cost"),
    ("twofinger_s2_ep00.npz", "policy: two-finger seed 2, failed the task"),
    ("fix5080_ep00.npz", "policy: early flick exploit"),
)


def build_env() -> CapTurn:
    cfg = capturn_config.default_config()
    # One world, so `naconmax` -- a total across worlds since M2 -- is sized for one.
    cfg.naconmax = capturn_config.NACONMAX_PER_WORLD * 2
    return CapTurn(cfg)


def load(path: pathlib.Path, env: CapTurn) -> dict:
    """A recording as `(qpos, qvel, ctrl, action)`, all of length T, plus notes on the recovery."""
    z = np.load(path)
    qpos = np.asarray(z["qpos"], dtype=float)
    dt = float(z["ctrl_dt"]) if "ctrl_dt" in z.files else env.dt
    scale = float(env._config.action_scale)
    lo, hi = np.asarray(env._lowers), np.asarray(env._uppers)

    saturated = None
    if "ctrl" in z.files:
        ctrl = np.asarray(z["ctrl"], dtype=float)
        # Invert the delta integrator to get the action that would have produced this ctrl.
        # `ctrl[0]` is the keyframe target, so the first action is zero by construction.
        raw = np.diff(ctrl, axis=0, prepend=ctrl[:1]) / scale
        action = np.clip(raw, -1.0, 1.0)
        saturated = float((np.abs(raw) > 1.0).mean())
    else:
        action = np.asarray(z["action"], dtype=float)
        # Replay the integrator forward from the keyframe to recover the realized targets.
        ctrl = np.empty_like(action)
        target = np.asarray(env._init_ctrl, dtype=float)
        for t in range(len(action)):
            target = np.clip(target + scale * action[t], lo, hi)
            ctrl[t] = target

    n = min(len(action), len(qpos) - 1) if len(qpos) > len(action) else min(len(action), len(qpos))
    qvel = np.diff(qpos, axis=0) / dt
    return {
        "qpos": qpos[:n],
        "qvel": np.vstack([np.zeros((1, qpos.shape[1])), qvel])[:n],
        "ctrl": ctrl[:n],
        "action": action[:n],
        "dt": dt,
        "saturated": saturated,
        "steps": n,
    }


def make_stepper(env: CapTurn):
    """A jitted `(qpos, qvel, ctrl, action, info) -> (aux, info)` over injected states."""
    base = mjx_env.make_data(
        env.mj_model,
        qpos=jp.array(env._init_q),
        qvel=jp.zeros(env.mjx_model.nv),
        ctrl=jp.array(env._init_ctrl),
        impl=env.mjx_model.impl.value,
        naconmax=env._config.naconmax,
        njmax=env._config.njmax,
    )

    def one(qpos, qvel, ctrl, action, info):
        data = mjx.forward(env.mjx_model, base.replace(qpos=qpos, qvel=qvel, ctrl=ctrl))
        aux = env.evaluate(data, action, info)
        return aux, info

    return jax.jit(one), base


def score(env: CapTurn, traj: dict) -> dict:
    """Every reward term's contribution over the recording, plus the gait diagnostics."""
    step, _ = make_stepper(env)
    scales = env._config.reward_config.scales
    cap_qid = env._cap_qid

    info = {
        "cap_angle_init": jp.array(traj["qpos"][0][cap_qid]),
        "last_cap_angle": jp.array(traj["qpos"][0][cap_qid]),
        "last_act": jp.zeros(env.action_size),
        "last_roll": jp.array(traj["qpos"][0][env._roll_qid]),
        "ungrasped_steps": jp.zeros((), dtype=jp.int32),
    }

    rows = []
    for t in range(traj["steps"]):
        aux, info = step(
            jp.array(traj["qpos"][t]),
            jp.array(traj["qvel"][t]),
            jp.array(traj["ctrl"][t]),
            jp.array(traj["action"][t]),
            info,
        )
        info["last_act"] = jp.array(traj["action"][t])
        rows.append({
            **{f"term/{k}": float(v) for k, v in aux["terms"].items()},
            **{k: float(aux[k]) for k in (
                "d_theta", "thumb_gap", "index_gap", "opposition", "pinch", "bottle",
                "tracked", "freewheel", "d_roll",
            )},
        })

    trace = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    dt, steps = traj["dt"], traj["steps"]
    # Playground's convention: the terms are rates, integrated over the control period.
    contributions = {
        k: float(trace[f"term/{k}"].sum() * scales[k] * dt) for k in scales
    }
    total = sum(contributions.values())
    gross = float(np.abs(trace["d_theta"]).sum())
    return {
        "steps": steps,
        "seconds": steps * dt,
        "contributions": contributions,
        "total": total,
        "per_second": total / (steps * dt),
        "roll_travel_deg": float(np.degrees(np.abs(trace["d_roll"]).sum())),
        "strokes": float(np.degrees(np.abs(trace["d_roll"]).sum()) / 180.0),
        "net_rotation_deg": float(np.degrees(trace["d_theta"].sum())),
        "gross_rotation_deg": float(np.degrees(gross)),
        "tracked_rotation_deg": float(np.degrees(trace["tracked"].sum())),
        "tracked_fraction": float(trace["tracked"].sum() / gross) if gross > 0 else 0.0,
        "freewheel_deg": float(np.degrees(trace["freewheel"].sum())),
        "pinch_median": float(np.median(trace["pinch"])),
        "opposition_median": float(np.median(trace["opposition"])),
        "thumb_gap_mm": float(np.median(trace["thumb_gap"]) * 1000),
        "index_gap_mm": float(np.median(trace["index_gap"]) * 1000),
        "bottle_steps": float((trace["bottle"] > 0).mean()),
        "saturated": traj["saturated"],
        "trace": trace,
    }


def report(name: str, label: str, res: dict, per_step: bool = False) -> None:
    print(f"\n{name}  --  {label}")
    print(f"  {res['steps']} steps ({res['seconds']:.1f} s)"
          + ("" if res["saturated"] is None
             else f", action recovery saturated on {res['saturated']*100:.1f}% of entries"))
    print("  reward term            contribution   per second")
    for k, v in sorted(res["contributions"].items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:<20} {v:+12.3f} {v / res['seconds']:+12.4f}")
    print(f"    {'TOTAL':<20} {res['total']:+12.3f} {res['per_second']:+12.4f}")
    print(f"  roll travel {res['roll_travel_deg']:.0f} deg   "
          f"rotation net {res['net_rotation_deg']:+.1f} deg, gross {res['gross_rotation_deg']:.1f}, "
          f"roll-tracked {res['tracked_rotation_deg']:.1f} ({res['tracked_fraction']*100:.1f}%), "
          f"freewheel {res['freewheel_deg']:.1f}")
    print(f"  pinch med {res['pinch_median']:.3f}   opposition med {res['opposition_median']:+.3f}   "
          f"gaps th/ff {res['thumb_gap_mm']:.2f}/{res['index_gap_mm']:.2f} mm   "
          f"bottle on {res['bottle_steps']*100:.1f}% of steps")
    if per_step:
        print("    t   th_mm   ff_mm     opp   pinch   dtheta_deg  track_deg  droll  twist")
        tr = res["trace"]
        for t in range(res["steps"]):
            print(f"  {t:3d} {tr['thumb_gap'][t]*1000:7.2f} {tr['index_gap'][t]*1000:7.2f} "
                  f"{tr['opposition'][t]:+7.3f} {tr['pinch'][t]:7.3f} "
                  f"{np.degrees(tr['d_theta'][t]):+11.2f} {np.degrees(tr['tracked'][t]):9.2f} "
                  f"{np.degrees(tr['d_roll'][t]):+6.2f} {tr['term/twist'][t]:+7.3f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", type=pathlib.Path,
                    help="trajectory .npz files (default: the known recordings in artifacts/)")
    ap.add_argument("--per-step", action="store_true", help="print the step-by-step trace")
    args = ap.parse_args(argv)

    if args.paths:
        wanted = [(p, "") for p in args.paths]
    else:
        wanted = [(ROOT / "artifacts" / n, label) for n, label in KNOWN]

    env = build_env()
    results = []
    for path, label in wanted:
        if not path.exists():
            print(f"\n{path.name}  --  MISSING (regenerate with scripts/squeeze_twist.py "
                  f"or scripts/eval.py)")
            continue
        res = score(env, load(path, env))
        report(path.name, label or "", res, args.per_step)
        results.append((path.name, res))

    if len(results) > 1:
        print("\n" + "=" * 78)
        print("ranking by return per second (the reference gait must come first)")
        for name, res in sorted(results, key=lambda kv: -kv[1]["per_second"]):
            print(f"  {res['per_second']:+9.4f}/s  {res['total']:+10.3f} over {res['seconds']:5.1f} s"
                  f"   strokes {res['strokes']:3.0f}   {name}")


if __name__ == "__main__":
    main()
