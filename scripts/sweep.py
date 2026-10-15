"""PRD M6: perturb one DR parameter at a time and record what it costs the nominal policy.

    python3 scripts/sweep.py logs/capturn-nominal-s1/checkpoints/from000000000000/000150732800
    python3 scripts/sweep.py --episodes 256 --out artifacts/sweep.json
    python3 scripts/sweep.py --only cap_frictionloss,tip_friction

The PRD's procedure, step 1: "run sweep.py on the nominal policy first: perturb one parameter at
a time across its range and record success rate. This shows which ranges matter." The output is a
markdown table meant to be pasted into NOTES.md.

This is *not* `randomize.py` with one field enabled. The randomiser samples every env
independently, which answers "can the policy survive the whole distribution"; a sweep sets one
parameter to one deterministic value across every env and walks it across its PRD range, which
answers "at what value does this policy start to fail". The second question is the one that says
which ranges matter, and it is also the one that produces a table a human can read.

Each point is a fresh `jax.jit` trace, because the model is a different pytree -- roughly a
minute of compile per point on an H200, against a few seconds of rollout. `--episodes` therefore
buys accuracy almost for free: 256 episodes resolve a success rate to about +-2.5 points, 1024 to
about +-1.3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from capturn import checkpoints  # noqa: E402
from capturn import config as capturn_config  # noqa: E402
from capturn import randomize  # noqa: E402
from capturn.env import CapTurn  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import eval as eval_script  # noqa: E402

RAD2DEG = 180.0 / math.pi


# Each entry: the field it writes, how to write it, and the points to walk. The ranges are the
# PRD's; the extra interior points are chosen so a cliff between two PRD bounds is visible rather
# than straddled.
def sweep_points(model):
    """`{name: (description, [(label, apply_fn), ...])}`, built against the nominal model."""
    import jax.numpy as jp

    geoms = randomize.IDS["friction_geoms"]
    cap_dof = randomize.IDS["cap_dof"]
    hand_dofs = randomize.IDS["hand_dofs"]
    hand_qids = randomize.IDS["hand_qids"]
    bottle = randomize.IDS["bottle"]

    def scale_field(field, index, factor):
        def apply(m):
            value = getattr(m, field)
            return m.tree_replace({field: value.at[index].set(value[index] * factor)})
        return apply

    def set_field(field, index, value):
        def apply(m):
            return m.tree_replace({field: getattr(m, field).at[index].set(value)})
        return apply

    def offset_qpos0(delta):
        def apply(m):
            return m.tree_replace(
                {"qpos0": m.qpos0.at[hand_qids].set(m.qpos0[hand_qids] + delta)})
        return apply

    def move_bottle(dx):
        def apply(m):
            pos = m.body_pos.at[bottle, 0].set(m.body_pos[bottle, 0] + dx)
            return m.tree_replace({"body_pos": pos})
        return apply

    def tilt_bottle(deg):
        def apply(m):
            half = 0.5 * jp.deg2rad(deg)
            tilt = jp.array([jp.cos(half), jp.sin(half), 0.0, 0.0])
            q = m.body_quat[bottle]
            w1, x1, y1, z1 = tilt
            w2, x2, y2, z2 = q
            new = jp.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ])
            return m.tree_replace({"body_quat": m.body_quat.at[bottle].set(new)})
        return apply

    nominal_fl = float(model.dof_frictionloss[cap_dof])
    return {
        "tip_friction": (
            "fingertip + cap sliding friction, PRD *U(0.6, 1.4)",
            [(f"x{f:g}", scale_field("geom_friction", (geoms, 0), f))
             for f in (0.6, 0.8, 1.0, 1.2, 1.4)]),
        "cap_frictionloss": (
            f"cap thread friction, PRD U(0.02, 0.15) N*m (nominal {nominal_fl:g})",
            [(f"{v:g}", set_field("dof_frictionloss", cap_dof, v))
             for v in (0.02, 0.035, 0.05, 0.08, 0.11, 0.15)]),
        "cap_damping": (
            "cap hinge damping, PRD *U(0.5, 2.0)",
            [(f"x{f:g}", scale_field("dof_damping", cap_dof, f)) for f in (0.5, 1.0, 2.0)]),
        "actuator_gain": (
            "actuator stiffness kp, PRD *U(0.8, 1.2)",
            [(f"x{f:g}", lambda m, f=f: m.tree_replace({
                "actuator_gainprm": m.actuator_gainprm.at[:, 0].set(m.actuator_gainprm[:, 0] * f),
                "actuator_biasprm": m.actuator_biasprm.at[:, 1].set(-m.actuator_gainprm[:, 0] * f),
            })) for f in (0.8, 0.9, 1.0, 1.1, 1.2)]),
        "hand_damping": (
            "hand joint damping, PRD *U(0.7, 1.3)",
            [(f"x{f:g}", scale_field("dof_damping", hand_dofs, f)) for f in (0.7, 1.0, 1.3)]),
        "hand_frictionloss": (
            "hand joint friction, PRD *U(0.5, 2.0)",
            [(f"x{f:g}", scale_field("dof_frictionloss", hand_dofs, f))
             for f in (0.5, 1.0, 1.5, 2.0)]),
        "hand_armature": (
            "hand armature, PRD *U(1.0, 1.2)",
            [(f"x{f:g}", scale_field("dof_armature", hand_dofs, f)) for f in (1.0, 1.1, 1.2)]),
        "qpos0_offset": (
            "joint zero offset, PRD +-0.03 rad (applied to every hand joint at once)",
            [(f"{d:+g}", offset_qpos0(d)) for d in (-0.03, -0.015, 0.0, 0.015, 0.03)]),
        "bottle_x": (
            "bottle placement along x, PRD +-1.5 cm",
            [(f"{d*100:+g} cm", move_bottle(d)) for d in (-0.015, -0.0075, 0.0, 0.0075, 0.015)]),
        "bottle_tilt": (
            "bottle tilt, PRD +-5 deg",
            [(f"{d:g} deg", tilt_bottle(d)) for d in (0.0, 2.5, 5.0)]),
    }


def evaluate(env: CapTurn, policy, episodes: int, seed: int) -> dict:
    """Success rate and rotation for one perturbed model."""
    traj = eval_script.rollout(env, policy, episodes, seed)
    d_theta = traj["d_theta"]
    reached = np.maximum.accumulate(np.cumsum(d_theta, axis=1), axis=1)[:, -1]
    success = reached >= float(env._config.success_rotation)
    rate = float(success.mean())
    net = d_theta.sum(axis=1)
    alive = traj["alive"]
    return {
        "success_rate": rate,
        "success_stderr": float(math.sqrt(max(rate * (1 - rate), 0.0) / max(len(success), 1))),
        "net_rotation_deg": float(net.mean() * RAD2DEG),
        "drop_rate": float((alive.sum(axis=1) < alive.shape[1]).mean()),
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", nargs="?", default=None,
                    help="default: the newest under logs/capturn-nominal-s1/checkpoints")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-separated parameter names")
    ap.add_argument("--out", default=None, help="write the table as JSON")
    ap.add_argument("--impl", default=None, choices=["warp", "jax"])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.checkpoint is None:
        tree = ROOT / "logs" / "capturn-nominal-s1" / "checkpoints"
        ckpt = checkpoints.find_latest(tree)
        if ckpt is None:
            raise SystemExit(f"no checkpoint under {tree}; pass one explicitly")
        path = pathlib.Path(ckpt.path).resolve()
    else:
        # Orbax refuses a relative path ("Checkpoint path should be absolute"), and a sweep is
        # most naturally invoked with the repo-relative one.
        path = pathlib.Path(args.checkpoint).resolve()

    cfg = capturn_config.default_config()
    if args.impl is not None:
        cfg.impl = args.impl
    # `naconmax` is a total across worlds (M2), so it follows the batch actually being run --
    # the same sizing `eval.py` does, and the reason a sweep at 1024 episodes is not simply the
    # training config with a different number.
    cfg.naconmax = capturn_config.NACONMAX_PER_WORLD * args.episodes
    env = CapTurn(cfg)
    nominal_model = env.mjx_model
    policy = eval_script.make_policy(path, deterministic=True)

    table = sweep_points(nominal_model)
    wanted = args.only.split(",") if args.only else list(table)
    unknown = [w for w in wanted if w not in table]
    if unknown:
        raise SystemExit(f"unknown parameter(s): {', '.join(unknown)}. "
                         f"Known: {', '.join(table)}")

    print(f"checkpoint: {path}")
    print(f"{args.episodes} episodes per point, deterministic policy\n")

    results = {}
    started = time.monotonic()
    for name in wanted:
        description, points = table[name]
        print(f"## {name} -- {description}")
        print(f"  {'value':>10} {'success':>9} {'net deg':>10} {'drops':>7}")
        rows = []
        for label, apply in points:
            env._mjx_model = apply(nominal_model)
            r = evaluate(env, policy, args.episodes, args.seed)
            rows.append({"value": label, **r})
            print(f"  {label:>10} {100*r['success_rate']:>8.1f}% {r['net_rotation_deg']:>10.1f} "
                  f"{100*r['drop_rate']:>6.1f}%")
        results[name] = {"description": description, "points": rows}
        print()
    env._mjx_model = nominal_model

    print(f"done in {time.monotonic() - started:.0f} s\n")
    print("markdown, one row per point:\n")
    print("| parameter | value | success | net deg | drops |")
    print("|---|---|---|---|---|")
    for name, block in results.items():
        for p in block["points"]:
            print(f"| `{name}` | {p['value']} | {100*p['success_rate']:.1f}% | "
                  f"{p['net_rotation_deg']:.0f} | {100*p['drop_rate']:.1f}% |")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"checkpoint": str(path), "episodes": args.episodes, "results": results}, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
