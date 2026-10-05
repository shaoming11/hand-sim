"""Open the CPU MuJoCo viewer on `assets/scene.xml`.

    python3 scripts/view_scene.py             # static, at the pregrasp keyframe
    python3 scripts/view_scene.py --play      # replay the open-loop gait, looping
    python3 scripts/view_scene.py --report    # headless: print the scene summary and exit

Static mode starts at the `pregrasp` keyframe and holds it. Ctrl+drag the cap to check that the
hinge turns and that the hand is not penetrating the bottle.

`--play` re-simulates `artifacts/squeeze_twist.npz` in real time and loops it, so the gait is
driven by physics rather than scrubbed from recorded positions. Generate that file first with
`python3 scripts/squeeze_twist.py`; `--play PATH` takes any other ctrl trajectory.

Fingertip sites and the cap site are in visualisation group 4 -- turn on "site" under the
Rendering panel to see them. The cameras are `front`, `cap` and `back`; cycle with `[` and `]`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# The scene path and the cap geometry live in `capturn.scene` so the env and these CPU scripts
# cannot drift apart about where the cap wall is. Re-exported here, because `squeeze_twist.py`,
# `fit_pregrasp.py` and the tests all import them from this module.
from capturn.scene import (  # noqa: E402
    ALL_TIPS, GAIT_TIPS, SCENE, cap_dimensions, surface_distance,
)

DEFAULT_TRAJECTORY = ROOT / "artifacts" / "squeeze_twist.npz"


def load(key: str = "pregrasp") -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(SCENE.as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe(key).id)
    mujoco.mj_forward(model, data)
    return model, data


def cap_surface_distance(model: mujoco.MjModel, data: mujoco.MjData, site: str) -> float:
    """Distance from a site to the cap's surface, from an `MjData`.

    The env gets the same number from the fingertip `framepos` sensors, which are already in the
    cap frame; here the site has to be rotated into it first.
    """
    rel = data.site(site).xpos - data.site("cap_site").xpos
    local = data.site("cap_site").xmat.reshape(3, 3).T @ rel
    return float(surface_distance(local, *cap_dimensions(model)))


def report(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    print(f"scene: nq={model.nq} nv={model.nv} nu={model.nu} ngeom={model.ngeom} "
          f"nsensor={model.nsensor} nexclude={model.nexclude}")
    print(f"timestep={model.opt.timestep} integrator={mujoco.mjtIntegrator(model.opt.integrator).name} "
          f"cone={mujoco.mjtCone(model.opt.cone).name} impratio={model.opt.impratio} "
          f"solver={mujoco.mjtSolver(model.opt.solver).name} "
          f"iterations={model.opt.iterations} ls_iterations={model.opt.ls_iterations}")

    print("\nactuators (forcerange is what limits fingertip force):")
    for a in range(model.nu):
        print(f"  {model.actuator(a).name:12s} kp={model.actuator_gainprm[a, 0]:5.2f} "
              f"ctrlrange=[{model.actuator_ctrlrange[a, 0]:+.4f}, {model.actuator_ctrlrange[a, 1]:+.4f}] "
              f"forcerange=[{model.actuator_forcerange[a, 0]:+.1f}, {model.actuator_forcerange[a, 1]:+.1f}]")

    radius, half_h = cap_dimensions(model)
    print(f"\ncap: {mujoco.mjtGeom(model.geom_type[model.geom('cap').id]).name} "
          f"radius {radius*1e3:.1f} mm, half-height {half_h*1e3:.1f} mm")
    print("fingertips at the pregrasp pose:")
    for tip in ALL_TIPS:
        mark = "*" if tip in GAIT_TIPS else " "
        print(f" {mark}{tip:10s} world={np.array2string(data.site(tip).xpos, precision=4)} "
              f"gap to cap surface={cap_surface_distance(model, data, tip)*1e3:6.1f} mm")

    print(f"\ncontacts at the pregrasp pose: {data.ncon}")
    for i in range(data.ncon):
        con = data.contact[i]
        names = []
        for gid in (con.geom1, con.geom2):
            gname = model.geom(gid).name or f"geom{gid}"
            names.append(f"{model.body(model.geom_bodyid[gid]).name}:{gname}")
        print(f"  dist={con.dist*1e3:+7.3f} mm  {names[0]} / {names[1]}")


def play(model: mujoco.MjModel, data: mujoco.MjData, ctrl: np.ndarray, n_substeps: int,
         realtime: float) -> None:
    """Re-simulate a ctrl trajectory in the passive viewer, looping."""
    import time

    import mujoco.viewer

    cap_adr = int(model.joint("cap_hinge").qposadr[0])
    step_wall = model.opt.timestep * n_substeps / max(realtime, 1e-6)
    print(f"\nreplaying {len(ctrl)} control steps "
          f"({len(ctrl) * model.opt.timestep * n_substeps:.1f} s) at {realtime:g}x, looping. "
          f"Close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_resetDataKeyframe(model, data, model.keyframe("pregrasp").id)
            for u in ctrl:
                if not viewer.is_running():
                    break
                start = time.time()
                for _ in range(n_substeps):
                    data.ctrl[:] = u
                    mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0.0, step_wall - (time.time() - start)))
            print(f"  loop finished: cap at {np.degrees(data.qpos[cap_adr]):+.1f} deg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="pregrasp")
    ap.add_argument("--report", action="store_true",
                    help="print a summary instead of opening a window")
    ap.add_argument("--play", nargs="?", const=str(DEFAULT_TRAJECTORY), default=None,
                    metavar="NPZ", help="replay a ctrl trajectory instead of holding the keyframe")
    ap.add_argument("--speed", type=float, default=1.0, help="playback rate, 1.0 = real time")
    args = ap.parse_args()

    model, data = load(args.key)
    if args.report:
        report(model, data)
        return

    report(model, data)
    if args.play:
        path = pathlib.Path(args.play)
        if not path.exists():
            raise SystemExit(
                f"{path} not found -- generate it with: python3 scripts/squeeze_twist.py"
            )
        npz = np.load(path)
        play(model, data, npz["ctrl"], int(npz["n_substeps"]), args.speed)
        return

    import mujoco.viewer

    print("\nopening viewer; ctrl+drag the cap to check the hinge turns")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
