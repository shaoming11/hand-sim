"""Open the CPU MuJoCo viewer on `assets/scene.xml` at the `pregrasp` keyframe.

    python3 scripts/view_scene.py                 # interactive viewer
    python3 scripts/view_scene.py --key pregrasp  # pick a keyframe by name
    python3 scripts/view_scene.py --report        # headless: print the scene summary and exit

In the viewer, ctrl+drag the cap to check that the hinge turns and that the hand is not
penetrating the bottle. Fingertip sites and the cap site are in visualisation group 4; press
`s` (or use the Rendering panel) to toggle site display.
"""

from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "assets" / "scene.xml"

GAIT_TIPS = ("rh_th_tip", "rh_ff_tip", "rh_mf_tip")
ALL_TIPS = GAIT_TIPS + ("rh_rf_tip", "rh_lf_tip")


def load(key: str = "pregrasp") -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(SCENE.as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe(key).id)
    mujoco.mj_forward(model, data)
    return model, data


def cap_surface_distance(model: mujoco.MjModel, data: mujoco.MjData, site: str) -> float:
    """Distance from a site to the cap's lateral surface, in the cap frame."""
    cap = model.geom("cap").id
    radius, half_h = model.geom_size[cap, 0], model.geom_size[cap, 1]
    rel = data.site(site).xpos - data.site("cap_site").xpos
    cap_mat = data.site("cap_site").xmat.reshape(3, 3)
    local = cap_mat.T @ rel
    return float(np.hypot(np.linalg.norm(local[:2]) - radius, max(0.0, abs(local[2]) - half_h)))


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

    print("\nfingertips at the pregrasp pose:")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="pregrasp")
    ap.add_argument("--report", action="store_true", help="print a summary instead of opening a window")
    args = ap.parse_args()

    model, data = load(args.key)
    if args.report:
        report(model, data)
        return

    report(model, data)
    import mujoco.viewer

    print("\nopening viewer; ctrl+drag the cap to check the hinge turns")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
