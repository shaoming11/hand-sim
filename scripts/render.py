"""Render a saved qpos trajectory to mp4. Runs on the Mac, no GPU needed.

    python3 scripts/render.py                                  # the open-loop gait, `cap` camera
    python3 scripts/render.py --camera front --width 1280
    python3 scripts/render.py --camera all                     # every camera side by side
    python3 scripts/render.py artifacts/eval_rollout.npz -o out.mp4

Takes any `.npz` holding a `qpos` array of shape (T, nq) -- `squeeze_twist.py` writes one, and
so will `eval.py` at M5. Playback speed follows `ctrl_dt` from the file, so the video runs in
real time unless `--speed` says otherwise.
"""

from __future__ import annotations

import argparse
import pathlib

import mediapy
import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENE = ROOT / "assets" / "scene.xml"
DEFAULT_TRAJECTORY = ROOT / "artifacts" / "squeeze_twist.npz"
CAMERAS = ("front", "cap", "back")


def render(qpos: np.ndarray, camera: str, width: int, height: int,
           show_sites: bool, show_contacts: bool) -> list[np.ndarray]:
    model = mujoco.MjModel.from_xml_path(SCENE.as_posix())
    model.vis.global_.offwidth = max(width, 1920)
    model.vis.global_.offheight = max(height, 1080)
    data = mujoco.MjData(model)

    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    if show_sites:
        options.sitegroup[4] = 1
    if show_contacts:
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    cameras = list(CAMERAS) if camera == "all" else [camera]
    renderer = mujoco.Renderer(model, height, width)
    frames = []
    try:
        for q in qpos:
            data.qpos[:] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            tiles = []
            for cam in cameras:
                renderer.update_scene(data, camera=cam, scene_option=options)
                tiles.append(renderer.render().copy())
            frames.append(np.concatenate(tiles, axis=1) if len(tiles) > 1 else tiles[0])
    finally:
        renderer.close()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory", nargs="?", default=str(DEFAULT_TRAJECTORY))
    ap.add_argument("-o", "--out", default=None, help="output mp4 (default: next to the .npz)")
    ap.add_argument("--camera", default="cap", choices=[*CAMERAS, "all"])
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    ap.add_argument("--sites", action="store_true", help="draw the fingertip and cap sites")
    ap.add_argument("--contacts", action="store_true", help="draw contact points and forces")
    args = ap.parse_args()

    path = pathlib.Path(args.trajectory)
    if not path.exists():
        raise SystemExit(f"{path} not found -- generate it with: python3 scripts/squeeze_twist.py")
    npz = np.load(path)
    if "qpos" not in npz.files:
        raise SystemExit(f"{path} has no 'qpos' array (found: {', '.join(npz.files)})")

    qpos = npz["qpos"]
    ctrl_dt = float(npz["ctrl_dt"]) if "ctrl_dt" in npz.files else 0.05
    fps = args.speed / ctrl_dt

    frames = render(qpos, args.camera, args.width, args.height, args.sites, args.contacts)
    out = pathlib.Path(args.out) if args.out else path.with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(out, frames, fps=fps)

    seconds = len(frames) / fps
    note = ""
    if "cap_angle" in npz.files:
        note = f", cap turns {np.degrees(npz['cap_angle'][-1]):+.1f} deg"
    print(f"wrote {out}  ({len(frames)} frames, {fps:.0f} fps, {seconds:.1f} s"
          f", camera={args.camera}{note})")


if __name__ == "__main__":
    main()
