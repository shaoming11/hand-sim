# hand-sim

Shadow Hand unscrewing a bottle cap by finger gaiting. Built from [`PRD.md`](PRD.md), one
milestone at a time; deviations, measured numbers and open questions live in
[`NOTES.md`](NOTES.md).

Status: **M0** (scene), **M1** (open-loop feasibility) and **M2** (MJX parity) done.
M3 onwards needs a GPU box.

## Setup (Mac, CPU)

```bash
pip install mujoco playground mediapy scipy pytest
```

## Watching it

```bash
# the open-loop gait, re-simulated live in the viewer, looping
python3 scripts/squeeze_twist.py          # generate the trajectory first (~2 s)
python3 scripts/view_scene.py --play
python3 scripts/view_scene.py --play --speed 0.25     # slow motion

# the static pregrasp pose -- ctrl+drag the cap to feel the hinge friction
python3 scripts/view_scene.py

# an mp4 you can scrub or share
python3 scripts/render.py --camera all --sites
open artifacts/squeeze_twist.mp4
```

In the viewer: turn on **site** under the Rendering panel to see the fingertip and cap markers,
and **contact force** to see where the grip is acting. `[` and `]` cycle the cameras
(`front`, `cap`, `back`).

## Checking it

```bash
python3 -m pytest tests/ -q               # 23 checks, ~7 s -- the M0/M1/M2 done-criteria
HAND_SIM_SLOW=1 python3 -m pytest tests/ -q   # adds the MJX-Warp rollouts (minutes on CPU)
python3 scripts/view_scene.py --report    # scene summary, actuator limits, pregrasp geometry

# CPU MuJoCo vs MJX-Warp, plus buffer sizing for training
python3 scripts/check_parity.py
python3 scripts/check_parity.py --probe   # just naconmax / njmax
```

`warp-lang` ships a CPU build, so MJX-Warp runs on the Mac -- slowly, but it is the real
backend. `impl='jax'` cannot run this scene (no cylinder/mesh collisions).

## Changing the scene

The hand placement and the `pregrasp` keyframe are solved numerically, not hand-tuned. After any
change to the bottle or cap geometry, re-fit them:

```bash
python3 scripts/fit_pregrasp.py           # report only
python3 scripts/fit_pregrasp.py --write   # patch assets/right_hand_mjx.xml and assets/scene.xml
```

**After any `--write`, re-run the gait before trusting the new pose.** The fitter's objective is
a proxy for gaiting quality, not a measure of it: one re-fit scored better on every term and
gaited at 5 degrees instead of 88, because it left the thumb 0.7 mm off the cap wall.

```bash
python3 scripts/squeeze_twist.py --sweep
python3 scripts/squeeze_twist.py --fingers th,ff     # the PRD's two-finger version
```

## Layout

```
assets/shadow_hand/        pristine mujoco_menagerie right hand (commit feadf76), LICENSE kept
assets/right_hand_mjx.xml  derived copy: fingertip sites, fitted mount, scene owns <option>
assets/scene.xml           hand + fixed bottle + hinged cap, sensors, pregrasp keyframe
scripts/                   fit_pregrasp, view_scene, squeeze_twist, check_parity, render
tests/                     M0 scene checks, M1 gait checks, M2 parity checks
artifacts/                 generated trajectories and video (gitignored)
```
