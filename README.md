# hand-sim

Shadow Hand unscrewing a bottle cap by finger gaiting. Built from [`PRD.md`](PRD.md), one
milestone at a time; deviations, measured numbers and open questions live in
[`NOTES.md`](NOTES.md).

Status: **M0** (scene), **M1** (open-loop feasibility), **M2** (MJX parity), **M3** (benchmark)
and **M4** (the env) done. **`num_envs = 8192`**, `sim_dt = 0.00125` -- 3.52M sim-steps/s,
**4,405x realtime** on an RTX PRO 6000 Blackwell (88,106 control-steps/s) once the fingertip
collision hulls are capped at 16 vertices. That is **2.72x** the same config on an H200
(32,358), GPU and geometry together. The H200 itself was worth only +2.9% over the H100 --
this workload is latency-bound, not bandwidth-bound, so the hulls are doing most of the work
(NOTES.md, M3 and M5).

**M5 (nominal training): criterion 1 passes, criterion 2 does not.** The scene and reward
changes did most of what they were meant to -- parity improves 2.2% -> **0.2%**, warp reports
**no overflow at all**, the pregrasp parking reward drops +0.978 -> **+0.038**, rotation falls
~370 -> **155 deg/s** and penetration is clean at -0.309 mm. But measured with the reward's own
`grasp_state` (which `eval.py` did not report until now), the policy is **not** doing a lateral
pinch: thumb and index sit on the **same side** of the cap (opposition -0.627 median, opposed
on 1.6% of steps), so the grasp term pays **0.000** and the policy is optimising rotation
alone, hovering at 9.8/12.2 mm to stay inside the 20 mm grasp-loss bound on 100% of steps.
The cause was structural -- `clip(opposition, 0, 1)` has no gradient below zero, so the term
could not teach the grip it asks for. **Reshaping that one factor to `0.5 * (1 + opposition)`
fixes it**: opposition goes -0.627 -> **+0.794** (opposed on 75% of steps), the grasp term
0.000 -> **0.550**, and the gaps collapse 9.8/12.2 mm -> **2.9/0.3 mm**, so the hover is gone
too. Penetration rises to -0.781 mm and `EPA_HORIZON` returns, both from making real contact.
Single seed. Details in [`NOTES.md`](NOTES.md), M5.

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
python3 -m pytest tests/ -q               # 73 checks, ~11 s -- the M0-M5 done-criteria
HAND_SIM_SLOW=1 python3 -m pytest tests/ -q   # adds the MJX-Warp rollouts (minutes on CPU)
python3 scripts/view_scene.py --report    # scene summary, actuator limits, pregrasp geometry

# CPU MuJoCo vs MJX-Warp, plus buffer sizing for training
python3 scripts/check_parity.py
python3 scripts/check_parity.py --probe   # just naconmax / njmax
```

`warp-lang` ships a CPU build, so MJX-Warp runs on the Mac -- slowly, but it is the real
backend. `impl='jax'` cannot run this scene (no cylinder/mesh collisions).

## On the GPU box

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset MUJOCO_GL          # EGL broke `import mujoco` on the M3 image; re-check on a new box
python3 -c "import jax; print(jax.default_backend())"   # must print: gpu

python3 scripts/squeeze_twist.py     # regenerate the trajectory at the current timestep
python3 scripts/check_parity.py      # M2 on CUDA
python3 scripts/benchmark.py --num-envs 8192 --out artifacts/bench_h200.json
```

**Run one `num_envs` per process.** With `XLA_PYTHON_CLIENT_PREALLOCATE=false` JAX grows its
pool and never releases it, so sweeping several env counts in one process starves Warp and the
largest one fails to allocate. This is why the H100's `num_envs = 16384` failure is not a real
ceiling, and is worth one fresh-process retry on the H200's 141 GiB (NOTES.md, M3).

There is no `git` on the box -- sync with `tar czf - . | ssh HOST 'tar xzf - -C ~/hand-sim'`.

## The training env

```bash
python3 -m pytest tests/test_env.py -q                    # layout, config, geometry
HAND_SIM_SLOW=1 python3 -m pytest tests/test_env.py -q    # + physics, ~7 min on the Mac
```

```python
from capturn import CapTurn, default_config, ppo_config

env = CapTurn()                       # nominal phase: no observation noise, no action delay
env.observation_size                  # {'state': (210,), 'privileged_state': (283,)}
```

Asymmetric actor-critic: `state` is 3 stacked frames of what a real hand could measure,
`privileged_state` adds velocities, cap rotation, fingertip geometry, actuator forces and a
zeroed slot for the DR parameters. The actor never sees the cap angle -- the cap is
rotationally symmetric, so it does not need it and could not measure it on hardware.

The action is a delta on the position targets, integrated out of `data.ctrl` rather than out of
a copy in `state.info`. That is deliberate: Playground's auto-reset wrapper restores `data` at
an episode boundary but leaves `info` alone, so anything cached there leaks into the next
episode. `ppo_config()` asks for `full_reset=True` as well, because otherwise every env restarts
from the one state cached at the first reset. Both are covered by tests and written up in
`NOTES.md`, M4.

`reach` was replaced by `grasp` at M5, after the first trained policy turned the cap with the
*side* of its index finger and no thumb at all -- the right contact surface, but a one-sided
push rather than a grasp. `reach` watched three fingertip sites and could not see it. `grasp`
asks for what a bottle-cap grasp actually is: the thumb and the index finger both on the cap,
anywhere along their length, and on opposite sides. See `NOTES.md`, "What the flick policy was
actually doing". `action_rate` at 0.01 is still a third of the reward signal for an untrained
policy. Measured numbers in `NOTES.md`.

## Training

```bash
# on the H200
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset MUJOCO_GL
tmux new -s train
python3 scripts/train.py --wandb                      # nominal, 150M steps, ~2 h
python3 scripts/train.py --num-timesteps 20_000_000   # a shorter shakeout first
python3 scripts/train.py --resume                     # after a preemption
```

The box is spot and its disk does not survive a stop, so `--resume` is the only thing standing
between a preemption and starting over. It needs no arguments: the run name is derived from the
phase and seed rather than a timestamp, so it finds the W&B run and its newest checkpoint
artifact on its own, and spends only the step budget that is left. A checkpoint is written and
uploaded roughly every 3.5 minutes (`--checkpoint-every MIN` to change that).

Nothing in `train.py` renders -- `MUJOCO_GL=egl` breaks `import mujoco` on the Daytona image.

```bash
# success rate and the exploit checks, on the box
python3 scripts/eval.py logs/capturn-nominal-s1/checkpoints --save 4 \
    --out artifacts/eval_nominal.json

# then, on the Mac
python3 scripts/eval.py --wandb-artifact ckpt-capturn-nominal-s1:latest --save 4
python3 scripts/render.py artifacts/eval_000150000000_ep00.npz --camera all --sites
```

`eval.py` answers both halves of the M5 criterion. The success rate is over 1024 episodes with
the deterministic policy. The second half -- "real finger gaiting with no exploit" -- is four
measurements instead of four videos: net/gross rotation (1 is one direction, 0 is shaking the
cap), regrips per gaiting fingertip (a gait has to let go), the wrist's share of the joint
travel (2 of 24 joints; more than that and the fingers are passengers), and the deepest contact
penetration, replayed through CPU MuJoCo so it is comparable to M1's -0.566 mm.

```bash
python3 -m pytest tests/test_train.py -q                  # 32 checks, ~5 s
HAND_SIM_SLOW=1 python3 -m pytest tests/test_train.py -q  # + train -> resume -> eval, ~5 min
python3 scripts/train.py --smoke                          # just the training half, ~1 min
```

`capturn/compat.py` carries two shims, because Brax 0.14.2 and JAX 0.11.2 cannot train and then
load what they trained: JAX removed `jax.device_put_replicated`, which `ppo.train` still calls,
and Brax's checkpoint `save` writes a `None` kernel initialiser that its own `load_config`
cannot read back. Both shims have tests that fail when the upstream bug is fixed, so they get
deleted rather than inherited. Details in `NOTES.md`, M5.

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
capturn/scene.py           scene path and cap geometry, shared with the scripts
capturn/env.py             the MjxEnv subclass -- obs, reward, termination
capturn/config.py          env config and the Brax PPO config
capturn/compat.py          two shims for the installed Brax/JAX pair
scripts/                   fit_pregrasp, view_scene, squeeze_twist, check_parity,
                           benchmark, train, eval, render
tests/                     M0 scene checks, M1 gait, M2 parity, M3 benchmark harness,
                           M4 env, M5 training and eval
artifacts/                 generated trajectories and video (gitignored)
logs/                      training runs: checkpoints and configs (gitignored)
```
