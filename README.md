# hand-sim

Shadow Hand unscrewing a bottle cap by finger gaiting. Built from [`PRD.md`](PRD.md), one
milestone at a time; deviations, measured numbers and open questions live in
[`NOTES.md`](NOTES.md).

Status: **M0** (scene), **M1** (open-loop feasibility), **M2** (MJX parity), **M3** (benchmark)
and **M4** (the env) done. **`num_envs = 8192`**, `sim_dt = 0.00125` -- 1.26M sim-steps/s,
1,572x realtime, 8% of an 80 GB GPU on the H100.

**M5 (nominal training): the pipeline is built and verified end to end, the run has not
happened.** `scripts/train.py` and `scripts/eval.py` train, checkpoint, upload, resume, load a
checkpoint back and evaluate -- checked by a round trip on warp-cpu. The milestone criterion
("success rate above 80% over 1000 eval episodes") needs the 150M-step run on an H100.

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
unset MUJOCO_GL          # EGL is broken on the Daytona image and breaks `import mujoco`
python3 -c "import jax; print(jax.default_backend())"   # must print: gpu

python3 scripts/squeeze_twist.py     # regenerate the trajectory at the current timestep
python3 scripts/check_parity.py      # M2 on CUDA
python3 scripts/benchmark.py --num-envs 8192 --out artifacts/bench.json
```

**Run one `num_envs` per process.** With `XLA_PYTHON_CLIENT_PREALLOCATE=false` JAX grows its
pool and never releases it, so sweeping several env counts in one process starves Warp and the
largest one fails to allocate. There is no `git` on the box -- sync with
`tar czf - . | ssh HOST 'tar xzf - -C ~/hand-sim'`.

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

Two reward weights are worth watching at M5: `reach` at 0.5 pays 8.93 of return for holding the
pregrasp pose and doing nothing, and `action_rate` at 0.01 is a third of the reward signal for an
untrained policy. Measured numbers in `NOTES.md`.

## Training

```bash
# on the H100
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
