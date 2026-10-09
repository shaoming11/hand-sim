# Shadow Hand cap unscrewing (MJX-Warp + Brax PPO)

Handoff spec for Claude Code. Build this repo from scratch, one milestone at a time. Stop and report at the end of each milestone before starting the next.

## Goal

Train an RL policy for a fixed-base Shadow Hand that unscrews a water bottle cap by finger gaiting. Two training phases:

1. Nominal: one fixed set of physics parameters.
2. Domain randomization (DR): fine-tuned from the nominal checkpoint.

## Decisions already made

Do not revisit these without asking.

| Topic | Decision |
|---|---|
| Hand | Shadow Hand (right) from `mujoco_menagerie`. Base fixed in the air. Wrist and fingers move (20 position actuators, 24 joints). |
| Bottle | Fixed in place, as if a person is holding it. No free joint, no table. |
| Cap | Only moving object. One hinge about the bottle axis. No thread model. |
| Cap state | "Already cracked open": running friction only, not the sealed breakaway torque. |
| Simulator | MuJoCo. Training runs on MJX with the Warp backend (`impl='warp'`). |
| RL | Brax PPO through the MuJoCo Playground env API. Asymmetric actor-critic. |
| Compute | Scene work on a Mac M1 (CPU). Training on one rented GPU over SSH: an H100 80GB through M3, an H200 141GB, and an RTX PRO 6000 Blackwell 96GB (sm_120) for the M5 retrain. See NOTES.md, M3 and M5. |

## Before you trust this file

API names, module paths and joint names below were written from memory. The installed source is the authority. Verify each of these before building on it:

- Playground module paths: `mujoco_playground._src.mjx_env` (`MjxEnv`, `State`, `init`, `step`), `mujoco_playground.wrapper.wrap_for_brax_training`, `mujoco_playground.registry`, `mujoco_playground.config.manipulation_params`, `learning/train_jax_ppo.py`.
- The Leap hand rotation env (`_src/manipulation/leap_hand/`, env name like `LeapCubeRotateZAxis`). This is the template. Read it fully before writing `env.py` and mirror its structure, config pattern and randomization function.
- How Playground selects the backend (an `impl` field in the env config, `--impl=warp` on the train script).
- `mjx.put_model(mj_model, impl='warp')` and `mjx.make_data(mj_model, impl='warp', naconmax=..., njmax=...)`. Confirm whether `naconmax` is a total across all worlds (scales with `num_envs`) and `njmax` is per world.
- Menagerie names (`rh_` prefix, `rh_forearm`, `rh_WRJ1`, `rh_A_FFJ0`, fingertip bodies). Read names from the compiled model. Never hardcode indices.
- Warp backend feature support for this scene: elliptic cones, `implicitfast`, `condim=4`, `frictionloss`, tendon coupling, and every hand-geom vs cap geom-type pair.
- How per-env model randomization works under the Warp backend (which fields can be batched per world).

If any of these differ from what is written here, follow the installed source and note the difference in `NOTES.md`.

## Stack and install

```bash
# GPU box (Linux, CUDA 12)
pip install -U "jax[cuda12]"
pip install playground          # may need: --extra-index-url=https://py.mujoco.org --extra-index-url=https://pypi.nvidia.com/warp-lang/
pip install "mujoco-mjx[warp]" wandb mediapy
python -c "import jax; print(jax.default_backend())"   # must print: gpu

# Mac (CPU only)
pip install mujoco jax playground mediapy
```

Environment variables on the GPU box:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # JAX otherwise grabs most of the VRAM before Warp gets any
export MUJOCO_GL=egl                          # fall back to osmesa if EGL is unavailable
                                              # (broken on the M3 image: leave it unset there, NOTES.md M3)
export WANDB_API_KEY=...
```

Make `impl` a config field everywhere. Use `impl='jax'` for CPU smoke tests of env logic on the Mac and `impl='warp'` on the GPU box. If the scene does not load under `impl='jax'`, skip local MJX tests and test on the GPU box.

## Repo layout

```
assets/
  shadow_hand/        # vendored copy of mujoco_menagerie/shadow_hand, keep its LICENSE
  scene.xml           # hand + fixed bottle + cap
capturn/
  config.py           # env config (ml_collections.ConfigDict), PPO config
  env.py              # MjxEnv subclass
  randomize.py        # DR function + range tables
scripts/
  view_scene.py       # CPU viewer, loads keyframe
  check_parity.py     # CPU MuJoCo vs MJX rollout comparison
  benchmark.py        # steps/s vs num_envs
  train.py            # nominal and DR training (flag selects)
  eval.py             # success rate + saves qpos trajectories (.npz)
  sweep.py            # one-parameter-at-a-time sensitivity sweep
  render.py           # renders saved trajectories to mp4 (runs on Mac too)
tests/
NOTES.md              # deviations from this spec, measured numbers, open questions
```

## Scene spec (`assets/scene.xml`)

**Hand.** Vendor the Menagerie right hand into `assets/shadow_hand/`. The forearm body has no joint to the world, which fixes it in the air. Orient it palm down above the bottle, either by editing the root body `pos`/`quat` in the vendored file or by composing with `mjSpec` and attaching at a posed frame.

**Bottle and cap.** Starting values, to be tuned:

```xml
<body name="bottle" pos="0 0 0.09">
  <geom name="bottle_body" type="cylinder" size="0.032 0.09"/>
  <geom name="bottle_neck" type="cylinder" size="0.013 0.012" pos="0 0 0.102"/>
  <body name="cap" pos="0 0 0.106">
    <joint name="cap_hinge" type="hinge" axis="0 0 1"
           frictionloss="0.05" damping="0.002" armature="1e-4"/>
    <geom name="cap" type="cylinder" size="0.016 0.008" mass="0.004"
          condim="4" friction="1.0 0.01 0.001"/>
    <site name="cap_site" pos="0 0 0"/>
  </body>
</body>
```

- Positive hinge angle is counterclockwise seen from above, which is the unscrew direction.
- `frictionloss` is the thread friction. Start at 0.05 N·m. At a 16 mm radius, 0.1 N·m needs about 6 N of total tangential fingertip force, which is feasible. A sealed cap near 1 N·m is out of scope.
- `armature` is there because the cap's own inertia is tiny and makes the hinge numerically stiff.
- If any hand-geom vs cylinder pair is unsupported on the Warp backend, replace the cap geom with a convex mesh (16-sided prism).

**Sites.** Add a site at each fingertip (thumb, first, middle, ring, little) and one on the palm if the Menagerie model lacks them.

**Contacts.** Needed: hand vs cap, hand vs bottle body and neck. Start with Menagerie's default hand self-collision. Prune it only if `benchmark.py` shows throughput is poor.

**Options.** Start from the Leap env's scene options in Playground, then adjust. Fallback values:

```xml
<option timestep="0.005" integrator="implicitfast" cone="elliptic" impratio="10"/>
```

**Keyframe.** Add a `pregrasp` keyframe: thumb, first and middle fingertips within about 1 cm of the cap rim, joints near mid-range. The hand placement is the most important design choice in the project. If these three fingertips cannot reach around the cap comfortably, move the bottle or the hand until they can.

## Env spec (`capturn/env.py`)

Subclass Playground's `MjxEnv`. Return dict observations with keys `state` (actor) and `privileged_state` (critic).

**Timing.** `ctrl_dt = 0.05` (20 Hz), `sim_dt = 0.005` (10 substeps), `episode_length = 400` steps (20 s).

**Action (20-dim).** Delta position targets:

```
target_t = clip(target_{t-1} + action_scale * a_t, ctrl_lower, ctrl_upper)
```

`action_scale = 0.1` rad per step to start. Store `target`, the last two actions and an action delay buffer in `state.info`.

**Actor observation (`state`).** Stack the last 3 frames of:

- joint positions (24), with a noise hook (zero noise in the nominal phase)
- previous action (20)
- current position targets (20)
- cap position in the palm frame (3) and cap axis in the palm frame (3)

No cap angle in the actor. The cap is rotationally symmetric, so the policy does not need it and could not observe it on hardware.

**Critic observation (`privileged_state`).** Actor obs plus: joint velocities, cap angle (unwrapped) and angular velocity, fingertip positions relative to the cap, actuator forces, and a fixed-size slot for DR parameters (zeros in the nominal phase).

**The observation layout must be identical in both phases.** The DR phase loads the nominal checkpoint. Noise scales, delay and DR parameter slots exist from day one and are set to zero or constant in the nominal phase.

**Reward.** Starting weights, expect to tune:

| Term | Definition | Weight |
|---|---|---|
| rotate | `clip(d_theta / ctrl_dt, -0.5, 1.0)`, `d_theta` from the hinge qpos | 1.0 |
| grasp | `exp(-100 * max(thumb gap, index gap)) * clip(opposition, 0, 1)`, surface-to-surface over every collision geom of the thumb and index finger | 0.15 |
| action rate | `-sum((a_t - a_{t-1})^2)` | 0.01 |
| torque | `-sum(actuator_force^2)` | 1e-3 |
| joint velocity | `-sum(qvel_hand^2)` | 1e-4 |
| success | one-time bonus when cumulative rotation passes `4*pi` | 10.0 |

**Termination.** NaN in qpos or qvel. All gaiting fingertips more than 10 cm from the cap, or the grasp lost for `grasp_grace_steps` consecutive steps. Otherwise fixed length.

**M5 revision: the gait is two-fingered.** This file specified three gaiting fingertips (thumb, index, middle) throughout. It is now the thumb and index finger only -- a lateral pinch, which is how a person unscrews a cap. The middle, ring and little fingers are costed for touching the cap (`reward_config.scales.idle_contact`) rather than merely unrewarded. The measurement that forced it: on the M1 scripted gait the middle finger is in contact on 93.8% of frames against the index's 64.6%, and the first policy trained against the opposition reward leaned on the middle finger hardest of all (71.3% of steps) while leaving the thumb pad 12.15 mm off the cap. Both are legitimate three-finger grasps and neither is the one this task is for. See NOTES.md, M5.

**Reset.** `pregrasp` keyframe plus uniform noise on hand joints (about ±0.05 rad, clipped to limits) and a random cap angle.

**Metrics.** Cumulative cap rotation, mean cap angular velocity, success flag, each reward term.

**Do not read the contact buffer directly.** It lives in different places on the two backends. Use site positions and the hinge state. If contact information becomes necessary, use contact sensors.

## Training

Brax PPO with Playground's wrapper (`wrap_for_brax_training`). Start from Playground's PPO config for the Leap rotation env. Fallback values if that is unavailable:

```
num_timesteps        150_000_000
num_envs             8192
episode_length       400
unroll_length        40
num_minibatches      32
num_updates_per_batch 4
batch_size           256
learning_rate        3e-4
entropy_cost         1e-2
discounting          0.99
normalize_observations True
policy / value hidden (512, 256, 128)
policy_obs_key "state", value_obs_key "privileged_state"
```

Requirements for `train.py`:

- `--phase nominal|dr`, `--impl`, `--load_checkpoint`, `--dr_scale`.
- Checkpoint every few minutes and upload each checkpoint off the machine (W&B artifact). The GPU box is preemptible and its disk is not durable.
- Resume from the latest uploaded checkpoint when started with `--resume`.
- Log all reward terms and metrics to W&B.

## Domain randomization (`capturn/randomize.py`)

Follow the Playground pattern: `randomization_fn(model, rng) -> (batched_model, in_axes)`, passed to PPO. After building the batched model, read every randomized field back and assert it varies across envs.

Model-level, full-scale ranges:

| Parameter | Field | Range |
|---|---|---|
| Fingertip and cap friction | `geom_friction[:, 0]` | scale U(0.6, 1.4) |
| Cap thread friction | `dof_frictionloss` (cap hinge) | U(0.02, 0.15) N·m |
| Cap damping | `dof_damping` (cap hinge) | scale U(0.5, 2.0) |
| Actuator stiffness | `actuator_gainprm[:, 0]`, `actuator_biasprm[:, 1] = -kp` | scale U(0.8, 1.2) |
| Hand joint damping | `dof_damping` (hand) | scale U(0.7, 1.3) |
| Hand joint friction | `dof_frictionloss` (hand) | scale U(0.5, 2.0) |
| Hand armature | `dof_armature` (hand) | scale U(1.0, 1.2) |
| Joint zero offset | `qpos0` (hand) | ±0.03 rad |
| Bottle placement | `body_pos`, `body_quat` (bottle) | ±1.5 cm, ±5° tilt |

Env-level, implemented in `env.py` and gated by config:

| Parameter | Range |
|---|---|
| Joint position observation noise | ±0.02 rad |
| Cap position observation noise | ±5 mm |
| Action delay | 0 to 2 control steps, sampled per episode |

Procedure:

1. Run `sweep.py` on the nominal policy first: perturb one parameter at a time across its range and record success rate. This shows which ranges matter. Save the table to `NOTES.md`.
2. Fine-tune from the nominal checkpoint with `--dr_scale` 0.25, then 0.5, then 1.0 (ranges interpolate between nominal and full).
3. Train one from-scratch run at `--dr_scale 1.0` as a baseline and compare.

## Milestones

**M0. Repo and scene (Mac).** `scene.xml` loads in CPU MuJoCo. `view_scene.py` opens the viewer at the `pregrasp` keyframe.
Done when: the cap turns when dragged in the viewer, the hand does not penetrate the bottle, and thumb, first and middle fingertips reach the cap rim. Record the actuator `forcerange` values in `NOTES.md`.

**M1. Feasibility check (Mac).** Script a crude open-loop squeeze-and-twist with the thumb and first finger.
Done when: the cap rotates at least a few degrees under `frictionloss=0.05`. If it cannot, fix the hand placement or contact settings before any RL.

**M2. MJX port and parity (GPU box).** `check_parity.py` replays the M1 action sequence on CPU MuJoCo and on MJX-Warp.
Done when: cap angle trajectories agree within about 10%, halving `sim_dt` does not change the result materially, and there are no contact or constraint overflow warnings. Set `naconmax`/`njmax` from measured counts with 2x headroom.

**M3. Benchmark (GPU box).** `benchmark.py` reports steps/s at 1024, 4096 and 8192 envs.
Done when: numbers are in `NOTES.md` and `num_envs` is chosen.

**M4. Env.** `env.py` passes tests: reset and step are jittable, obs shapes are fixed, reward terms are finite, random actions run for 1000 steps without NaN.

**M5. Nominal training.** Done when: success rate (cumulative rotation ≥ `4*pi` within the episode) is above 80% over 1000 eval episodes, and rendered rollouts show real finger gaiting with no exploit (flicking, penetration, vibrating the cap).

**M6. Sensitivity sweep.** Table in `NOTES.md`.

**M7. DR training.** Done when: success rate above 80% at `--dr_scale 1.0`, compared against the from-scratch baseline and against the nominal policy evaluated under the same DR.

## Compute (Daytona GPU machine)

| Setting | Value |
|---|---|
| GPU | Whatever the sandbox provides; all three measured so far run unmodified. 1x H100 80GB through M3, 1x H200 141GB, 1x RTX PRO 6000 Blackwell 96GB (sm_120) for the M5 retrain. |
| Capacity | Spot (can be killed with no warning) |
| OS | Linux |
| vCPU / memory / storage | 8 / 32 GiB / 100 GiB |
| Image | Custom: `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` (not the vLLM image) |
| Auto-stop | 0 during training runs. Auto-stop fires even while processes run; only external activity such as a live SSH session resets it. Stop the machine manually afterwards. |
| Delete disk on stop | Unchecked |

Run training inside `tmux`. Assume the machine can disappear at any moment: nothing that matters lives only on its disk.

The boxes have come up with wildly different host specs (52 vCPU / 442 GiB on the H100, 24 / 235 on the H200, 30 / 88 on the Blackwell), none of them matching this row. Re-check per box; the simulation needs almost none of it (6.5 GiB of VRAM at `num_envs = 8192`).

**Assume preemption within the hour.** Four boxes have been killed mid-run so far, twice inside twenty minutes. The disk has survived every restart, and `scripts/train.py --resume` reads the local checkpoint directory before it asks W&B, so a restarted box recovers at the last checkpoint with the spent budget subtracted. Set `WANDB_API_KEY` anyway: it is the only thing that protects against a box whose disk does *not* come back.

## Stretch, not in scope yet

- Compliant bottle hold: mount the bottle on stiff spring joints to mimic a human grip.
- Cap radius and height variants.
- Screw model: hinge coupled to a slide through a joint equality so the cap rises as it turns.
- Free bottle on a table.