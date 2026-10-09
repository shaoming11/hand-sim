"""M5: success rate for a trained policy, plus the exploit checks the PRD's criterion names.

    # on the GPU box, after training
    python3 scripts/eval.py logs/capturn-nominal-s1/checkpoints --save 4
    python3 scripts/eval.py CKPT --episodes 1024 --out artifacts/eval_nominal.json

    # pull the checkpoint from W&B instead of the box's disk
    python3 scripts/eval.py --wandb-artifact ckpt-capturn-nominal-s1:latest --save 4

    # M7: the nominal policy under DR, and the DR policy under DR
    python3 scripts/eval.py CKPT --dr-scale 1.0

    # on the Mac, to look at what came back
    python3 scripts/render.py artifacts/eval_nominal_ep00.npz --camera all --sites

PRD M5 is two criteria, and the second one is the one that is easy to fake:

    "success rate ... above 80% over 1000 eval episodes, and rendered rollouts show real finger
     gaiting with no exploit (flicking, penetration, vibrating the cap)"

A success rate alone cannot distinguish a gait from a policy that has found a way to spin the
cap without ever letting go of it, so this reports five numbers that separate them, none of
which needs the contact buffer:

* **net / gross rotation.** `sum(d_theta) / sum(|d_theta|)`. A gait pushes one way and resets
  its fingers while barely touching; it scores near 1. Vibrating the cap to exploit a
  rectified reward scores near 0.
* **regrips per finger.** A gaiting fingertip has to release the cap and come back. Counted
  with hysteresis on the gap to the cap surface -- below `--contact-gap` is holding, above
  `--release-gap` is released -- so chatter around one threshold does not inflate it. Zero
  regrips with a high success rate means the cap is being turned without gaiting, which at
  this wrist range means a wrist twist or a flick.
* **simultaneous contacts.** How many gaiting tips are inside `--contact-gap` at the same
  instant. `regrips` counts each tip on its own, so it cannot tell one finger working alone
  from three taking turns; this can. Over the two pinch fingers the M1 scripted gait averages
  1.55 of 2 and holds both on the cap 63.5% of the time; the first opposition-trained policy
  averages 0.67 and manages both only 11.2%.
* **wrist share of the joint travel.** The wrist has 2 of the 24 joints. If it accounts for
  most of the motion, the fingers are passengers.
* **deepest penetration.** Measured on the saved trajectories by replaying them through CPU
  MuJoCo, which is the same measurement M1 and M2 reported, so the numbers are comparable.

Episodes run under `VmapWrapper` only -- no `EpisodeWrapper`, no auto-reset -- for
`episode_length` steps, with a per-env alive mask. That is deliberate: Brax's eval path
auto-resets on termination and then keeps stepping, so a dropped cap would silently contribute
a second, partial episode to the averages.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from capturn import checkpoints  # noqa: E402
from capturn import config as capturn_config  # noqa: E402
from capturn import scene  # noqa: E402
from capturn.env import CapTurn  # noqa: E402

RAD2DEG = 180.0 / np.pi

# The `OverflowType` flags that mean the sim silently stopped being the sim M2
# verified, as opposed to the solver merely working hard. Same set as benchmark.py.
BUFFER_OVERFLOWS = {"NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "NVMAX"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", nargs="?", default=None,
                    help="a step dir, one from<n>/ attempt dir, or a whole checkpoints/ tree "
                         "(the newest global step wins)")
    ap.add_argument("--wandb-artifact", default=None,
                    help="instead of a path: 'ckpt-<run>:latest', downloaded from W&B")
    ap.add_argument("--wandb-project", default="hand-sim")
    ap.add_argument("--wandb-entity", default=None)

    ap.add_argument("--episodes", type=int, default=1024,
                    help="PRD M5 judges success over 1000 episodes")
    ap.add_argument("--batch", type=int, default=None,
                    help="episodes per vmapped rollout (default: all of them at once)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample from the policy instead of taking its mean action")
    ap.add_argument("--impl", default=None, choices=["warp", "jax"])
    ap.add_argument("--dr-scale", type=float, default=0.0,
                    help="evaluate under env-level DR: observation noise and action delay")
    ap.add_argument("--env-override", default=None, help="JSON of env config overrides")

    ap.add_argument("--save", type=int, default=0, metavar="N",
                    help="save N episodes as .npz for scripts/render.py")
    ap.add_argument("--save-best", action="store_true",
                    help="save the N best episodes by rotation instead of the first N")
    ap.add_argument("--save-prefix", default=None,
                    help="default: artifacts/eval_<checkpoint step>")
    ap.add_argument("--out", default=None, help="write the summary as JSON")

    # Hysteresis band for "holding" vs "released". The pregrasp gaps are 3.5-7.2 mm (M0), so
    # 10 mm is comfortably "on the cap" and 20 mm is unambiguously off it.
    ap.add_argument("--contact-gap", type=float, default=0.010)
    ap.add_argument("--release-gap", type=float, default=0.020)
    return ap.parse_args(argv)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def resolve_checkpoint(args: argparse.Namespace) -> checkpoints.Checkpoint:
    """The newest checkpoint to evaluate, and the global step it was written at.

    Takes a step directory, an attempt directory, a whole `checkpoints/` tree or a downloaded
    W&B artifact. The global step is not `int(dirname)`: a resumed run's directories are named
    relative to the resume, which is what `capturn/checkpoints.py` is for.
    """
    if args.wandb_artifact:
        import wandb

        entity = f"{args.wandb_entity}/" if args.wandb_entity else ""
        path = pathlib.Path(
            wandb.Api().artifact(f"{entity}{args.wandb_project}/{args.wandb_artifact}",
                                 type="model").download()
        )
    elif args.checkpoint:
        path = pathlib.Path(args.checkpoint).resolve()
    else:
        raise SystemExit("give a checkpoint path or --wandb-artifact")

    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    latest = checkpoints.find_latest(path)
    if latest is None:
        raise SystemExit(f"no step-named checkpoint under {path}")
    return latest


def build_env(args: argparse.Namespace, num_envs: int) -> CapTurn:
    cfg = capturn_config.default_config()
    if args.impl is not None:
        cfg.impl = args.impl
    cfg.noise_config.level = args.dr_scale
    cfg.action_delay.enabled = args.dr_scale > 0.0
    if args.env_override:
        for dotted, value in json.loads(args.env_override).items():
            node = cfg
            *path, leaf = dotted.split(".")
            for part in path:
                node = node[part]
            node[leaf] = value
    # `naconmax` is a total across worlds (M2), so it follows the batch actually being run.
    cfg.naconmax = capturn_config.NACONMAX_PER_WORLD * num_envs
    return CapTurn(cfg)


# --------------------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------------------
def sensor_slice(model, name: str) -> slice:
    """Where one sensor lives in `sensordata`.

    `mjx_env.get_sensor_data` indexes the first axis, which is the batch axis here, so the
    batched rollout slices `sensordata[:, adr:adr+dim]` itself rather than vmapping the env's
    accessors over a Warp `Data`.
    """
    import mujoco

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise SystemExit(f"the scene has no sensor named {name!r}")
    adr = int(model.sensor_adr[sid])
    return slice(adr, adr + int(model.sensor_dim[sid]))


def rollout(env: CapTurn, policy, num_envs: int, seed: int) -> dict[str, np.ndarray]:
    """One episode per env, `episode_length` steps, no auto-reset.

    Returns, with a leading env axis: `qpos` (T+1 frames), and per-step `d_theta`, `angvel`,
    `gaps` (3 fingertips), `action` and `alive`.
    """
    import jax
    import jax.numpy as jp
    import mujoco
    from brax.envs.wrappers import training as brax_training

    model = env.mj_model
    steps = int(env._config.episode_length)
    cap_qid = int(model.joint(scene.CAP_JOINT).qposadr[0])
    angvel_at = sensor_slice(model, "cap_angvel")
    tips_at = [sensor_slice(model, s) for s in scene.GAIT_TIP_SENSORS]
    radius, half_h = scene.cap_dimensions(model)
    del mujoco

    overflow_available = env._config.impl == "warp"
    batched = brax_training.VmapWrapper(env)

    def one_step(carry, _):
        state, rng, alive = carry
        rng, act_rng = jax.random.split(rng)
        action = policy(state.obs, jax.random.split(act_rng, num_envs))
        cap_before = state.data.qpos[:, cap_qid]
        nstate = batched.step(state, action)
        sensors = nstate.data.sensordata
        # The fingertip `framepos` sensors are already relative to `cap_site`, i.e. in the cap
        # frame, which is what `surface_distance` expects.
        tips = jp.stack([sensors[:, at] for at in tips_at], axis=1)  # (envs, 3 tips, 3)
        # `alive` gates the step that *follows* a termination, so the terminating step itself
        # still counts -- that is the step on which the cap was dropped.
        was_alive = alive
        alive = alive * (1.0 - nstate.done)
        out = {
            "qpos": nstate.data.qpos,
            "d_theta": (nstate.data.qpos[:, cap_qid] - cap_before) * was_alive,
            "angvel": sensors[:, angvel_at][:, 0] * was_alive,
            "gaps": scene.surface_distance(tips, radius, half_h),
            "action": action,
            "alive": was_alive,
        }
        # M5: the fingertip `gaps` above cannot see the grasp this task actually asks for -- a
        # lateral pinch puts the *side* of the index finger on the cap, not its tip -- so every
        # tip-based number here reads as "off the cap" for a perfectly good pinch. These are the
        # three quantities the reward itself is built from, which is what `grasp_state`'s
        # docstring promises and what was missing when the first Blackwell run was evaluated.
        thumb_gap, index_gap, opposition = jax.vmap(env.grasp_state)(nstate.data)
        out["thumb_gap"] = thumb_gap
        out["index_gap"] = index_gap
        out["opposition"] = opposition
        # Warp reports solver and buffer overflow through a bitmask on the Data rather than an
        # exception, and prints its warnings from inside kernels where Python cannot see them.
        # M2 checked this on the open-loop gait; a policy loads the contacts differently, so it
        # is worth carrying out of the rollout rather than trusting the earlier result.
        if overflow_available:
            out["overflow"] = nstate.data._impl.overflow
        return (nstate, rng, alive), out

    @jax.jit
    def run(rng):
        reset_rng, rng = jax.random.split(rng)
        state = batched.reset(jax.random.split(reset_rng, num_envs))
        alive = jp.ones(num_envs)
        _, traj = jax.lax.scan(one_step, (state, rng, alive), None, length=steps)
        return state.data.qpos, traj

    qpos0, traj = run(jax.random.PRNGKey(seed))
    overflow = traj.pop("overflow", None)
    out = {k: np.asarray(jp.swapaxes(v, 0, 1)) for k, v in traj.items()}
    # Prepend the reset frame so a saved trajectory starts from the pose the episode started in.
    out["qpos"] = np.concatenate([np.asarray(qpos0)[:, None], out["qpos"]], axis=1)
    if overflow is not None:
        out["overflow"] = np.bitwise_or.reduce(
            np.asarray(overflow).astype(np.int64).reshape(-1)
        )
    return out


def overflow_flags(mask: int) -> list[str]:
    """The `OverflowType` bits set in a rollout's accumulated mask."""
    from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType

    return [f.name for f in OverflowType
            if f.value and f.name != "ALL" and (mask & f.value) == f.value]


def make_policy(ckpt: pathlib.Path, deterministic: bool):
    """The checkpoint's inference function, vmapped over the env batch.

    The hidden sizes and both obs keys come back from `ppo_network_config.json`, which Brax
    writes from the `functools.partial` that `train.py` passed -- so nothing here has to be
    told the architecture, and a checkpoint from a differently shaped run still loads correctly.
    """
    import jax
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    from capturn import compat

    shims = compat.apply()
    if shims:
        print(f"compat shims applied: {', '.join(shims)}  (see capturn/compat.py)")

    inference_fn = ppo_checkpoint.load_policy(ckpt, deterministic=deterministic)
    jitted = jax.jit(jax.vmap(inference_fn))
    return lambda obs, keys: jitted(obs, keys)[0]


# --------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------
def grasp_summary(traj: dict[str, np.ndarray], live: np.ndarray, env_config) -> dict:
    """The lateral pinch, measured the way the reward measures it.

    `opposition` is +1 when the thumb and the index finger meet the cap from opposite sides and
    -1 when they crowd the same side, and the reward multiplies by `clip(opposition, 0, 1)` --
    so a negative median means the grasp term is contributing *nothing* and the policy is
    succeeding on rotation alone. `held` is the fraction of steps inside `contact_distance`,
    the same threshold the grasp-loss termination uses: a policy can keep that satisfied, and
    so never terminate, while still never closing an opposed pinch.
    """
    if "thumb_gap" not in traj:          # a trajectory recorded before these were collected
        return {}
    thumb, index = traj["thumb_gap"][live], traj["index_gap"][live]
    opposition = traj["opposition"][live]
    worst = np.maximum(thumb, index)
    contact = float(env_config.contact_distance)
    sharpness = float(env_config.reward_config.grasp_sharpness)
    # Mirrors `CapTurn._get_reward`'s `grasp` term and must be changed with it. It is NOT
    # `clip(opposition, 0, 1)` any more: that gate is what made the term inert, and reporting
    # the clipped form against the reshaped reward understates it wherever opposition is
    # negative -- which is exactly the regime worth reading. NOTES.md, M5.
    term = np.exp(-sharpness * worst) * (0.5 * (1.0 + opposition))
    return {
        "thumb_gap_mm": {"median": float(np.median(thumb)) * 1e3,
                         "p90": float(np.percentile(thumb, 90)) * 1e3},
        "index_gap_mm": {"median": float(np.median(index)) * 1e3,
                         "p90": float(np.percentile(index, 90)) * 1e3},
        "opposition": {"median": float(np.median(opposition)),
                       "frac_opposed": float((opposition > 0).mean())},
        "frac_steps_both_within_contact_distance": float((worst <= contact).mean()),
        "contact_distance_mm": contact * 1e3,
        "grasp_term_median": float(np.median(term)),
    }


def count_regrips(gaps: np.ndarray, alive: np.ndarray,
                  contact: float, release: float) -> np.ndarray:
    """Release-then-regrip events per (episode, fingertip), with hysteresis.

    `gaps` is (episodes, steps, 3). A finger is "holding" below `contact` and "released" above
    `release`; a regrip is a holding -> released -> holding round trip. The band is what keeps
    a tip resting right at the threshold from being counted hundreds of times.
    """
    holding = gaps < contact
    released = gaps > release
    live = alive[..., None] > 0
    counts = np.zeros((gaps.shape[0], gaps.shape[2]), dtype=int)  # (episodes, tips)
    state = np.where(holding[:, 0], 1, 0)  # 1 holding, 0 unknown/released
    seen_release = np.zeros_like(state, dtype=bool)
    for t in range(gaps.shape[1]):
        step_live = live[:, t]
        rel = released[:, t] & step_live
        hold = holding[:, t] & step_live
        seen_release |= rel & (state == 1)
        regrip = hold & seen_release
        counts += regrip
        seen_release &= ~regrip
        state = np.where(hold, 1, np.where(rel, 0, state))
    return counts


def summarize(env: CapTurn, traj: dict[str, np.ndarray], args: argparse.Namespace) -> dict:
    alive = traj["alive"]
    steps_alive = alive.sum(axis=1)
    net = traj["d_theta"].sum(axis=1)
    gross = np.abs(traj["d_theta"]).sum(axis=1)
    # The running maximum is what the reward latches on, so success is "ever reached 4*pi",
    # not "ended past 4*pi" -- a policy that overshoots and slips back still succeeded.
    reached = np.maximum.accumulate(np.cumsum(traj["d_theta"], axis=1), axis=1)[:, -1]
    threshold = float(env._config.success_rotation)
    success = reached >= threshold

    n = len(success)
    rate = float(success.mean())
    regrips = count_regrips(traj["gaps"], alive, args.contact_gap, args.release_gap)

    # M5: the number that separates a two-finger grasp from the flick exploit. `regrips` counts
    # each tip's events independently, so one finger working alone and three taking turns can
    # produce similar totals; this counts how many are on the cap *at the same instant*. The M1
    # scripted gait holds 1.55 of the two pinch fingers on the cap on average, both of them on
    # 63.5% of frames; the first opposition-trained policy manages 0.67 and 11.2%.
    live_steps = alive > 0
    simultaneous = (traj["gaps"] < args.contact_gap).sum(axis=2)[live_steps]

    from mujoco_playground._src import mjx_env

    model = env.mj_model
    hand_joints = [model.joint(j).name for j in range(model.njnt)
                   if model.joint(j).name != scene.CAP_JOINT]
    wrist = [i for i, name in enumerate(hand_joints) if "WRJ" in name]
    qpos_hand = traj["qpos"][:, :, mjx_env.get_qpos_ids(model, hand_joints)]
    travel = np.abs(np.diff(qpos_hand, axis=1)).sum(axis=1)  # (episodes, 24)
    wrist_share = travel[:, wrist].sum(axis=1) / np.maximum(travel.sum(axis=1), 1e-9)

    dropped = steps_alive < traj["alive"].shape[1]
    return {
        "episodes": n,
        "success_rate": rate,
        # Binomial standard error: 1024 episodes resolve 80% to about +-1.3 points, which is
        # the margin the 80% criterion has to be read against.
        "success_stderr": float(np.sqrt(max(rate * (1 - rate), 0.0) / n)),
        "success_threshold_deg": threshold * RAD2DEG,
        "net_rotation_deg": {
            "mean": float(net.mean() * RAD2DEG),
            "median": float(np.median(net) * RAD2DEG),
            "p10": float(np.percentile(net, 10) * RAD2DEG),
            "p90": float(np.percentile(net, 90) * RAD2DEG),
        },
        "gross_rotation_deg_mean": float(gross.mean() * RAD2DEG),
        "net_over_gross": float(net.sum() / max(gross.sum(), 1e-9)),
        "mean_angvel_deg_s": float(
            (traj["angvel"].sum() / max(alive.sum(), 1.0)) * RAD2DEG
        ),
        "drop_rate": float(dropped.mean()),
        "mean_episode_steps": float(steps_alive.mean()),
        "regrips_per_episode": {
            tip: float(regrips[:, i].mean()) for i, tip in enumerate(scene.GAIT_TIPS)
        },
        "episodes_with_no_regrip": float((regrips.sum(axis=1) == 0).mean()),
        "grasp": grasp_summary(traj, live_steps, env._config),
        "gait_tip_count": len(scene.GAIT_TIPS),
        "simultaneous_contacts": {
            "median": float(np.median(simultaneous)),
            "mean": float(simultaneous.mean()),
            "frac_steps_multi_contact": float((simultaneous >= 2).mean()),
            "frac_steps_no_contact": float((simultaneous == 0).mean()),
        },
        "wrist_share_of_travel": float(wrist_share.mean()),
        "wrist_joint_count": len(wrist),
        "hand_joint_count": len(hand_joints),
        "action_chatter": float(
            np.abs(np.diff(traj["action"], axis=1)).mean()
        ),
        "overflow_flags": overflow_flags(int(traj["overflow"])) if "overflow" in traj else [],
        "gap_thresholds": {
            "contact_gap_m": args.contact_gap,
            "release_gap_m": args.release_gap,
            "median_gap_m": float(np.median(traj["gaps"][alive > 0])),
        },
    }


def penetration_mm(qpos: np.ndarray) -> tuple[float, float]:
    """Deepest contact penetration over a saved trajectory, via CPU MuJoCo.

    Returns `(policy_mm, reset_mm)`: the deepest penetration the *policy* drives the hand into,
    and the one already present in frame 0.

    Splitting the two is not cosmetic. Frame 0 is the reset pose, and `reset_noise.hand_qpos`
    perturbs 24 joints by up to 0.05 rad with no collision check, so some resets start with a
    fingertip several mm inside the cap. That has nothing to do with what the policy learned,
    and because the reset states are seeded it is the *same* number for every policy -- it was
    reported as -4.531 mm for two completely different policies at M5 before this was split
    out, which is what gave the first H200 run a penetration exploit it did not have. The
    policy figure is the one comparable to M1's -0.566 mm CPU / -0.104 mm warp.

    `mj_forward` on each frame gives the contact set for that pose. Only the frames that get
    saved are replayed, so this is cheap.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(scene.SCENE.as_posix())
    data = mujoco.MjData(model)

    def deepest(q: np.ndarray) -> float:
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return min((float(data.contact[i].dist) for i in range(data.ncon)), default=0.0)

    reset = deepest(qpos[0]) if len(qpos) else 0.0
    policy = min((deepest(q) for q in qpos[1:]), default=0.0)
    return policy * 1e3, reset * 1e3


def save_episodes(traj: dict[str, np.ndarray], which: np.ndarray, prefix: pathlib.Path,
                  ctrl_dt: float) -> list[pathlib.Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for rank, ep in enumerate(which):
        steps = int(traj["alive"][ep].sum())
        qpos = traj["qpos"][ep][: steps + 1]
        path = prefix.with_name(f"{prefix.name}_ep{rank:02d}.npz")
        np.savez_compressed(
            path,
            qpos=qpos,
            # Cumulative rotation per frame, which is what render.py prints and the same
            # quantity the reward integrates.
            cap_angle=np.concatenate([[0.0], np.cumsum(traj["d_theta"][ep][:steps])]),
            gaps=traj["gaps"][ep][:steps],
            action=traj["action"][ep][:steps],
            ctrl_dt=ctrl_dt,
            episode=ep,
        )
        paths.append(path)
    return paths


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ckpt, step = resolve_checkpoint(args)

    import jax

    print(f"jax backend: {jax.default_backend()}   devices: {jax.devices()}")
    print(f"checkpoint: {ckpt}  (step {step:,})")

    batch = args.batch or args.episodes
    env = build_env(args, batch)
    policy = make_policy(ckpt, deterministic=not args.stochastic)
    print(f"policy: {'mean action' if not args.stochastic else 'sampled'}   "
          f"obs {env.observation_size}")
    print(f"episodes: {args.episodes} in batches of {batch}, "
          f"{env._config.episode_length} steps each"
          + (f"   dr_scale={args.dr_scale:g}" if args.dr_scale else ""))
    print()

    chunks = []
    done = 0
    while done < args.episodes:
        size = min(batch, args.episodes - done)
        if size != batch:
            env = build_env(args, size)
        chunks.append(rollout(env, policy, size, args.seed + done))
        done += size
        print(f"  {done}/{args.episodes} episodes", flush=True)
    # Every array is per-episode and concatenates; `overflow` is a single accumulated bitmask
    # per batch, so it is OR-ed rather than stacked.
    traj = {k: np.concatenate([c[k] for c in chunks])
            for k in chunks[0] if k != "overflow"}
    if "overflow" in chunks[0]:
        traj["overflow"] = np.bitwise_or.reduce([int(c["overflow"]) for c in chunks])

    summary = summarize(env, traj, args)
    summary["checkpoint"] = ckpt.as_posix()
    summary["step"] = step
    summary["dr_scale"] = args.dr_scale
    summary["deterministic"] = not args.stochastic

    # ------------------------------------------------------------------ save + penetration
    saved: list[pathlib.Path] = []
    if args.save:
        net = traj["d_theta"].sum(axis=1)
        order = np.argsort(-net) if args.save_best else np.arange(len(net))
        which = order[: args.save]
        prefix = pathlib.Path(args.save_prefix) if args.save_prefix else (
            ROOT / "artifacts" / f"eval_{step:012d}"
        )
        saved = save_episodes(traj, which, prefix, float(env.dt))
        summary["saved"] = [p.as_posix() for p in saved]
        summary["saved_net_rotation_deg"] = [float(net[i] * RAD2DEG) for i in which]
        measured = [penetration_mm(np.load(p)["qpos"]) for p in saved]
        summary["worst_penetration_mm"] = min(policy for policy, _ in measured)
        summary["reset_penetration_mm"] = min(reset for _, reset in measured)

    # ------------------------------------------------------------------ report
    rate, err = summary["success_rate"], summary["success_stderr"]
    print()
    print(f"success rate      {rate * 100:6.2f}% +- {err * 100:.2f}  "
          f"({int(rate * summary['episodes'])}/{summary['episodes']} episodes reached "
          f"{summary['success_threshold_deg']:.0f} deg)")
    print(f"PRD M5 criterion  {'PASS' if rate > 0.80 else 'FAIL'}  (needs > 80%)")
    print()
    rot = summary["net_rotation_deg"]
    print(f"net rotation      mean {rot['mean']:+8.1f} deg   median {rot['median']:+8.1f}   "
          f"p10 {rot['p10']:+8.1f}   p90 {rot['p90']:+8.1f}")
    print(f"mean angvel       {summary['mean_angvel_deg_s']:+8.2f} deg/s")
    print(f"drop rate         {summary['drop_rate'] * 100:6.2f}%   "
          f"mean episode {summary['mean_episode_steps']:.0f} steps")
    print()
    print("gaiting evidence (the PRD's second M5 criterion):")
    print(f"  net/gross rotation      {summary['net_over_gross']:+.3f}   "
          "(1 = one direction, 0 = shaking the cap)")
    for tip, count in summary["regrips_per_episode"].items():
        print(f"  regrips, {tip:<10}     {count:7.2f} per episode")
    print(f"  episodes with none      {summary['episodes_with_no_regrip'] * 100:6.2f}%   "
          "(high = turning the cap without gaiting)")
    sim = summary["simultaneous_contacts"]
    print(f"  tips on the cap at once {sim['median']:7.2f}   "
          f"median of {summary['gait_tip_count']} (M1 scripted gait 1.55 mean, both on 63.5%)")
    print(f"  steps with 2+ tips      {sim['frac_steps_multi_contact'] * 100:6.2f}%   "
          f"(steps with none {sim['frac_steps_no_contact'] * 100:.2f}%)")
    g = summary["grasp"]
    if g:
        print(f"  thumb gap               {g['thumb_gap_mm']['median']:6.2f} mm median   "
              f"(p90 {g['thumb_gap_mm']['p90']:.2f})")
        print(f"  index gap               {g['index_gap_mm']['median']:6.2f} mm median   "
              f"(p90 {g['index_gap_mm']['p90']:.2f})")
        print(f"  opposition              {g['opposition']['median']:+6.3f} median   "
              f"opposed on {g['opposition']['frac_opposed'] * 100:.1f}% of steps "
              "(+1 = opposite sides, -1 = same side)")
        print(f"  both within {g['contact_distance_mm']:.0f} mm        "
              f"{g['frac_steps_both_within_contact_distance'] * 100:6.2f}% of steps   "
              "(the grasp-loss termination threshold)")
        print(f"  grasp reward term       {g['grasp_term_median']:6.3f} median   "
              "(0 = the grasp term is paying nothing)")
    print(f"  wrist share of travel   {summary['wrist_share_of_travel'] * 100:6.2f}%   "
          f"({summary['wrist_joint_count']} of {summary['hand_joint_count']} joints)")
    print(f"  action chatter          {summary['action_chatter']:7.4f} per step")
    if "worst_penetration_mm" in summary:
        print(f"  deepest penetration     {summary['worst_penetration_mm']:+7.3f} mm   "
              "(M1 open-loop: -0.566 CPU, -0.104 warp)")
        print(f"  reset-pose penetration  {summary['reset_penetration_mm']:+7.3f} mm   "
              "(frame 0, from reset_noise -- not the policy's doing)")
    flags = summary["overflow_flags"]
    print(f"  warp overflow           {', '.join(flags) or 'none'}")
    # The PRD's M2 criterion, re-checked on a policy trajectory: a policy loads the contacts
    # differently from the open-loop gait M2 was validated on, and an overflowed buffer means
    # the physics being evaluated is not the physics that was benchmarked.
    buffers = sorted(BUFFER_OVERFLOWS.intersection(flags))
    if buffers:
        print(f"    BUFFER OVERFLOW: {', '.join(buffers)} -- raise naconmax/njmax; the "
              "numbers above are not trustworthy")
    if {"ITERATIONS", "LS_ITERATIONS"}.intersection(flags):
        print("    the solver hit its iteration cap; raise iterations/ls_iterations")
    if "EPA_HORIZON" in flags:
        print("    EPA_HORIZON is a known warp limit (M2): a compile-time 24-entry convex "
              "collision horizon, not configurable from the model")
    if saved:
        print()
        for path, deg in zip(saved, summary["saved_net_rotation_deg"]):
            print(f"wrote {path}  ({deg:+.1f} deg)")
        print(f"render with: python3 scripts/render.py {saved[0]} --camera all --sites")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
