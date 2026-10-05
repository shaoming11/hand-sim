"""M5/M7: Brax PPO training for the cap-unscrewing task.

    # the H100, nominal phase (M5)
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    unset MUJOCO_GL                       # EGL is broken on the Daytona image
    tmux new -s train
    python3 scripts/train.py --wandb

    python3 scripts/train.py --resume                    # after a preemption
    python3 scripts/train.py --num-timesteps 20_000_000  # a short shakeout first

    # M7, fine-tuned from the nominal checkpoint
    python3 scripts/train.py --phase dr --dr-scale 0.25 \
        --load-checkpoint logs/capturn-nominal/checkpoints

    # the Mac: does the whole pipeline run at all (minutes, warp-cpu)
    python3 scripts/train.py --smoke

Nothing here renders. `MUJOCO_GL=egl` makes `import mujoco` fail on the GPU image, so videos
come from `scripts/eval.py --save` on the box and `scripts/render.py` on the Mac.

Four things this does that Playground's own `train_jax_ppo.py` does not, each because the
installed source made it necessary:

* **Binds `full_reset`.** `ppo.train` calls `wrap_env_fn(env, episode_length=, action_repeat=,
  randomization_fn=)` and never passes `full_reset`, so Playground's script trains every env
  from the one state cached at the first reset. `ppo_config()` asks for `full_reset=True`
  (NOTES.md, M4), which only happens if it is bound onto the wrapper here.
* **Uploads checkpoints from `progress_fn`, not `policy_params_fn`.** Brax's eval loop runs
  `policy_params_fn` -> `checkpoint.save` -> `progress_fn`, so the checkpoint for the current
  step does not exist yet when `policy_params_fn` fires. Uploading from there would ship the
  previous one every time, silently off by one eval.
* **Subtracts the restored step count from the budget.** `restore_checkpoint_path` restores the
  normalizer and the policy and value params -- not `env_steps` and not the optimizer state. A
  naive `--resume` would therefore train a second full `num_timesteps`. The step count is
  recoverable from the checkpoint directory name, so the budget is reduced by it and W&B is
  logged at the true global step.
* **Keeps the run name deterministic.** The box is preemptible, so `--resume` has to be able to
  find the W&B run and its checkpoint artifact without being told an id. The name is derived
  from the phase, the DR scale and the seed; pass `--run-name` for a deliberately separate run.

The Adam moments are the one piece of state a resume genuinely loses -- Brax does not checkpoint
the optimizer. Each resume therefore costs a short transient while the moments refill. That is
cheap once and expensive if the machine dies every ten minutes; `--checkpoint-every` trades
checkpoint density against eval overhead if preemption turns out to be frequent.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import os
import pathlib
import sys
import time

# Must be set before jax initialises, or JAX takes the VRAM Warp needs and 8192 envs fails to
# allocate (NOTES.md, M3).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capturn import checkpoints  # noqa: E402
from capturn import config as capturn_config  # noqa: E402
from capturn.env import CapTurn  # noqa: E402

# M3, at num_envs=8192 and sim_dt=0.00125. Only used to turn `--checkpoint-every` minutes into a
# `num_evals`, and reported alongside the real rate so the estimate is checkable.
MEASURED_CONTROL_STEPS_PER_S = 31_449

RAD2DEG = 180.0 / math.pi


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="nominal", choices=["nominal", "dr"])
    ap.add_argument("--impl", default=None, choices=["warp", "jax"],
                    help="default: the env config's ('warp'; 'jax' cannot load this scene)")
    ap.add_argument("--dr-scale", type=float, default=1.0,
                    help="DR phase only: 0 is nominal, 1 is the full ranges")
    ap.add_argument("--load-checkpoint", default=None,
                    help="checkpoint to start from: a step dir, one from<n>/ attempt dir, or "
                         "a whole checkpoints/ tree (the newest global step wins)")
    ap.add_argument("--resume", action="store_true",
                    help="continue this run name: reuse its latest checkpoint (local, else the "
                         "W&B artifact) and spend only the remaining step budget")
    ap.add_argument("--no-restore-value", action="store_true",
                    help="load the policy but reinitialise the critic")

    ap.add_argument("--num-timesteps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--num-eval-envs", type=int, default=None)
    ap.add_argument("--num-evals", type=int, default=None,
                    help="checkpoints are written once per eval; default: the PPO config's 24")
    ap.add_argument("--checkpoint-every", type=float, default=None,
                    help="instead of --num-evals: target minutes between checkpoints, "
                         "converted using the M3 throughput")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--env-override", default=None,
                    help="JSON of env config overrides, e.g. "
                         '\'{"reward_config.scales.reach": 0.15}\'')
    ap.add_argument("--ppo-override", default=None, help="JSON of PPO config overrides")

    ap.add_argument("--logdir", default=str(ROOT / "logs"))
    ap.add_argument("--run-name", default=None, help="default: capturn-<phase>[-dr<scale>]-s<seed>")
    ap.add_argument("--wandb", action="store_true", help="default: on when WANDB_API_KEY is set")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="hand-sim")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--warp-cache", default=None, help="Warp kernel cache dir; survives a restart")
    ap.add_argument("--log-training-metrics", action="store_true",
                    help="log per-rollout episode metrics too; slows training")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run that completes on a CPU Mac -- checks the pipeline, "
                         "not the task")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
def build_configs(args: argparse.Namespace):
    """The env config, the eval env's config and the PPO config, with every flag applied."""
    env_cfg = capturn_config.default_config()
    if args.impl is not None:
        env_cfg.impl = args.impl

    if args.phase == "dr":
        # Env-level DR, from the PRD's table: the scale interpolates between nominal and full.
        env_cfg.noise_config.level = args.dr_scale
        env_cfg.action_delay.enabled = args.dr_scale > 0.0

    ppo_cfg = capturn_config.ppo_config(env_cfg)
    if args.smoke:
        apply_smoke(env_cfg, ppo_cfg)
    for name, value in (
        ("num_timesteps", args.num_timesteps),
        ("num_envs", args.num_envs),
        ("num_eval_envs", args.num_eval_envs),
        ("num_evals", args.num_evals),
    ):
        if value is not None:
            ppo_cfg[name] = value

    if args.env_override:
        apply_overrides(env_cfg, json.loads(args.env_override))
    if args.ppo_override:
        apply_overrides(ppo_cfg, json.loads(args.ppo_override))

    if args.checkpoint_every is not None:
        if args.num_evals is not None:
            raise SystemExit("pass --num-evals or --checkpoint-every, not both")
        ppo_cfg.num_evals = derive_num_evals(ppo_cfg.num_timesteps, args.checkpoint_every)

    # `naconmax` is a total across worlds, so it has to follow the world count or the contact
    # buffer silently overflows and the physics quietly stops being the physics M2 verified.
    # The eval env is vmapped over `num_eval_envs`, which is a different count, so it needs its
    # own config rather than a shared one sized for training.
    env_cfg.naconmax = capturn_config.NACONMAX_PER_WORLD * ppo_cfg.num_envs
    eval_cfg = copy.deepcopy(env_cfg)
    eval_cfg.naconmax = capturn_config.NACONMAX_PER_WORLD * ppo_cfg.num_eval_envs

    check_shapes(env_cfg, ppo_cfg)
    return env_cfg, eval_cfg, ppo_cfg


def check_shapes(env_cfg, ppo_cfg) -> None:
    """The divisibility rules, checked here so they fail with a reason rather than an assert."""
    if env_cfg.episode_length != ppo_cfg.episode_length:
        raise SystemExit(
            f"episode_length disagrees: env {env_cfg.episode_length}, "
            f"ppo {ppo_cfg.episode_length}. Both come from the env config, so an override has "
            "to set the env one."
        )
    rollout = ppo_cfg.batch_size * ppo_cfg.num_minibatches
    if rollout % ppo_cfg.num_envs:
        raise SystemExit(
            f"batch_size * num_minibatches ({rollout}) must be divisible by num_envs "
            f"({ppo_cfg.num_envs}) -- Brax asserts this. Scale batch_size with num_envs."
        )


def apply_smoke(env_cfg, ppo_cfg) -> None:
    """A run small enough to finish on warp-cpu. Not a training run -- a pipeline check."""
    env_cfg.episode_length = 20
    ppo_cfg.episode_length = 20
    ppo_cfg.num_timesteps = 160
    ppo_cfg.num_envs = 8
    ppo_cfg.num_eval_envs = 4
    ppo_cfg.num_evals = 2
    ppo_cfg.unroll_length = 5
    ppo_cfg.batch_size = 8
    ppo_cfg.num_minibatches = 1
    ppo_cfg.num_updates_per_batch = 1
    ppo_cfg.network_factory.policy_hidden_layer_sizes = (32, 32)
    ppo_cfg.network_factory.value_hidden_layer_sizes = (32, 32)


def apply_overrides(cfg, overrides: dict) -> None:
    """`{"reward_config.scales.reach": 0.15}` -> cfg.reward_config.scales.reach = 0.15."""
    for dotted, value in overrides.items():
        node = cfg
        *path, leaf = dotted.split(".")
        for part in path:
            node = node[part]
        if leaf not in node:
            raise SystemExit(f"--*-override: no config field {dotted!r}")
        node[leaf] = value


def derive_num_evals(num_timesteps: int, minutes: float) -> int:
    """`num_evals` that puts roughly `minutes` of wall clock between checkpoints.

    Brax writes one checkpoint per eval, so checkpoint density *is* `num_evals`. The estimate
    uses the M3 throughput and ignores the SGD time, which makes it conservative: the real gap
    is a little longer than asked for, never shorter.
    """
    if minutes <= 0:
        raise SystemExit("--checkpoint-every must be positive")
    seconds = num_timesteps / MEASURED_CONTROL_STEPS_PER_S
    return max(2, round(seconds / (minutes * 60)) + 1)


def run_name_for(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    name = f"capturn-{args.phase}"
    if args.phase == "dr":
        name += f"-dr{args.dr_scale:g}"
    return f"{name}-s{args.seed}"


# --------------------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------------------
def resolve_checkpoint(path_str: str) -> checkpoints.Checkpoint:
    """The newest checkpoint at `path_str`, and the global step it was written at.

    Accepts a step directory, an attempt directory or the whole `checkpoints/` tree, so all of
    `logs/<run>/checkpoints`, `.../checkpoints/from000000000000` and
    `.../checkpoints/from000000000000/000006553600` work.
    """
    path = pathlib.Path(path_str).resolve()
    if not path.exists():
        raise SystemExit(f"--load-checkpoint: {path} does not exist")
    latest = checkpoints.find_latest(path)
    if latest is None:
        raise SystemExit(f"--load-checkpoint: no step-named checkpoint under {path}")
    return latest


def pull_from_wandb(run_name: str, project: str, entity: str | None) -> pathlib.Path | None:
    """The latest checkpoint artifact for this run name, downloaded. None if there is none.

    This is the half of `--resume` that matters: the GPU box's disk does not survive a stop, so
    after a preemption the only copy of the run is the one W&B holds.
    """
    try:
        import wandb
    except ImportError:
        return None
    qualified = f"{entity + '/' if entity else ''}{project}/{artifact_name(run_name)}:latest"
    try:
        artifact = wandb.Api().artifact(qualified, type="model")
        return pathlib.Path(artifact.download())
    except Exception as exc:  # no such artifact, not logged in, offline
        print(f"  no checkpoint artifact at {qualified} ({type(exc).__name__}: {exc})")
        return None


def artifact_name(run_name: str) -> str:
    return f"ckpt-{wandb_id(run_name)}"


def wandb_id(run_name: str) -> str:
    """A W&B run id and artifact name derived from the run name.

    Used as the run *id*, not just its display name, so `--resume` can rejoin the run after a
    preemption without being told anything. W&B ids and artifact names allow letters, digits,
    dashes, underscores and dots; the default DR run name (`capturn-dr-dr0.5-s1`) already has a
    dot, so anything outside that set is folded to a dash rather than risking a rejected id
    halfway through a run.
    """
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in run_name)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_name = run_name_for(args)
    env_cfg, eval_cfg, ppo_cfg = build_configs(args)

    if args.warp_cache:
        import warp as wp

        wp.config.kernel_cache_dir = args.warp_cache

    if os.environ.get("MUJOCO_GL"):
        print(f"note: MUJOCO_GL={os.environ['MUJOCO_GL']} is set. Nothing here renders, and "
              "`egl` breaks `import mujoco` on the Daytona image -- unset it if this fails.")

    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from capturn import compat

    shims = compat.apply()
    if shims:
        print(f"compat shims applied: {', '.join(shims)}  (see capturn/compat.py)")

    print(f"jax backend: {jax.default_backend()}   devices: {jax.devices()}")
    if jax.default_backend() != "gpu" and not args.smoke:
        print("WARNING: no GPU. A real run needs the H100; --smoke is the CPU pipeline check.")

    logdir = pathlib.Path(args.logdir).resolve() / run_name
    ckpt_dir = logdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # `save_dir` is set once `step_offset` is known -- see capturn/checkpoints.py for why each
    # attempt needs its own directory.

    # --------------------------------------------------------------------- restore
    restore_path, step_offset = None, 0
    if args.load_checkpoint:
        restore_path, step_offset = resolve_checkpoint(args.load_checkpoint)
    if args.resume:
        local = checkpoints.find_latest(ckpt_dir)
        if local is not None:
            restore_path, step_offset = local
            print(f"resuming from local checkpoint {restore_path} (global step {step_offset:,})")
        else:
            print(f"no local checkpoint under {ckpt_dir}; asking W&B")
            pulled = pull_from_wandb(run_name, args.wandb_project, args.wandb_entity)
            if pulled is not None:
                restore_path, step_offset = resolve_checkpoint(str(pulled))
                print(f"resuming from W&B artifact {restore_path} "
                      f"(global step {step_offset:,})")
            else:
                print("nothing to resume from; starting fresh")

    # A fine-tune (`--load-checkpoint`) spends a fresh budget; a `--resume` continues one. Only
    # the latter subtracts, because only the latter is the same run.
    budget = int(ppo_cfg.num_timesteps)
    if args.resume and step_offset:
        budget = max(budget - step_offset, 0)
        if budget == 0:
            print(f"run already reached {step_offset:,} of {ppo_cfg.num_timesteps:,} steps; "
                  "nothing to do")
            return
        if args.checkpoint_every is not None:
            # The remaining budget is shorter, so the same minutes-per-checkpoint needs fewer
            # evals. Without this, a resume near the end would checkpoint on every update.
            ppo_cfg.num_evals = derive_num_evals(budget, args.checkpoint_every)
    else:
        step_offset = 0

    save_dir = checkpoints.attempt_dir(ckpt_dir, step_offset)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- env
    if args.phase == "dr":
        try:
            from capturn import randomize  # noqa: F401
        except ImportError:
            raise SystemExit(
                "--phase dr needs capturn/randomize.py, which is M7. The env-level hooks "
                "(observation noise, action delay) are wired here already, but running the DR "
                "phase without the model-level randomisation would quietly be a different "
                "experiment from the one the PRD asks for."
            )
        randomization_fn = randomize.domain_randomize
        if restore_path is None:
            print("note: --phase dr without a checkpoint is the PRD's from-scratch DR baseline")
    else:
        randomization_fn = None

    env = CapTurn(env_cfg)
    eval_env = CapTurn(eval_cfg)

    print(f"\nrun: {run_name}")
    print(f"logdir: {logdir}")
    print(f"phase: {args.phase}" + (f"  dr_scale={args.dr_scale:g}" if args.phase == "dr" else ""))
    print(f"obs: {env.observation_size}   action: {env.action_size}")
    print(f"budget: {budget:,} steps from step {step_offset:,} "
          f"(of {ppo_cfg.num_timesteps:,} total)")
    print(f"num_envs: {ppo_cfg.num_envs}   naconmax: {env_cfg.naconmax:,}   "
          f"njmax: {env_cfg.njmax}")
    print(f"num_evals: {ppo_cfg.num_evals}  -> a checkpoint every "
          f"{budget / max(ppo_cfg.num_evals - 1, 1):,.0f} steps "
          f"(~{budget / max(ppo_cfg.num_evals - 1, 1) / MEASURED_CONTROL_STEPS_PER_S / 60:.1f} "
          "min at the M3 rate)")
    print()

    (logdir / "env_config.json").write_text(json.dumps(env_cfg.to_dict(), indent=2))
    (logdir / "ppo_config.json").write_text(json.dumps(ppo_cfg.to_dict(), indent=2))
    (save_dir / "config.json").write_text(json.dumps(env_cfg.to_dict(), indent=2))

    # --------------------------------------------------------------------- wandb
    use_wandb = args.wandb or (bool(os.environ.get("WANDB_API_KEY")) and not args.no_wandb)
    run = None
    if use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            # The id is derived from the run name so that `--resume` lands in the same run
            # after a preemption without being told anything.
            id=wandb_id(run_name),
            name=run_name,
            resume="allow",
            config={
                "phase": args.phase,
                "dr_scale": args.dr_scale,
                "seed": args.seed,
                "env": env_cfg.to_dict(),
                "ppo": ppo_cfg.to_dict(),
            },
        )
        print(f"wandb: {run.url}")

    # --------------------------------------------------------------------- callbacks
    uploaded: set[int] = set()

    def upload_latest_checkpoint(global_step: int) -> None:
        """Ship the newest checkpoint off the box. Called after Brax has written it."""
        if run is None:
            return
        newest = checkpoints.find_latest(save_dir)
        if newest is None or newest.step in uploaded:
            return
        latest = newest.path
        import wandb

        artifact = wandb.Artifact(
            artifact_name(run_name),
            type="model",
            metadata={
                "step": newest.step,
                "step_in_attempt": int(latest.name),
                "step_offset": step_offset,
                "phase": args.phase,
                "run": run_name,
            },
        )
        # Published under the attempt directory it came from, so a resume that downloads this
        # artifact recovers the global step from the layout rather than from the metadata.
        # The step dir already carries Brax's `ppo_network_config.json`, which is all `eval.py`
        # needs to rebuild the network; `config.json` is the env config, which Brax does not
        # write, so it is added alongside.
        artifact.add_dir(latest.as_posix(), name=f"{save_dir.name}/{latest.name}")
        if (save_dir / "config.json").exists():
            artifact.add_file((save_dir / "config.json").as_posix(),
                              name=f"{save_dir.name}/config.json")
        # A failed upload must not end the run. This whole mechanism exists to survive the
        # machine disappearing, and losing two hours of training to a transient network error
        # while trying to guard against preemption would be a poor trade. The checkpoint is
        # still on disk, and the next eval tries again -- `uploaded` is only marked on success.
        try:
            run.log_artifact(artifact)
        except Exception as exc:
            print(f"  WARNING: checkpoint upload failed ({type(exc).__name__}: {exc}). "
                  f"{latest} is still on disk; retrying at the next eval.", flush=True)
            return
        uploaded.add(newest.step)

    def progress(num_steps: int, metrics: dict) -> None:
        global_step = step_offset + num_steps

        reward = metrics.get("eval/episode_reward")
        success = metrics.get("eval/episode_success")
        net = metrics.get("eval/episode_cap_rotation")
        gross = metrics.get("eval/episode_cap_rotation_abs")
        line = [f"{global_step:>12,}"]
        if reward is not None:
            line.append(f"reward {float(reward):+8.3f}")
        if success is not None:
            line.append(f"success {float(success) * 100:5.1f}%")
        if net is not None:
            line.append(f"net {float(net) * RAD2DEG:+8.1f} deg")
        if net is not None and gross is not None and float(gross) > 1e-9:
            # Near 1 is a gait; near 0 is the cap being shaken back and forth.
            line.append(f"net/gross {float(net) / float(gross):+.2f}")
        print("   ".join(line), flush=True)

        if run is not None:
            # Brax's metrics are scalars, but anything that is not stays out rather than
            # raising from inside a callback and taking the run down with it.
            scalars = {}
            for key, value in metrics.items():
                try:
                    scalars[key] = float(value)
                except (TypeError, ValueError):
                    continue
            try:
                run.log({**scalars, "train/global_step": global_step}, step=global_step)
            except Exception as exc:
                print(f"  WARNING: wandb.log failed ({type(exc).__name__}: {exc})", flush=True)
        upload_latest_checkpoint(global_step)

    # --------------------------------------------------------------------- train
    # `full_reset` belongs to the wrapper, `network_factory` and `num_eval_envs` are passed
    # explicitly, and `num_timesteps` is the resume-adjusted budget. Everything else in the PPO
    # config is a `ppo.train` keyword verbatim.
    passed_separately = ("network_factory", "num_eval_envs", "full_reset", "num_timesteps")
    training_params = {k: v for k, v in dict(ppo_cfg).items() if k not in passed_separately}
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_cfg.network_factory
    )
    # `ppo.train` calls the wrapper without `full_reset`, so bind it (NOTES.md, M4).
    wrap_env_fn = functools.partial(
        wrapper.wrap_for_brax_training, full_reset=bool(ppo_cfg.full_reset)
    )

    started = time.monotonic()
    ppo.train(
        environment=env,
        eval_env=eval_env,
        num_timesteps=budget,
        **training_params,
        num_eval_envs=ppo_cfg.num_eval_envs,
        network_factory=network_factory,
        wrap_env_fn=wrap_env_fn,
        randomization_fn=randomization_fn,
        seed=args.seed,
        restore_checkpoint_path=restore_path.as_posix() if restore_path else None,
        restore_value_fn=not args.no_restore_value,
        save_checkpoint_path=save_dir.as_posix(),
        progress_fn=progress,
        log_training_metrics=args.log_training_metrics,
    )

    print(f"\ndone in {time.monotonic() - started:.1f} s "
          f"({budget / max(time.monotonic() - started, 1e-9):,.0f} steps/s overall)")
    final = checkpoints.find_latest(ckpt_dir)
    if final is not None:
        print(f"final checkpoint: {final.path}  (global step {final.step:,})")
        print(f"next: python3 scripts/eval.py {final.path} --save 4")
    if run is not None:
        upload_latest_checkpoint(step_offset + budget)
        run.finish()


if __name__ == "__main__":
    main()
