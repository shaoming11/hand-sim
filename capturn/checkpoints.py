"""Where checkpoints live, and how to find the newest one. Shared by `train.py` and `eval.py`.

Brax names a checkpoint directory after `training_state.env_steps`, and `restore_checkpoint_path`
does not restore that counter -- so after a resume the step count starts at zero again and the
names Brax writes are *relative to the resume*, not global. Resuming a run at 6,553,600 steps
and training on would produce

    checkpoints/000006553600      <- the checkpoint that was resumed from
    checkpoints/000000327680      <- written after it, but named lower
    checkpoints/000000655360

in one directory. "The newest checkpoint" is then not the largest name, and a `--resume` that
picked the largest would resume the same stale checkpoint forever, losing every subsequent
preemption's work silently.

So each attempt gets its own `from<offset>/` directory and Brax names freely inside it. The
global step of a checkpoint is `offset + int(name)`, which makes the newest one well defined
again, and the layout records the resume history instead of flattening it:

    checkpoints/from000000000000/000000327680      global step    327,680
    checkpoints/from000000000000/000006553600      global step  6,553,600
    checkpoints/from000006553600/000000327680      global step  6,881,280

A flat `checkpoints/<step>` layout is still read correctly, with offset zero, so checkpoints
from before this existed and any written by Playground's own train script still load.
"""

from __future__ import annotations

import pathlib
from typing import NamedTuple

ATTEMPT_PREFIX = "from"


class Checkpoint(NamedTuple):
    """A checkpoint directory and the global step it was written at."""

    path: pathlib.Path
    step: int


def attempt_dir(root: pathlib.Path, step_offset: int) -> pathlib.Path:
    """Where Brax should write this attempt's checkpoints."""
    return root / f"{ATTEMPT_PREFIX}{step_offset:012d}"


def _step_dirs(path: pathlib.Path) -> list[pathlib.Path]:
    if not path.is_dir():
        return []
    return sorted(
        (d for d in path.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )


def find_all(root: str | pathlib.Path) -> list[Checkpoint]:
    """Every checkpoint under `root`, oldest global step first.

    `root` may be the `checkpoints/` directory (either layout), one `from*/` attempt directory,
    or a single step directory.
    """
    root = pathlib.Path(root)
    if not root.exists():
        return []
    if root.name.isdigit():
        parent = root.parent.name
        offset = (
            int(parent[len(ATTEMPT_PREFIX):])
            if parent.startswith(ATTEMPT_PREFIX) and parent[len(ATTEMPT_PREFIX):].isdigit()
            else 0
        )
        return [Checkpoint(root, offset + int(root.name))]

    found = [Checkpoint(d, int(d.name)) for d in _step_dirs(root)]  # flat layout, offset 0
    for attempt in sorted(root.glob(f"{ATTEMPT_PREFIX}*")):
        suffix = attempt.name[len(ATTEMPT_PREFIX):]
        if not attempt.is_dir() or not suffix.isdigit():
            continue
        offset = int(suffix)
        found += [Checkpoint(d, offset + int(d.name)) for d in _step_dirs(attempt)]
    return sorted(found, key=lambda c: c.step)


def find_latest(root: str | pathlib.Path) -> Checkpoint | None:
    """The checkpoint with the highest global step under `root`, or None if there is none."""
    found = find_all(root)
    return found[-1] if found else None
