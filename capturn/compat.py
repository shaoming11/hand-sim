"""M5: the two shims this Brax/JAX pair needs to train and then load what it trained.

Both were found by running the pipeline end to end on warp-cpu (`train.py --smoke` followed by
`eval.py`), and neither is reachable by importing anything -- the first fires minutes into a
run, the second only when a checkpoint is read back.

1. `jax.device_put_replicated`, removed by JAX, still called by Brax.
2. `KERNEL_INITIALIZER[None]`, a save/load asymmetry in Brax's own checkpoint code.

Shim 1
------

`brax.training.agents.ppo.train` replicates the training state across devices with
`jax.device_put_replicated`. JAX 0.11.2 removed that public alias as part of the `pmap`
migration -- not deprecated-with-a-warning, *removed*: `jax/__init__.py` registers it with a
`None` handler, so touching `jax.device_put_replicated` raises

    AttributeError: jax.device_put_replicated is deprecated; use jax.device_put instead.

Brax 0.14.2 is the newest release and still has the call, so there is no version of the pair
that works out of the box. The two ways out:

1. Pin JAX down to a release where the alias still resolves. That changes the stack underneath
   MuJoCo 3.14, MJX-Warp and `warp-lang` 1.17 -- the stack M2's parity result and M3's 1.26M
   sim-steps/s were measured on. Every one of those numbers would need re-measuring to still
   mean anything.
2. Restore the one symbol. The implementation was never deleted, only the alias:
   `jax._src.api.device_put_replicated` is present and works.

This takes (2), because the blast radius is one name instead of a whole numerical stack. It is
a monkeypatch on a dependency, so it is deliberately loud about what it is doing and it checks
first that the symbol is actually missing -- when a later Brax or JAX fixes this, the shim
becomes a no-op and `tests/test_train.py` says so.

Import this before calling `brax.training.agents.ppo.train`. Importing Brax is not enough to
trigger the problem: the lookup happens at call time, several minutes into a run, after the
env is built and the Warp kernels are compiled.

Shim 2
------
Brax's checkpoint `save` and `load_config` disagree about kernel initialisers that are `None`.
`make_ppo_networks` has `mean_kernel_init_fn=None` by default, so `network_config` captures
`None`, and `save` deliberately leaves it alone:

    if config_cp_dict['network_factory_kwargs'][init_fn_name] is None:
        continue

while `load_config` looks the stored value up unconditionally:

    networks.KERNEL_INITIALIZER[init_fn_name_]        # KeyError: None

So *every* PPO checkpoint this Brax writes is unreadable by its own `load_policy` unless that
initialiser is bound to something registered -- and binding it would change the network's
initialisation to work around a serialisation bug, which is the wrong trade.

The registry is a plain name -> function dict, and the stored `None` already means "no
initialiser", so teaching it that `None` resolves to `None` makes the round trip agree with
itself. One dict entry, no behaviour change, and `save`'s reverse lookup never asks for it.
"""

from __future__ import annotations


def patch_jax_device_put_replicated() -> bool:
    """Restore `jax.device_put_replicated` if this JAX removed it. True if it was patched."""
    import jax

    try:
        jax.device_put_replicated  # noqa: B018
        return False
    except AttributeError:
        pass

    from jax._src import api

    fn = getattr(api, "device_put_replicated", None)
    if fn is None:
        raise RuntimeError(
            "jax.device_put_replicated is gone and jax._src.api has no replacement either. "
            "Brax's ppo.train needs it; pin jax to a release that still has it, or patch "
            "brax to use jax.device_put. See capturn/compat.py."
        )
    jax.device_put_replicated = fn
    return True


def patch_brax_kernel_initializer_none() -> bool:
    """Let Brax read back the `None` kernel initialiser it writes. True if it was patched."""
    from brax.training import networks

    if None in networks.KERNEL_INITIALIZER:
        return False
    networks.KERNEL_INITIALIZER[None] = None
    return True


def apply() -> list[str]:
    """Every shim this stack needs, applied. Returns the names of the ones that were used."""
    applied = []
    if patch_jax_device_put_replicated():
        applied.append("jax.device_put_replicated")
    if patch_brax_kernel_initializer_none():
        applied.append("brax KERNEL_INITIALIZER[None]")
    return applied
