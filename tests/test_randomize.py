"""PRD M7 checks for `capturn/randomize.py`.

The PRD asks for one thing explicitly -- "after building the batched model, read every randomized
field back and assert it varies across envs" -- and that is `test_every_randomized_field_varies`.
The rest pin the two properties that would make a DR run quietly meaningless: that `--dr-scale`
actually narrows the ranges (a scale that does nothing is the failure M7's `functools.partial`
in `train.py` exists to prevent), and that `scale=0` reproduces the nominal model exactly, so the
DR phase at zero and the nominal phase are the same experiment.

Nothing here steps physics, so the whole file runs on the Mac in a few seconds.
"""

from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jp
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from capturn import config as capturn_config  # noqa: E402
from capturn import randomize  # noqa: E402
from capturn.env import CapTurn  # noqa: E402

N_ENVS = 8


@pytest.fixture(scope="module")
def model():
    return CapTurn(capturn_config.default_config()).mjx_model


@pytest.fixture(scope="module")
def rng():
    return jax.random.split(jax.random.PRNGKey(0), N_ENVS)


def test_every_randomized_field_varies(model, rng):
    batched, _ = randomize.domain_randomize(model, rng, scale=1.0)
    spreads = randomize.assert_varies(batched)          # raises if any field is constant
    assert set(spreads) == set(randomize.RANDOMIZED_FIELDS)


def test_in_axes_marks_exactly_the_randomized_fields(model, rng):
    _, in_axes = randomize.domain_randomize(model, rng, scale=1.0)
    for field in randomize.RANDOMIZED_FIELDS:
        assert getattr(in_axes, field) == 0, f"{field} is randomized but not mapped"
    # Anything the sampler does not touch must stay unbatched, or vmap would try to map a
    # field that has no env axis.
    assert in_axes.opt.timestep is None
    assert in_axes.geom_size is None


def test_scale_zero_is_the_nominal_model(model, rng):
    batched, _ = randomize.domain_randomize(model, rng, scale=0.0)
    for field in randomize.RANDOMIZED_FIELDS:
        value = getattr(batched, field)
        nominal = getattr(model, field)
        assert jp.allclose(value, jp.broadcast_to(nominal, value.shape)), field


def test_scale_narrows_every_range(model, rng):
    """A quarter-scale run must be strictly inside a full-scale one, field by field."""
    full, _ = randomize.domain_randomize(model, rng, scale=1.0)
    quarter, _ = randomize.domain_randomize(model, rng, scale=0.25)
    for field in randomize.RANDOMIZED_FIELDS:
        f, q = getattr(full, field), getattr(quarter, field)
        spread_full = float(jp.sum(jp.max(f, axis=0) - jp.min(f, axis=0)))
        spread_quarter = float(jp.sum(jp.max(q, axis=0) - jp.min(q, axis=0)))
        assert spread_quarter < spread_full, f"{field}: {spread_quarter} not inside {spread_full}"


def test_cap_thread_friction_lands_in_the_prd_range(model, rng):
    """The one absolute range in the PRD table: U(0.02, 0.15) N*m at full scale."""
    batched, _ = randomize.domain_randomize(model, rng, scale=1.0)
    values = batched.dof_frictionloss[:, randomize.IDS["cap_dof"]]
    lo, hi = randomize.CAP_FRICTIONLOSS
    assert float(values.min()) >= lo and float(values.max()) <= hi


def test_position_actuators_keep_bias_equal_to_minus_gain(model, rng):
    """Randomizing the gain without the bias turns a position servo into something else."""
    batched, _ = randomize.domain_randomize(model, rng, scale=1.0)
    assert jp.allclose(batched.actuator_biasprm[:, :, 1], -batched.actuator_gainprm[:, :, 0])


def test_the_cap_hinge_is_not_jittered_by_the_hand_ranges(model, rng):
    """`qpos0` jitter is the hand's; moving the cap's zero would move the task itself."""
    import mujoco

    from capturn import scene
    batched, _ = randomize.domain_randomize(model, rng, scale=1.0)
    m = mujoco.MjModel.from_xml_path(scene.SCENE.as_posix())
    cap_qid = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, scene.CAP_JOINT)])
    assert jp.allclose(batched.qpos0[:, cap_qid], model.qpos0[cap_qid])
