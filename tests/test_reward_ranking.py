"""M6 acceptance checks: does the reward prefer the behaviour the project wants?

Through M5 that question was only ever answered after a two-hour GPU run. Twice the answer was
wrong for a reason that was visible on disk the whole time -- an `idle_contact` term that cost
the reference gait -5.3 of return, and a `rotate` term that paid an exploit 60.0 against the
reference gait's 54.9. Both were reward-design errors, and both are the kind that a reward
scored against a known-good trajectory catches in seconds.

So these tests score trajectories with the live reward, via `scripts/score_reward.py`, and pin
the ordering. If a weight changes such that a grind outranks a gait again, this is what fails.

Two tiers, because `artifacts/*.npz` is gitignored:

* the scripted reference gait and its variants are *generated here*, on CPU, in a few seconds,
  so the ordering they pin is always checked;
* the recorded policy episodes cannot be regenerated without a GPU run, so the tests that use
  them skip when they are absent, and are gated behind `HAND_SIM_SLOW` as well.

The three always-on tests cost about 30 s between them, which is the deliberate trade: they are
what stands between a reward change and a two-hour run that was never going to work.
`scripts/score_reward.py` is the same measurement with the numbers printed.
"""

from __future__ import annotations

import os
import pathlib
import sys

import jax.numpy as jp
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import score_reward  # noqa: E402
from scripts.squeeze_twist import run  # noqa: E402

# The gait parameters the pregrasp was re-fitted around: a light grip (0.35), two fingers, and a
# release that really lets go. NOTES.md, M5 -- the opposed pose wants a *lighter* grip than the
# old tripod, because the tips already bracket the cap and winding in harder drives them into
# the neck.
GAIT = dict(fingers=["th", "ff"], grip=0.35, twist=0.9, cycles=12)


slow = pytest.mark.skipif(
    not os.environ.get("HAND_SIM_SLOW"),
    reason="set HAND_SIM_SLOW=1 to score the recorded policy episodes",
)


@pytest.fixture(scope="module")
def env():
    return score_reward.build_env()


def score_ctrl(env, ctrl, qpos, angle, tmp: pathlib.Path, name: str) -> dict:
    """Score a CPU rollout by the route `score_reward` uses on a saved recording."""
    path = tmp / f"{name}.npz"
    np.savez(path, qpos=qpos, ctrl=ctrl, cap_angle=angle, ctrl_dt=0.05)
    return score_reward.score(env, score_reward.load(path, env))


@pytest.fixture(scope="module")
def reference(env, tmp_path_factory):
    _, ctrl, res = run(release=0.008, use_wrist=False, **GAIT)
    tmp = tmp_path_factory.mktemp("ranking")
    return score_ctrl(env, ctrl, res["qpos"], res["angle"], tmp, "reference")


@pytest.fixture(scope="module")
def no_release(env, tmp_path_factory):
    """The same gait with the release amplitude set to zero -- it grips and keeps gripping."""
    _, ctrl, res = run(release=0.0, use_wrist=False, **GAIT)
    tmp = tmp_path_factory.mktemp("ranking")
    return score_ctrl(env, ctrl, res["qpos"], res["angle"], tmp, "no_release")


def test_a_single_grip_is_paid_for_one_stroke_and_no_more(env):
    """The core of M6, with the physics taken out of it.

    The hand is frozen at the `pregrasp` pose -- which is a closed, opposed pinch -- and only the
    cap's hinge angle advances. So this is the pure grind: one grip, held forever, while the cap
    turns without limit and the fingers never release. Exactly the strategy every M5 seed found,
    and under the old `rotate` term it was paid for every radian.

    `stroke_budget` is what bounds it now. Two full turns of grinding must be paid for one
    stroke, and the twelfth turn must be paid the same as the second: nothing.
    """
    step, _ = score_reward.make_stepper(env)
    budget = float(env._config.reward_config.stroke_budget)
    qpos = np.asarray(env._init_q, dtype=float)
    ctrl = jp.array(env._init_ctrl)
    d_theta = 0.02  # rad per control step, a steady 0.4 rad/s grind

    info = {
        "cap_angle_init": jp.zeros(()),
        "last_cap_angle": jp.zeros(()),
        "last_act": jp.zeros(env.action_size),
        "grip_held": jp.zeros((), dtype=bool),
        "grip_angle_start": jp.zeros(()),
        "ungrasped_steps": jp.zeros((), dtype=jp.int32),
    }
    paid, pinch, twist, strokes = [], [], [], 0
    for t in range(1, 401):  # 8 rad, a little over two full turns
        q = qpos.copy()
        q[env._cap_qid] = t * d_theta
        aux, info = step(jp.array(q), jp.zeros(env.mjx_model.nv), ctrl,
                         jp.zeros(env.action_size), info)
        paid.append(float(aux["paid"]))
        pinch.append(float(aux["pinch"]))
        twist.append(float(aux["terms"]["twist"]))
        strokes += int(aux["new_grip"])

    turned = 400 * d_theta
    assert turned > 2 * budget, "the test has to out-turn the budget for this to mean anything"
    assert pinch[0] > 0.3, (
        "the pregrasp pose must be a real pinch for this to be testing the budget rather than "
        f"the gate -- it reads {pinch[0]:.3f}"
    )
    assert strokes == 1, "one grip, never released"
    assert sum(paid) == pytest.approx(budget, rel=0.05), (
        f"a single grip must be paid for {budget:.3f} rad of the {turned:.2f} it turned, "
        f"not {sum(paid):.3f}"
    )
    # and the pay really stops rather than tapering: the last turn earns nothing at all
    assert sum(twist[-100:]) == pytest.approx(0.0, abs=1e-9)


def test_the_reward_prefers_a_gait_to_the_same_gait_that_never_lets_go(reference, no_release):
    """Same hand, same cycle, same rotation -- only the releases differ.

    The no-release variant turns the cap +399 deg against the reference's +420, so under any
    reward that pays for rotation alone the two are within 5% of each other. They are not meant
    to be: one regrips twelve times and the other holds on. (It is not a *pure* grind -- the
    return sweep backs the tips off incidentally, so four of its twelve cycles still register as
    releases, which is why the separation here is 1.8x rather than the 5-20x the recorded
    one-sided policies show. `test_a_single_grip_is_paid_for_one_stroke_and_no_more` is the clean
    version of this.)
    """
    assert reference["net_rotation_deg"] > 350 and no_release["net_rotation_deg"] > 350
    assert abs(reference["net_rotation_deg"] - no_release["net_rotation_deg"]) < 50, (
        "the premise: these two turn the cap about equally far"
    )
    assert reference["strokes"] >= 10, "twelve scripted cycles, and the machine should see them"
    assert no_release["strokes"] <= 6

    assert reference["paid_fraction"] > 0.9, (
        "the reference gait must be paid for everything it turns -- `stroke_budget` is set so "
        f"its best cycle fits, and it reads {reference['paid_fraction']*100:.1f}%"
    )
    assert no_release["paid_fraction"] < 0.6
    assert reference["total"] > 1.5 * no_release["total"] > 0.0


def test_the_reference_gait_is_not_punished_by_any_cost_term(reference):
    """The check that caught the broad `idle_contact` at M5, kept and widened.

    A cost that fires on the best behaviour in the project is measuring the wrong thing. At
    `idle_sharpness = 100` the idle term cost this trajectory -5.3 of return; at 500 it costs
    -0.4. The bottle term is new at M6 and has to clear the same bar: the gait's closest
    approach to the bottle is 3.41 mm against a 2 mm margin, so it should be exactly zero.
    """
    costs = {k: v for k, v in reference["contributions"].items() if v < 0}
    assert reference["contributions"]["bottle"] == pytest.approx(0.0, abs=1e-9), (
        "the reference gait never touches the bottle, so the bottle cost must be zero on it"
    )
    assert reference["contributions"]["idle_contact"] > -0.5, (
        f"the idle cost must not punish the reference gait: {costs.get('idle_contact')}"
    )
    # 1e-4 rather than exactly zero: the cap angle is float32, and the gait reads -2.9e-6 of
    # backdrive, which is rounding rather than the cap springing back.
    assert reference["contributions"]["backdrive"] == pytest.approx(0.0, abs=1e-4), (
        "the gait's rotation is monotonic -- it holds its angle through every release"
    )
    assert reference["contributions"]["twist"] > 3 * -sum(costs.values()), (
        "and the costs together must stay well under what the gait earns for gaiting"
    )


@slow
@pytest.mark.parametrize("recording", [
    "seed1_best_ep00.npz",
    "sharp_s1_ep00.npz",
    "twofinger_s2_ep00.npz",
    "fix5080_ep00.npz",
])
def test_recorded_policies_rank_below_the_reference_gait(env, reference, recording):
    """Every policy M5 produced, scored against the gait none of them learned.

    These are the behaviours the old reward preferred: a one-sided hover, a knuckle-driven
    grind, a flick. Each turns the cap several times further than the scripted gait and each
    does it without a pinch, so `twist` pays almost none of it and `slip` charges for the rest.

    Skipped rather than failed when the recordings are absent -- `artifacts/*.npz` is gitignored
    and these need a GPU run to regenerate (`scripts/eval.py --save-episodes`).
    """
    path = ROOT / "artifacts" / recording
    if not path.exists():
        pytest.skip(f"{recording} not on disk; regenerate with scripts/eval.py")

    policy = score_reward.score(env, score_reward.load(path, env))
    assert policy["per_second"] < reference["per_second"], (
        f"{recording} earns {policy['per_second']:+.4f}/s against the reference gait's "
        f"{reference['per_second']:+.4f}/s"
    )
    # and the reason, not just the ranking: the rotation it produces is not pinched rotation
    assert policy["paid_fraction"] < 0.5 < reference["paid_fraction"]
