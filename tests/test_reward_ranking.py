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
from scripts.squeeze_twist import roll_gait, run  # noqa: E402

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
    """M7's reference: grip, roll the forearm, release, roll back. The human motion."""
    _, ctrl, res = roll_gait(grip=0.50, cycles=10)
    tmp = tmp_path_factory.mktemp("ranking")
    return score_ctrl(env, ctrl, res["qpos"], res["angle"], tmp, "roll_gait")


@pytest.fixture(scope="module")
def finger_gait(env, tmp_path_factory):
    """M1's gait, which turns the cap with the fingers and never rolls the forearm.

    This was the reference through M6 and is now the *counter*-example: it is exactly the
    behaviour `freewheel` exists to stop paying for.
    """
    _, ctrl, res = run(release=0.008, use_wrist=False, **GAIT)
    tmp = tmp_path_factory.mktemp("ranking")
    return score_ctrl(env, ctrl, res["qpos"], res["angle"], tmp, "finger_gait")


def test_rotation_the_forearm_did_not_produce_pays_nothing(env):
    """The core of M7, with the physics taken out of it.

    The hand is frozen at the `pregrasp` pose -- a closed, opposed pinch -- the forearm roll is
    held at zero, and only the cap's hinge advances. So the cap turns a long way under a perfect
    grip with the arm completely still. That is the M6 policy's strategy exactly, and it is the
    thing pronation was added to stop rewarding: `twist` must pay nothing for any of it, and
    `freewheel` must charge for all of it.
    """
    step, _ = score_reward.make_stepper(env)
    qpos = np.asarray(env._init_q, dtype=float)
    ctrl = jp.array(env._init_ctrl)
    d_theta = 0.02  # rad per control step, a steady 0.4 rad/s finger-driven spin

    info = {
        "cap_angle_init": jp.zeros(()),
        "last_cap_angle": jp.zeros(()),
        "last_act": jp.zeros(env.action_size),
        "last_roll": jp.array(qpos[env._roll_qid]),
        "ungrasped_steps": jp.zeros((), dtype=jp.int32),
    }
    twist, freewheel, tracked, pinch = [], [], [], []
    for t in range(1, 201):                      # 4 rad, most of a full turn
        q = qpos.copy()
        q[env._cap_qid] = t * d_theta            # only the cap moves; the roll stays put
        aux, info = step(jp.array(q), jp.zeros(env.mjx_model.nv), ctrl,
                         jp.zeros(env.action_size), info)
        twist.append(float(aux["terms"]["twist"]))
        freewheel.append(float(aux["terms"]["freewheel"]))
        tracked.append(float(aux["tracked"]))
        pinch.append(float(aux["pinch"]))

    turned = 200 * d_theta
    assert pinch[0] > 0.3, (
        "the pregrasp must be a real pinch, or this tests the grasp gate rather than no-slip -- "
        f"it reads {pinch[0]:.3f}"
    )
    assert sum(tracked) == pytest.approx(0.0, abs=1e-9), (
        f"the forearm never moved, so none of the {turned:.2f} rad is roll-tracked"
    )
    assert sum(twist) == pytest.approx(0.0, abs=1e-9), "and none of it may be paid"
    # charged, and charged in proportion to the whole of it
    charged = sum(freewheel) * env.dt
    assert charged == pytest.approx(turned * pinch[0], rel=0.15), (
        f"freewheel should charge ~pinch * {turned:.2f} rad, not {charged:.3f}"
    )


def test_the_reward_prefers_the_forearm_to_the_fingers(reference, finger_gait):
    """The M7 inversion, and the sharpest statement of what this milestone changed.

    Both trajectories unscrew the cap at a similar rate -- 56 deg/s against 44 -- with the same
    two fingers on the same pose. The only difference is *what turns it*: the reference rolls
    the forearm and the finger gait does not. Through M6 the finger gait was the gold standard
    and the reward was tuned to rank it first. It must now come last, and negative: it is a
    policy turning a cap with a motion no hand makes.
    """
    assert reference["tracked_fraction"] > 0.9, (
        "the reference's rotation must be almost entirely roll-driven -- it reads "
        f"{reference['tracked_fraction']*100:.1f}%"
    )
    assert finger_gait["tracked_fraction"] < 0.05, "the finger gait never rolls the forearm"

    assert reference["per_second"] > 0.0 > finger_gait["per_second"], (
        f"roll {reference['per_second']:+.3f}/s must beat fingers "
        f"{finger_gait['per_second']:+.3f}/s, and the fingers must be net-negative"
    )
    # and the reason, not just the ordering
    assert reference["contributions"]["twist"] > 50 * finger_gait["contributions"]["twist"]
    assert finger_gait["contributions"]["freewheel"] < -1.0 < reference["contributions"]["freewheel"]


def test_the_reference_gait_is_not_punished_by_any_cost_term(reference):  # noqa: D401
    """The check that caught the broad `idle_contact` at M5, kept and widened.

    A cost that fires on the best behaviour in the project is measuring the wrong thing. At
    `idle_sharpness = 100` the idle term cost this trajectory -5.3 of return; at 500 it costs
    -0.4. The bottle term is new at M6 and has to clear the same bar: the gait's closest
    approach to the bottle is 3.41 mm against a 2 mm margin, so it should be exactly zero.
    """
    costs = {k: v for k, v in reference["contributions"].items() if v < 0}
    # M7: the pronation gait does brush the bottle neck -- the index fingertip rides onto it
    # during the roll, because the pregrasp was fitted with the forearm welded and has not been
    # re-fitted. Known, recorded in NOTES.md M7, and small: the term costs it under 0.05.
    assert reference["contributions"]["bottle"] > -0.05, (
        f"bottle cost on the reference: {reference['contributions']['bottle']}"
    )
    assert reference["contributions"]["idle_contact"] > -0.5, (
        f"the idle cost must not punish the reference gait: {costs.get('idle_contact')}"
    )
    # M7: stated against `twist` rather than as an absolute epsilon. The finger gait read
    # -2.9e-6 here, which was float32 rounding; the pronation gait loses a real but trivial
    # 0.4 deg against 617 gained as the cap decelerates through each release, costing -0.007.
    # What matters is that the cap is not being dragged back, so the test is a ratio.
    assert abs(reference["contributions"]["backdrive"]) < 0.01 * reference["contributions"]["twist"], (
        "the gait must hold its angle through each release, give or take the cap settling: "
        f"backdrive {reference['contributions']['backdrive']:.4f} against twist "
        f"{reference['contributions']['twist']:.2f}"
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
    """Every policy M5 and M6 produced, scored against the gait none of them could perform.

    M7 added a joint, so these recordings have 25-wide `qpos` against the model's 26 and cannot
    be replayed at all -- they are skipped rather than silently truncated. They are kept here
    because the comparison is still the one worth making once post-M7 episodes exist: every one
    of them turns the cap without a forearm, which is now a negative return by construction.
    """
    path = ROOT / "artifacts" / recording
    if not path.exists():
        pytest.skip(f"{recording} not on disk; regenerate with scripts/eval.py")
    width = np.load(path)["qpos"].shape[1]
    if width != env.mj_model.nq:
        pytest.skip(
            f"{recording} has {width}-wide qpos against the model's {env.mj_model.nq}: it "
            "predates M7's forearm roll and cannot be replayed"
        )

    policy = score_reward.score(env, score_reward.load(path, env))
    assert policy["per_second"] < reference["per_second"]
    assert policy["tracked_fraction"] < 0.5 < reference["tracked_fraction"]
