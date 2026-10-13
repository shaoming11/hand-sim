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

The always-on tests cost well under a minute between them, which is the deliberate trade: they
are what stands between a reward change and a two-hour run that was never going to work.
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
from scripts.squeeze_twist import pinch_twist, roll_gait, run  # noqa: E402

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
    """M8's reference: grip, then turn the cap with the fingers *and* the forearm together.

    The index straightens and the thumb draws across it while `rh_WRJ0` pronates -- what a person
    does. On the re-fitted pose, and at this grip, the finger contribution is worth +606.9 deg
    against the same stroke schedule's +326.8 without it.
    """
    _, ctrl, res = pinch_twist(grip=0.50, sweep=0.35, cycles=10)
    tmp = tmp_path_factory.mktemp("ranking")
    return score_ctrl(env, ctrl, res["qpos"], res["angle"], tmp, "pinch_twist")


@pytest.fixture(scope="module")
def roll_only(env, tmp_path_factory):
    """M7's reference: a fixed pinch and the forearm doing all the turning.

    Kept, and still a legitimate way to unscrew a cap -- the pads do travel round the cap axis,
    so the reward pays it. It is simply the weaker half of the motion: the forearm roll sweeps
    the index pad 38.5 deg of cap azimuth per radian where `rh_FFJ3` sweeps it 106.7.
    """
    _, ctrl, res = roll_gait(grip=0.50, cycles=10)
    tmp = tmp_path_factory.mktemp("ranking")
    return score_ctrl(env, ctrl, res["qpos"], res["angle"], tmp, "roll_gait")


@pytest.fixture(scope="module")
def finger_gait(env, tmp_path_factory):
    """M1's gait: two fingers, no forearm roll at all.

    Through M6 this was the reference; M7 made it the counter-example and charged it, which was
    wrong -- `freewheel` charged every bit of rotation the forearm did not produce, and on this
    hand the fingers are the stronger lever by 2.8x. Scored with M8's no-slip measure it is 80%
    pad-carried: a real grip really turning the cap, which is what it always was.
    """
    _, ctrl, res = run(release=0.008, use_wrist=False, **GAIT)
    tmp = tmp_path_factory.mktemp("ranking")
    return score_ctrl(env, ctrl, res["qpos"], res["angle"], tmp, "finger_gait")


def fresh_info(env, qpos):
    return {
        "cap_angle_init": jp.zeros(()),
        "last_cap_angle": jp.zeros(()),
        "last_act": jp.zeros(env.action_size),
        "last_roll": jp.array(qpos[env._roll_qid]),
        "ungrasped_steps": jp.zeros((), dtype=jp.int32),
    }


def drive_cap(env, qvel, steps: int = 200, d_theta: float = 0.02) -> dict:
    """Advance the cap by hand at a fixed rate while the joints move at `qvel`. No physics.

    The hand is injected at the `pregrasp` pose -- a closed, opposed pinch -- and only the cap's
    hinge is stepped, so the cap turns a long way under a perfect grip whatever the joints are
    doing. That isolates the no-slip accounting from the dynamics: what `twist` pays depends
    entirely on whether the two pads were travelling around the cap axis with it.
    """
    step, _ = score_reward.make_stepper(env)
    qpos = np.asarray(env._init_q, dtype=float)
    ctrl = jp.array(env._init_ctrl)
    info = fresh_info(env, qpos)
    rows = {k: [] for k in ("twist", "grind", "tracked", "pinch", "thumb_sweep", "index_sweep")}
    for t in range(1, steps + 1):
        q = qpos.copy()
        q[env._cap_qid] = t * d_theta            # only the cap moves
        aux, info = step(jp.array(q), jp.array(qvel), ctrl, jp.zeros(env.action_size), info)
        rows["twist"].append(float(aux["terms"]["twist"]))
        rows["grind"].append(float(aux["terms"]["grind"]))
        rows["tracked"].append(float(aux["tracked"]))
        rows["pinch"].append(float(aux["pinch"]))
        rows["thumb_sweep"].append(float(aux["sweep"][0]))
        rows["index_sweep"].append(float(aux["sweep"][1]))
    out = {k: np.array(v) for k, v in rows.items()}
    out["turned"] = steps * d_theta
    return out


def test_rotation_the_pads_did_not_carry_pays_nothing(env):
    """The core of M8, with the physics taken out of it.

    The hand is completely still -- every joint velocity zero -- and the cap turns 4 rad under a
    closed, opposed pinch. Nothing on the hand is moving with it, so the cap is sliding under the
    fingers: `twist` must pay nothing and `grind` must charge for all of it.

    This was M7's test and it still passes, but it now proves something stronger. M7 asked only
    that the *forearm roll* be zero, which a hand turning the cap with its fingers also satisfies
    -- so M7 charged the fingers for working. The condition here is that no contact *surface*
    moved, which is slip and nothing else.
    """
    res = drive_cap(env, np.zeros(env.mjx_model.nv))

    assert res["pinch"][0] > 0.3, (
        "the pregrasp must be a real pinch, or this tests the grasp gate rather than no-slip -- "
        f"it reads {res['pinch'][0]:.3f}"
    )
    assert np.allclose(res["thumb_sweep"], 0.0) and np.allclose(res["index_sweep"], 0.0), (
        "a motionless hand sweeps nothing"
    )
    assert res["tracked"].sum() == pytest.approx(0.0, abs=1e-9), (
        f"no surface moved, so none of the {res['turned']:.2f} rad was carried"
    )
    assert res["twist"].sum() == pytest.approx(0.0, abs=1e-9), "and none of it may be paid"
    charged = res["grind"].sum() * env.dt
    assert charged == pytest.approx(res["turned"] * res["pinch"][0], rel=0.15), (
        f"grind should charge ~pinch * {res['turned']:.2f} rad, not {charged:.3f}"
    )


def test_shearing_the_pinch_carries_nothing_however_fast(env):
    """A refusal M7's reward did not have, and the reason `min` is the aggregator.

    `rh_WRJ1` is the wrist flexion, and it is a huge lever on both pads -- it sweeps the thumb
    +199 deg of cap azimuth per radian and the index -226. But it sweeps them *opposite ways*.
    That is a shear, not a rotation: a cap cannot be carried round by two surfaces travelling in
    opposite directions, one of them has to slide. So the pair carries nothing, and this is why
    the ceiling is the lesser of the two sweeps rather than their sum or their mean -- both of
    which would pay generously for this.
    """
    qvel = np.zeros(env.mjx_model.nv)
    qvel[int(env.mj_model.joint("rh_WRJ1").dofadr[0])] = 1.0
    res = drive_cap(env, qvel)

    assert res["thumb_sweep"][0] > 1.0 > -1.0 > res["index_sweep"][0], (
        "the premise: the two pads must be sweeping fast and in opposite directions -- "
        f"thumb {res['thumb_sweep'][0]:+.2f}, index {res['index_sweep'][0]:+.2f} rad/s"
    )
    assert res["twist"].sum() == pytest.approx(0.0, abs=1e-9), (
        "a shear carries no rotation, however fast the surfaces move"
    )
    assert res["grind"].sum() > 0.0, "and the rotation that happened anyway is charged"


def test_the_fingers_carry_more_than_the_forearm_can(env):
    """The measurement behind this milestone, as an assertion.

    Per unit joint rate the index finger straightening while the thumb draws across it carries
    the cap at 1.88 rad/s; the forearm roll alone carries it at 0.68. Nearly three times the
    lever, which is why a reward that paid only the forearm was training out the stronger
    contribution. (On the M7 pose, which gripped the cap's bottom lip, it was 2.14 against 0.34
    -- a 6x gap. Re-fitting the pose narrowed it by improving the roll, not by weakening the
    fingers, and the ordering is what this pins rather than the margin.)

    Both are driven at the same unit joint rate here, and the cap is turned at 1.2 rad/s --
    chosen to sit between the two ceilings, since at a slow enough rotation both motions cover it
    and the test would prove nothing. The finger motion covers it entirely; the forearm cannot
    keep up and the shortfall is charged as slip.
    """
    m = env.mj_model
    fingers = np.zeros(env.mjx_model.nv)
    fingers[int(m.joint("rh_FFJ3").dofadr[0])] = -1.0   # the index straightening
    fingers[int(m.joint("rh_THJ4").dofadr[0])] = +1.0   # the thumb drawing across it
    roll = np.zeros(env.mjx_model.nv)
    roll[int(m.joint("rh_WRJ0").dofadr[0])] = -1.0      # pronation, the M7 motion

    # 1.2 rad/s: above the forearm's ceiling of 0.68 rad/s, below the fingers' 1.88.
    by_fingers = drive_cap(env, fingers, d_theta=0.06)
    by_forearm = drive_cap(env, roll, d_theta=0.06)

    assert min(by_fingers["thumb_sweep"][0], by_fingers["index_sweep"][0]) > 2 * min(
        by_forearm["thumb_sweep"][0], by_forearm["index_sweep"][0]
    ), (
        "the fingers must out-carry the forearm by a wide margin: "
        f"{min(by_fingers['thumb_sweep'][0], by_fingers['index_sweep'][0]):.3f} against "
        f"{min(by_forearm['thumb_sweep'][0], by_forearm['index_sweep'][0]):.3f} rad/s"
    )
    # the finger motion carries the whole of the cap's rotation ...
    assert by_fingers["tracked"].sum() == pytest.approx(by_fingers["turned"], rel=0.01)
    assert by_fingers["grind"].sum() == pytest.approx(0.0, abs=1e-9)
    # ... and the forearm, at the same joint rate, cannot
    assert 0.0 < by_forearm["tracked"].sum() < 0.75 * by_forearm["turned"]
    assert by_forearm["grind"].sum() > 0.0

    assert by_fingers["twist"].sum() > by_forearm["twist"].sum(), (
        "and so the fingers are paid more, which is the whole correction M8 makes to M7"
    )


def test_every_gait_that_really_grips_is_paid(reference, roll_only, finger_gait):
    """What replaces M7's "the forearm must beat the fingers", which was the wrong question.

    M7 ranked the pure-finger gait last and *negative*, on the grounds that it never pronates.
    But a hand that grips a cap and turns it by straightening its fingers is doing the thing the
    task asks for -- scored by whether the pads carry the cap, that gait is 80% carried. Charging
    it was charging the contribution that does most of the work.

    So every gait that really grips the cap and really turns it must be paid, whichever joints it
    used. The refusals live in the three tests above, where they can be stated without a
    trajectory at all: slip pays nothing, shear pays nothing.
    """
    rates = {
        "fingers + forearm": reference["per_second"],
        "forearm only": roll_only["per_second"],
        "fingers only": finger_gait["per_second"],
    }
    for name, rate in rates.items():
        assert rate > 0.0, f"{name} grips and turns the cap, so it must be paid: {rate:+.3f}/s"
    # `pinch_twist` is `roll_gait` with the fingers added on the same stroke schedule and the
    # same forearm travel, so at a matched grip this isolates what the fingers contributed.
    assert reference["rotation_per_roll"] > 1.5 * roll_only["rotation_per_roll"], (
        "at a matched grip, adding the finger motion must buy real rotation: "
        f"{reference['rotation_per_roll']:.3f} vs {roll_only['rotation_per_roll']:.3f} deg of cap "
        "per deg of forearm"
    )
    # Deliberately *not* asserted: that the combined gait outranks pronation at whatever grip
    # suits each best. It does not, and the reason is a limitation of the script rather than of
    # the reward -- `grip_offset` and `sweep_offset` draw on the same pinch actuators, so sweeping
    # the fingers spends grip, and grip is what stops the roll slipping. Tuned independently,
    # `roll_gait` at grip 0.60 holds a pinch of 0.597 and scores +1.150/s against this gait's
    # 0.465 and +0.474. A policy is not under that constraint: it can grip hard through the
    # stroke and ease off only to reset, which is what a hand does and what no fixed envelope
    # can express. Pinning an ordering the script cannot honestly produce would be tuning the
    # test to the conclusion.
    for name, res in (("reference", reference), ("finger gait", finger_gait)):
        assert res["tracked_fraction"] > 0.5, (
            f"{name} should carry most of the rotation it produces: "
            f"{res['tracked_fraction']*100:.1f}%"
        )


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


def test_the_finger_share_metric_reads_the_route_not_the_score(
    reference, roll_only, finger_gait
):
    """`finger_carried_rotation` has to separate the two routes, because nothing else does.

    The reward pays a non-slipping carry equally whether the fingers or the forearm produced it,
    deliberately -- it states what counts as turning the cap, not how to do it. That makes the
    route unobservable from the return, and a training run where the arm does everything would
    look identical to one where the hand does. This metric is the instrument: of the rotation the
    pads carried, the share they carried while the forearm was still.

    Pinned at both ends. The M1 gait never rolls the forearm at all, so it must read ~100%; the
    pronation gaits must read low. If this ever stops discriminating, the first training run
    becomes unreadable, which is the failure M6 built `score_reward.py` to stop repeating.
    """
    assert finger_gait["finger_share"] > 0.95, (
        "a gait with no forearm motion must credit the fingers with all of it: "
        f"{finger_gait['finger_share']*100:.0f}%"
    )
    for name, res in (("fingers + forearm", reference), ("forearm only", roll_only)):
        assert res["finger_share"] < 0.5, (
            f"{name} is forearm-driven and must read as such: {res['finger_share']*100:.0f}%"
        )
    # and it is not just the reward in disguise -- the lowest finger share scores *highest*
    assert roll_only["finger_share"] < finger_gait["finger_share"]
    assert roll_only["per_second"] > finger_gait["per_second"], (
        "the point of a separate metric: the most forearm-driven gait is also the best-scoring "
        "one, so the return cannot be used to read the route"
    )


@slow
def test_the_finger_motion_makes_the_gait_robust_to_grip():
    """Why the finger contribution is worth having, stated as the thing that actually holds.

    Pronation alone only works in a narrow window of grip: too light and the cap slips through
    the stroke, too firm and the pads bind. Across grip 0.40 to 0.80 `roll_gait` ranges from
    +24 deg to +927 -- a factor of 39 -- and two of the five settings barely turn the cap at all.
    Adding the finger motion flattens that: `pinch_twist` stays between +235 and +678 over the
    same range, a factor of 3.

    That matters for training rather than for the scripts. A reward whose only route to the cap
    is a narrow grip window is a reward a policy has to find by luck; one with a broad plateau is
    one it can climb. Slow because it is ten CPU rollouts.
    """
    grips = (0.40, 0.50, 0.60, 0.70, 0.80)
    rolled, combined = [], []
    for grip in grips:
        rolled.append(roll_gait(grip=grip, cycles=10)[2]["total_deg"])
        combined.append(pinch_twist(grip=grip, sweep=0.35, cycles=10)[2]["total_deg"])

    def spread(xs):
        return max(xs) / max(min(xs), 1.0)

    assert min(combined) > 2 * min(rolled), (
        "the worst grip must still unscrew the cap when the fingers help: "
        f"{min(combined):.0f} deg against {min(rolled):.0f}"
    )
    assert spread(combined) < 0.25 * spread(rolled), (
        f"and the gait must be far less grip-sensitive: spread {spread(combined):.1f}x "
        f"against {spread(rolled):.1f}x"
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
    be replayed at all -- they are skipped rather than silently truncated. They are kept because
    the comparison is still the one worth making once post-M7 episodes exist, though M8 changes
    what it is testing: not "did the forearm turn the cap" -- the M6 policies had no forearm to
    turn it with and should not be charged for that -- but "did the grip carry the cap, or grind
    against it". The flick and the one-sided hover should still fail that; a clean two-finger
    gait no longer has to.
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
    assert policy["tracked_fraction"] < reference["tracked_fraction"]
