"""M4: the training environment -- a Playground `MjxEnv` that unscrews the cap.

Structure mirrors Playground's `LeapCubeRotateZAxis`, which the PRD names as the template. The
task differs in one way that drives most of the design: the cap's hinge angle is *cumulative*,
so the thing being rewarded is progress along an unbounded coordinate rather than a velocity to
be held. That makes the episode's start angle a per-episode constant the reward has to subtract,
and it is why the notes below care so much about what survives an auto-reset.

Observations
------------
One actor frame is 70 numbers, and `state` is the last `history_len` (3) of them, newest first:

    hand joint positions      24   + observation noise (zero in the nominal phase)
    previous action           20
    position targets          20   `data.ctrl`, the realized targets
    cap position, palm frame   3   + observation noise
    cap axis, palm frame       3

`privileged_state` is that whole history followed by 73 more:

    hand joint velocities     24
    cumulative cap rotation    1   unwrapped, relative to this episode's start angle
    cap angular velocity       1
    fingertip positions       15   all five, in the cap frame
    actuator forces           20
    DR parameter slot         12   zeros until M7 fills it

The actor deliberately cannot see the cap angle. The cap is rotationally symmetric, so the
policy does not need it and could not measure it on hardware.

Metrics, and why they are rates rather than levels
--------------------------------------------------
Brax reads `state.metrics` through `EvalWrapper`, which **sums** each one over the episode's
active steps and only divides by the episode length for names ending in `_per_step`. A metric
defined as a level therefore logs as nonsense: a per-step "cumulative rotation so far" sums to
the area under the rotation curve, and a per-step "is past 4*pi" sums to a step count. So every
metric here is defined so that its episode sum is the quantity worth reading:

    cap_rotation          d_theta        -> net radians turned in the episode
    cap_rotation_abs      |d_theta|      -> gross radians turned
    cap_angvel_per_step   angular vel.   -> mean angular velocity (Brax divides)
    success               4*pi crossing  -> 1 for a successful episode
    reward/<term>         the term       -> that term's contribution to the return

`cap_rotation / cap_rotation_abs` is then a free exploit detector: a policy that turns the cap
by shaking it back and forth scores near zero on the ratio while a real gait scores near one.

Auto-reset and what may live in `info`
--------------------------------------
Playground's `BraxAutoResetWrapper` with `full_reset=False` restores `data` and `obs` at an
episode boundary but leaves `info` alone. Anything in `info` that *evolves during* an episode is
therefore stale on the first step of the next one. Both `info` and the cached reset state come
from the same single `reset` call, so per-episode *constants* in `info` stay consistent forever.

So the rule here is: `info` holds only per-episode constants (`cap_angle_init`, `action_delay`,
`dr_params`), and everything that evolves is derived from `data`:

* the integrator's previous target is `data.ctrl`, not a copy in `info`
* cumulative rotation is `cap_angle - cap_angle_init`, both consistent across a reset
* success is recomputed from that rotation rather than latched in a flag

Three fields are exceptions -- `last_act`, `last_last_act` and `action_buffer`, which feed the
action-rate cost, and `last_cap_angle`, which feeds the finite difference behind the rotate
reward. They genuinely evolve and have nowhere else to live. Stale, they cost one spurious
action-rate term on the first step of an episode (as in Playground's own envs) and one rotate
term, which the term's own clip bounds at -0.5 however far the previous episode had turned.
`ppo_config()` sets `full_reset=True`, where `info` is reset too and none of this arises; the
separate and stronger reason for that setting is that the reset randomisation is otherwise
frozen after the first episode.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from capturn import scene
from capturn.config import default_config

N_HAND_JOINTS = 24
N_ACTUATORS = 20
# Zeros in the nominal phase. Sized for the eight model-level scalars in the PRD's DR table plus
# the bottle's 3-vector offset and its tilt: fingertip friction, cap friction, cap frictionloss,
# cap damping, actuator gain, hand damping, hand frictionloss, hand armature, bottle_pos (3),
# bottle tilt. The per-joint `qpos0` offsets are left out -- 24 more numbers would dominate the
# slot, and a critic can infer a joint offset from the joint's own behaviour.
DR_PARAM_SLOTS = 12

FRAME_SIZE = N_HAND_JOINTS + N_ACTUATORS + N_ACTUATORS + 3 + 3  # 70
PRIVILEGED_EXTRA = (
    N_HAND_JOINTS + 1 + 1 + 3 * len(scene.ALL_TIPS) + N_ACTUATORS + DR_PARAM_SLOTS
)  # 73


class CapTurn(mjx_env.MjxEnv):
    """Shadow Hand unscrewing a bottle cap by finger gaiting."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        # Built fresh rather than taken as a default argument: Playground's envs default to a
        # module-level `default_config()` instance, and `config_overrides` then mutates that
        # shared dict for every later instance.
        super().__init__(default_config() if config is None else config, config_overrides)

        self._xml_path = scene.SCENE.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self._config.sim_dt
        # Unlike Playground's Leap base class this does *not* touch `ccd_iterations`. The
        # scene's 50 / 1e-8 is an M1 finding: every fingertip here is a mesh against a mesh cap,
        # and at Leap's setting of 10 the tips sink 7.4 mm in rather than 1.3 mm.
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        m = self._mj_model

        # Read names from the compiled model; never hardcode indices.
        hand_joints = [
            m.joint(j).name for j in range(m.njnt) if m.joint(j).name != scene.CAP_JOINT
        ]
        if len(hand_joints) != N_HAND_JOINTS:
            raise ValueError(f"expected {N_HAND_JOINTS} hand joints, found {len(hand_joints)}")
        self._hand_qids = mjx_env.get_qpos_ids(m, hand_joints)
        self._hand_dqids = mjx_env.get_qvel_ids(m, hand_joints)
        self._cap_qid = int(m.joint(scene.CAP_JOINT).qposadr[0])

        limits = np.array([m.joint(j).range for j in hand_joints], dtype=float)
        unlimited = np.array([not m.joint(j).limited for j in hand_joints], dtype=bool)
        limits[unlimited] = (-np.inf, np.inf)
        self._joint_lowers, self._joint_uppers = jp.array(limits[:, 0]), jp.array(limits[:, 1])

        key = m.keyframe("pregrasp")
        self._init_q = jp.array(key.qpos)
        self._init_ctrl = jp.array(key.ctrl)
        lowers, uppers = m.actuator_ctrlrange.T
        self._lowers, self._uppers = jp.array(lowers), jp.array(uppers)

        self._cap_radius, self._cap_half_height = scene.cap_dimensions(m)
        self._cap_site = m.site("cap_site").id

        # M5: the two surfaces a bottle-cap grasp actually uses. Resolved from body names, and
        # every collision geom on those bodies counts -- the index finger's *side* is a
        # legitimate contact, so this must not be narrowed to the fingertip.
        self._thumb_geoms = self._collision_geoms(m, scene.THUMB_BODIES)
        self._index_geoms = self._collision_geoms(m, scene.INDEX_BODIES)

        self._buffer_len = int(self._config.action_delay.max_steps) + 1

    @staticmethod
    def _collision_geoms(m: mujoco.MjModel, bodies: tuple[str, ...]) -> dict[str, Any]:
        """Geom ids on `bodies` that can collide, with the radius and half-length each needs."""
        ids = [
            g for g in range(m.ngeom)
            if m.body(m.geom_bodyid[g]).name in bodies
            and (m.geom_contype[g] or m.geom_conaffinity[g])
        ]
        if not ids:
            raise ValueError(f"no collision geoms on {bodies}")
        capsule = mujoco.mjtGeom.mjGEOM_CAPSULE
        return {
            "ids": np.array(ids),
            "radius": jp.array([float(m.geom_size[g, 0]) for g in ids]),
            # Only a capsule has an axial extent to sample along; anything else is a point.
            "half_length": jp.array(
                [float(m.geom_size[g, 1]) if m.geom_type[g] == capsule else 0.0 for g in ids]
            ),
        }

    # ---------------------------------------------------------------------------------
    # rollout
    # ---------------------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, hand_rng, cap_rng, delay_rng = jax.random.split(rng, 4)

        jitter = self._config.reset_noise.hand_qpos
        q_hand = jp.clip(
            self._init_q[self._hand_qids]
            + jax.random.uniform(hand_rng, (N_HAND_JOINTS,), minval=-jitter, maxval=jitter),
            self._joint_lowers,
            self._joint_uppers,
        )
        cap_angle = jax.random.uniform(cap_rng, (), maxval=self._config.reset_noise.cap_angle)

        qpos = self._init_q.at[self._hand_qids].set(q_hand).at[self._cap_qid].set(cap_angle)
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=jp.zeros(self.mjx_model.nv),
            # The keyframe's `ctrl` is the fitted target that holds the pregrasp pose, and it is
            # also the integrator's starting point. Only `qpos` is jittered: the policy moves
            # the target from step one anyway, so jittering both would just add a transient.
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        # Playground's Leap env reads its sensors straight out of `make_data`, where
        # `sensordata` is still zeros, so its first observation of the cube is fiction. One
        # forward pass costs a fortieth of a control step and makes the first frame real.
        data = mjx.forward(self.mjx_model, data)

        if self._config.action_delay.enabled:
            delay = jax.random.randint(delay_rng, (), 0, self._buffer_len)
        else:
            delay = jp.zeros((), dtype=jp.int32)

        info = {
            "rng": rng,
            # per-episode constants -- safe to keep here across an auto-reset
            "cap_angle_init": cap_angle,
            "action_delay": delay,
            "dr_params": jp.zeros(DR_PARAM_SLOTS),
            # evolving, and stale for one step when `full_reset=False`
            "last_act": jp.zeros(N_ACTUATORS),
            "last_last_act": jp.zeros(N_ACTUATORS),
            "action_buffer": jp.zeros((self._buffer_len, N_ACTUATORS)),
            "last_cap_angle": cap_angle,
            "ungrasped_steps": jp.zeros((), dtype=jp.int32),
        }

        # Every metric here is defined so that *summing it over an episode* is the
        # quantity worth reading -- see the module docstring. `_per_step` is Brax's own
        # suffix for "divide by the episode length", which turns a sum into a mean.
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales}
        metrics["cap_rotation"] = jp.zeros(())  # d_theta -> net radians turned
        metrics["cap_rotation_abs"] = jp.zeros(())  # |d_theta| -> gross radians turned
        metrics["cap_angvel_per_step"] = jp.zeros(())  # -> mean angular velocity
        metrics["success"] = jp.zeros(())  # the 4*pi crossing -> the success flag
        # M5: the grasp reward is a product, so a zero is ambiguous -- log the three parts
        # separately to say whether the thumb is off the cap, the index is, or they are on the
        # same side. `_per_step` is Brax's suffix for "divide by the episode length".
        metrics["thumb_gap_per_step"] = jp.zeros(())
        metrics["index_gap_per_step"] = jp.zeros(())
        metrics["opposition_per_step"] = jp.zeros(())

        obs = self._get_obs(data, info, jp.zeros(self._config.history_len * FRAME_SIZE))
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = state.info

        # The *action* is delayed, not the target, so a delayed step still integrates exactly
        # once. Index 0 is the newest action; nominal reads index 0 and is therefore undelayed.
        buffer = jp.roll(info["action_buffer"], shift=1, axis=0).at[0].set(action)
        effective = buffer[info["action_delay"]]

        # The previous target is `data.ctrl` rather than a field in `info`, so that an episode
        # that starts from a restored `data` also starts from that data's target.
        target = jp.clip(
            state.data.ctrl + self._config.action_scale * effective, self._lowers, self._uppers
        )
        data = mjx_env.step(self.mjx_model, state.data, target, self.n_substeps)

        cap_angle = self._cap_angle(data)
        d_theta = cap_angle - info["last_cap_angle"]
        rotation = cap_angle - info["cap_angle_init"]

        # Computed once and threaded through: the reward, the termination and three metrics
        # all want it, and it walks every collision geom on two fingers.
        grasp = self.grasp_state(data)
        thumb_gap, index_gap, opposition = grasp

        # M5: a gait *has* to let go to regrip, so losing contact for an instant is not a
        # failure -- terminating on it would end the M1 gait on 15% of its own steps. What
        # separates the two is how long the grasp stays off: the scripted gait's longest
        # release is 3 control steps, the flick policy hovers for 18 at a time, 11 times an
        # episode. The counter resets to zero the moment both surfaces are back on the cap.
        ungrasped = jp.maximum(thumb_gap, index_gap) > self._config.contact_distance
        info["ungrasped_steps"] = (info["ungrasped_steps"] + 1) * ungrasped

        done = self._get_termination(data, info["ungrasped_steps"])
        terms = self._get_reward(data, action, info, d_theta, rotation, grasp)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in terms.items()}
        # Playground's convention: the terms are rates and the sum is integrated over the
        # control period. The rotate term is then literally radians turned this step.
        reward = sum(rewards.values()) * self.dt

        info["action_buffer"] = buffer
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["last_cap_angle"] = cap_angle

        # Built after `last_act` is updated, so the frame carries the action that produced the
        # state it describes. Playground's Leap env builds it before, and is one step behind.
        obs = self._get_obs(data, info, state.obs["state"])

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["cap_rotation"] = d_theta
        state.metrics["cap_rotation_abs"] = jp.abs(d_theta)
        state.metrics["cap_angvel_per_step"] = self._cap_angvel(data)
        state.metrics["success"] = terms["success"]
        state.metrics["thumb_gap_per_step"] = thumb_gap
        state.metrics["index_gap_per_step"] = index_gap
        state.metrics["opposition_per_step"] = opposition

        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

    # ---------------------------------------------------------------------------------
    # observation, reward, termination
    # ---------------------------------------------------------------------------------
    def _get_obs(
        self, data: mjx.Data, info: Dict[str, Any], history: jax.Array
    ) -> Dict[str, jax.Array]:
        joint_pos = data.qpos[self._hand_qids]
        cap_pos = self._sensor(data, "cap_position")
        cap_axis = self._sensor(data, "cap_axis")

        info["rng"], joint_rng, cap_rng = jax.random.split(info["rng"], 3)
        level = self._config.noise_config.level
        scales = self._config.noise_config.scales
        joint_pos = joint_pos + level * scales.joint_pos * jax.random.uniform(
            joint_rng, joint_pos.shape, minval=-1.0, maxval=1.0
        )
        cap_pos = cap_pos + level * scales.cap_pos * jax.random.uniform(
            cap_rng, cap_pos.shape, minval=-1.0, maxval=1.0
        )

        frame = jp.concatenate([joint_pos, info["last_act"], data.ctrl, cap_pos, cap_axis])
        history = jp.roll(history, frame.size).at[: frame.size].set(frame)

        rotation = self._cap_angle(data) - info["cap_angle_init"]
        privileged = jp.concatenate([
            history,
            data.qvel[self._hand_dqids],
            rotation[None],
            self._cap_angvel(data)[None],
            self._tip_positions(data, scene.ALL_TIP_SENSORS).reshape(-1),
            data.actuator_force,
            info["dr_params"],
        ])
        return {"state": history, "privileged_state": privileged}

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: Dict[str, Any],
        d_theta: jax.Array,
        rotation: jax.Array,
        grasp: tuple[jax.Array, jax.Array, jax.Array],
    ) -> Dict[str, jax.Array]:
        thumb_gap, index_gap, opposition = grasp
        threshold = self._config.success_rotation
        # Fires on the control step that crosses the threshold going forward. A latch in `info`
        # would be exactly one-time, but it evolves during the episode and so would go stale
        # high under `full_reset=False` and then never fire again -- silently, which is worse
        # than a bonus that can fire twice if the cap happens to oscillate across 4*pi.
        crossed = (rotation >= threshold) & (rotation - d_theta < threshold) & (d_theta > 0)
        return {
            "rotate": jp.clip(d_theta / self.dt, -0.5, self._config.reward_config.rotate_clip),
            # M5: replaces `reach`, which measured how close three fingertip *sites* were to the
            # cap and so could not see the grasp the first policy actually used -- the side of
            # the index finger, with the thumb absent. This asks for the real thing: both
            # surfaces on the cap (the worse of the two sets the exponential) *and* opposed.
            #
            # M5, after the first Blackwell run: the opposition factor was
            # `jp.clip(opposition, 0.0, 1.0)`, and that gate made the whole term inert. `clip`
            # is flat below zero -- value *and* derivative are exactly 0.0 -- and the trained
            # policy sat at an opposition of -0.627, on the same side of the cap, 98.4% of the
            # time. So the term paid 0.000 at the median and, more to the point, pointed
            # nowhere: there was no gradient by which the policy could discover opposition from
            # where it started. It optimised `rotate` alone and hovered just inside the
            # grasp-loss bound. Measured in NOTES.md, M5.
            #
            # `0.5 * (1 + opposition)` is the same preference without the cliff: it maps
            # opposition onto (0, 1], is monotone, so more opposed is always worth more, and
            # has a constant non-zero derivative everywhere, including the same-side region the
            # policy actually starts in. Crowding the same side still scores poorly (0.19 at
            # -0.627 against 1.0 fully opposed) -- it is no longer *unlearnable*.
            "grasp": (
                jp.exp(-self._config.reward_config.grasp_sharpness
                       * jp.maximum(thumb_gap, index_gap))
                * (0.5 * (1.0 + opposition))
            ),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "torques": jp.sum(jp.square(data.actuator_force)),
            "joint_vel": jp.sum(jp.square(data.qvel[self._hand_dqids])),
            "success": crossed.astype(float),
        }

    def _get_termination(self, data: mjx.Data, ungrasped_steps: jax.Array) -> jax.Array:
        nan = ~jp.isfinite(data.qpos).all() | ~jp.isfinite(data.qvel).all()
        if not self._config.early_termination:
            return nan
        dropped = (self.tip_gaps(data) > self._config.drop_distance).all()
        # M5: `drop_distance` (10 cm) only fires once the cap is gone entirely, and the first
        # policy sat well inside it -- thumb 20 mm out, pushing the cap one-sidedly with the
        # index finger. This is the tighter test, on a grace period so that a legitimate
        # regrip survives it. Opposition is deliberately not part of it: it is shaped by the
        # reward, and a hard rule on it would end an episode in the middle of a regrip.
        let_go = ungrasped_steps >= self._config.grasp_grace_steps
        return nan | dropped | let_go

    # ---------------------------------------------------------------------------------
    # sensor readings
    # ---------------------------------------------------------------------------------
    def _sensor(self, data: mjx.Data, name: str) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, name)

    def _cap_angle(self, data: mjx.Data) -> jax.Array:
        """The hinge angle, as a scalar. Does not wrap, so it is the cumulative angle."""
        return self._sensor(data, "cap_angle")[0]

    def _cap_angvel(self, data: mjx.Data) -> jax.Array:
        return self._sensor(data, "cap_angvel")[0]

    def _tip_positions(self, data: mjx.Data, sensors: tuple[str, ...]) -> jax.Array:
        """Fingertip positions in the cap frame, (len(sensors), 3)."""
        return jp.stack([self._sensor(data, s) for s in sensors])

    def grasp_state(self, data: mjx.Data) -> tuple[jax.Array, jax.Array, jax.Array]:
        """`(thumb_gap, index_gap, opposition)` -- what a bottle-cap grasp is made of.

        The gaps are surface-to-surface, in metres, to the closest point on any collision geom
        of that finger; `opposition` is +1 when the two touch the cap from exactly opposite
        sides and -1 when they crowd the same side. A real lateral pinch is thumb pad against
        the side of the index finger, so what matters is that both are on the cap and that they
        are opposed -- not which part of the finger is doing it.

        Public because `scripts/eval.py` reports all three.
        """
        origin = data.site_xpos[self._cap_site]
        mat = data.site_xmat[self._cap_site].reshape(3, 3)

        def closest(group: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
            ids = group["ids"]
            points = scene.segment_points(
                data.geom_xpos[ids], data.geom_xmat[ids].reshape(-1, 3, 3), group["half_length"]
            )
            return scene.grasp_geometry(
                points, group["radius"], origin, mat, self._cap_radius, self._cap_half_height
            )

        thumb_gap, thumb_dir = closest(self._thumb_geoms)
        index_gap, index_dir = closest(self._index_geoms)
        return thumb_gap, index_gap, -jp.dot(thumb_dir, index_dir)

    def tip_gaps(self, data: mjx.Data) -> jax.Array:
        """Gap from each gaiting fingertip to the cap surface, (3,).

        Public because `scripts/eval.py` reads it to tell gaiting from a static grip.
        """
        return scene.surface_distance(
            self._tip_positions(data, scene.GAIT_TIP_SENSORS),
            self._cap_radius,
            self._cap_half_height,
        )

    # ---------------------------------------------------------------------------------
    # accessors
    # ---------------------------------------------------------------------------------
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
