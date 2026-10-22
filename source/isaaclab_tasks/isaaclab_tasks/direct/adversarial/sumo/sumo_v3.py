"""Visual skill-conditioned Go2 curriculum. Ground truth is restricted to scoring/drill resets."""

import torch
import torch.nn.functional as F

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from .sumo_v2 import (
    DogArenaMixin,
    Go2ArenaLocomotionCfg,
    Go2ArenaPushCfg,
    Go2ArenaFightCfg,
    QuadrupedGoToPointSumo,
    QuadrupedBlocksPushEnv,
    SumoStage2Quadruped1v1Env,
)
from .sensor_history import DepthHistory
from .tactics_events import TacticsEvents, EVENTS

OBSERVATIONS = 45 + 4 * 3 * 36 * 64 + 4 * 96 + 2
FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
TARGETS = (
    "base",
    "Head_upper",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    *FEET,
)


class FilteredPeakContact(ContactSensor):
    """Keep brief filtered contacts across all four physics substeps of a control."""

    def __init__(self, cfg):
        self.peak_forces = None
        super().__init__(cfg)

    def update(self, dt, force_recompute=False):
        super().update(dt, force_recompute=True)
        matrix = self.data.force_matrix_w
        if matrix is not None:
            magnitude = torch.linalg.vector_norm(matrix, dim=-1)
            if self.peak_forces is None:
                self.peak_forces = magnitude.clone()
            else:
                self.peak_forces.copy_(torch.maximum(self.peak_forces, magnitude))

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self.peak_forces is not None:
            self.peak_forces[slice(None) if env_ids is None else env_ids] = 0


def tactics_config(cfg):
    cfg.observation_spaces = {key: OBSERVATIONS for key in cfg.action_spaces}
    cfg.observation_space = OBSERVATIONS
    cfg.drill_mode = False
    cfg.drill_fraction = 0.0
    cfg.learner_role = 0
    cfg.fixed_drill = -1
    if cfg.sumo_stage == 2:
        cfg.flip_reward_scale = 0.0  # replaced by contact-qualified, sustained pin events
        cfg.flat_orientation_reward_scale = 0.0  # allow controlled lean/three-leg support
        cfg.back_loss_hold_steps = 80  # 1.6 s recovery opportunity; pin can end it earlier
        cfg.spawn_separation_min = 1.3
        cfg.spawn_separation_max = 2.2


@configclass
class Go2TacticsLocomotionCfg(Go2ArenaLocomotionCfg):
    def __post_init__(self):
        super().__post_init__()
        tactics_config(self)


@configclass
class Go2TacticsPushCfg(Go2ArenaPushCfg):
    def __post_init__(self):
        super().__post_init__()
        tactics_config(self)


@configclass
class Go2TacticsFightCfg(Go2ArenaFightCfg):
    def __post_init__(self):
        super().__post_init__()
        tactics_config(self)


class TacticsMixin(DogArenaMixin):
    def _setup_onboard_sensors(self):
        super()._setup_onboard_sensors()
        self._tactical_images = {key: DepthHistory(self.num_envs, 3 * 36 * 64, 4, self.device) for key in self.robots}
        self._tactics_tick = -1
        self.skills = torch.zeros(self.num_envs, len(self.robots), dtype=torch.long, device=self.device)
        self.drills = torch.zeros_like(self.skills)
        self.drill_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.drill_passive = torch.zeros_like(self.drill_active)
        self.tactical_transition = {}
        self._tactic_events = TacticsEvents(self.num_envs, self.device)
        self._event_tick = -1
        self._pair_contacts = {}
        if self.cfg.sumo_stage != 2:
            return
        # One source rigid body per sensor is mandatory for IsaacLab filtered contacts.
        for i in range(2):
            targets = [f"/World/envs/env_.*/Robot_{1-i}/{name}" for name in TARGETS]
            targets += ["/World/ground", "/World/envs/env_.*/ArenaSurface"]
            for link in ("base", *FEET):
                sensor = FilteredPeakContact(
                    ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Robot_{i}/{link}",
                        update_period=0.0,
                        history_length=4,
                        filter_prim_paths_expr=targets,
                    )
                )
                self._pair_contacts[i, link] = sensor
                self.scene.sensors[f"tactics_{i}_{link}"] = sensor

    def _pre_physics_step(self, actions):
        for sensor in self._pair_contacts.values():
            if sensor.peak_forces is not None:
                sensor.peak_forces.zero_()
        super()._pre_physics_step(actions)

    def _get_observations(self):
        base = super()._get_observations()
        tick = self.common_step_counter
        if self._tactics_tick == tick:
            return self._tactics_cache
        result = {}
        for team, agents in base.items():
            result[team] = {}
            for key, old in agents.items():
                history = self._tactical_images[key]
                if tick % 5 == 0:
                    raw = self.rig.rgb[key].data.output["rgb"][..., :3].float().permute(0, 3, 1, 2) / 255.0
                    current = F.adaptive_avg_pool2d(raw, (36, 64)).flatten(1)
                    current = (current * self._camera_gain + 0.008 * torch.randn_like(current)).clamp(0, 1)
                    rendered = tick - (self._sim_step_counter % self.cfg.sim.render_interval) // self.cfg.decimation
                    eligible = (rendered >= self._sensor_ready) & (torch.rand(self.num_envs, device=self.device) > 0.02)
                    history.update(current, rendered, eligible)
                tail = old[:, 45 + 2 * 432 :].clone()
                age = ((tick - history.last_tick).float() * self.step_dt).clamp(0, 1)
                age[~history.initialized] = 1.0
                tail[:, -2] = age
                result[team][key] = torch.cat((old[:, :45], history.values.flatten(1), tail), -1)
        self._tactics_tick, self._tactics_cache = tick, result
        return result

    def _reset_idx(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        super()._reset_idx(ids)
        for history in self._tactical_images.values():
            history.reset(ids)
        self._tactics_tick = -1
        self._tactic_events.reset(ids)
        self.skills[ids] = 0
        self.drills[ids] = 0
        self.drill_active[ids] = False
        self.drill_passive[ids] = False
        # Finite-horizon PPO includes episode deadlines as terminal task outcomes.
        self.episode_length_buf[ids] = 0
        if self.cfg.drill_mode:
            role = self.cfg.learner_role
            chosen = torch.randint(1, 9, (len(ids),), device=self.device)
            if self.cfg.fixed_drill >= 0:
                chosen[:] = self.cfg.fixed_drill
            self.drill_active[ids] = torch.rand(len(ids), device=self.device) < self.cfg.drill_fraction
            chosen[~self.drill_active[ids]] = 0
            self.drill_passive[ids] = (torch.rand(len(ids), device=self.device) < 0.5) & (
                (chosen == 3) | (chosen == 4) | (chosen == 7) | (chosen == 8)
            )
            self.drills[ids, role] = chosen
            # Start disengagement/counter drills already engaged, without overlapping torsos.
            close = ids[(chosen == 5) | (chosen == 6) | (chosen == 7)]
            if len(close):
                r0, r1 = self.robots.values()
                midpoint = (r0.data.root_pos_w[close, :2] + r1.data.root_pos_w[close, :2]) / 2
                direction = F.normalize(r1.data.root_pos_w[close, :2] - r0.data.root_pos_w[close, :2], dim=-1)
                for i, robot in enumerate(self.robots.values()):
                    pose = robot.data.root_state_w[close, :7].clone()
                    pose[:, :2] = midpoint + (-1 if i == 0 else 1) * 0.38 * direction
                    robot.write_root_pose_to_sim(pose, close)
            # Recover starts leaning; pin starts with the defender already on its side.
            for skill, target, angle in ((8, role, 1.15), (7, 1 - role, 1.60)):
                selected = ids[chosen == skill]
                if len(selected):
                    robot = self.robots[f"robot_{target}"]
                    pose = robot.data.root_state_w[selected, :7].clone()
                    zeros = torch.zeros(len(selected), device=self.device)
                    roll = quat_from_euler_xyz(zeros + angle, zeros, zeros)
                    pose[:, 3:7] = quat_mul(pose[:, 3:7], roll)
                    pose[:, 2] += 0.10
                    robot.write_root_pose_to_sim(pose, selected)

    def _contact_measurements(self):
        values = {}
        for key, sensor in self._pair_contacts.items():
            matrix = sensor.data.force_matrix_w
            if matrix is None or matrix.shape[1:3] != (1, len(TARGETS) + 2):
                raise RuntimeError(f"Invalid filtered contact layout for {key}; refusing unqualified tactic rewards")
            values[key] = (
                sensor.peak_forces[:, 0]
                if sensor.peak_forces is not None
                else torch.linalg.vector_norm(matrix[:, 0], dim=-1)
            )
        return values

    def _update_tactics(self):
        if self._event_tick == self.common_step_counter:
            return
        robots = list(self.robots.values())
        positions = torch.stack([r.data.root_pos_w for r in robots], 1)
        up = torch.stack([-r.data.projected_gravity_b[:, 2] for r in robots], 1)
        direction = positions.flip(1)[..., :2] - positions[..., :2]
        gap = torch.linalg.vector_norm(direction[:, 0], dim=-1)
        unit = F.normalize(direction, dim=-1)
        headings = torch.stack(
            [
                quat_apply(
                    r.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
                )[:, :2]
                for r in robots
            ],
            1,
        )
        face = (headings * unit).sum(-1)
        opponent_angle = (headings.flip(1) * -unit).sum(-1)
        flank = (opponent_angle.abs() < 0.4) & (face > 0.5) & (gap[:, None] > 0.55) & (gap[:, None] < 1.4)
        flank_side = -(headings.flip(1)[..., 0] * -unit[..., 1] - headings.flip(1)[..., 1] * -unit[..., 0])
        contacts = self._contact_measurements()
        leg_hits, moving, stance, body_contacts, grounded, impact = [], [], [], [], [], []
        for i, robot in enumerate(robots):
            floors = torch.stack([contacts[i, link][:, -2:].sum(-1) for link in FEET], -1)
            stance.append((floors > 3.0).sum(-1))
            leg_hits.append(
                torch.stack([contacts[i, link][:, 6 : len(TARGETS)].amax(-1) > 3.0 for link in FEET[:2]], -1)
            )
            foot_ids, _ = robot.find_bodies(list(FEET[:2]), preserve_order=True)
            velocity = robot.data.body_lin_vel_w[:, foot_ids] - robot.data.root_lin_vel_w[:, None]
            moving.append((torch.linalg.vector_norm(velocity, dim=-1) > 0.15) & (floors[:, :2] < 3.0))
            all_contact = torch.stack([contacts[i, link][:, : len(TARGETS)].sum(-1) for link in ("base", *FEET)], -1)
            body_contacts.append(all_contact.sum(-1) > 5.0)
            impact.append(all_contact.amax(-1))
            grounded.append(contacts[i, "base"][:, -2:].sum(-1) > 5.0)
        radius = torch.linalg.vector_norm(positions[..., :2] - self.scene.env_origins[:, None, :2], dim=-1)
        velocities = torch.stack([r.data.root_lin_vel_w[:, :2] for r in robots], 1)
        incoming = ((velocities.flip(1) * unit).sum(-1) < -0.15) & ((velocities * unit).sum(-1) < 0.10)
        body_contacts = torch.stack(body_contacts, 1)
        impact = torch.stack(impact, 1)
        events, bonus = self._tactic_events.update(
            up=up,
            gap=gap,
            height=positions[..., 2],
            leg_hit=torch.stack(leg_hits, 1),
            foot_moving=torch.stack(moving, 1),
            stance=torch.stack(stance, 1),
            body_contact=body_contacts,
            grounded=torch.stack(grounded, 1),
            flank=flank,
            flank_side=flank_side,
            radius=radius,
            incoming=body_contacts & incoming & (up > 0.65),
            skills=self.skills,
            drills=self.drill_active,
        )
        # Exploration shaping is confined to drills, capped per control and faded in full games.
        dense = torch.zeros_like(bonus)
        if self.cfg.drill_mode:
            for i, robot in enumerate(robots):
                skill = self.skills[:, i]
                lateral = robot.data.root_lin_vel_b[:, 1]
                signed = torch.where(skill == 1, lateral, -lateral)
                dense[:, i] += 0.01 * signed.clamp(-1, 1) * ((skill == 1) | (skill == 2)) * (up[:, i] > 0.65)
                ids, _ = robot.find_bodies(list(FEET[:2]), preserve_order=True)
                foot_z = robot.data.body_pos_w[:, ids, 2] - self.scene.env_origins[:, 2:3]
                # A small balanced-lift term helps discover one-leg control; no impact bonus.
                lift = torch.where(skill == 3, foot_z[:, 0], foot_z[:, 1])
                dense[:, i] += (
                    0.003 * (lift > 0.06) * ((skill == 3) | (skill == 4)) * (torch.stack(stance, 1)[:, i] >= 2)
                )
        dense *= self.drill_active[:, None]
        dense = torch.minimum(dense.clamp_min(0), (3.0 - self._tactic_events.budget).clamp_min(0))
        self._tactic_events.budget += dense
        self._tactic_bonus = bonus + dense
        self._tactic_events_now = events
        self._impact_cost = 0.002 * ((impact - 150.0).clamp_min(0) / 150.0).square().clamp_max(10)
        self._event_tick = self.common_step_counter
        self.tactical_transition = {
            "events": events.clone(),
            "upright": up.clone(),
            "impact": impact.clone(),
            "skills": self.skills.clone(),
            "outcome": torch.zeros_like(up),
        }

    def _get_dones(self):
        if self.cfg.sumo_stage != 2:
            return super()._get_dones()
        self._update_tactics()
        dones, timeouts = super()._get_dones()
        pinned = self._tactic_events_now[..., EVENTS.index("pin")].flip(1)
        self._back_lost |= pinned
        for team in dones:
            dones[team] |= pinned.any(-1)
        return dones, timeouts

    def _get_rewards(self):
        rewards = super()._get_rewards()
        if self.cfg.sumo_stage == 2:
            for i, robot in enumerate(self.robots.values()):
                power = (robot.data.applied_torque * robot.data.joint_vel).abs().sum(-1)
                lean = (-robot.data.projected_gravity_b[:, 2] < 0.4).float()
                rewards[f"team_{i}"] += (
                    self._tactic_bonus[:, i] - self._impact_cost[:, i] - 0.00001 * power - 0.02 * lean
                )
            out = self._robots_out_of_ring()
            lost = torch.stack(
                [
                    out[f"robot_{i}"] | self._back_lost[:, i] | (r.data.root_com_pos_w[:, 2] < 0.1)
                    for i, r in enumerate(self.robots.values())
                ],
                1,
            )
            self.tactical_transition["outcome"] = (lost.flip(1) & ~lost).float() - (lost & ~lost.flip(1)).float()
        else:
            self.tactical_transition = {
                "upright": torch.stack([-r.data.projected_gravity_b[:, 2] for r in self.robots.values()], 1)
            }
        return rewards


class Go2TacticsLocomotionEnv(TacticsMixin, QuadrupedGoToPointSumo):
    pass


class Go2TacticsPushEnv(TacticsMixin, QuadrupedBlocksPushEnv):
    pass


class Go2TacticsFightEnv(TacticsMixin, SumoStage2Quadruped1v1Env):
    pass
