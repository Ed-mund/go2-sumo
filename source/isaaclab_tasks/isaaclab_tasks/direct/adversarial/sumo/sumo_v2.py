"""Versioned dog-sumo tasks with reusable sensors, controls, and arena presentation.

AIR/PRO sensor types: RGB, range points, IMU and encoders. No opponent truth,
perfect localization/body velocity, depth camera or foot-force actor inputs.
The old tasks remain available for reproducible checkpoint playback.
"""

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from .go2_sensor_curriculum import (
    Go2SensorBlocksPushCfg,
    Go2SensorGoToPointCfg,
    Go2SensorSumo1v1Cfg,
)
from .quadruped_curriculum import QuadrupedBlocksPushEnv, QuadrupedGoToPointSumo
from .quadruped_stage2 import SumoStage2Quadruped1v1Env
from .robot_profiles import GO2, apply_dog_profile
from .onboard_rig import OnboardRig
from .arena_surface import install_surfaces
from .sensor_history import DepthHistory, gravity_estimate, joint_targets, lidar_features, rgb_features, stall_update

# Same actor contract for every stage: proprio, RGB history, LiDAR history, ages.
OBS_DIM = 45 + 2 * 432 + 4 * 96 + 2


def configure(cfg, stage):
    """Common contract; morphology comes from the base configuration/profile."""
    apply_dog_profile(cfg)
    cfg.sumo_stage = stage
    cfg.arena_variant_offset = 0
    cfg.observation_spaces = {key: OBS_DIM for key in cfg.action_spaces}
    cfg.observation_space = OBS_DIM
    cfg.sim.render_interval = 8  # 200 Hz physics, 50 Hz controls, 25 Hz RGB
    cfg.sim.render.antialiasing_mode = "OFF"
    cfg.sim.enable_scene_query_support = True
    cfg.sim.use_fabric = False  # PhysX range-sensor parent transforms must follow the moving USD bodies
    cfg.viewer.eye = (5.8, -6.0, 4.3)
    cfg.viewer.lookat = (0.0, 0.0, 0.35)
    cfg.viewer.resolution = (1280, 720)
    cfg.scene.env_spacing = 16.0  # separation beyond range + arena/goal travel extents
    cfg.scene.replicate_physics = False  # each arena has its own collision surface
    cfg.episode_length_s = 24.0
    cfg.action_delay_probability = 0.25
    cfg.heading_jitter = 0.30
    cfg.stage_reward_cost = 0.005
    if stage == 0:
        cfg.reverse_goal_probability = 0.15
        cfg.goal_spawn_radius_min = 1.5
        cfg.goal_spawn_radius_max = 4.0
        cfg.dist_to_goal_reward_scale = 0.0  # progress and success, no standing-near-goal income
    else:
        cfg.ring_radius_min = 3.0
        cfg.ring_radius_max = 3.0  # fixed, visible sport boundary; no hidden shrinking ring
    if stage == 2:
        cfg.reward_scale = 20.0
        cfg.flip_reward_scale = 8.0  # winning is worth more than a transient mount
        cfg.flip_hold_steps = 10
        cfg.back_loss_hold_steps = 40  # 0.8 s to recover
        cfg.spawn_separation_min = 1.7
        cfg.spawn_separation_max = 2.7
        cfg.backward_facing_penalty_scale = -0.01  # allow tactical reversals
    # Consistent material randomization with low bounce for contact sport.
    for i in range(len(cfg.action_spaces)):
        material = getattr(cfg.events, f"physics_material_{i}")
        material.params["static_friction_range"] = (0.8, 1.0)
        material.params["dynamic_friction_range"] = (0.6, 0.8)
        material.params["restitution_range"] = (0.0, 0.02)


@configclass
class Go2ArenaLocomotionCfg(Go2SensorGoToPointCfg):
    def __post_init__(self):
        configure(self, 0)


@configclass
class Go2ArenaPushCfg(Go2SensorBlocksPushCfg):
    def __post_init__(self):
        configure(self, 1)


@configclass
class Go2ArenaFightCfg(Go2SensorSumo1v1Cfg):
    def __post_init__(self):
        configure(self, 2)


class DogArenaMixin:
    """Shared observation/action contract for robot-specific curriculum configs."""

    def _setup_onboard_sensors(self):
        self.rig = OnboardRig(self, GO2)
        self._histories = {key: DepthHistory(self.num_envs, device=self.device) for key in self.robots}
        self._rgb_histories = {
            key: DepthHistory(self.num_envs, features=432, frames=2, device=self.device) for key in self.robots
        }
        self._gravity = {
            key: torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1) for key in self.robots
        }
        self._imu_bias = {key: torch.zeros(self.num_envs, 6, device=self.device) for key in self.robots}
        self._camera_gain = torch.ones(self.num_envs, 1, device=self.device)
        self._sensor_ready = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._delayed = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
        self._stall_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._v2_cache = None
        self._v2_tick = -1

    def _setup_scene(self):
        super()._setup_scene()
        install_surfaces(self)
        self._surface_max = torch.tensor(self._surface_max, device=self.device)

    def _pre_physics_step(self, actions):
        # Preserve sampled actions for PPO's likelihoods; only the motor command is bounded.
        super()._pre_physics_step(actions)
        for key, robot in self.robots.items():
            command = torch.where(self._delayed, self.previous_actions[key], self.actions[key])
            scale = getattr(self.cfg, "action_scales", {}).get(key, self.cfg.action_scale)
            self.processed_actions[key] = joint_targets(
                command, robot.data.default_joint_pos, robot.data.soft_joint_pos_limits, scale
            )

    def _get_observations(self):
        tick = self.common_step_counter
        if self._v2_tick == tick and self._v2_cache is not None:
            return self._v2_cache
        obs = {}
        for team, agents in self.cfg.teams.items():
            obs[team] = {}
            for key in agents:
                robot = self.robots[key]
                history = self._histories[key]
                rgb_history = self._rgb_histories[key]
                if tick % 5 == 0 and tick > 0:  # 10 Hz processed cloud reference rate
                    points = self.rig.points(key)
                    current = lidar_features(points + torch.randn_like(points) * 0.01)
                    current = current.masked_fill(torch.rand_like(current) < 0.01, 0)
                    history.update(current, tick, tick >= self._sensor_ready)
                if self._sim_step_counter % self.cfg.sim.render_interval == 0:
                    current = rgb_features(self.rig.rgb[key].data.output["rgb"])
                    current = (current * self._camera_gain + torch.randn_like(current) * 0.01).clamp(0, 1)
                    # Lost frames retain their timestamp: policies can detect stale input.
                    eligible = (tick >= self._sensor_ready) & (torch.rand(self.num_envs, device=self.device) > 0.02)
                    rgb_history.update(current, tick, eligible)
                ages = []
                for buffer in (rgb_history, history):
                    age = ((tick - buffer.last_tick).float() * self.step_dt).clamp(0, 1)[:, None]
                    age[~buffer.initialized] = 1.0
                    ages.append(age)
                imu = self.rig.imu[key].data
                accel = imu.lin_acc_b + self._imu_bias[key][:, :3] + torch.randn_like(imu.lin_acc_b) * 0.05
                gyro = imu.ang_vel_b + self._imu_bias[key][:, 3:] + torch.randn_like(imu.ang_vel_b) * 0.005
                warming = tick < self._sensor_ready
                accel[warming] = torch.tensor([0.0, 0.0, 9.81], device=self.device)
                gyro[warming] = 0
                self._gravity[key] = gravity_estimate(self._gravity[key], gyro, accel, self.step_dt)
                proprio = torch.cat(
                    (
                        accel.clamp(-50, 50) / 9.81,
                        gyro.clamp(-30, 30),
                        self._gravity[key],
                        robot.data.joint_pos
                        - robot.data.default_joint_pos
                        + torch.randn_like(robot.data.joint_pos) * 0.002,
                        (robot.data.joint_vel + torch.randn_like(robot.data.joint_vel) * 0.02) * 0.05,
                        self.actions[key].clamp(-3, 3),
                    ),
                    dim=-1,
                )
                obs[team][key] = torch.cat(
                    (
                        proprio,
                        rgb_history.values.flatten(1),
                        history.values.flatten(1),
                        *ages,
                    ),
                    dim=-1,
                )
        self._v2_cache, self._v2_tick = obs, tick
        return obs

    def _reset_idx(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        push_metrics = {}
        if self.cfg.sumo_stage == 1 and len(ids):
            out = self._robots_out_of_ring()
            successes = []
            for i in range(2):
                success = out[f"block_{i}"][ids] & ~out[f"robot_{i}"][ids]
                successes.append(success)
                push_metrics[f"Metrics/team_{i}_push_success"] = success.float().mean().item()
            push_metrics["Metrics/push_success"] = torch.stack(successes).any(0).float().mean().item()
        super()._reset_idx(ids)
        self.extras.setdefault("log", {}).update(push_metrics)
        for history in (*self._histories.values(), *self._rgb_histories.values()):
            history.reset(ids)
        # The installed DirectMARLEnv does not rerender after partial resets.
        # Discard the previous episode's image until the next scheduled render.
        self._sensor_ready[ids] = self.common_step_counter + 2
        self._camera_gain[ids] = torch.empty((len(ids), 1), device=self.device).uniform_(0.8, 1.2)
        for key in self.robots:
            self._gravity[key][ids] = torch.tensor([0.0, 0.0, -1.0], device=self.device)
            self._imu_bias[key][ids, :3] = torch.randn((len(ids), 3), device=self.device) * 0.08
            self._imu_bias[key][ids, 3:] = torch.randn((len(ids), 3), device=self.device) * 0.008
        self._delayed[ids] = torch.rand((len(ids), 1), device=self.device) < self.cfg.action_delay_probability
        self._stall_counter[ids] = 0
        self._v2_tick = -1
        # Starts mostly face the target; small independent offsets teach steering.
        for robot in self.robots.values():
            pose = robot.data.root_state_w[ids, :7].clone()
            pose[:, 2] += self._surface_max[ids] + 0.005
            q = pose[:, 3:7]
            yaw = torch.atan2(
                2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2].square() + q[:, 3].square())
            )
            yaw += torch.empty_like(yaw).uniform_(-self.cfg.heading_jitter, self.cfg.heading_jitter)
            pose[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
            robot.write_root_pose_to_sim(pose, ids)
        for block in getattr(self, "blocks", {}).values():
            pose = block.data.root_state_w[ids, :7].clone()
            pose[:, 2] += self._surface_max[ids] + 0.005
            block.write_root_pose_to_sim(pose, ids)

    def _draw_team_dots(self):
        # No floating spheres in the robots' sensor images.
        return

    def _draw_markers(self):
        # A visible training beacon replaces the old oracle goal vector. It is
        # only present in locomotion, never an opponent detector in the game.
        scales = torch.full((self.num_envs, 3), 1.8, device=self.device)
        self.my_visualizer.visualize(
            self._desired_pos,
            scales=scales,
            marker_indices=torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
        )

    def sensor_preview(self):
        """Read-only first-environment panel; never feeds the actor or advances history."""
        from PIL import Image, ImageDraw
        import numpy as np

        panel = Image.new("RGB", (320 * len(self.robots), 240), (8, 15, 24))
        draw = ImageDraw.Draw(panel)
        for index, key in enumerate(self.robots):
            image = self.rig.rgb[key].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            panel.paste(Image.fromarray(image).resize((256, 144)), (index * 320, 25))
            ranges = self._histories[key].values[0, 0].reshape(3, 32).cpu().numpy()
            heat = (np.stack((ranges, ranges * 0.55, 1 - ranges), axis=-1) * 255).astype(np.uint8)
            panel.paste(Image.fromarray(heat).resize((256, 36)), (index * 320, 185))
            draw.text((index * 320 + 4, 5), f"{key}: onboard RGB", fill="white")
            draw.text((index * 320 + 4, 224), "LiDAR bins: red=near blue=no return", fill="white")
        return panel

    def _draw_ring_markers(self):
        # Paint on the physical mesh follows its height, unlike a floating flat disc.
        self.ring_markers.set_visibility(False)

    def _get_rewards(self):
        rewards = super()._get_rewards()
        extra = {}
        for team, agents in self.cfg.teams.items():
            robot = self.robots[agents[0]]
            # Cost also covers full inversion (XY gravity alone misses it).
            inversion = robot.data.projected_gravity_b[:, 2].clamp_min(0).square()
            cost = 0.15 * inversion + self.cfg.stage_reward_cost
            if self.cfg.sumo_stage > 0:
                cost += 1e-5 * robot.data.applied_torque.square().sum(-1)
            rewards[team] = rewards[team] - cost
            extra[f"{team}/inverted_fraction"] = (inversion > 0).float().mean()
        if self.cfg.sumo_stage == 2:
            r0, r1 = self.robots.values()
            gap = torch.linalg.vector_norm(r0.data.root_pos_w[:, :2] - r1.data.root_pos_w[:, :2], dim=-1)
            # Two dogs translating together while pushing are not a frozen engagement.
            speed = torch.maximum(
                torch.linalg.vector_norm(r0.data.root_lin_vel_w[:, :2], dim=-1),
                torch.linalg.vector_norm(r1.data.root_lin_vel_w[:, :2], dim=-1),
            )
            self._stall_counter, stalled = stall_update(self._stall_counter, gap < 1.0, speed, self.step_dt)
            for team in rewards:
                rewards[team] -= 0.02 * stalled.float()
            extra["sumo/stalled_fraction"] = stalled.float().mean()
        self.extras.setdefault("log", {}).update(extra)
        return rewards


class DogArenaLocomotionEnv(DogArenaMixin, QuadrupedGoToPointSumo):
    pass


class DogArenaPushEnv(DogArenaMixin, QuadrupedBlocksPushEnv):
    pass


class DogArenaFightEnv(DogArenaMixin, SumoStage2Quadruped1v1Env):
    pass
