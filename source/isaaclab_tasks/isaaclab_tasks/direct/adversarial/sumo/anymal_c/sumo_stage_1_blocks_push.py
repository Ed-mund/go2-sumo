# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, subtract_frame_transforms

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from ..heading import backward_facing_penalty


def get_quaternion_tuple_from_xyz(x, y, z):
    quat_tensor = quat_from_euler_xyz(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])).flatten()
    return (quat_tensor[0].item(), quat_tensor[1].item(), quat_tensor[2].item(), quat_tensor[3].item())


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material_0 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass_0 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    physics_material_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass_1 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


@configclass
class AnymalCBlocksPushEnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 20.0

    action_spaces = {f"robot_{i}": 12 for i in range(2)}
    observation_spaces = {f"robot_{i}": 56 for i in range(2)}

    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(2)}

    possible_agents = [f"robot_{i}" for i in range(2)]
    teams = {"team_0": ["robot_0"], "team_1": ["robot_1"]}
    root_body_pattern = "base"
    feet_body_pattern = ".*FOOT"
    undesired_body_pattern = ".*THIGH"
    # Optional per-agent overrides used by heterogeneous match-ups.  Empty
    # mappings preserve the original homogeneous behaviour.
    root_body_patterns = {}
    feet_body_patterns = {}
    undesired_body_patterns = {}
    collision_profile = "anymal_c"

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            enable_ccd=True,
            min_position_iteration_count=8,
            min_velocity_iteration_count=4,
            gpu_max_rigid_contact_count=2**24,
            gpu_found_lost_pairs_capacity=2**22,
        ),
    )

    events: EventCfg = EventCfg()

    _push_rigid_props = ANYMAL_C_CFG.spawn.rigid_props.replace(max_depenetration_velocity=10.0)
    _push_articulation_props = ANYMAL_C_CFG.spawn.articulation_props.replace(
        solver_position_iteration_count=24, solver_velocity_iteration_count=4
    )

    robot_0: ArticulationCfg = ANYMAL_C_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_0",
        spawn=ANYMAL_C_CFG.spawn.replace(
            rigid_props=_push_rigid_props, articulation_props=_push_articulation_props
        ),
    )
    robot_0.init_state.rot = get_quaternion_tuple_from_xyz(0, 0, torch.pi / 2)
    robot_0.init_state.pos = (0.0, 0.0, 0.6)

    robot_1: ArticulationCfg = ANYMAL_C_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_1",
        spawn=ANYMAL_C_CFG.spawn.replace(
            rigid_props=_push_rigid_props, articulation_props=_push_articulation_props
        ),
    )
    robot_1.init_state.rot = get_quaternion_tuple_from_xyz(0, 0, torch.pi / 2)
    robot_1.init_state.pos = (0.0, 0.0, 0.6)

    contact_sensor_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_0/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )
    contact_sensor_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_1/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    env_spacing = 10.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

    action_scale = 0.5
    action_scales = {}
    ring_radius_min = 3.0
    ring_radius_max = 4.5
    reward_scale = 10
    block_progress_reward_scale = 4.0
    approach_progress_reward_scale = 0.75
    action_rate_reward_scale = -0.002
    flat_orientation_reward_scale = -0.02
    # time penalty
    time_penalty = -0.01
    box_velocity_scale = 0.01
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    backward_facing_penalty_scale = -0.1
    forward_yaw_offsets = {}

    block_0 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/block_0",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.5, 0.2), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    block_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/block_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, -0.5, 0.2), rot=(1.0, 0.0, 0.0, 0.0)),
    )


class AnymalCBlocksPushEnv(DirectMARLEnv):
    cfg: AnymalCBlocksPushEnvCfg

    def __init__(
        self,
        cfg: AnymalCBlocksPushEnvCfg,
        render_mode: str | None = None,
        headless: bool | None = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.headless = headless
        self.debug = debug

        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.previous_actions = {agent: action.clone() for agent, action in self.actions.items()}
        self._prev_block_radius = {
            block_id: torch.zeros(self.num_envs, device=self.device) for block_id in ("block_0", "block_1")
        }
        self._prev_robot_block_distance = {
            robot_id: torch.zeros(self.num_envs, device=self.device) for robot_id in ("robot_0", "robot_1")
        }
        self.env_spacing = self.cfg.env_spacing

        self.ring_radius = torch.full(
            (self.num_envs,),
            (self.cfg.ring_radius_min + self.cfg.ring_radius_max) * 0.5,
            dtype=torch.float32,
            device=self.device,
        )

        self._ring_segments = 64
        markers = {
            f"ring_{i}": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            )
            for i in range(self._ring_segments)
        }

        ring_marker_cfg = VisualizationMarkersCfg(prim_path="/World/RingMarkers", markers=markers)
        self.ring_markers = VisualizationMarkers(ring_marker_cfg)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # Team 0
                "team_0_block0_center_reward",
                "team_0_dist_r0_b0_reward",
                "team_0_action_rate_reward",
                "team_0_flat_orientation_reward",
                "team_0_backward_facing_penalty",
                # "team_0_time_penalty",
                "team_0_push_out_reward",
                # Team 1
                "team_1_block1_center_reward",
                "team_1_dist_r1_b1_reward",
                "team_1_action_rate_reward",
                "team_1_flat_orientation_reward",
                "team_1_backward_facing_penalty",
                # "team_1_time_penalty",
                "team_1_push_out_reward",
            ]
        }

        self.base_ids = {}
        self.feet_ids = {}
        self.undesired_body_contact_ids = {}

        for robot_id, contact_sensor in self.contact_sensors.items():
            root_pattern = self.cfg.root_body_patterns.get(robot_id, self.cfg.root_body_pattern)
            feet_pattern = self.cfg.feet_body_patterns.get(robot_id, self.cfg.feet_body_pattern)
            undesired_pattern = self.cfg.undesired_body_patterns.get(robot_id, self.cfg.undesired_body_pattern)
            _base_id, _ = contact_sensor.find_bodies(root_pattern)
            _feet_ids, _ = contact_sensor.find_bodies(feet_pattern)
            _undesired_contact_body_ids, _ = contact_sensor.find_bodies([undesired_pattern])
            self.base_ids[robot_id] = _base_id
            self.feet_ids[robot_id] = _feet_ids
            self.undesired_body_contact_ids[robot_id] = _undesired_contact_body_ids

    @torch.no_grad()
    def _draw_ring_markers(self):
        """
        Draws a ring per environment by placing N tiny spheres on a circle
        centered at each env origin with radius given by self.ring_radius.
        Uses marker indices 0..N-1 that correspond to your ring_i markers.
        """
        device = self.device
        E = self.num_envs
        N = self._ring_segments
        z0 = float(getattr(self.cfg, "ring_z", 0.0))

        # Angles for the circle (excluded endpoint to avoid duplicate point)
        theta = torch.linspace(0, 2 * torch.pi, steps=N + 1, device=device)[:-1]  # (N,)
        cs = torch.cos(theta)  # (N,)
        sn = torch.sin(theta)  # (N,)

        # Env centers and radii
        origins_xy = self.scene.env_origins[:, :2].to(device)  # (E, 2)
        radii = self.ring_radius.view(E, 1)  # (E, 1)

        # Build batched positions: stack per marker index (ring_k) across all envs.
        # For marker k, we place E positions at angle theta[k].
        # Result: positions shape = (N*E, 3), marker_indices length = N*E
        pos_chunks = []
        idx_chunks = []

        # Precompute z column for all envs
        z_col = torch.full((E, 1), z0, device=device)

        for k in range(N):
            dir_k = torch.tensor([cs[k].item(), sn[k].item()], device=device)  # (2,)
            xy_k = origins_xy + radii * dir_k  # (E, 2)
            pos_k = torch.cat([xy_k, z_col], dim=1)  # (E, 3)
            pos_chunks.append(pos_k)
            idx_chunks.append(k * torch.ones(E, dtype=torch.long, device=device))

        marker_positions = torch.cat(pos_chunks, dim=0)  # (N*E, 3)
        marker_indices = torch.cat(idx_chunks, dim=0)  # (N*E,)

        marker_scales = torch.ones((marker_positions.shape[0], 3), device=device)

        # Orientations: identity quaternions since these are spheres.
        marker_orientations = torch.zeros((marker_positions.shape[0], 4), device=device)
        marker_orientations[:, 0] = 1.0  # w=1, x=y=z=0

        # Visualize. Your API already accepts marker_indices like your velocity viz.
        self.ring_markers.visualize(
            marker_positions,
            marker_orientations,
            scales=marker_scales,
            marker_indices=marker_indices,
        )

    def _setup_scene(self):

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(500.0, 500.0),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # Setup rest of the scene
        self.blocks = {}

        for i in range(2):
            self.blocks[f"block_{i}"] = RigidObject(self.cfg.__dict__[f"block_{i}"])

        self.robots = {}
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.contact_sensors = {}
        for i in range(self.num_robots):
            self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
            self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]
            self.contact_sensors[f"robot_{i}"] = ContactSensor(self.cfg.__dict__["contact_sensor_" + str(i)])
            self.scene.sensors[f"robot_{i}"] = self.contact_sensors[f"robot_{i}"]

        # Stage 1 must learn to push with the same physical body envelope it
        # will use in the final match. Reuse the Stage 2 asset repair before
        # cloning (torso/thigh/hip proxies plus per-body CCD).
        from .sumo_stage_2_anymal_1v1 import SumoStage2Anymal1v1Env

        self._repair_robot_colliders()

        # Sensor-aware task variants install their onboard sensors after the
        # robot prims exist but before env_0 is cloned.
        setup_onboard_sensors = getattr(self, "_setup_onboard_sensors", None)
        if setup_onboard_sensors is not None:
            setup_onboard_sensors()

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _repair_robot_colliders(self) -> None:
        SumoStage2Anymal1v1Env._deinstance_and_set_contact_offsets(self)

    def _pre_physics_step(self, actions: dict) -> None:
        self.processed_actions = {}
        self.previous_actions = {agent: action.clone() for agent, action in self.actions.items()}
        self.actions = copy.deepcopy(actions)
        for robot_id, robot in self.robots.items():
            action_scale = self.cfg.action_scales.get(robot_id, self.cfg.action_scale)
            self.processed_actions[robot_id] = action_scale * actions[robot_id] + robot.data.default_joint_pos

        for block in self.blocks.values():
            block.update(self.step_dt)

    def _apply_action(self) -> None:
        for robot_id, robot in self.robots.items():
            robot.set_joint_position_target(self.processed_actions[robot_id])

    def _get_observations(self) -> dict:
        rcol = self.ring_radius.view(-1, 1)

        robot_0_desired_pos, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3],
            self.robots["robot_0"].data.root_state_w[:, 3:7],
            self.blocks["block_0"].data.root_pos_w,
        )
        robot_0_block_velocity = quat_rotate_inverse(
            self.robots["robot_0"].data.root_quat_w,
            self.blocks["block_0"].data.root_lin_vel_w
            - self.robots["robot_0"].data.root_lin_vel_w,
        )
        empty_slot = torch.zeros(self.num_envs, 3, device=self.device)
        robot_0_dist_center = torch.linalg.norm(
            self.robots["robot_0"].data.root_pos_w[:, :2]
            - self.scene.env_origins[:, :2], dim=-1, keepdim=True
        )
        robot_0_obs = torch.cat(
            [
                self.robots["robot_0"].data.root_lin_vel_b,
                self.robots["robot_0"].data.root_ang_vel_b,
                self.robots["robot_0"].data.projected_gravity_b,
                self.robots["robot_0"].data.joint_pos - self.robots["robot_0"].data.default_joint_pos,
                self.robots["robot_0"].data.joint_vel,
                self.actions["robot_0"],
                robot_0_desired_pos,
                robot_0_block_velocity,
                empty_slot,
                robot_0_dist_center,
                rcol,
                # time_remaining
            ],
            dim=-1,
        )

        robot_1_desired_pos, _ = subtract_frame_transforms(
            self.robots["robot_1"].data.root_state_w[:, :3],
            self.robots["robot_1"].data.root_state_w[:, 3:7],
            self.blocks["block_1"].data.root_pos_w,
        )
        robot_1_block_velocity = quat_rotate_inverse(
            self.robots["robot_1"].data.root_quat_w,
            self.blocks["block_1"].data.root_lin_vel_w
            - self.robots["robot_1"].data.root_lin_vel_w,
        )
        robot_1_dist_center = torch.linalg.norm(
            self.robots["robot_1"].data.root_pos_w[:, :2]
            - self.scene.env_origins[:, :2], dim=-1, keepdim=True
        )
        robot_1_obs = torch.cat(
            [
                self.robots["robot_1"].data.root_lin_vel_b,
                self.robots["robot_1"].data.root_ang_vel_b,
                self.robots["robot_1"].data.projected_gravity_b,
                self.robots["robot_1"].data.joint_pos - self.robots["robot_1"].data.default_joint_pos,
                self.robots["robot_1"].data.joint_vel,
                self.actions["robot_1"],
                robot_1_desired_pos,
                robot_1_block_velocity,
                empty_slot,
                robot_1_dist_center,
                rcol,
                # time_remaining
            ],
            dim=-1,
        )

        robot_0_obs = torch.nan_to_num(robot_0_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        robot_1_obs = torch.nan_to_num(robot_1_obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return {"team_0": {"robot_0": robot_0_obs}, "team_1": {"robot_1": robot_1_obs}}

    def _get_rewards(self) -> dict:
        circle_centers_xy = self.scene.env_origins[:, :2]

        # --- Distance of each block from center ---
        block0_dist = torch.linalg.norm(
            self.blocks["block_0"].data.root_pos_w[:, :2] - circle_centers_xy, dim=-1
        )
        block1_dist = torch.linalg.norm(
            self.blocks["block_1"].data.root_pos_w[:, :2] - circle_centers_xy, dim=-1
        )
        block0_progress = (block0_dist - self._prev_block_radius["block_0"]).clamp(-0.1, 0.1)
        block1_progress = (block1_dist - self._prev_block_radius["block_1"]).clamp(-0.1, 0.1)

        # --- Distances robot<->its block ---
        robot0_pos = self.robots["robot_0"].data.root_pos_w[:, :2]
        robot1_pos = self.robots["robot_1"].data.root_pos_w[:, :2]
        block0_pos = self.blocks["block_0"].data.root_pos_w[:, :2]
        block1_pos = self.blocks["block_1"].data.root_pos_w[:, :2]

        dist_r0_b0 = torch.linalg.norm(robot0_pos - block0_pos, dim=-1)
        dist_r1_b1 = torch.linalg.norm(robot1_pos - block1_pos, dim=-1)
        approach0_progress = (
            self._prev_robot_block_distance["robot_0"] - dist_r0_b0
        ).clamp(-0.1, 0.1)
        approach1_progress = (
            self._prev_robot_block_distance["robot_1"] - dist_r1_b1
        ).clamp(-0.1, 0.1)

        out = self._robots_out_of_ring()
        block0_out = out["block_0"].to(torch.float32)
        block1_out = out["block_1"].to(torch.float32)
        robot0_out = out["robot_0"].to(torch.float32)
        robot1_out = out["robot_1"].to(torch.float32)

        push_out_reward_team0 = (block0_out - robot0_out) * self.cfg.reward_scale
        push_out_reward_team1 = (block1_out - robot1_out) * self.cfg.reward_scale

        action_rate_0 = self.cfg.action_rate_reward_scale * torch.sum(
            torch.square(self.actions["robot_0"] - self.previous_actions["robot_0"]), dim=-1
        )
        action_rate_1 = self.cfg.action_rate_reward_scale * torch.sum(
            torch.square(self.actions["robot_1"] - self.previous_actions["robot_1"]), dim=-1
        )
        flat_0 = self.cfg.flat_orientation_reward_scale * torch.sum(
            torch.square(self.robots["robot_0"].data.projected_gravity_b[:, :2]), dim=-1
        )
        flat_1 = self.cfg.flat_orientation_reward_scale * torch.sum(
            torch.square(self.robots["robot_1"].data.projected_gravity_b[:, :2]), dim=-1
        )
        facing_0 = backward_facing_penalty(
            self.robots["robot_0"].data.root_quat_w,
            self.robots["robot_0"].data.root_pos_w,
            self.blocks["block_0"].data.root_pos_w,
            self.cfg.backward_facing_penalty_scale,
            self.cfg.forward_yaw_offsets.get("robot_0", 0.0),
        )
        facing_1 = backward_facing_penalty(
            self.robots["robot_1"].data.root_quat_w,
            self.robots["robot_1"].data.root_pos_w,
            self.blocks["block_1"].data.root_pos_w,
            self.cfg.backward_facing_penalty_scale,
            self.cfg.forward_yaw_offsets.get("robot_1", 0.0),
        )

        # --- Rewards per team ---
        rewards_team0 = {
            "block0_center_reward": block0_progress * self.cfg.block_progress_reward_scale,
            "dist_r0_b0_reward": approach0_progress * self.cfg.approach_progress_reward_scale,
            "action_rate_reward": action_rate_0,
            "flat_orientation_reward": flat_0,
            "backward_facing_penalty": facing_0,
            # "time_penalty": time_penalty,
            "push_out_reward": push_out_reward_team0,
        }
        # rewards_team0.update(all_rewards["robot_0"])

        rewards_team1 = {
            "block1_center_reward": block1_progress * self.cfg.block_progress_reward_scale,
            "dist_r1_b1_reward": approach1_progress * self.cfg.approach_progress_reward_scale,
            "action_rate_reward": action_rate_1,
            "flat_orientation_reward": flat_1,
            "backward_facing_penalty": facing_1,
            # "time_penalty": time_penalty,
            "push_out_reward": push_out_reward_team1,
        }
        # rewards_team1.update(all_rewards["robot_1"])

        # Sum per team
        reward_team0 = torch.sum(torch.stack(list(rewards_team0.values())), dim=0)
        reward_team1 = torch.sum(torch.stack(list(rewards_team1.values())), dim=0)

        # Log episode sums
        for key, val in rewards_team0.items():
            self._episode_sums[f"team_0_{key}"] += val
        for key, val in rewards_team1.items():
            self._episode_sums[f"team_1_{key}"] += val

        self._prev_block_radius["block_0"].copy_(block0_dist)
        self._prev_block_radius["block_1"].copy_(block1_dist)
        self._prev_robot_block_distance["robot_0"].copy_(dist_r0_b0)
        self._prev_robot_block_distance["robot_1"].copy_(dist_r1_b1)

        return {
            "team_0": reward_team0,
            "team_1": reward_team1,
        }

    def _robots_out_of_ring(self) -> dict[str, torch.Tensor]:
        env_xy = self.scene.env_origins[:, :2].to(self.device)
        out = {}
        for robot_id in self.robots.keys():
            pos_xy = self.robots[robot_id].data.root_pos_w[:, :2]
            dist = torch.linalg.norm(pos_xy - env_xy, dim=1)
            out[robot_id] = dist > self.ring_radius
        for block_id in self.blocks.keys():
            pos_xy = self.blocks[block_id].data.root_com_pos_w[:, :2]
            dist = torch.linalg.norm(pos_xy - env_xy, dim=1)
            out[block_id] = dist > self.ring_radius

        return out

    def _get_dones(self) -> tuple[dict, dict]:
        out_map = self._robots_out_of_ring()
        team0_out = torch.any(torch.stack([out_map["robot_0"], out_map["robot_1"]]), dim=0)
        team1_out = torch.any(torch.stack([out_map["block_0"], out_map["block_1"]]), dim=0)
        fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for robot_id, robot in self.robots.items():
            died = robot.data.root_com_pos_w[:, 2] < 0.2
            fallen |= died

        done = team0_out | team1_out | fallen
        dones = {team: done for team in self.cfg.teams.keys()}

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeouts = {team: time_out for team in self.cfg.teams.keys()}
        return dones, timeouts

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        episode_times = self.episode_length_buf[env_ids].to(torch.float32) + 1
        super()._reset_idx(env_ids)

        # spread out the updates
        if not self.debug and len(env_ids) == self.num_envs:  # type:ignore
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Randomize ring radius per env
        low, high = self.cfg.ring_radius_min, self.cfg.ring_radius_max
        self.ring_radius[env_ids] = torch.empty(env_ids.shape[0], device=self.device).uniform_(low, high)  # type:ignore

        origins = self.scene.env_origins[env_ids]  # (N, 3)
        N = env_ids.shape[0]  # type:ignore

        # Two mirrored robot/block lanes. Every robot starts upright, directly
        # behind its target and facing outward. This is both fair and far
        # cheaper than the old Python loop over thousands of environments.
        lane_angle = torch.empty(N, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        lane_directions = torch.stack((torch.cos(lane_angle), torch.sin(lane_angle)), dim=-1)
        block_radius = torch.empty(N, device=self.device).uniform_(1.5, 1.8)
        robot_gap = torch.empty(N, device=self.device).uniform_(0.8, 1.0)

        block_slots = torch.zeros(N, 2, 3, device=self.device)
        robot_slots = torch.zeros(N, 2, 3, device=self.device)
        block_slots[:, 0, :2] = block_radius.unsqueeze(-1) * lane_directions
        block_slots[:, 1, :2] = -block_slots[:, 0, :2]
        robot_slots[:, 0, :2] = (block_radius - robot_gap).unsqueeze(-1) * lane_directions
        robot_slots[:, 1, :2] = -robot_slots[:, 0, :2]
        zeros = torch.zeros_like(lane_angle)
        target_headings = (lane_angle, lane_angle + torch.pi)
        robot_yaws = tuple(
            target_headings[i] - self.cfg.forward_yaw_offsets.get(f"robot_{i}", 0.0)
            for i in range(2)
        )

        # Apply robot positions
        for i, robot_id in enumerate(self.robots):

            # Reset robot state
            self.robots[robot_id].reset(env_ids)
            joint_pos = self.robots[robot_id].data.default_joint_pos[env_ids]
            joint_vel = self.robots[robot_id].data.default_joint_vel[env_ids]
            default_root_state = self.robots[robot_id].data.default_root_state[env_ids].clone()
            default_root_state[:, :2] = origins[:, :2]
            default_root_state[:, 0:2] += robot_slots[:, i, 0:2]
            default_root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, robot_yaws[i])
            self.robots[robot_id].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[robot_id].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robots[robot_id].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            self.actions[robot_id][env_ids] = 0.0
            self.previous_actions[robot_id][env_ids] = 0.0

        # Apply block positions
        for j, (block_id, block) in enumerate(self.blocks.items()):
            block.reset(env_ids)
            default_root_state = block.data.default_root_state[env_ids].clone()
            default_root_state[:, :2] = origins[:, :2]
            default_root_state[:, 0:2] += block_slots[:, j, 0:2]
            block.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            block.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._prev_block_radius["block_0"][env_ids] = block_radius
        self._prev_block_radius["block_1"][env_ids] = block_radius
        self._prev_robot_block_distance["robot_0"][env_ids] = robot_gap
        self._prev_robot_block_distance["robot_1"][env_ids] = robot_gap

        # Draw markers & reset episode logs
        self._draw_ring_markers()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids] / (episode_times))
            extras["Episode_Reward/" + key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

    def _sample_positions_grid(self, N, radii, num_samples, min_dist=0.6, grid_spacing=0.5):
        """
        Sample positions from a grid inside the circle.

        Args:
            N: number of environments
            radii: tensor of ring radii (N,)
            num_samples: number of positions needed per environment
            min_dist: minimum allowed spacing (still enforced between chosen grid spots)
            grid_spacing: spacing between candidate grid points
        Returns:
            (N, num_samples, 3) tensor of sampled positions
        """
        device = radii.device
        offsets = torch.zeros((N, num_samples, 3), device=device)

        for i in range(N):
            r = radii[i].item() * 0.9

            # Build grid within bounding square [-r, r]
            xs = torch.arange(-r, r + grid_spacing, grid_spacing, device=device)
            ys = torch.arange(-r, r + grid_spacing, grid_spacing, device=device)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
            grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

            # Keep only points inside circle
            mask = torch.linalg.norm(grid, dim=1) < r
            candidates = grid[mask]

            # Shuffle candidates
            perm = torch.randperm(candidates.shape[0], device=device)
            candidates = candidates[perm]

            chosen = []
            for pt in candidates:
                if len(chosen) == num_samples:
                    break
                if all(torch.norm(pt - c) >= min_dist for c in chosen):
                    chosen.append(pt)

            if len(chosen) < num_samples:
                raise RuntimeError(f"Not enough valid grid positions for env {i}.")

            offsets[i, :, 0:2] = torch.stack(chosen, dim=0)

        return offsets
