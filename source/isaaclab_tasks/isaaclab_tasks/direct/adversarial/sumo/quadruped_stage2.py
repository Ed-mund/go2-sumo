"""Spot and Unitree one-versus-one sumo environments."""

from __future__ import annotations

import copy

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG

from .anymal_c.sumo_stage_2_anymal_1v1 import (
    SumoStage2Anymal1v1Env,
    SumoStage2Anymal1v1EnvCfg,
)
from .quadruped_colliders import PROFILES, repair_quadruped_colliders


def _sumo_robot(source: ArticulationCfg, prim_path: str) -> ArticulationCfg:
    rigid_props = source.spawn.rigid_props.replace(max_depenetration_velocity=10.0)
    articulation_props = source.spawn.articulation_props.replace(
        solver_position_iteration_count=24,
        solver_velocity_iteration_count=4,
    )
    return source.replace(
        prim_path=prim_path,
        spawn=source.spawn.replace(rigid_props=rigid_props, articulation_props=articulation_props),
    )


def _events_for(root_body: str, mass_range: tuple[float, float]):
    events = copy.deepcopy(SumoStage2Anymal1v1EnvCfg().events)
    for index in (0, 1):
        term = getattr(events, f"add_base_mass_{index}")
        term.params["asset_cfg"].body_names = root_body
        term.params["mass_distribution_params"] = mass_range
    return events


def _events_for_pair(root_bodies, mass_ranges):
    events = copy.deepcopy(SumoStage2Anymal1v1EnvCfg().events)
    for index in (0, 1):
        term = getattr(events, f"add_base_mass_{index}")
        term.params["asset_cfg"].body_names = root_bodies[index]
        term.params["mass_distribution_params"] = mass_ranges[index]
    return events


class SumoStage2Quadruped1v1Env(SumoStage2Anymal1v1Env):
    """Shared dynamics/rewards with model-specific measured collider repair."""

    def _deinstance_and_set_contact_offsets(self) -> None:
        profiles = getattr(self.cfg, "collision_profiles", self.cfg.collision_profile)
        repair_quadruped_colliders(self, profiles)


@configclass
class SumoStage2Spot1v1EnvCfg(SumoStage2Anymal1v1EnvCfg):
    collision_profile = "spot"
    root_body_pattern = PROFILES[collision_profile].root_body
    action_scale = 0.2
    forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    events = _events_for(root_body_pattern, (-2.5, 2.5))
    robot_0 = _sumo_robot(SPOT_CFG, "/World/envs/env_.*/Robot_0")
    robot_1 = _sumo_robot(SPOT_CFG, "/World/envs/env_.*/Robot_1")
    robot_0.init_state.pos = (0.0, 1.0, SPOT_CFG.init_state.pos[2])
    robot_1.init_state.pos = (0.0, -1.0, SPOT_CFG.init_state.pos[2])
    robot_0.init_state.rot = (0.70710677, 0.0, 0.0, 0.70710677)
    robot_1.init_state.rot = robot_0.init_state.rot


@configclass
class SumoStage2A11v1EnvCfg(SumoStage2Anymal1v1EnvCfg):
    collision_profile = "a1"
    root_body_pattern = PROFILES[collision_profile].root_body
    action_scale = 0.25
    forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    events = _events_for(root_body_pattern, (-1.0, 1.0))
    robot_0 = _sumo_robot(UNITREE_A1_CFG, "/World/envs/env_.*/Robot_0")
    robot_1 = _sumo_robot(UNITREE_A1_CFG, "/World/envs/env_.*/Robot_1")
    robot_0.init_state.pos = (0.0, 1.0, UNITREE_A1_CFG.init_state.pos[2])
    robot_1.init_state.pos = (0.0, -1.0, UNITREE_A1_CFG.init_state.pos[2])
    robot_0.init_state.rot = (0.70710677, 0.0, 0.0, 0.70710677)
    robot_1.init_state.rot = robot_0.init_state.rot


@configclass
class SumoStage2Go11v1EnvCfg(SumoStage2Anymal1v1EnvCfg):
    collision_profile = "go1"
    root_body_pattern = PROFILES[collision_profile].root_body
    action_scale = 0.25
    forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    events = _events_for(root_body_pattern, (-1.0, 1.0))
    robot_0 = _sumo_robot(UNITREE_GO1_CFG, "/World/envs/env_.*/Robot_0")
    robot_1 = _sumo_robot(UNITREE_GO1_CFG, "/World/envs/env_.*/Robot_1")
    robot_0.init_state.pos = (0.0, 1.0, UNITREE_GO1_CFG.init_state.pos[2])
    robot_1.init_state.pos = (0.0, -1.0, UNITREE_GO1_CFG.init_state.pos[2])
    robot_0.init_state.rot = (0.70710677, 0.0, 0.0, 0.70710677)
    robot_1.init_state.rot = robot_0.init_state.rot


@configclass
class SumoStage2Go21v1EnvCfg(SumoStage2Anymal1v1EnvCfg):
    collision_profile = "go2"
    root_body_pattern = PROFILES[collision_profile].root_body
    action_scale = 0.25
    forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    events = _events_for(root_body_pattern, (-1.0, 1.0))
    robot_0 = _sumo_robot(UNITREE_GO2_CFG, "/World/envs/env_.*/Robot_0")
    robot_1 = _sumo_robot(UNITREE_GO2_CFG, "/World/envs/env_.*/Robot_1")
    robot_0.init_state.pos = (0.0, 1.0, UNITREE_GO2_CFG.init_state.pos[2])
    robot_1.init_state.pos = (0.0, -1.0, UNITREE_GO2_CFG.init_state.pos[2])
    robot_0.init_state.rot = (0.70710677, 0.0, 0.0, 0.70710677)
    robot_1.init_state.rot = robot_0.init_state.rot


@configclass
class SumoStage2Go2VsSpotEnvCfg(SumoStage2Anymal1v1EnvCfg):
    """Unitree Go2 (Team 0) versus Boston Dynamics Spot (Team 1)."""

    collision_profile = "go2"
    collision_profiles = ("go2", "spot")
    root_body_pattern = "base"
    root_body_patterns = {"robot_0": "base", "robot_1": "body"}
    action_scale = 0.25
    action_scales = {"robot_0": 0.25, "robot_1": 0.2}
    forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    events = _events_for_pair(("base", "body"), ((-1.0, 1.0), (-2.5, 2.5)))

    robot_0 = _sumo_robot(UNITREE_GO2_CFG, "/World/envs/env_.*/Robot_0")
    robot_1 = _sumo_robot(SPOT_CFG, "/World/envs/env_.*/Robot_1")
    robot_0.init_state.pos = (0.0, 1.0, UNITREE_GO2_CFG.init_state.pos[2])
    robot_1.init_state.pos = (0.0, -1.0, SPOT_CFG.init_state.pos[2])
    robot_0.init_state.rot = (0.70710677, 0.0, 0.0, 0.70710677)
    robot_1.init_state.rot = robot_0.init_state.rot


def _anymal_robot(prim_path: str) -> ArticulationCfg:
    """Reuse the already validated high-contact ANYmal-C spawn properties."""
    rigid_props = ANYMAL_C_CFG.spawn.rigid_props.replace(max_depenetration_velocity=10.0)
    articulation_props = ANYMAL_C_CFG.spawn.articulation_props.replace(
        enabled_self_collisions=True,
        solver_position_iteration_count=24,
        solver_velocity_iteration_count=4,
    )
    return ANYMAL_C_CFG.replace(
        prim_path=prim_path,
        spawn=ANYMAL_C_CFG.spawn.replace(
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        ),
    )


@configclass
class SumoStage2SpotVsAnymalEnvCfg(SumoStage2Anymal1v1EnvCfg):
    """Spot (Team 0) versus ANYmal-C (Team 1)."""

    collision_profile = "spot"
    collision_profiles = ("spot", "anymal_c")
    root_body_pattern = "body"
    root_body_patterns = {"robot_0": "body", "robot_1": "base"}
    action_scale = 0.2
    action_scales = {"robot_0": 0.2, "robot_1": 0.5}
    forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    events = _events_for_pair(("body", "base"), ((-2.5, 2.5), (-5.0, 5.0)))

    robot_0 = _sumo_robot(SPOT_CFG, "/World/envs/env_.*/Robot_0")
    robot_1 = _anymal_robot("/World/envs/env_.*/Robot_1")
    robot_0.init_state.pos = (0.0, 1.0, SPOT_CFG.init_state.pos[2])
    robot_1.init_state.pos = (0.0, -1.0, 0.56)
    robot_0.init_state.rot = (0.70710677, 0.0, 0.0, 0.70710677)
    robot_1.init_state.rot = robot_0.init_state.rot


@configclass
class SumoStage2AnymalVsGo2EnvCfg(SumoStage2Anymal1v1EnvCfg):
    """ANYmal-C (Team 0) versus Unitree Go2 (Team 1)."""

    collision_profile = "anymal_c"
    collision_profiles = ("anymal_c", "go2")
    root_body_pattern = "base"
    root_body_patterns = {"robot_0": "base", "robot_1": "base"}
    action_scale = 0.5
    action_scales = {"robot_0": 0.5, "robot_1": 0.25}
    forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    events = _events_for_pair(("base", "base"), ((-5.0, 5.0), (-1.0, 1.0)))

    robot_0 = _anymal_robot("/World/envs/env_.*/Robot_0")
    robot_1 = _sumo_robot(UNITREE_GO2_CFG, "/World/envs/env_.*/Robot_1")
    robot_0.init_state.pos = (0.0, 1.0, 0.56)
    robot_1.init_state.pos = (0.0, -1.0, UNITREE_GO2_CFG.init_state.pos[2])
    robot_0.init_state.rot = (0.70710677, 0.0, 0.0, 0.70710677)
    robot_1.init_state.rot = robot_0.init_state.rot


@configclass
class SumoStage2SpotVsGo2EnvCfg(SumoStage2Anymal1v1EnvCfg):
    """Boston Dynamics Spot (Team 0) versus Unitree Go2 (Team 1)."""

    collision_profile = "spot"
    collision_profiles = ("spot", "go2")
    root_body_pattern = "body"
    root_body_patterns = {"robot_0": "body", "robot_1": "base"}
    action_scale = 0.2
    action_scales = {"robot_0": 0.2, "robot_1": 0.25}
    forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    events = _events_for_pair(("body", "base"), ((-2.5, 2.5), (-1.0, 1.0)))

    robot_0 = _sumo_robot(SPOT_CFG, "/World/envs/env_.*/Robot_0")
    robot_1 = _sumo_robot(UNITREE_GO2_CFG, "/World/envs/env_.*/Robot_1")
    robot_0.init_state.pos = (0.0, 1.0, SPOT_CFG.init_state.pos[2])
    robot_1.init_state.pos = (0.0, -1.0, UNITREE_GO2_CFG.init_state.pos[2])
    robot_0.init_state.rot = (0.70710677, 0.0, 0.0, 0.70710677)
    robot_1.init_state.rot = robot_0.init_state.rot


@configclass
class SumoStage2AnymalVsSpotEnvCfg(SumoStage2Anymal1v1EnvCfg):
    """ANYmal-C (Team 0) versus Boston Dynamics Spot (Team 1)."""

    collision_profile = "anymal_c"
    collision_profiles = ("anymal_c", "spot")
    root_body_pattern = "base"
    root_body_patterns = {"robot_0": "base", "robot_1": "body"}
    action_scale = 0.5
    action_scales = {"robot_0": 0.5, "robot_1": 0.2}
    forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    events = _events_for_pair(("base", "body"), ((-5.0, 5.0), (-2.5, 2.5)))

    robot_0 = _anymal_robot("/World/envs/env_.*/Robot_0")
    robot_1 = _sumo_robot(SPOT_CFG, "/World/envs/env_.*/Robot_1")
    robot_0.init_state.pos = (0.0, 1.0, 0.56)
    robot_1.init_state.pos = (0.0, -1.0, SPOT_CFG.init_state.pos[2])
    robot_0.init_state.rot = (0.70710677, 0.0, 0.0, 0.70710677)
    robot_1.init_state.rot = robot_0.init_state.rot


@configclass
class SumoStage2Go2VsAnymalEnvCfg(SumoStage2Anymal1v1EnvCfg):
    """Unitree Go2 (Team 0) versus ANYmal-C (Team 1)."""

    collision_profile = "go2"
    collision_profiles = ("go2", "anymal_c")
    root_body_pattern = "base"
    root_body_patterns = {"robot_0": "base", "robot_1": "base"}
    action_scale = 0.25
    action_scales = {"robot_0": 0.25, "robot_1": 0.5}
    forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    events = _events_for_pair(("base", "base"), ((-1.0, 1.0), (-5.0, 5.0)))

    robot_0 = _sumo_robot(UNITREE_GO2_CFG, "/World/envs/env_.*/Robot_0")
    robot_1 = _anymal_robot("/World/envs/env_.*/Robot_1")
    robot_0.init_state.pos = (0.0, 1.0, UNITREE_GO2_CFG.init_state.pos[2])
    robot_1.init_state.pos = (0.0, -1.0, 0.56)
    robot_0.init_state.rot = (0.70710677, 0.0, 0.0, 0.70710677)
    robot_1.init_state.rot = robot_0.init_state.rot
