"""Locomotion and block-push curriculum stages for Spot and Unitree dogs."""

from __future__ import annotations

import copy

from isaaclab.utils import configclass

from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG

from .anymal_c.anymal_c_go_to_point_sumo import AnymalCGoToPointSumo, AnymalCGoToPointSumoCfg
from .anymal_c.sumo_stage_1_blocks_push import AnymalCBlocksPushEnv, AnymalCBlocksPushEnvCfg
from .quadruped_colliders import PROFILES, repair_quadruped_colliders
from .quadruped_stage2 import _sumo_robot


_ROTATED = (0.70710677, 0.0, 0.0, 0.70710677)


def _stage0_events(root_body: str, mass_range):
    events = copy.deepcopy(AnymalCGoToPointSumoCfg().events)
    events.add_base_mass_0.params["asset_cfg"].body_names = root_body
    events.add_base_mass_0.params["mass_distribution_params"] = mass_range
    return events


def _stage1_events(root_body: str, mass_range):
    events = copy.deepcopy(AnymalCBlocksPushEnvCfg().events)
    for index in (0, 1):
        term = getattr(events, f"add_base_mass_{index}")
        term.params["asset_cfg"].body_names = root_body
        term.params["mass_distribution_params"] = mass_range
    return events


def _mixed_stage1_events():
    events = copy.deepcopy(AnymalCBlocksPushEnvCfg().events)
    events.add_base_mass_0.params["asset_cfg"].body_names = PROFILES["go2"].root_body
    events.add_base_mass_0.params["mass_distribution_params"] = (-1.0, 1.0)
    events.add_base_mass_1.params["asset_cfg"].body_names = PROFILES["spot"].root_body
    events.add_base_mass_1.params["mass_distribution_params"] = (-2.5, 2.5)
    return events


def _block(index: int, mass: float):
    cfg = copy.deepcopy(getattr(AnymalCBlocksPushEnvCfg(), f"block_{index}"))
    cfg.spawn.mass_props.mass = mass
    return cfg


class QuadrupedGoToPointSumo(AnymalCGoToPointSumo):
    def _repair_robot_colliders(self) -> None:
        repair_quadruped_colliders(self, self.cfg.collision_profile)


class QuadrupedBlocksPushEnv(AnymalCBlocksPushEnv):
    def _repair_robot_colliders(self) -> None:
        repair_quadruped_colliders(self, self.cfg.collision_profile)


def _configure_stage0(cls, profile, source, action_scale, mass_range):
    cls.collision_profile = profile
    cls.root_body_pattern = PROFILES[profile].root_body
    cls.feet_body_pattern = ".*_foot"
    cls.undesired_body_pattern = ".*_uleg" if profile == "spot" else ".*_thigh"
    cls.action_scale = action_scale
    cls.forward_yaw_offset = 0.0
    cls.events = _stage0_events(cls.root_body_pattern, mass_range)
    cls.robot_0 = _sumo_robot(source, "/World/envs/env_.*/Robot_0")
    cls.robot_0.init_state.pos = (0.0, 0.0, source.init_state.pos[2])
    cls.robot_0.init_state.rot = _ROTATED
    return configclass(cls)


def _configure_stage1(cls, profile, source, action_scale, mass_range, block_mass):
    cls.collision_profile = profile
    cls.root_body_pattern = PROFILES[profile].root_body
    cls.feet_body_pattern = ".*_foot"
    cls.undesired_body_pattern = ".*_uleg" if profile == "spot" else ".*_thigh"
    cls.action_scale = action_scale
    cls.forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    cls.events = _stage1_events(cls.root_body_pattern, mass_range)
    cls.robot_0 = _sumo_robot(source, "/World/envs/env_.*/Robot_0")
    cls.robot_1 = _sumo_robot(source, "/World/envs/env_.*/Robot_1")
    cls.robot_0.init_state.pos = (0.0, 0.0, source.init_state.pos[2])
    cls.robot_1.init_state.pos = (0.0, 0.0, source.init_state.pos[2])
    cls.robot_0.init_state.rot = _ROTATED
    cls.robot_1.init_state.rot = _ROTATED
    cls.block_0 = _block(0, block_mass)
    cls.block_1 = _block(1, block_mass)
    return configclass(cls)


class SpotGoToPointCfg(AnymalCGoToPointSumoCfg):
    pass


SpotGoToPointCfg = _configure_stage0(SpotGoToPointCfg, "spot", SPOT_CFG, 0.2, (-2.5, 2.5))


class A1GoToPointCfg(AnymalCGoToPointSumoCfg):
    pass


A1GoToPointCfg = _configure_stage0(A1GoToPointCfg, "a1", UNITREE_A1_CFG, 0.25, (-1.0, 1.0))


class Go1GoToPointCfg(AnymalCGoToPointSumoCfg):
    pass


Go1GoToPointCfg = _configure_stage0(Go1GoToPointCfg, "go1", UNITREE_GO1_CFG, 0.25, (-1.0, 1.0))


class Go2GoToPointCfg(AnymalCGoToPointSumoCfg):
    pass


Go2GoToPointCfg = _configure_stage0(Go2GoToPointCfg, "go2", UNITREE_GO2_CFG, 0.25, (-1.0, 1.0))


class SpotBlocksPushCfg(AnymalCBlocksPushEnvCfg):
    pass


SpotBlocksPushCfg = _configure_stage1(SpotBlocksPushCfg, "spot", SPOT_CFG, 0.2, (-2.5, 2.5), 6.0)


class A1BlocksPushCfg(AnymalCBlocksPushEnvCfg):
    pass


A1BlocksPushCfg = _configure_stage1(A1BlocksPushCfg, "a1", UNITREE_A1_CFG, 0.25, (-1.0, 1.0), 3.0)


class Go1BlocksPushCfg(AnymalCBlocksPushEnvCfg):
    pass


Go1BlocksPushCfg = _configure_stage1(Go1BlocksPushCfg, "go1", UNITREE_GO1_CFG, 0.25, (-1.0, 1.0), 3.0)


class Go2BlocksPushCfg(AnymalCBlocksPushEnvCfg):
    pass


Go2BlocksPushCfg = _configure_stage1(Go2BlocksPushCfg, "go2", UNITREE_GO2_CFG, 0.25, (-1.0, 1.0), 3.0)


@configclass
class Go2VsSpotBlocksPushCfg(AnymalCBlocksPushEnvCfg):
    """Stage 1 transfer task with Go2 on team 0 and Spot on team 1."""

    collision_profile = ("go2", "spot")
    root_body_patterns = {"robot_0": PROFILES["go2"].root_body, "robot_1": PROFILES["spot"].root_body}
    feet_body_patterns = {"robot_0": ".*_foot", "robot_1": ".*_foot"}
    undesired_body_patterns = {"robot_0": ".*_thigh", "robot_1": ".*_uleg"}
    action_scales = {"robot_0": 0.25, "robot_1": 0.2}
    forward_yaw_offsets = {"robot_0": 0.0, "robot_1": 0.0}
    events = _mixed_stage1_events()

    robot_0 = _sumo_robot(UNITREE_GO2_CFG, "/World/envs/env_.*/Robot_0")
    robot_0.init_state.pos = (0.0, 0.0, UNITREE_GO2_CFG.init_state.pos[2])
    robot_0.init_state.rot = _ROTATED

    robot_1 = _sumo_robot(SPOT_CFG, "/World/envs/env_.*/Robot_1")
    robot_1.init_state.pos = (0.0, 0.0, SPOT_CFG.init_state.pos[2])
    robot_1.init_state.rot = _ROTATED

    # Match each target to the approximate mass and pushing ability of its
    # assigned robot so neither species starts Stage 1 at a large disadvantage.
    block_0 = _block(0, 3.0)
    block_1 = _block(1, 6.0)
