"""Morphology descriptions, separate from game rules and observations.

Adding a profile does not make its checkpoints compatible with another dog.
Every new profile must pass collision, heading, sensor and optimizer audits.
"""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class DogProfile:
    name: str
    asset_module: str
    asset_name: str
    root_body: str
    feet_pattern: str
    upper_leg_pattern: str
    action_scale: float
    mass_delta: float
    front_sensor_xyz: tuple[float, float, float]
    imu_xyz: tuple[float, float, float]
    lidar_xyz: tuple[float, float, float]
    lidar_pitch: float
    forward_yaw: float = 0.0


# Reference transforms, not measurements of Ed's hardware. Unitree ROS commit
# 5994d4faef0a9cadd3287f8de0199a67eeb2a259, go2_description.urdf:
# front_camera_joint, imu_joint, radar_joint (all relative to base).
GO2 = DogProfile(
    "go2",
    "isaaclab_assets.robots.unitree",
    "UNITREE_GO2_CFG",
    "base",
    ".*_foot",
    ".*_thigh",
    0.25,
    1.0,
    (0.32715, -0.00003, 0.04297),
    (-0.02557, 0.0, 0.04232),
    (0.28945, 0.0, -0.046825),
    2.8782,
)


def apply_dog_profile(cfg, profile=GO2):
    """Install one homogeneous morphology across any of the three base stages."""
    from .onboard_rig import front_rgb_camera
    from .quadruped_stage2 import _sumo_robot

    source = getattr(import_module(profile.asset_module), profile.asset_name)
    cfg.collision_profile = profile.name
    cfg.root_body_pattern = profile.root_body
    cfg.feet_body_pattern = profile.feet_pattern
    cfg.undesired_body_pattern = profile.upper_leg_pattern
    cfg.action_scale = profile.action_scale
    cfg.forward_yaw_offset = profile.forward_yaw
    cfg.forward_yaw_offsets = {key: profile.forward_yaw for key in cfg.action_spaces}
    for index, key in enumerate(cfg.action_spaces):
        robot = _sumo_robot(source, f"/World/envs/env_.*/Robot_{index}")
        robot.init_state.pos = (0.0, 0.0, source.init_state.pos[2])
        setattr(cfg, key, robot)
        setattr(cfg, f"rgb_camera_{index}", front_rgb_camera(index, profile))
        mass = getattr(cfg.events, f"add_base_mass_{index}")
        mass.params["asset_cfg"].body_names = profile.root_body
        mass.params["mass_distribution_params"] = (-profile.mass_delta, profile.mass_delta)
