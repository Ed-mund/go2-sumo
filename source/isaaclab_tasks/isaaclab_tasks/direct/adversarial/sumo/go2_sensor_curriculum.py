"""Go2 curriculum whose policies use robot-mounted depth cameras.

The reward functions retain simulator ground truth, which is allowed during
training.  Actor observations do not contain ground-truth block or opponent
poses: the only exteroception is a finite-range, forward-facing depth camera.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from .quadruped_curriculum import Go2BlocksPushCfg, Go2GoToPointCfg, QuadrupedBlocksPushEnv, QuadrupedGoToPointSumo
from .quadruped_stage2 import SumoStage2Go21v1EnvCfg, SumoStage2Quadruped1v1Env

DEPTH_HEIGHT = 36
DEPTH_WIDTH = 64
DEPTH_BANDS = 3
DEPTH_COLUMNS = 32
DEPTH_FEATURES_PER_FRAME = DEPTH_BANDS * DEPTH_COLUMNS
# Current proximity and its one-control-step change provide deployable visual
# motion information without relying on the broken recurrent path in this HARL build.
DEPTH_FEATURES = 2 * DEPTH_FEATURES_PER_FRAME
PROPRIOCEPTIVE_FEATURES = 45
NAVIGATION_FEATURES = 3
RING_FEATURES = 2
SENSOR_OBSERVATION_DIM = PROPRIOCEPTIVE_FEATURES + NAVIGATION_FEATURES + DEPTH_FEATURES + RING_FEATURES


def go2_front_depth_camera(robot_index: int) -> TiledCameraCfg:
    """Return a forward camera mounted on the Go2 base frame."""
    return TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/Robot_{robot_index}/base/front_depth_camera",
        update_period=0.04,
        height=DEPTH_HEIGHT,
        width=DEPTH_WIDTH,
        data_types=["distance_to_camera"],
        depth_clipping_behavior="max",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            focus_distance=6.0,
            horizontal_aperture=20.955,
            clipping_range=(0.08, 6.0),
        ),
        # Go2's base +X axis is anatomical forward.  The "world" camera
        # convention likewise uses +X forward and +Z up.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.28, 0.0, 0.03),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )


def compress_depth_image(
    depth: torch.Tensor,
    max_range: float,
    output_size: tuple[int, int] = (DEPTH_BANDS, DEPTH_COLUMNS),
) -> torch.Tensor:
    """Convert metric depth to a compact near-obstacle profile in [0, 1].

    Max-pooling proximity preserves small nearby objects better than averaging
    metric depth.  The result remains a deterministic function of a deployable
    depth sensor and is cheap enough for the existing MLP/RNN policy.
    """
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shaped (N,H,W) or (N,H,W,1), got {tuple(depth.shape)}")

    finite_depth = torch.nan_to_num(depth, nan=max_range, posinf=max_range, neginf=0.0)
    finite_depth = finite_depth.clamp_(0.0, max_range)
    proximity = 1.0 - finite_depth / max_range
    pooled = functional.adaptive_max_pool2d(proximity.unsqueeze(1), output_size)
    return pooled.flatten(start_dim=1)


class Go2OnboardDepthMixin:
    """Install and read forward depth cameras without privileged detections."""

    depth_max_range = 6.0
    depth_noise_std = 0.01
    depth_dropout_probability = 0.01

    def _setup_onboard_sensors(self) -> None:
        self.depth_cameras: dict[str, TiledCamera] = {}
        self._previous_depth_features: dict[str, torch.Tensor] = {}
        for robot_id in self.robots:
            camera_cfg = getattr(self.cfg, f"depth_camera_{robot_id.removeprefix('robot_')}")
            camera = TiledCamera(camera_cfg)
            self.depth_cameras[robot_id] = camera
            self.scene.sensors[f"{robot_id}_front_depth"] = camera
            self._previous_depth_features[robot_id] = torch.zeros(
                (self.num_envs, DEPTH_FEATURES_PER_FRAME), device=self.device
            )

    def _depth_features(self, robot_id: str) -> torch.Tensor:
        depth = self.depth_cameras[robot_id].data.output["distance_to_camera"].clone()
        if self.depth_noise_std > 0.0:
            depth.add_(torch.randn_like(depth) * self.depth_noise_std)
        current = compress_depth_image(depth, self.depth_max_range)
        if self.depth_dropout_probability > 0.0:
            dropped = torch.rand_like(current) < self.depth_dropout_probability
            current.masked_fill_(dropped, 0.0)
        delta = (current - self._previous_depth_features[robot_id]).clamp_(-1.0, 1.0)
        self._previous_depth_features[robot_id].copy_(current)
        return torch.cat((current, delta), dim=-1)

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        for previous in self._previous_depth_features.values():
            previous[env_ids] = 0.0

    def _sensor_observation(self, robot_id: str, navigation: torch.Tensor) -> torch.Tensor:
        robot = self.robots[robot_id]
        dist_center = torch.linalg.norm(
            robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2],
            dim=-1,
            keepdim=True,
        )
        ring_radius = getattr(self, "ring_radius", None)
        if ring_radius is None:
            radius_feature = torch.zeros((self.num_envs, 1), device=self.device)
        else:
            radius_feature = ring_radius.view(-1, 1)

        observation = torch.cat(
            (
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                robot.data.projected_gravity_b,
                robot.data.joint_pos - robot.data.default_joint_pos,
                robot.data.joint_vel,
                self.actions[robot_id],
                navigation,
                self._depth_features(robot_id),
                dist_center,
                radius_feature,
            ),
            dim=-1,
        )
        return torch.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

    def _ring_centre_in_body(self, robot_id: str) -> torch.Tensor:
        centre = self.scene.env_origins.clone()
        centre[:, 2] = self.robots[robot_id].data.root_pos_w[:, 2]
        centre_b, _ = subtract_frame_transforms(
            self.robots[robot_id].data.root_pos_w,
            self.robots[robot_id].data.root_quat_w,
            centre,
        )
        return centre_b


@configclass
class Go2SensorGoToPointCfg(Go2GoToPointCfg):
    observation_space = SENSOR_OBSERVATION_DIM
    observation_spaces = {"robot_0": SENSOR_OBSERVATION_DIM}
    depth_camera_0 = go2_front_depth_camera(0)
    rerender_on_reset = True


class Go2SensorGoToPointEnv(Go2OnboardDepthMixin, QuadrupedGoToPointSumo):
    cfg: Go2SensorGoToPointCfg

    def _get_observations(self) -> dict:
        goal_b, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_pos_w,
            self.robots["robot_0"].data.root_quat_w,
            self._desired_pos,
        )
        return {"team_0": {"robot_0": self._sensor_observation("robot_0", goal_b)}}


@configclass
class Go2SensorBlocksPushCfg(Go2BlocksPushCfg):
    observation_spaces = {
        "robot_0": SENSOR_OBSERVATION_DIM,
        "robot_1": SENSOR_OBSERVATION_DIM,
    }
    depth_camera_0 = go2_front_depth_camera(0)
    depth_camera_1 = go2_front_depth_camera(1)
    rerender_on_reset = True


class Go2SensorBlocksPushEnv(Go2OnboardDepthMixin, QuadrupedBlocksPushEnv):
    cfg: Go2SensorBlocksPushCfg

    def _get_observations(self) -> dict:
        return {
            "team_0": {"robot_0": self._sensor_observation("robot_0", self._ring_centre_in_body("robot_0"))},
            "team_1": {"robot_1": self._sensor_observation("robot_1", self._ring_centre_in_body("robot_1"))},
        }


@configclass
class Go2SensorSumo1v1Cfg(SumoStage2Go21v1EnvCfg):
    observation_spaces = {
        "robot_0": SENSOR_OBSERVATION_DIM,
        "robot_1": SENSOR_OBSERVATION_DIM,
    }
    depth_camera_0 = go2_front_depth_camera(0)
    depth_camera_1 = go2_front_depth_camera(1)
    rerender_on_reset = True


class Go2SensorSumo1v1Env(Go2OnboardDepthMixin, SumoStage2Quadruped1v1Env):
    cfg: Go2SensorSumo1v1Cfg

    def _get_observations(self) -> dict:
        # Rewards and termination still use simulator truth, but neither actor
        # receives opponent position, velocity, heading, or upright state.
        return {
            "team_0": {"robot_0": self._sensor_observation("robot_0", self._ring_centre_in_body("robot_0"))},
            "team_1": {"robot_1": self._sensor_observation("robot_1", self._ring_centre_in_body("robot_1"))},
        }
