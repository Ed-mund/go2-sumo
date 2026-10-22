"""AIR/PRO-compatible sensor types at published Go2 frame transforms.

RGB intrinsics and sensor error distributions are reference assumptions awaiting
calibration. PhysX LiDAR sees moving collision geometry, unlike IsaacLab's static
mesh RayCaster. Its regular instantaneous rays are NOT Unitree's scanning pattern.
"""

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Imu, ImuCfg, TiledCamera, TiledCameraCfg


class PhysicsRateImu(Imu):
    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._prev_lin_vel_w[ids] = 0
        self._prev_ang_vel_w[ids] = 0

    def update(self, dt, force_recompute=False):
        # Acceleration is a finite difference over dt. Lazy control-rate reads
        # with a physics-rate dt would otherwise overestimate acceleration.
        super().update(dt, force_recompute=True)


def front_rgb_camera(index, profile):
    return TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/Robot_{index}/{profile.root_body}/front_rgb_camera",
        update_period=0.04,
        update_latest_camera_pose=True,
        height=72,
        width=128,
        data_types=["rgb"],
        # 90-degree pinhole is an explicit uncalibrated approximation of the wide-angle camera.
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, horizontal_aperture=20.0, clipping_range=(0.01, 12.0)),
        offset=TiledCameraCfg.OffsetCfg(pos=profile.front_sensor_xyz, convention="world", rot=(1.0, 0.0, 0.0, 0.0)),
    )


class OnboardRig:
    def __init__(self, env, profile):
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.sensors.physx")
        import omni.kit.commands
        from isaacsim.sensors.physx import _range_sensor
        from pxr import Gf, UsdGeom

        self.env = env
        self.rgb, self.imu, self.paths = {}, {}, {}
        self.interface = _range_sensor.acquire_lidar_sensor_interface()
        for key in env.robots:
            index = int(key.split("_")[-1])
            camera = TiledCamera(getattr(env.cfg, f"rgb_camera_{index}"))
            imu = PhysicsRateImu(
                ImuCfg(
                    prim_path=f"/World/envs/env_.*/Robot_{index}/{profile.root_body}",
                    update_period=0.005,
                    offset=ImuCfg.OffsetCfg(pos=profile.imu_xyz),
                )
            )
            self.rgb[key], self.imu[key] = camera, imu
            env.scene.sensors[f"{key}_rgb"] = camera
            env.scene.sensors[f"{key}_imu"] = imu
            path = f"/World/envs/env_0/Robot_{index}/{profile.root_body}/reference_lidar"
            ok, lidar = omni.kit.commands.execute(
                "RangeSensorCreateLidar",
                path=path,
                parent=None,
                min_range=0.05,
                max_range=6.0,
                horizontal_fov=360.0,
                vertical_fov=180.0,
                horizontal_resolution=10.0,
                vertical_resolution=10.0,
                rotation_rate=0.0,
                high_lod=True,
                draw_points=False,
                draw_lines=False,
                enable_semantics=False,
            )
            if not ok:
                raise RuntimeError(f"Failed to create moving-scene LiDAR: {path}")
            # Command creates translate/rotateXYZ ops; rotateXYZ is identity on Z-up stages.
            xform = UsdGeom.Xformable(lidar.GetPrim())
            ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}
            translate = ops.get(UsdGeom.XformOp.TypeTranslate)
            if translate is None:
                translate = xform.AddTranslateOp()
            translate.Set(Gf.Vec3d(*profile.lidar_xyz))
            orient = ops.get(UsdGeom.XformOp.TypeOrient)
            if orient is None:
                orient = xform.AddOrientOp()
            half = profile.lidar_pitch / 2
            orient.Set(Gf.Quatf(math.cos(half), 0, math.sin(half), 0))
            self.paths[key] = [path.replace("env_0/", f"env_{i}/") for i in range(env.num_envs)]

    def points(self, key):
        clouds = [
            np.asarray(self.interface.get_point_cloud_data(path)).reshape(-1, 3).copy() for path in self.paths[key]
        ]
        if not clouds or any(len(cloud) == 0 for cloud in clouds):
            raise RuntimeError("LiDAR returned no rays. Check extension, cloned prims and scene synchronization.")
        array = np.stack(clouds)
        if not np.isfinite(array).all():
            raise RuntimeError("Non-finite LiDAR points")
        return torch.as_tensor(array, dtype=torch.float32, device=self.env.device)
