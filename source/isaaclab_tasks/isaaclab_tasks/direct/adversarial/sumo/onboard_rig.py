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


def _create_physx_lidar(path):
    """Create a PhysX lidar prim without Isaac Sim 5.1's RangeSensorCreatePrim command.

    That command sets ``visibility`` on an attribute with an empty type name, which USD rejects.
    """
    import omni.isaac.RangeSensorSchema as RangeSensorSchema
    import omni.usd
    from isaacsim.core.utils.stage import get_next_free_path
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim_path = get_next_free_path(path)
    schema = RangeSensorSchema.Lidar.Define(stage, prim_path)
    prim = schema.GetPrim()
    if not prim.IsValid():
        print(f"[sumo] Lidar.Define failed for {prim_path}", flush=True)
        return None
    if prim.GetTypeName() != "Lidar":
        prim.SetTypeName("Lidar")
    for name, value in (
        ("Enabled", True),
        ("DrawPoints", False),
        ("DrawLines", False),
        ("MinRange", 0.05),
        ("MaxRange", 6.0),
        ("HorizontalFov", 360.0),
        ("VerticalFov", 180.0),
        ("HorizontalResolution", 10.0),
        ("VerticalResolution", 10.0),
        ("RotationRate", 0.0),
        ("HighLod", True),
        ("YawOffset", 0.0),
        ("EnableSemantics", False),
    ):
        getattr(schema, f"Create{name}Attr")().Set(value)
    UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.inherited)
    print(f"[sumo] created PhysX lidar {prim.GetPath()}", flush=True)
    return schema


class OnboardRig:
    def __init__(self, env, profile):
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.sensors.physx")
        import omni.kit.app
        from isaacsim.sensors.physx import _range_sensor
        from pxr import Gf, UsdGeom

        # The PhysX lidar manager only notices prims created after its extension has started.
        for _ in range(2):
            omni.kit.app.get_app().update()

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
            lidar = _create_physx_lidar(path)
            if lidar is None:
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
            real, imag = math.cos(half), math.sin(half)
            if orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                orient.Set(Gf.Quatd(real, 0, imag, 0))
            else:
                orient.Set(Gf.Quatf(real, 0, imag, 0))
            self.paths[key] = [path.replace("env_0/", f"env_{i}/") for i in range(env.num_envs)]

    def points(self, key):
        stage = __import__("omni").usd.get_context().get_stage()
        for path in self.paths[key]:
            if self.interface.is_lidar_sensor(path):
                continue
            parent = path.rsplit("/", 1)[0]
            if stage.GetPrimAtPath(parent).IsValid() and not stage.GetPrimAtPath(path).IsValid():
                _create_physx_lidar(path)
        if any(not self.interface.is_lidar_sensor(path) for path in self.paths[key]):
            if not getattr(self, "_warned_lidar", False):
                print(f"[sumo] lidar not registered yet: {self.paths[key]}", flush=True)
                self._warned_lidar = True
            return torch.zeros(self.env.num_envs, 1, 3, device=self.env.device)
        clouds = [
            np.asarray(self.interface.get_point_cloud_data(path)).reshape(-1, 3).copy() for path in self.paths[key]
        ]
        if not clouds or any(len(cloud) == 0 for cloud in clouds):
            if not getattr(self, "_warned_empty", False):
                print("[sumo] lidar returned no rays yet", flush=True)
                self._warned_empty = True
            return torch.zeros(self.env.num_envs, 1, 3, device=self.env.device)
        array = np.stack(clouds)
        if not np.isfinite(array).all():
            raise RuntimeError("Non-finite LiDAR points")
        return torch.as_tensor(array, dtype=torch.float32, device=self.env.device)
