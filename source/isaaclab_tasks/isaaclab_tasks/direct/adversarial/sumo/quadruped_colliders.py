"""Measured collision repairs for quadrupeds used in adversarial sumo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuadrupedCollisionProfile:
    root_body: str
    hip_links: tuple[str, ...]
    upper_links: tuple[str, ...]
    lower_links: tuple[str, ...]
    torso_boxes: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]
    hip_radius: float
    upper_end: tuple[float, float, float]
    upper_radius: float
    lower_end: tuple[float, float, float]
    lower_radius: float
    torso_capsules: tuple[
        tuple[str, float, float, str, tuple[float, float, float]], ...
    ] = ()
    upper_capsule: tuple[float, float, str, tuple[float, float, float]] | None = None
    lower_ends: tuple[tuple[float, float, float], ...] = ()
    lower_clearances: tuple[float, float] = (0.035, 0.035)


_UNITREE_LEGS = ("FL", "FR", "RL", "RR")

PROFILES = {
    "anymal_c": QuadrupedCollisionProfile(
        root_body="base",
        hip_links=("LF_HIP", "RF_HIP", "LH_HIP", "RH_HIP"),
        upper_links=("LF_THIGH", "RF_THIGH", "LH_THIGH", "RH_THIGH"),
        lower_links=("LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"),
        torso_boxes=(
            ("front", (0.12, 0.31, 0.18), (0.468, 0.0, -0.015)),
            ("rear", (0.12, 0.31, 0.18), (-0.468, 0.0, -0.015)),
        ),
        hip_radius=0.07,
        upper_end=(0.0, 0.0, -0.285),
        upper_radius=0.0525,
        lower_end=(0.01305, -0.08795, -0.33797),
        lower_radius=0.032,
        torso_capsules=(("core", 0.14, 0.47, "x", (0.0, 0.0, 0.0)),),
        upper_capsule=(0.0525, 0.18, "z", (0.0, 0.0, -0.1425)),
        lower_ends=(
            (0.01305, -0.08795, -0.33797),
            (0.01305, 0.08795, -0.33797),
            (0.01305, 0.08795, -0.33797),
            (0.01305, -0.08795, -0.33797),
        ),
        lower_clearances=(0.055, 0.045),
    ),
    "spot": QuadrupedCollisionProfile(
        root_body="body",
        hip_links=("fl_hip", "fr_hip", "hl_hip", "hr_hip"),
        upper_links=("fl_uleg", "fr_uleg", "hl_uleg", "hr_uleg"),
        lower_links=("fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"),
        torso_boxes=(
            ("core", (0.42, 0.18, 0.16), (0.0, 0.0, -0.005)),
            ("front", (0.08, 0.22, 0.18), (0.402, 0.0, -0.005)),
            ("rear", (0.08, 0.22, 0.18), (-0.389, 0.0, -0.005)),
        ),
        hip_radius=0.048,
        upper_end=(0.025, 0.0, -0.3205),
        upper_radius=0.045,
        lower_end=(0.0, 0.0, -0.3365),
        lower_radius=0.024,
    ),
    "a1": QuadrupedCollisionProfile(
        root_body="trunk",
        hip_links=tuple(f"{leg}_hip" for leg in _UNITREE_LEGS),
        upper_links=tuple(f"{leg}_thigh" for leg in _UNITREE_LEGS),
        lower_links=tuple(f"{leg}_calf" for leg in _UNITREE_LEGS),
        torso_boxes=(
            ("core", (0.24, 0.16, 0.10), (0.0, 0.0, 0.0)),
            ("front", (0.055, 0.18, 0.105), (0.2525, 0.0, 0.0)),
        ),
        hip_radius=0.038,
        upper_end=(0.0, 0.0, -0.2),
        upper_radius=0.036,
        lower_end=(0.0, 0.0, -0.2),
        lower_radius=0.023,
    ),
    "go1": QuadrupedCollisionProfile(
        root_body="trunk",
        hip_links=tuple(f"{leg}_hip" for leg in _UNITREE_LEGS),
        upper_links=tuple(f"{leg}_thigh" for leg in _UNITREE_LEGS),
        lower_links=tuple(f"{leg}_calf" for leg in _UNITREE_LEGS),
        torso_boxes=(
            ("core", (0.24, 0.16, 0.115), (0.0, 0.0, 0.0)),
            ("front", (0.055, 0.18, 0.12), (0.27, 0.0, 0.0)),
        ),
        hip_radius=0.043,
        upper_end=(0.0, 0.0, -0.213),
        upper_radius=0.04,
        lower_end=(0.0, 0.0, -0.213),
        lower_radius=0.025,
    ),
    "go2": QuadrupedCollisionProfile(
        root_body="base",
        hip_links=tuple(f"{leg}_hip" for leg in _UNITREE_LEGS),
        upper_links=tuple(f"{leg}_thigh" for leg in _UNITREE_LEGS),
        lower_links=tuple(f"{leg}_calf" for leg in _UNITREE_LEGS),
        torso_boxes=(
            ("core", (0.25, 0.16, 0.15), (0.01, 0.0, -0.004)),
            ("front", (0.07, 0.18, 0.17), (0.292, 0.0, -0.004)),
        ),
        hip_radius=0.045,
        upper_end=(0.0, 0.0, -0.213),
        upper_radius=0.041,
        lower_end=(0.0, 0.0, -0.213),
        lower_radius=0.026,
    ),
}


def repair_quadruped_colliders(env, profile_name: str | tuple[str, str]) -> None:
    """Inject measured analytic proxies into env_0 before environment cloning."""
    import os

    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    profile_names = (profile_name, profile_name) if isinstance(profile_name, str) else profile_name
    if len(profile_names) != 2:
        raise ValueError(f"Expected two collision profiles, got {profile_names!r}")
    stage = env.sim.stage
    contact_offset = 0.012
    show = os.environ.get("ISAACLAB_SUMO_SHOW_PROXY_GEOMETRY", "0") == "1"

    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        if ("Robot_0" in path or "Robot_1" in path) and prim.IsInstance():
            prim.SetInstanceable(False)

    def finish(shape, path: Sdf.Path) -> bool:
        color = Gf.Vec3f(0.1, 0.9, 1.0) if "Robot_0" in str(path) else Gf.Vec3f(1.0, 0.8, 0.1)
        shape.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if show else UsdGeom.Tokens.invisible)
        if show:
            shape.GetDisplayColorAttr().Set([color])
            shape.GetDisplayOpacityAttr().Set([0.55])
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        api = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        api.GetContactOffsetAttr().Set(contact_offset)
        api.GetRestOffsetAttr().Set(0.0)
        return True

    def add_box(path: Sdf.Path, size, translation) -> bool:
        if stage.GetPrimAtPath(path).IsValid():
            return False
        shape = UsdGeom.Cube.Define(stage, path)
        shape.GetSizeAttr().Set(1.0)
        shape.AddTranslateOp().Set(Gf.Vec3d(*translation))
        shape.AddScaleOp().Set(Gf.Vec3d(*size))
        return finish(shape, path)

    def add_sphere(path: Sdf.Path, radius: float) -> bool:
        if stage.GetPrimAtPath(path).IsValid():
            return False
        shape = UsdGeom.Sphere.Define(stage, path)
        shape.GetRadiusAttr().Set(radius)
        return finish(shape, path)

    def add_capsule(path: Sdf.Path, radius: float, height: float, axis: str, translation) -> bool:
        if stage.GetPrimAtPath(path).IsValid():
            return False
        axis_token = {"x": UsdGeom.Tokens.x, "y": UsdGeom.Tokens.y, "z": UsdGeom.Tokens.z}[axis]
        shape = UsdGeom.Capsule.Define(stage, path)
        shape.GetRadiusAttr().Set(radius)
        shape.GetHeightAttr().Set(height)
        shape.GetAxisAttr().Set(axis_token)
        shape.AddTranslateOp().Set(Gf.Vec3d(*translation))
        return finish(shape, path)

    def add_capsule_between(
        path: Sdf.Path,
        end,
        radius: float,
        start_clearance: float | None = None,
        end_clearance: float | None = None,
    ) -> bool:
        if stage.GetPrimAtPath(path).IsValid():
            return False
        start_v = Gf.Vec3d(0.0, 0.0, 0.0)
        end_v = Gf.Vec3d(*end)
        delta = end_v - start_v
        length = delta.GetLength()
        direction = delta / length
        start_clearance = max(0.035, radius) if start_clearance is None else start_clearance
        end_clearance = max(0.035, radius) if end_clearance is None else end_clearance
        start_v += start_clearance * direction
        end_v -= end_clearance * direction
        covered = (end_v - start_v).GetLength()
        shape = UsdGeom.Capsule.Define(stage, path)
        shape.GetRadiusAttr().Set(radius)
        shape.GetHeightAttr().Set(max(0.001, covered - 2.0 * radius))
        shape.GetAxisAttr().Set(UsdGeom.Tokens.z)
        shape.AddTranslateOp().Set(0.5 * (start_v + end_v))
        rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), direction)
        shape.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rotation.GetQuat())
        return finish(shape, path)

    collision_count = rigid_count = added = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "Robot_0" not in path and "Robot_1" not in path:
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            api = PhysxSchema.PhysxCollisionAPI(prim) if prim.HasAPI(PhysxSchema.PhysxCollisionAPI) else PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.GetContactOffsetAttr().Set(contact_offset)
            api.GetRestOffsetAttr().Set(0.0)
            collision_count += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = PhysxSchema.PhysxRigidBodyAPI(prim) if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI) else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            api.GetEnableCCDAttr().Set(True)
            rigid_count += 1

    per_robot_added = {}
    for robot_tag, current_profile_name in zip(("Robot_0", "Robot_1"), profile_names):
        profile = PROFILES[current_profile_name]
        root = f"/World/envs/env_0/{robot_tag}"
        if not stage.GetPrimAtPath(root).IsValid():
            continue
        before = added
        for name, size, translation in profile.torso_boxes:
            added += int(add_box(Sdf.Path(f"{root}/{profile.root_body}/sumo_torso_{name}"), size, translation))
        for name, radius, height, axis, translation in profile.torso_capsules:
            added += int(
                add_capsule(
                    Sdf.Path(f"{root}/{profile.root_body}/sumo_torso_{name}"),
                    radius,
                    height,
                    axis,
                    translation,
                )
            )
        for link in profile.hip_links:
            added += int(add_sphere(Sdf.Path(f"{root}/{link}/sumo_hip_sphere"), profile.hip_radius))
        for link in profile.upper_links:
            path = Sdf.Path(f"{root}/{link}/sumo_upper_capsule")
            if profile.upper_capsule is None:
                added += int(add_capsule_between(path, profile.upper_end, profile.upper_radius))
            else:
                radius, height, axis, translation = profile.upper_capsule
                added += int(add_capsule(path, radius, height, axis, translation))
        lower_ends = profile.lower_ends or (profile.lower_end,) * len(profile.lower_links)
        for link, lower_end in zip(profile.lower_links, lower_ends):
            added += int(
                add_capsule_between(
                    Sdf.Path(f"{root}/{link}/sumo_lower_capsule"),
                    lower_end,
                    profile.lower_radius,
                    *profile.lower_clearances,
                )
            )
        per_robot_added[robot_tag] = added - before

    print(
        f"[sumo-{'-vs-'.join(profile_names)}] collision prims={collision_count}; "
        f"rigid links={rigid_count}; measured proxies added={added} {per_robot_added}; "
        f"CCD enabled; contact_offset={contact_offset}",
        flush=True,
    )
