# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2 sumo ring: one ANYMAL-C vs one ANYMAL-C (1v1, no multi-robot teams)."""

from __future__ import annotations

import copy
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_rotate_inverse, subtract_frame_transforms

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from .flip_detection import inverted_back_penalty, mounted_flip_candidates, sustained_inversion_update
from ..heading import backward_facing_penalty

# Proprio(45) + opponent position(3) + opponent velocity(3) + opponent heading(3)
# + distance from centre(1) + ring radius(1) = 56. The shape remains compatible
# with the earlier checkpoints, while the formerly empty slots now make the
# other robot's motion observable.
_OBS_DIM = 56


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
class SumoStage2Anymal1v1EnvCfg(DirectMARLEnvCfg):
    """Stage 2 ring: push opponent out or survive; two quadrupeds only."""

    decimation = 4
    episode_length_s = 20.0

    action_spaces = {"robot_0": 12, "robot_1": 12}
    observation_spaces = {"robot_0": _OBS_DIM, "robot_1": _OBS_DIM}

    state_space = 0
    state_spaces = {"robot_0": 0, "robot_1": 0}

    possible_agents = ["robot_0", "robot_1"]

    # HARL adversarial API: one agent per side (1v1).
    teams = {"team_0": ["robot_0"], "team_1": ["robot_1"]}
    root_body_pattern = "base"
    root_body_patterns = {}
    collision_profile = "anymal_c"
    forward_yaw_offsets = {}

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            # CCD prevents robots from tunnelling through each other when moving fast.
            enable_ccd=True,
            # Scene-level solver iteration floors — the per-articulation count below
            # drives the actual resolution quality.
            min_position_iteration_count=8,
            min_velocity_iteration_count=4,
            # Larger GPU buffers for inter-robot contact pairs across many parallel envs.
            gpu_max_rigid_contact_count=2**24,
            gpu_found_lost_pairs_capacity=2**22,
        ),
    )

    events: EventCfg = EventCfg()

    # ANYMAL-C USD uses instanced collision prims, so CollisionPropertiesCfg cannot
    # repair its incomplete collision geometry through the spawn config. The scene
    # setup below de-instances env_0, adds conservative proxy geometry, and enables
    # CCD on each rigid body before Isaac Lab clones the environments.
    _sumo_rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=10.0,
    )
    _sumo_articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        # The default ANYmal-C value is 4; sumo needs substantially tighter contact resolution.
        solver_position_iteration_count=24,
        solver_velocity_iteration_count=4,
    )

    robot_0: ArticulationCfg = ANYMAL_C_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_0",
        spawn=ANYMAL_C_CFG.spawn.replace(
            rigid_props=_sumo_rigid_props,
            articulation_props=_sumo_articulation_props,
        ),
    )
    robot_0.init_state.rot = get_quaternion_tuple_from_xyz(0, 0, torch.pi / 2)
    robot_0.init_state.pos = (0.0, 1.0, 0.56)

    robot_1: ArticulationCfg = ANYMAL_C_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_1",
        spawn=ANYMAL_C_CFG.spawn.replace(
            rigid_props=_sumo_rigid_props,
            articulation_props=_sumo_articulation_props,
        ),
    )
    robot_1.init_state.rot = get_quaternion_tuple_from_xyz(0, 0, torch.pi / 2)
    robot_1.init_state.pos = (0.0, -1.0, 0.56)

    action_scale = 0.5
    ring_radius_min = 3
    ring_radius_max = 4.5
    reward_scale = 10
    time_penalty = -0.01

    # Dense potentials retain the terminal zero-sum objective while teaching
    # the useful intermediate behaviours that a sparse ring-out reward cannot.
    ring_progress_reward_scale = 4.0
    closing_progress_reward_scale = 0.75
    action_rate_reward_scale = -0.002
    flat_orientation_reward_scale = -0.02
    # A timeout can never earn this term: it only penalizes presenting the rear
    # half of the robot to the opponent and gives zero reward when nose-on.
    backward_facing_penalty_scale = -0.05

    # The XY-only flat-orientation term above is zero both upright and exactly
    # upside down. Penalize the inverted half explicitly, and count a robot
    # that remains on its back for 0.3 s as having lost. This permits a brief
    # recovery attempt but prevents inverted robots waiting for the timeout.
    inverted_back_penalty_scale = -0.2
    back_loss_up_max = -0.25
    back_loss_hold_steps = 15

    # A one-shot, zero-sum bonus for an upright robot mounting an inverted
    # opponent. Requiring several consecutive control steps rejects momentary
    # contact and the per-episode latch prevents reward farming.
    flip_reward_scale = 25.0
    flip_victim_up_max = -0.25
    flip_attacker_up_min = 0.35
    flip_max_horizontal_distance = 0.65
    flip_min_height_advantage = 0.16
    flip_hold_steps = 5

    spawn_separation_min = 1.8
    spawn_separation_max = 2.4

    # Clipping is prevented at the physics level with per-body CCD, real proxy
    # capsules for the missing torso/thigh geometry, and a tighter contact solver.

    env_spacing = 10.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=ring_radius_max * 2, replicate_physics=True
    )

    # Center viewport on env 0 (first ring); default world camera often misses cloned envs.
    viewer: ViewerCfg = ViewerCfg(
        origin_type="env",
        env_index=0,
        eye=(8.0, 0.0, 5.0),
        lookat=(0.0, 0.0, 0.6),
    )


class SumoStage2Anymal1v1Env(DirectMARLEnv):
    cfg: SumoStage2Anymal1v1EnvCfg

    def __init__(
        self,
        cfg: SumoStage2Anymal1v1EnvCfg,
        render_mode: str | None = None,
        headless: bool | None = None,
        debug: bool | None = False,
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
        self._prev_ring_advantage = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._prev_opponent_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._flip_hold_counter = torch.zeros(
            (self.num_envs, 2), dtype=torch.int32, device=self.device
        )
        self._flip_awarded = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._back_hold_counter = torch.zeros(
            (self.num_envs, 2), dtype=torch.int32, device=self.device
        )
        self._back_lost = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )

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

        team_dot_markers = {
            "blue": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.8)),
            ),
            "red": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
        }
        self.team_markers = VisualizationMarkers(
            VisualizationMarkersCfg(prim_path="/World/TeamDots", markers=team_dot_markers)
        )

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "team_0_terminal_reward",
                "team_1_terminal_reward",
                "team_0_ring_progress_reward",
                "team_1_ring_progress_reward",
                "team_0_closing_progress_reward",
                "team_1_closing_progress_reward",
                "team_0_action_rate_reward",
                "team_1_action_rate_reward",
                "team_0_flat_orientation_reward",
                "team_1_flat_orientation_reward",
                "team_0_backward_facing_penalty",
                "team_1_backward_facing_penalty",
                "team_0_inverted_back_penalty",
                "team_1_inverted_back_penalty",
                "team_0_flip_reward",
                "team_1_flip_reward",
            ]
        }

        self.base_ids = {}

        for robot_id, contact_sensor in self.contact_sensors.items():
            root_pattern = self.cfg.root_body_patterns.get(robot_id, self.cfg.root_body_pattern)
            _base_id, _ = contact_sensor.find_bodies(root_pattern)
            self.base_ids[robot_id] = _base_id

        self._opponent = {"robot_0": "robot_1", "robot_1": "robot_0"}

    @torch.no_grad()
    def _draw_team_dots(self):
        positions, indices = [], []
        for robot_id, robot in self.robots.items():
            pos = robot.data.root_pos_w.clone()
            pos[:, 2] += 0.5
            positions.append(pos)

            team_idx = 0 if robot_id == "robot_0" else 1
            indices.append(torch.full((self.num_envs,), team_idx, device=self.device))

        marker_positions = torch.cat(positions, dim=0)
        marker_indices = torch.cat(indices, dim=0)
        marker_orientations = torch.zeros((marker_positions.shape[0], 4), device=self.device)
        marker_orientations[:, 0] = 1.0
        marker_scales = torch.ones((marker_positions.shape[0], 3), device=self.device)

        self.team_markers.visualize(
            marker_positions, marker_orientations, scales=marker_scales, marker_indices=marker_indices
        )

    @torch.no_grad()
    def _draw_ring_markers(self):
        device = self.device
        E = self.num_envs
        N = self._ring_segments
        z0 = float(getattr(self.cfg, "ring_z", 0.0))

        theta = torch.linspace(0, 2 * torch.pi, steps=N + 1, device=device)[:-1]
        cs = torch.cos(theta)
        sn = torch.sin(theta)

        origins_xy = self.scene.env_origins[:, :2].to(device)
        radii = self.ring_radius.view(E, 1)

        pos_chunks = []
        idx_chunks = []

        z_col = torch.full((E, 1), z0, device=device)

        for k in range(N):
            dir_k = torch.tensor([cs[k].item(), sn[k].item()], device=device)
            xy_k = origins_xy + radii * dir_k
            pos_k = torch.cat([xy_k, z_col], dim=1)
            pos_chunks.append(pos_k)
            idx_chunks.append(k * torch.ones(E, dtype=torch.long, device=device))

        marker_positions = torch.cat(pos_chunks, dim=0)
        marker_indices = torch.cat(idx_chunks, dim=0)

        marker_scales = torch.ones((marker_positions.shape[0], 3), device=device)

        marker_orientations = torch.zeros((marker_positions.shape[0], 4), device=device)
        marker_orientations[:, 0] = 1.0

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
                size=(500.0, 500.0),
                color=(0.2, 0.2, 0.2),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        self.robots = {}

        self.num_robots = 2
        self.contact_sensors = {}
        for i in range(self.num_robots):
            self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
            self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]

        self.animals = {"robot_0": self.robots["robot_0"], "robot_1": self.robots["robot_1"]}

        # Break USD instanceable on robot collision containers to set contact_offset
        # and to add missing collision shapes to uncovered links.
        #
        # ANYmal-C collision geometry gaps found by inspection:
        #   base     : Cylinder r=0.075 m, h=0.750 m — visual torso is ~0.15 m wide,
        #              leaving a ~7.5 cm gap on each side where legs pass through.
        #   THIGH ×4 : Cylinder r=0.060 m, h=0.080 m — an 8 cm joint-cap disk, NOT a
        #              thigh arm; the full thigh length has no collision coverage.
        #   SHANK ×4 : Cylinder r=0.018 m, h=0.200 m — thin rod, close to visual size.
        #   FOOT  ×4 : Sphere   r=0.030 m — correct.
        #   HAA   ×4 : (LF/RF/LH/RH_HIP) — NO collision shape at all!
        #
        # Fix: add conservative torso/thigh capsules and HAA spheres, use a small
        # real contact offset, and enable CCD on every rigid link.
        # This always runs; num_envs is controlled at launch time to fit VRAM
        # (e.g. --num_envs 3500 on a 24 GiB card leaves ~3 GiB headroom).
        self._deinstance_and_set_contact_offsets()

        # Sensor-aware task variants install their onboard sensors after the
        # robot prims exist but before env_0 is cloned.
        setup_onboard_sensors = getattr(self, "_setup_onboard_sensors", None)
        if setup_onboard_sensors is not None:
            setup_onboard_sensors()

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _deinstance_and_set_contact_offsets(self) -> None:
        """Fix ANYmal-C collision geometry gaps, then allow clone_environments().

        ANYmal-C has three collision problems that allow legs to pass through
        torsos and other legs (found by USD inspection):
          1. The base cylinder is long enough but much narrower than its visual.
          2. Each thigh collider covers only a short distal joint cap.
          3. The four HAA hip links have no collision shape at all.

        Contact offset is not geometry, so increasing it alone gives early-looking
        contacts but does not stop overlap at rest. Here we add actual analytic
        proxy geometry and keep the contact offset close to the physical surface.

        USD edits happen on env_0 before clone_environments() so every clone
        inherits the fixes automatically.
        """
        try:
            from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
            stage = self.sim.stage

            # Contact offsets start collision generation; they do not expand a
            # too-small collider. Keep them physically modest and add actual
            # proxy geometry below for the missing volume.
            DEFAULT_OFFSET = 0.015
            REST_OFFSET = 0.0

            # Pass 1: break USD instancing on every robot collision container prim
            n_deinstanced = 0
            for prim in stage.Traverse():
                if not prim.IsInstance():
                    continue
                path = str(prim.GetPath())
                if "Robot_0" in path or "Robot_1" in path:
                    prim.SetInstanceable(False)
                    n_deinstanced += 1

            # Optional one-env geometry dump used when tuning the sumo proxy
            # colliders. Keep this behind an environment variable so normal
            # training does not pay for verbose USD inspection.
            import os

            show_proxy_geometry = os.environ.get("ISAACLAB_SUMO_SHOW_PROXY_GEOMETRY", "0") == "1"

            def style_proxy(gprim, path: Sdf.Path) -> None:
                """Hide proxies normally, or tint them for collision audits."""
                if not show_proxy_geometry:
                    gprim.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                    return
                gprim.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
                color = Gf.Vec3f(0.1, 0.9, 1.0) if "Robot_0" in str(path) else Gf.Vec3f(1.0, 0.8, 0.1)
                gprim.GetDisplayColorAttr().Set([color])
                gprim.GetDisplayOpacityAttr().Set([0.55])

            if os.environ.get("ISAACLAB_SUMO_COLLIDER_DIAGNOSTICS") == "1":
                for prim in stage.Traverse():
                    path = str(prim.GetPath())
                    if "Robot_0" not in path:
                        continue
                    if prim.IsA(UsdPhysics.Joint):
                        joint = UsdPhysics.Joint(prim)
                        print(
                            f"[sumo-joint] {path} type={prim.GetTypeName()} "
                            f"body0={joint.GetBody0Rel().GetTargets()} "
                            f"body1={joint.GetBody1Rel().GetTargets()} "
                            f"local_pos0={joint.GetLocalPos0Attr().Get()} "
                            f"local_pos1={joint.GetLocalPos1Attr().Get()}",
                            flush=True,
                        )
                        continue
                    relevant_link = any(
                        name in path for name in ("SHANK", "THIGH", "FOOT", "/base/")
                    )
                    if not relevant_link or not prim.IsA(UsdGeom.Boundable):
                        continue
                    attrs = {}
                    for attr_name in ("radius", "height", "axis", "size", "extent"):
                        attr = prim.GetAttribute(attr_name)
                        if attr and attr.IsValid():
                            attrs[attr_name] = attr.Get()
                    local_xform = UsdGeom.Xformable(prim).GetLocalTransformation()
                    print(
                        f"[sumo-geometry] {path} type={prim.GetTypeName()} "
                        f"collision={prim.HasAPI(UsdPhysics.CollisionAPI)} attrs={attrs} "
                        f"local_xform={local_xform}",
                        flush=True,
                    )

            # Pass 2: use realistic offsets and enable CCD on every rigid link.
            n_set = 0
            n_ccd = 0
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if "Robot_0" not in path and "Robot_1" not in path:
                    continue
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                        col_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    else:
                        col_api = PhysxSchema.PhysxCollisionAPI(prim)
                    col_api.GetContactOffsetAttr().Set(DEFAULT_OFFSET)
                    col_api.GetRestOffsetAttr().Set(REST_OFFSET)
                    n_set += 1
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    else:
                        rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
                    rigid_api.GetEnableCCDAttr().Set(True)
                    n_ccd += 1

            def add_capsule(
                path: Sdf.Path,
                radius: float,
                height: float,
                axis,
                translation: tuple[float, float, float],
            ) -> bool:
                """Create an invisible analytic collision capsule."""
                if stage.GetPrimAtPath(path).IsValid():
                    return False
                capsule = UsdGeom.Capsule.Define(stage, path)
                capsule.GetRadiusAttr().Set(radius)
                capsule.GetHeightAttr().Set(height)
                capsule.GetAxisAttr().Set(axis)
                style_proxy(capsule, path)
                if translation != (0.0, 0.0, 0.0):
                    capsule.AddTranslateOp().Set(Gf.Vec3d(*translation))
                UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
                pxcol = PhysxSchema.PhysxCollisionAPI.Apply(capsule.GetPrim())
                pxcol.GetContactOffsetAttr().Set(DEFAULT_OFFSET)
                pxcol.GetRestOffsetAttr().Set(REST_OFFSET)
                return True

            def add_sphere(path: Sdf.Path, radius: float) -> bool:
                """Create an invisible analytic collision sphere."""
                if stage.GetPrimAtPath(path).IsValid():
                    return False
                sphere = UsdGeom.Sphere.Define(stage, path)
                sphere.GetRadiusAttr().Set(radius)
                style_proxy(sphere, path)
                UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
                pxcol = PhysxSchema.PhysxCollisionAPI.Apply(sphere.GetPrim())
                pxcol.GetContactOffsetAttr().Set(DEFAULT_OFFSET)
                pxcol.GetRestOffsetAttr().Set(REST_OFFSET)
                return True

            def add_capsule_between(
                path: Sdf.Path,
                start: tuple[float, float, float],
                end: tuple[float, float, float],
                radius: float,
                start_clearance: float = 0.055,
                end_clearance: float = 0.045,
            ) -> bool:
                """Create a capsule whose tips follow a measured link-local segment."""
                if stage.GetPrimAtPath(path).IsValid():
                    return False
                start_v = Gf.Vec3d(*start)
                end_v = Gf.Vec3d(*end)
                delta = end_v - start_v
                segment_length = delta.GetLength()
                if segment_length <= start_clearance + end_clearance + 2.0 * radius:
                    return False
                direction = delta / segment_length
                start_v += start_clearance * direction
                end_v -= end_clearance * direction
                covered_length = (end_v - start_v).GetLength()

                capsule = UsdGeom.Capsule.Define(stage, path)
                capsule.GetRadiusAttr().Set(radius)
                capsule.GetHeightAttr().Set(covered_length - 2.0 * radius)
                capsule.GetAxisAttr().Set(UsdGeom.Tokens.z)
                style_proxy(capsule, path)
                capsule.AddTranslateOp().Set(0.5 * (start_v + end_v))
                align_z = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), direction)
                capsule.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(align_z.GetQuat())
                UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
                pxcol = PhysxSchema.PhysxCollisionAPI.Apply(capsule.GetPrim())
                pxcol.GetContactOffsetAttr().Set(DEFAULT_OFFSET)
                pxcol.GetRestOffsetAttr().Set(REST_OFFSET)
                return True

            def add_box(
                path: Sdf.Path,
                size: tuple[float, float, float],
                translation: tuple[float, float, float],
            ) -> bool:
                """Create an invisible analytic box collider with XYZ dimensions."""
                if stage.GetPrimAtPath(path).IsValid():
                    return False
                cube = UsdGeom.Cube.Define(stage, path)
                cube.GetSizeAttr().Set(1.0)
                style_proxy(cube, path)
                cube.AddTranslateOp().Set(Gf.Vec3d(*translation))
                cube.AddScaleOp().Set(Gf.Vec3d(*size))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                pxcol = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
                pxcol.GetContactOffsetAttr().Set(DEFAULT_OFFSET)
                pxcol.GetRestOffsetAttr().Set(REST_OFFSET)
                return True

            # Pass 3: fill actual holes in the asset's analytic colliders.
            # The base's stock cylinder is 0.75 m long but only 0.15 m wide,
            # while the rendered chassis/bumper union is roughly 1.05 x 0.32 m.
            # End boxes cover the real front/rear bumper extents; the capsule
            # covers the central core. A full-width central box was rejected
            # because it overlaps the articulated hips in the standing pose.
            # Each stock thigh
            # cylinder is merely a distal knee cap at z=-0.285, so a capsule is
            # added along the otherwise uncovered upper leg.
            HAA_LINKS = ("LF_HIP", "RF_HIP", "LH_HIP", "RH_HIP")
            # These switches are deliberately diagnostic-only. They let us
            # isolate any unstable self-contact caused by a newly added group
            # without maintaining separate USD/code variants.
            full_torso_proxy = os.environ.get("ISAACLAB_SUMO_FULL_TORSO_PROXY", "1") == "1"
            chassis_proxy = os.environ.get("ISAACLAB_SUMO_CHASSIS_PROXY", "0") == "1"
            shank_proxy = os.environ.get("ISAACLAB_SUMO_SHANK_PROXY", "1") == "1"
            hip_proxy = os.environ.get("ISAACLAB_SUMO_HIP_PROXY", "1") == "1"
            n_added = 0
            for robot_tag in ("Robot_0", "Robot_1"):
                robot_root = stage.GetPrimAtPath(f"/World/envs/env_0/{robot_tag}")
                if not robot_root or not robot_root.IsValid():
                    continue
                base_path = Sdf.Path(f"/World/envs/env_0/{robot_tag}/base/sumo_torso_capsule")
                n_added += int(add_capsule(base_path, 0.14, 0.47, UsdGeom.Tokens.x, (0.0, 0.0, 0.0)))
                if full_torso_proxy:
                    if chassis_proxy:
                        chassis_path = Sdf.Path(
                            f"/World/envs/env_0/{robot_tag}/base/sumo_torso_chassis_box"
                        )
                        n_added += int(add_box(chassis_path, (0.83, 0.31, 0.20), (0.0, 0.0, -0.005)))
                    face_specs = (("front", 0.468, (0.12, 0.31, 0.18)), ("rear", -0.468, (0.12, 0.31, 0.18)))
                else:
                    # Previously tested stable geometry, retained only as the
                    # control condition for collider isolation.
                    face_specs = (("front", 0.335, (0.10, 0.25, 0.18)), ("rear", -0.335, (0.10, 0.25, 0.18)))
                for face_name, x_pos, face_size in face_specs:
                    face_path = Sdf.Path(
                        f"/World/envs/env_0/{robot_tag}/base/sumo_torso_{face_name}_box"
                    )
                    n_added += int(add_box(face_path, face_size, (x_pos, 0.0, -0.015)))

                for leg in ("LF", "RF", "LH", "RH"):
                    thigh_path = Sdf.Path(
                        f"/World/envs/env_0/{robot_tag}/{leg}_THIGH/sumo_thigh_capsule"
                    )
                    n_added += int(
                        add_capsule(thigh_path, 0.0525, 0.18, UsdGeom.Tokens.z, (0.0, 0.0, -0.1425))
                    )

                    # The visible lower-leg rod spans the SHANK link from the
                    # knee origin to the measured fixed-foot joint below. Put
                    # the proxy on SHANK too, avoiding self-contact between
                    # overlapping shapes on two fixed-together rigid links.
                    if shank_proxy:
                        lower_leg_path = Sdf.Path(
                            f"/World/envs/env_0/{robot_tag}/{leg}_SHANK/sumo_full_shank_capsule"
                        )
                        shank_y = -0.08795 if leg in ("LF", "RH") else 0.08795
                        n_added += int(
                            add_capsule_between(
                                lower_leg_path,
                                (0.0, 0.0, 0.0),
                                (0.01305, shank_y, -0.33797),
                                0.032,
                            )
                        )

                if hip_proxy:
                    for haa in HAA_LINKS:
                        link_path_str = f"/World/envs/env_0/{robot_tag}/{haa}"
                        link_prim = stage.GetPrimAtPath(link_path_str)
                        if not link_prim or not link_prim.IsValid():
                            continue
                        sphere_path = Sdf.Path(f"{link_path_str}/haa_collision_sphere")
                        n_added += int(add_sphere(sphere_path, 0.07))

            print(
                f"[sumo] De-instanced {n_deinstanced} collision containers; "
                f"contact_offset set on {n_set} prims; CCD enabled on {n_ccd} rigid links; "
                f"proxy colliders injected: {n_added} "
                f"(full_torso={full_torso_proxy}, chassis={chassis_proxy}, "
                f"shank={shank_proxy}, hip={hip_proxy}).",
                flush=True,
            )
        except Exception as exc:
            import traceback
            print(f"[sumo] WARNING: _deinstance_and_set_contact_offsets failed: {exc}", flush=True)
            traceback.print_exc()

    def _pre_physics_step(self, actions: dict) -> None:
        self.processed_actions = {}
        self.previous_actions = {agent: action.clone() for agent, action in self.actions.items()}
        self.actions = copy.deepcopy(actions)
        for robot_id, robot in self.animals.items():
            action_scale = getattr(self.cfg, "action_scales", {}).get(robot_id, self.cfg.action_scale)
            self.processed_actions[robot_id] = action_scale * actions[robot_id] + robot.data.default_joint_pos

    def _apply_action(self) -> None:
        for robot_id, robot in self.animals.items():
            robot.set_joint_position_target(self.processed_actions[robot_id])

    def _get_observations(self) -> dict:
        obs: dict = {}
        rcol = self.ring_radius.view(-1, 1)

        for team, agents in self.cfg.teams.items():
            obs[team] = {}
            for robot_id in agents:
                oid = self._opponent[robot_id]
                enemy_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3],
                    self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.robots[oid].data.root_pos_w,
                )

                relative_velocity_w = (
                    self.robots[oid].data.root_lin_vel_w
                    - self.robots[robot_id].data.root_lin_vel_w
                )
                enemy_velocity = quat_rotate_inverse(
                    self.robots[robot_id].data.root_quat_w, relative_velocity_w
                )

                forward_b = torch.zeros(self.num_envs, 3, device=self.device)
                forward_b[:, 0] = 1.0
                enemy_forward_w = quat_apply(self.robots[oid].data.root_quat_w, forward_b)
                enemy_heading = quat_rotate_inverse(
                    self.robots[robot_id].data.root_quat_w, enemy_forward_w
                )
                # Stage 1 reserves these three slots. In Stage 2, keep two
                # relative-heading components and use the third for the
                # opponent's world-up alignment: +1 upright, -1 inverted.
                enemy_up_alignment = -self.robots[oid].data.projected_gravity_b[:, 2:3]
                enemy_pose = torch.cat((enemy_heading[:, :2], enemy_up_alignment), dim=-1)

                dist_center = torch.linalg.norm(
                    self.robots[robot_id].data.root_pos_w[:, :2]
                    - self.scene.env_origins[:, :2],
                    dim=-1,
                    keepdim=True,
                )
                robot_obs = torch.cat(
                    [
                        self.robots[robot_id].data.root_lin_vel_b,        # 3
                        self.robots[robot_id].data.root_ang_vel_b,        # 3
                        self.robots[robot_id].data.projected_gravity_b,   # 3
                        self.robots[robot_id].data.joint_pos - self.robots[robot_id].data.default_joint_pos,  # 12
                        self.robots[robot_id].data.joint_vel,             # 12
                        self.actions[robot_id],                           # 12
                        enemy_pos,                                         # 3
                        enemy_velocity,                                    # 3
                        enemy_pose,                                        # 3
                        dist_center,                                       # 1
                        rcol,                                              # 1
                    ],
                    dim=-1,
                )  # total = 56

                robot_obs = torch.nan_to_num(robot_obs, nan=0.0, posinf=1e6, neginf=-1e6)

                obs[team][robot_id] = robot_obs

        return obs

    def _update_flip_events(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one-shot mounted-flip events for robot 0 and robot 1."""
        pos_0 = self.robots["robot_0"].data.root_com_pos_w
        pos_1 = self.robots["robot_1"].data.root_com_pos_w
        candidates = mounted_flip_candidates(
            pos_0,
            pos_1,
            self.robots["robot_0"].data.projected_gravity_b,
            self.robots["robot_1"].data.projected_gravity_b,
            victim_up_max=self.cfg.flip_victim_up_max,
            attacker_up_min=self.cfg.flip_attacker_up_min,
            max_horizontal_distance=self.cfg.flip_max_horizontal_distance,
            min_height_advantage=self.cfg.flip_min_height_advantage,
        )
        self._flip_hold_counter = torch.where(
            candidates,
            self._flip_hold_counter + 1,
            torch.zeros_like(self._flip_hold_counter),
        )
        events = (
            (self._flip_hold_counter >= self.cfg.flip_hold_steps)
            & candidates
            & ~self._flip_awarded
        )
        self._flip_awarded |= events
        return events[:, 0], events[:, 1]

    def _get_rewards(self) -> dict:
        self._draw_team_dots()

        fallen = {
            "team_0": (self.robots["robot_0"].data.root_com_pos_w[:, 2] < 0.1)
            | self._back_lost[:, 0],
            "team_1": (self.robots["robot_1"].data.root_com_pos_w[:, 2] < 0.1)
            | self._back_lost[:, 1],
        }

        out = self._robots_out_of_ring()
        team0_out = out["robot_0"]
        team1_out = out["robot_1"]

        team0_lost = team0_out | fallen["team_0"]
        team1_lost = team1_out | fallen["team_1"]
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        unresolved_timeout = time_out & ~team0_lost & ~team1_lost

        terminal_0 = self.cfg.reward_scale * (
            (team1_lost & ~team0_lost).float()
            - (team0_lost & ~team1_lost).float()
            - unresolved_timeout.float()
        )
        terminal_1 = self.cfg.reward_scale * (
            (team0_lost & ~team1_lost).float()
            - (team1_lost & ~team0_lost).float()
            - unresolved_timeout.float()
        )

        env_xy = self.scene.env_origins[:, :2]
        radius_0 = torch.linalg.norm(self.robots["robot_0"].data.root_pos_w[:, :2] - env_xy, dim=-1)
        radius_1 = torch.linalg.norm(self.robots["robot_1"].data.root_pos_w[:, :2] - env_xy, dim=-1)
        ring_advantage = radius_1 - radius_0
        ring_delta = (ring_advantage - self._prev_ring_advantage).clamp(-0.1, 0.1)

        opponent_distance = torch.linalg.norm(
            self.robots["robot_1"].data.root_pos_w[:, :2]
            - self.robots["robot_0"].data.root_pos_w[:, :2],
            dim=-1,
        )
        closing_delta = (self._prev_opponent_distance - opponent_distance).clamp(-0.1, 0.1)

        ring_0 = self.cfg.ring_progress_reward_scale * ring_delta
        ring_1 = -ring_0
        closing_0 = self.cfg.closing_progress_reward_scale * closing_delta
        closing_1 = closing_0

        rate_0 = self.cfg.action_rate_reward_scale * torch.sum(
            torch.square(self.actions["robot_0"] - self.previous_actions["robot_0"]), dim=-1
        )
        rate_1 = self.cfg.action_rate_reward_scale * torch.sum(
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
            self.robots["robot_1"].data.root_pos_w,
            self.cfg.backward_facing_penalty_scale,
            self.cfg.forward_yaw_offsets.get("robot_0", 0.0),
        )
        facing_1 = backward_facing_penalty(
            self.robots["robot_1"].data.root_quat_w,
            self.robots["robot_1"].data.root_pos_w,
            self.robots["robot_0"].data.root_pos_w,
            self.cfg.backward_facing_penalty_scale,
            self.cfg.forward_yaw_offsets.get("robot_1", 0.0),
        )
        back_0 = inverted_back_penalty(
            self.robots["robot_0"].data.projected_gravity_b,
            scale=self.cfg.inverted_back_penalty_scale,
        )
        back_1 = inverted_back_penalty(
            self.robots["robot_1"].data.projected_gravity_b,
            scale=self.cfg.inverted_back_penalty_scale,
        )
        robot_0_flip, robot_1_flip = self._update_flip_events()
        flip_0 = self.cfg.flip_reward_scale * (robot_0_flip.float() - robot_1_flip.float())
        flip_1 = -flip_0

        components = {
            "team_0_terminal_reward": terminal_0,
            "team_1_terminal_reward": terminal_1,
            "team_0_ring_progress_reward": ring_0,
            "team_1_ring_progress_reward": ring_1,
            "team_0_closing_progress_reward": closing_0,
            "team_1_closing_progress_reward": closing_1,
            "team_0_action_rate_reward": rate_0,
            "team_1_action_rate_reward": rate_1,
            "team_0_flat_orientation_reward": flat_0,
            "team_1_flat_orientation_reward": flat_1,
            "team_0_backward_facing_penalty": facing_0,
            "team_1_backward_facing_penalty": facing_1,
            "team_0_inverted_back_penalty": back_0,
            "team_1_inverted_back_penalty": back_1,
            "team_0_flip_reward": flip_0,
            "team_1_flip_reward": flip_1,
        }
        for key, value in components.items():
            self._episode_sums[key] += value

        self._prev_ring_advantage.copy_(ring_advantage)
        self._prev_opponent_distance.copy_(opponent_distance)

        return {
            "team_0": terminal_0 + ring_0 + closing_0 + rate_0 + flat_0 + facing_0 + back_0 + flip_0,
            "team_1": terminal_1 + ring_1 + closing_1 + rate_1 + flat_1 + facing_1 + back_1 + flip_1,
        }

    def _robots_out_of_ring(self) -> dict[str, torch.Tensor]:
        env_xy = self.scene.env_origins[:, :2].to(self.device)
        out = {}
        for robot_id in self.robots.keys():
            pos_xy = self.robots[robot_id].data.root_pos_w[:, :2]
            dist = torch.linalg.norm(pos_xy - env_xy, dim=1)
            out[robot_id] = dist > self.ring_radius
        return out

    def _get_dones(self) -> tuple[dict, dict]:
        out_map = self._robots_out_of_ring()
        team0_out = out_map["robot_0"]
        team1_out = out_map["robot_1"]

        gravity = torch.stack(
            (
                self.robots["robot_0"].data.projected_gravity_b,
                self.robots["robot_1"].data.projected_gravity_b,
            ),
            dim=1,
        )
        self._back_hold_counter, self._back_lost = sustained_inversion_update(
            self._back_hold_counter,
            gravity,
            up_max=self.cfg.back_loss_up_max,
            hold_steps=self.cfg.back_loss_hold_steps,
        )

        height_fallen = (
            (self.robots["robot_0"].data.root_com_pos_w[:, 2] < 0.1)
            | (self.robots["robot_1"].data.root_com_pos_w[:, 2] < 0.1)
        )
        fallen = height_fallen | torch.any(self._back_lost, dim=1)

        done = team0_out | team1_out | fallen
        dones = {team: done for team in self.cfg.teams.keys()}

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeouts = {team: time_out for team in self.cfg.teams.keys()}
        return dones, timeouts

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        episode_lengths = self.episode_length_buf[env_ids].to(torch.float32) + 1
        out_map = self._robots_out_of_ring()
        N = env_ids.shape[0]

        team0_back_lost = self._back_lost[:, 0].clone()
        team1_back_lost = self._back_lost[:, 1].clone()
        team0_fallen = (self.robots["robot_0"].data.root_com_pos_w[:, 2] < 0.1) | team0_back_lost
        team1_fallen = (self.robots["robot_1"].data.root_com_pos_w[:, 2] < 0.1) | team1_back_lost
        team0_lost = out_map["robot_0"] | team0_fallen
        team1_lost = out_map["robot_1"] | team1_fallen
        timed_out = self.episode_length_buf >= self.max_episode_length - 1

        super()._reset_idx(env_ids)

        if not self.debug:
            if len(env_ids) == self.num_envs:  # type: ignore
                self.episode_length_buf[:] = torch.randint_like(
                    self.episode_length_buf, high=int(self.max_episode_length)
                )

        low, high = self.cfg.ring_radius_min, self.cfg.ring_radius_max
        self.ring_radius[env_ids] = torch.empty(env_ids.shape[0], device=self.device).uniform_(
            low, high
        )  # type: ignore

        origins = self.scene.env_origins[env_ids]

        # Symmetric, central, face-to-face starts remove the accidental advantage
        # from random world headings and prevent episodes beginning near the edge.
        angles = torch.empty(N, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        direction = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        separation = torch.empty(N, device=self.device).uniform_(
            self.cfg.spawn_separation_min, self.cfg.spawn_separation_max
        )
        offsets = torch.zeros(N, self.num_robots, 3, device=self.device)
        offsets[:, 0, :2] = -0.5 * separation.unsqueeze(-1) * direction
        offsets[:, 1, :2] = 0.5 * separation.unsqueeze(-1) * direction
        target_headings = (angles, angles + torch.pi)
        yaws = tuple(
            target_headings[i] - self.cfg.forward_yaw_offsets.get(f"robot_{i}", 0.0)
            for i in range(2)
        )
        zeros = torch.zeros_like(angles)

        for i, robot_id in enumerate(self.robots):
            self.robots[robot_id].reset(env_ids)
            default_root_state = self.robots[robot_id].data.default_root_state[env_ids].clone()
            default_root_state[:, :2] = origins[:, :2]
            default_root_state[:, 0:2] += offsets[:, i, 0:2]
            default_root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaws[i])
            self.robots[robot_id].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[robot_id].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            joint_pos = self.robots[robot_id].data.default_joint_pos[env_ids]
            joint_vel = self.robots[robot_id].data.default_joint_vel[env_ids]
            self.robots[robot_id].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

            self.actions[robot_id][env_ids] = 0.0
            self.previous_actions[robot_id][env_ids] = 0.0

        self._prev_ring_advantage[env_ids] = 0.0
        self._prev_opponent_distance[env_ids] = separation

        self._draw_ring_markers()
        self._draw_team_dots()
        extras = dict()

        team0_wins = team1_lost[env_ids] & ~team0_lost[env_ids]
        team1_wins = team0_lost[env_ids] & ~team1_lost[env_ids]
        team0_flips = self._flip_awarded[env_ids, 0].clone()
        team1_flips = self._flip_awarded[env_ids, 1].clone()
        extras["team0_win_percentage"] = torch.mean(team0_wins.float()).item() if N > 0 else 0
        extras["team1_win_percentage"] = torch.mean(team1_wins.float()).item() if N > 0 else 0
        extras["team0_flip_percentage"] = torch.mean(team0_flips.float()).item() if N > 0 else 0
        extras["team1_flip_percentage"] = torch.mean(team1_flips.float()).item() if N > 0 else 0
        extras["team0_back_loss_percentage"] = (
            torch.mean(team0_back_lost[env_ids].float()).item() if N > 0 else 0
        )
        extras["team1_back_loss_percentage"] = (
            torch.mean(team1_back_lost[env_ids].float()).item() if N > 0 else 0
        )
        extras["timeout_percentage"] = torch.mean(timed_out[env_ids].float()).item() if N > 0 else 0

        self._flip_hold_counter[env_ids] = 0
        self._flip_awarded[env_ids] = False
        self._back_hold_counter[env_ids] = 0
        self._back_lost[env_ids] = False

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids] / episode_lengths)
            extras["Episode_Reward/" + key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

    def _sample_positions_grid(self, N, radii, num_samples, min_dist=0.6, grid_spacing=0.5):
        device = radii.device
        offsets = torch.zeros((N, num_samples, 3), device=device)

        for i in range(N):
            r = radii[i].item()

            xs = torch.arange(-r, r + grid_spacing, grid_spacing, device=device)
            ys = torch.arange(-r, r + grid_spacing, grid_spacing, device=device)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
            grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

            mask = torch.linalg.norm(grid, dim=1) < r
            candidates = grid[mask]

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
