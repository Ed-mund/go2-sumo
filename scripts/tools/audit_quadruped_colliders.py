#!/usr/bin/env python3
"""Audit resolved visual and collision coverage for supported quadrupeds.

This intentionally loads the live USD through Isaac Sim. Source URDFs and unresolved
USD references are not sufficient for deciding whether a sumo collider covers the
rendered robot.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--robot", required=True, choices=("spot", "a1", "go1", "go2"))
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG
from pxr import Usd, UsdGeom, UsdPhysics


ROBOT_CFGS = {
    "spot": SPOT_CFG,
    "a1": UNITREE_A1_CFG,
    "go1": UNITREE_GO1_CFG,
    "go2": UNITREE_GO2_CFG,
}


def _as_list(value) -> list[float]:
    return [float(value[i]) for i in range(3)]


def _owning_rigid_body(prim: Usd.Prim, root_path: str) -> Usd.Prim | None:
    current = prim
    while current and current.IsValid() and str(current.GetPath()).startswith(root_path):
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _has_collision_ancestor(prim: Usd.Prim, owner: Usd.Prim) -> bool:
    current = prim
    while current and current.IsValid():
        if current.HasAPI(UsdPhysics.CollisionAPI):
            return True
        if current == owner:
            return False
        current = current.GetParent()
    return False


def _union_ranges(ranges: list[tuple[list[float], list[float]]]) -> dict | None:
    ranges = [
        bounds
        for bounds in ranges
        if all(bounds[0][axis] <= bounds[1][axis] and abs(bounds[0][axis]) < 1.0e20 for axis in range(3))
    ]
    if not ranges:
        return None
    minimum = [min(bounds[0][axis] for bounds in ranges) for axis in range(3)]
    maximum = [max(bounds[1][axis] for bounds in ranges) for axis in range(3)]
    return {
        "min": minimum,
        "max": maximum,
        "size": [maximum[axis] - minimum[axis] for axis in range(3)],
        "center": [0.5 * (maximum[axis] + minimum[axis]) for axis in range(3)],
    }


def _shape_attributes(prim: Usd.Prim) -> dict:
    values = {}
    for name in ("radius", "height", "axis", "size", "extent"):
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid() and attr.HasAuthoredValueOpinion():
            value = attr.Get()
            if value is not None:
                values[name] = str(value)
    return values


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 200.0))
    robot_cfg = ROBOT_CFGS[args_cli.robot].replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    stage = sim.stage
    root_path = "/World/Robot"
    # The shipped robot assets keep geometry below instanceable containers.
    # Break those instances in the one-off audit stage so traversal reaches the
    # resolved visual and collision prims. This does not alter the source USD.
    for prim in list(stage.Traverse()):
        if str(prim.GetPath()).startswith(root_path) and prim.IsInstance():
            prim.SetInstanceable(False)
    # Only create PhysX tensor views after the audit copy has been de-instanced.
    sim.reset()
    robot.update(sim.get_physics_dt())
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )
    grouped = defaultdict(lambda: {"visual": [], "collision": [], "shapes": []})

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root_path) or not prim.IsA(UsdGeom.Boundable):
            continue
        owner = _owning_rigid_body(prim, root_path)
        if owner is None:
            continue
        try:
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            bounds = (_as_list(aligned.GetMin()), _as_list(aligned.GetMax()))
        except Exception:
            continue
        owner_name = owner.GetName()
        is_collision = _has_collision_ancestor(prim, owner)
        kind = "collision" if is_collision else "visual"
        grouped[owner_name][kind].append(bounds)
        if is_collision:
            grouped[owner_name]["shapes"].append(
                {
                    "path": path,
                    "type": prim.GetTypeName(),
                    "attributes": _shape_attributes(prim),
                    "world_bounds": {"min": bounds[0], "max": bounds[1]},
                }
            )

    bodies = {}
    for body_name in sorted(grouped):
        visual = _union_ranges(grouped[body_name]["visual"])
        collision = _union_ranges(grouped[body_name]["collision"])
        coverage = None
        if visual and collision:
            coverage = [
                collision["size"][axis] / max(visual["size"][axis], 1.0e-8)
                for axis in range(3)
            ]
        bodies[body_name] = {
            "visual_bounds": visual,
            "collision_bounds": collision,
            "axis_span_ratio": coverage,
            "collision_shapes": grouped[body_name]["shapes"],
        }

    joints = []
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root_path) or not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        joints.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "body0": [str(path) for path in joint.GetBody0Rel().GetTargets()],
                "body1": [str(path) for path in joint.GetBody1Rel().GetTargets()],
                "local_pos0": str(joint.GetLocalPos0Attr().Get()),
                "local_pos1": str(joint.GetLocalPos1Attr().Get()),
            }
        )

    report = {
        "robot": args_cli.robot,
        "usd_path": robot_cfg.spawn.usd_path,
        "root_body_names": list(robot.body_names),
        "joint_names": list(robot.joint_names),
        "default_root_height": float(robot_cfg.init_state.pos[2]),
        "bodies": bodies,
        "joints": joints,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[audit] wrote {args_cli.output}")
    for name, body in bodies.items():
        print(
            f"[audit] {name}: visual={body['visual_bounds']} "
            f"collision={body['collision_bounds']} ratio={body['axis_span_ratio']}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
