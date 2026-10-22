"""Controlled runtime tests. Only imported after Isaac starts; never a hardware driver."""

import math
import torch


def audit_onboard_sensors(env):
    """Move an actual articulated opponent; reject empty, static or self-blinded sensors.

    This diagnostic uses simulator truth to position/check the test objects.
    Neither the actor observations nor the range calculation receives opponent poses.
    The caller must reset afterwards. Failure blocks the long campaign.
    """
    if env.num_envs != 1 or not hasattr(env, "rig") or len(env.robots) != 2:
        raise ValueError("Sensor audit requires a one-environment v2 two-robot task")
    from isaaclab_tasks.direct.adversarial.sumo.robot_profiles import GO2
    from isaaclab.utils.math import quat_apply

    zeros = {key: torch.zeros_like(robot.data.default_joint_pos) for key, robot in env.robots.items()}
    ids = torch.tensor([0], device=env.device)
    results = {}
    pictures = {}
    for name, distance, shift in (("far", 2.2, 0.0), ("near", 1.2, 0.0), ("translated", 1.2, 0.5)):
        for _ in range(20):
            for index, robot in enumerate(env.robots.values()):
                pose = robot.data.default_root_state[:, :7].clone()
                pose[:, :3] += env.scene.env_origins
                pose[:, 0] += shift + (distance if index else 0.0)
                pose[:, 2] += 0.01
                pose[:, 3:7] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0] if index == 0 else [0.0, 0.0, 0.0, 1.0], device=env.device
                )
                robot.write_root_pose_to_sim(pose, ids)
                robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), ids)
                robot.write_joint_state_to_sim(
                    robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_pos), env_ids=ids
                )
            env.step(zeros)
        cloud = env.rig.points("robot_0")[0]
        pitch = GO2.lidar_pitch
        rotation = torch.tensor([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0], device=env.device)
        body = quat_apply(rotation.expand(len(cloud), -1), cloud) + torch.tensor(GO2.lidar_xyz, device=env.device)
        mask = (
            (cloud[:, 2] >= 0)
            & (body[:, 0] > 0.45)
            & (body[:, 0] < 3.0)
            & (body[:, 1].abs() < 0.30)
            & (body[:, 2] > -0.10)
            & (body[:, 2] < 0.20)
        )
        if not mask.any():
            raise RuntimeError(f"LiDAR moving-opponent audit failed ({name}): no torso returns")
        results[name] = float(body[mask, 0].min())
        robot = env.robots["robot_0"]
        camera = env.rig.rgb["robot_0"].data
        expected = robot.data.root_pos_w + quat_apply(
            robot.data.root_quat_w, torch.tensor(GO2.front_sensor_xyz, device=env.device).expand(1, -1)
        )
        if torch.linalg.vector_norm(camera.pos_w - expected) > 0.025:
            raise RuntimeError("RGB camera does not follow the published base-frame mount")
        pictures[name] = camera.output["rgb"][0, ..., :3].float().clone()
        if pictures[name].std() < 2.0:
            raise RuntimeError("RGB camera image is blank/uninformative")
    if results["far"] - results["near"] < 0.4:
        raise RuntimeError(f"LiDAR does not resolve the approaching articulated robot: {results}")
    if abs(results["near"] - results["translated"]) > 0.2:
        raise RuntimeError(f"LiDAR origin does not follow the robot: {results}")
    difference = float((pictures["near"] - pictures["far"]).abs().mean())
    if difference < 0.5:
        raise RuntimeError("RGB failed moving-target response check")
    return {
        "passed": True,
        "torso_forward_ranges_m": results,
        "rgb_change_0_255": difference,
        "scope": "reference RGB mount, nonblank images, articulated LiDAR target and moving sensor origin; not hardware calibration",
    }
