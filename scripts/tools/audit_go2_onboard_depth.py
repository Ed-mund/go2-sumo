"""Verify that the Go2 onboard cameras see a nose-to-nose opponent."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, default=Path("artifacts/go2_sensor_audit"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import imageio.v2 as imageio

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg(
        "Sumo-Stage2-Unitree-Go2-Sensor-1v1-v0",
        device=args_cli.device,
        num_envs=1,
    )
    env = gym.make("Sumo-Stage2-Unitree-Go2-Sensor-1v1-v0", cfg=cfg)
    env.reset()

    zero_actions = {
        agent: torch.zeros((1, space.shape[-1]), device=env.unwrapped.device)
        for agent, space in env.unwrapped.action_spaces.items()
    }
    for _ in range(8):
        env.step(zero_actions)

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    for robot_id, camera in env.unwrapped.depth_cameras.items():
        depth = camera.data.output["distance_to_camera"][0, ..., 0]
        depth = torch.nan_to_num(depth, nan=6.0, posinf=6.0, neginf=0.0).clamp(0.0, 6.0)
        centre = depth[
            depth.shape[0] // 3 : 2 * depth.shape[0] // 3,
            depth.shape[1] // 3 : 2 * depth.shape[1] // 3,
        ]
        image = ((1.0 - depth / 6.0) * 255.0).to(torch.uint8).cpu().numpy()
        imageio.imwrite(args_cli.output_dir / f"{robot_id}_depth_proximity.png", image)
        print(
            f"{robot_id}: min_depth={depth.min().item():.3f}m "
            f"centre_min={centre.min().item():.3f}m "
            f"centre_mean={centre.mean().item():.3f}m",
            flush=True,
        )

    np.savetxt(
        args_cli.output_dir / "spawn_root_positions.txt",
        np.stack(
            [env.unwrapped.robots[robot_id].data.root_pos_w[0].cpu().numpy() for robot_id in ("robot_0", "robot_1")]
        ),
        header="x y z",
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
