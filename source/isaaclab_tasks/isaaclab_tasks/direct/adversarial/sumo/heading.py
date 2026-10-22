"""Shared quadruped heading rewards.

All supported ANYmal, Spot, and Unitree assets use local +X as their anatomical
forward direction (their front hip joints are on the positive-X side).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def anatomical_forward_xy(
    root_quat_w: torch.Tensor,
    forward_yaw_offset: float = 0.0,
) -> torch.Tensor:
    """Return the robot's configured anatomical-forward direction in world XY."""
    # Rotate the configured anatomical-forward vector into world coordinates.
    # Isaac Lab quaternions are wxyz.
    w, x, y, z = root_quat_w.unbind(dim=-1)
    offset = torch.as_tensor(forward_yaw_offset, dtype=root_quat_w.dtype, device=root_quat_w.device)
    local_x = torch.cos(offset)
    local_y = torch.sin(offset)
    return torch.stack(
        (
            (1.0 - 2.0 * (y.square() + z.square())) * local_x + 2.0 * (x * y - w * z) * local_y,
            2.0 * (x * y + w * z) * local_x + (1.0 - 2.0 * (x.square() + z.square())) * local_y,
        ),
        dim=-1,
    )


def resolve_heading_target_position(
    robot_id: str,
    robot_positions: Mapping[str, torch.Tensor],
    desired_position: torch.Tensor | None = None,
    block_positions: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor | None, str | None]:
    """Choose the task-relevant heading target used by policy audits.

    Locomotion tasks expose a desired position, block-pushing tasks assign each
    robot a same-numbered block, and direct two-robot sumo targets the opponent.
    The order is important: a block-pushing environment also contains two
    robots, but they are not each other's target during that curriculum stage.
    """
    if desired_position is not None:
        return desired_position, "desired_position"

    block_id = robot_id.replace("robot_", "block_", 1)
    if block_positions is not None and block_id in block_positions:
        return block_positions[block_id], "assigned_block"

    opponents = [position for other_id, position in robot_positions.items() if other_id != robot_id]
    if len(robot_positions) == 2 and len(opponents) == 1:
        return opponents[0], "opponent"

    return None, None


def target_heading_alignment(
    root_quat_w: torch.Tensor,
    source_pos_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    forward_yaw_offset: float = 0.0,
) -> torch.Tensor:
    """Cosine alignment between the robot's nose and an XY target direction."""
    forward_xy = anatomical_forward_xy(root_quat_w, forward_yaw_offset)
    target_xy = target_pos_w[:, :2] - source_pos_w[:, :2]
    target_xy = target_xy / torch.linalg.norm(target_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    return torch.sum(forward_xy * target_xy, dim=-1)


def backward_facing_penalty(
    root_quat_w: torch.Tensor,
    source_pos_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    scale: float,
    forward_yaw_offset: float = 0.0,
) -> torch.Tensor:
    """Penalize facing more than 90 degrees away; facing forward earns nothing."""
    alignment = target_heading_alignment(
        root_quat_w, source_pos_w, target_pos_w, forward_yaw_offset=forward_yaw_offset
    )
    return float(scale) * torch.relu(-alignment)
