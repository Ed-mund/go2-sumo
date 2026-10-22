"""Torch-only geometry helpers for quadruped sumo flip events."""

from __future__ import annotations

import torch


def inverted_back_penalty(
    projected_gravity: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Penalize only the inverted half of the robot's orientation.

    ``projected_gravity[..., 2]`` is ``-1`` when upright and ``+1`` when
    perfectly upside down.  The older XY-only flat-orientation cost was zero
    at both of those poses, so an ANYmal lying squarely on its back was not
    penalized.  This term is zero while the base up-axis is above horizontal
    and ramps quadratically to ``scale`` when fully inverted.
    """
    up_alignment = -projected_gravity[..., 2]
    inverted_amount = torch.clamp(-up_alignment, min=0.0, max=1.0)
    return scale * torch.square(inverted_amount)


def sustained_inversion_update(
    hold_counter: torch.Tensor,
    projected_gravity: torch.Tensor,
    *,
    up_max: float,
    hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a consecutive inversion counter and return sustained back losses."""
    inverted = -projected_gravity[..., 2] < up_max
    updated_counter = torch.where(inverted, hold_counter + 1, torch.zeros_like(hold_counter))
    return updated_counter, updated_counter >= hold_steps


def mounted_flip_candidates(
    pos_0: torch.Tensor,
    pos_1: torch.Tensor,
    projected_gravity_0: torch.Tensor,
    projected_gravity_1: torch.Tensor,
    *,
    victim_up_max: float,
    attacker_up_min: float,
    max_horizontal_distance: float,
    min_height_advantage: float,
) -> torch.Tensor:
    """Classify which robot is upright above a nearby inverted opponent.

    Returns a boolean tensor shaped ``(num_envs, 2)``. Column zero means robot
    0 is over robot 1; column one is the mirrored event. Temporal persistence
    and the one-shot episode latch are handled by the environment.
    """
    up_0 = -projected_gravity_0[:, 2]
    up_1 = -projected_gravity_1[:, 2]
    horizontal_distance = torch.linalg.norm(pos_1[:, :2] - pos_0[:, :2], dim=-1)

    robot_0_over_robot_1 = (
        (up_0 > attacker_up_min)
        & (up_1 < victim_up_max)
        & (horizontal_distance < max_horizontal_distance)
        & (pos_0[:, 2] > pos_1[:, 2] + min_height_advantage)
    )
    robot_1_over_robot_0 = (
        (up_1 > attacker_up_min)
        & (up_0 < victim_up_max)
        & (horizontal_distance < max_horizontal_distance)
        & (pos_1[:, 2] > pos_0[:, 2] + min_height_advantage)
    )
    return torch.stack((robot_0_over_robot_1, robot_1_over_robot_0), dim=-1)
