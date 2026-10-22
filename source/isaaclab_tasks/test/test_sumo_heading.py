"""Tests for anatomically forward quadruped sumo behaviour."""

import math

import torch

from isaaclab_tasks.direct.adversarial.sumo.heading import (
    anatomical_forward_xy,
    backward_facing_penalty,
    resolve_heading_target_position,
    target_heading_alignment,
)


def test_target_heading_alignment_uses_local_positive_x_as_the_nose():
    source = torch.zeros(3, 3)
    target = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    quaternions = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
        ]
    )
    alignment = target_heading_alignment(quaternions, source, target)
    torch.testing.assert_close(alignment, torch.tensor([1.0, -1.0, 0.0]), atol=1.0e-6, rtol=0.0)


def test_backward_penalty_never_rewards_facing_the_target():
    source = torch.zeros(2, 3)
    target = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    penalty = backward_facing_penalty(quaternions, source, target, scale=-0.1)
    torch.testing.assert_close(penalty, torch.tensor([0.0, -0.1]))


def test_asset_forward_yaw_offset_is_applied():
    source = torch.zeros(1, 3)
    target = torch.tensor([[-1.0, 0.0, 0.0]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    alignment = target_heading_alignment(identity, source, target, forward_yaw_offset=math.pi)
    torch.testing.assert_close(alignment, torch.ones(1), atol=1.0e-6, rtol=0.0)


def test_anatomical_forward_xy_applies_asset_offset():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    forward = anatomical_forward_xy(identity, forward_yaw_offset=math.pi / 2.0)
    torch.testing.assert_close(forward, torch.tensor([[0.0, 1.0]]), atol=1.0e-6, rtol=0.0)


def test_stage1_heading_audit_targets_assigned_block_instead_of_opponent():
    robot_positions = {
        "robot_0": torch.tensor([[0.0, 0.0, 0.0]]),
        "robot_1": torch.tensor([[2.0, 0.0, 0.0]]),
    }
    block_positions = {
        "block_0": torch.tensor([[-1.0, 0.0, 0.0]]),
        "block_1": torch.tensor([[3.0, 0.0, 0.0]]),
    }

    target, source = resolve_heading_target_position(
        "robot_0", robot_positions, block_positions=block_positions
    )

    assert source == "assigned_block"
    torch.testing.assert_close(target, block_positions["block_0"])


def test_direct_sumo_heading_audit_targets_opponent_without_a_goal_object():
    robot_positions = {
        "robot_0": torch.tensor([[0.0, 0.0, 0.0]]),
        "robot_1": torch.tensor([[2.0, 0.0, 0.0]]),
    }

    target, source = resolve_heading_target_position("robot_0", robot_positions)

    assert source == "opponent"
    torch.testing.assert_close(target, robot_positions["robot_1"])
