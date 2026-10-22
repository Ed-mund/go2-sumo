"""Pure tensor checks for the Go2 depth observation encoder."""

import torch

import pytest

from isaaclab_tasks.direct.adversarial.sumo.go2_sensor_curriculum import (
    DEPTH_FEATURES,
    DEPTH_FEATURES_PER_FRAME,
    SENSOR_OBSERVATION_DIM,
    compress_depth_image,
)


def test_sensor_observation_layout_is_stable_across_curriculum():
    assert SENSOR_OBSERVATION_DIM == 45 + 3 + DEPTH_FEATURES + 2
    assert DEPTH_FEATURES_PER_FRAME == 96
    assert DEPTH_FEATURES == 192


def test_depth_encoder_preserves_near_obstacle_and_bounds_output():
    depth = torch.full((2, 36, 64, 1), 6.0)
    depth[0, 12:24, 28:36, 0] = 0.6

    features = compress_depth_image(depth, max_range=6.0)

    assert features.shape == (2, DEPTH_FEATURES_PER_FRAME)
    assert torch.all((features >= 0.0) & (features <= 1.0))
    assert features[0].max().item() == pytest.approx(0.9)
    assert features[1].max().item() == pytest.approx(0.0)


def test_depth_encoder_maps_invalid_returns_to_sensor_limits():
    depth = torch.tensor([[[[float("inf")], [float("nan")], [-1.0], [3.0]]]])

    features = compress_depth_image(depth, max_range=6.0, output_size=(1, 4))

    assert torch.allclose(features, torch.tensor([[0.0, 0.0, 1.0, 0.5]]))
