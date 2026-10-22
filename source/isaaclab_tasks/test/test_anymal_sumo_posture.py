"""Unit tests for ANYmal sumo posture rewards and sustained back losses."""

import torch

from isaaclab_tasks.direct.adversarial.sumo.anymal_c.flip_detection import (
    inverted_back_penalty,
    sustained_inversion_update,
)


def test_inverted_back_penalty_distinguishes_upright_from_upside_down():
    gravity = torch.tensor(
        [
            [0.0, 0.0, -1.0],  # upright
            [0.0, 0.0, 1.0],  # upside down
            [1.0, 0.0, 0.0],  # on its side
        ]
    )

    penalty = inverted_back_penalty(gravity, scale=-0.2)

    torch.testing.assert_close(penalty, torch.tensor([0.0, -0.2, 0.0]))


def test_sustained_inversion_requires_consecutive_steps_and_recovers():
    counter = torch.zeros(2, dtype=torch.int32)
    upright = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    inverted = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    for _ in range(14):
        counter, lost = sustained_inversion_update(
            counter, inverted, up_max=-0.25, hold_steps=15
        )
        assert not torch.any(lost)

    counter, lost = sustained_inversion_update(counter, inverted, up_max=-0.25, hold_steps=15)
    assert torch.all(lost)

    counter, lost = sustained_inversion_update(counter, upright, up_max=-0.25, hold_steps=15)
    torch.testing.assert_close(counter, torch.zeros_like(counter))
    assert not torch.any(lost)
