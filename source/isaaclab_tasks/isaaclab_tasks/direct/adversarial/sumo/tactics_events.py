"""Reward-only contact event logic. Never an actor observation or hardware sensor.

Temporal contact attribution is a heuristic, not proof of causation. Every event
is bounded and one-shot per robot per episode; contact magnitude itself earns no reward.
"""

import torch

EVENTS = ("sweep_left", "sweep_right", "flank_left", "flank_right", "counter", "escape", "pin", "recovery")


class TacticsEvents:
    def __init__(self, count, device="cpu"):
        self.device = device
        self.awarded = torch.zeros(count, 2, len(EVENTS), dtype=torch.bool, device=device)
        self.sweep_window = torch.zeros(count, 2, 2, dtype=torch.long, device=device)
        self.sweep_stable = torch.zeros_like(self.sweep_window)
        self.previous_hit = torch.zeros_like(self.sweep_window, dtype=torch.bool)
        self.pin_hold = torch.zeros(count, 2, dtype=torch.long, device=device)
        self.recovery_hold = torch.zeros_like(self.pin_hold)
        self.was_unstable = torch.zeros_like(self.pin_hold, dtype=torch.bool)
        self.was_close = torch.zeros(count, dtype=torch.bool, device=device)
        self.counter_window = torch.zeros_like(self.pin_hold)
        self.counter_radius = torch.zeros(count, 2, device=device)
        self.previous_incoming = torch.zeros(count, 2, dtype=torch.bool, device=device)
        self.budget = torch.zeros(count, 2, device=device)

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    def update(
        self,
        *,
        up,
        gap,
        height,
        leg_hit,
        foot_moving,
        stance,
        body_contact,
        grounded,
        flank,
        flank_side,
        radius,
        incoming,
        skills,
        drills=False
    ):
        victim_up = up.flip(1)
        good = up > 0.65
        fresh_hit = leg_hit & ~self.previous_hit
        self.previous_hit.copy_(leg_hit)
        qualified = fresh_hit & foot_moving & (stance >= 2)[..., None] & good[..., None] & (victim_up > 0.75)[..., None]
        self.sweep_window = torch.where(qualified, 30, (self.sweep_window - 1).clamp_min(0))
        destabilized = (victim_up < 0.55)[..., None] & good[..., None] & (self.sweep_window > 0)
        self.sweep_stable = torch.where(destabilized, self.sweep_stable + 1, 0)
        pin = good & (victim_up < 0.15) & (gap < 0.8)[:, None] & (height - height.flip(1) > 0.08)
        pin &= body_contact & grounded.flip(1)
        self.pin_hold = torch.where(pin, self.pin_hold + 1, 0)
        self.was_unstable |= up < 0.5
        self.recovery_hold = torch.where((up > 0.85) & self.was_unstable, self.recovery_hold + 1, 0)
        self.was_close |= gap < 0.8
        fresh_incoming = incoming & ~self.previous_incoming
        self.previous_incoming.copy_(incoming)
        self.counter_radius = torch.where(fresh_incoming, radius.flip(1), self.counter_radius)
        self.counter_window = torch.where(fresh_incoming, 100, (self.counter_window - 1).clamp_min(0))
        candidates = torch.stack(
            (
                self.sweep_stable[..., 0] >= 8,
                self.sweep_stable[..., 1] >= 8,
                flank & (flank_side > 0) & good,
                flank & (flank_side < 0) & good,
                (self.counter_window > 0) & good & (radius.flip(1) - self.counter_radius > 0.2),
                self.was_close[:, None] & (gap > 1.3)[:, None] & good & (radius < 2.6),
                self.pin_hold >= 25,
                self.recovery_hold >= 10,
            ),
            -1,
        )
        events = candidates & ~self.awarded
        self.awarded |= events
        # Modest combat bonuses; only drill recovery is paid, to avoid deliberate falls.
        weights = torch.tensor([0.6, 0.6, 0.2, 0.2, 0.6, 0.2, 2.0, 0.0], device=self.device)
        requested = torch.stack(
            (skills == 3, skills == 4, skills == 1, skills == 2, skills == 5, skills == 6, skills == 7, skills == 8), -1
        )
        drill_reward = (events & requested).float().sum(-1) * 2.0
        mask = torch.as_tensor(drills, device=self.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask[:, None]
        reward = torch.where(mask, drill_reward, (events.float() * weights).sum(-1))
        reward = torch.minimum(reward, (3.0 - self.budget).clamp_min(0))
        self.budget += reward
        return events, reward
