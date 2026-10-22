"""CPU-testable visual option policy and PPO. No Isaac imports.

The manager chooses a skill every 10 controls (5 Hz). The motor policy remains
closed-loop at 50 Hz. Both use only onboard observations. Manager likelihood is
included ONLY at actual decisions; forced drill labels never train the manager.
"""

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from tactics_contract import CONTRACT, FRAMES, HEIGHT, WIDTH, PIXELS, VECTOR, SKILLS


def unpack(observation):
    """Layout stays compatible with env sensor-vector first, image history, ranges/ages."""
    pixels = observation[:, 45 : 45 + PIXELS].reshape(-1, FRAMES * 3, HEIGHT, WIDTH)
    vector = torch.cat((observation[:, :45], observation[:, 45 + PIXELS :]), -1)
    return (pixels.clamp(0, 1) * 255).round().to(torch.uint8), vector


class TacticsPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Four ordered RGB frames, spanning 0.3 s at the actor's 10 Hz image sampling.
        self.vision = nn.Sequential(
            nn.Conv2d(12, 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 32, 3, 2),
            nn.ELU(),
            nn.Conv2d(32, 48, 3, 2),
            nn.ELU(),
            nn.Flatten(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(48 * 3 * 6 + VECTOR + len(SKILLS) + 1, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU()
        )
        self.manager = nn.Linear(256, len(SKILLS))
        self.motor = nn.Sequential(nn.Linear(256 + len(SKILLS), 256), nn.ELU(), nn.Linear(256, 12))
        self.value = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((12,), -0.7))
        nn.init.normal_(self.motor[-1].weight, std=0.01)
        nn.init.zeros_(self.motor[-1].bias)

    def features(self, pixels, vector, previous, remaining):
        visual = self.vision(pixels.float() / 255)
        skill = nn.functional.one_hot(previous, len(SKILLS)).float()
        return self.trunk(torch.cat((visual, vector, skill, remaining[:, None].float() / 10), -1))

    def act(self, pixels, vector, previous, remaining, forced=None, deterministic=False):
        features = self.features(pixels, vector, previous, remaining)
        manager = Categorical(logits=self.manager(features))
        decisions = remaining == 0
        chosen = manager.logits.argmax(-1) if deterministic else manager.sample()
        skills = torch.where(decisions, chosen, previous)
        if forced is not None:
            forced_mask = forced >= 0
            skills = torch.where(forced_mask, forced.clamp_min(0), skills)
            decisions &= ~forced_mask
        motor = Normal(
            self.motor(torch.cat((features, nn.functional.one_hot(skills, len(SKILLS)).float()), -1)),
            self.log_std.clamp(-2.5, 0.5).exp(),
        )
        actions = motor.mean if deterministic else motor.sample()
        logprob = motor.log_prob(actions).sum(-1) + decisions.float() * manager.log_prob(skills)
        return actions, skills, decisions, logprob, self.value(features).squeeze(-1)

    def evaluate(self, data):
        features = self.features(data["pixels"], data["vector"], data["previous"], data["remaining"])
        manager = Categorical(logits=self.manager(features))
        selected = nn.functional.one_hot(data["skills"], len(SKILLS)).float()
        motor = Normal(self.motor(torch.cat((features, selected), -1)), self.log_std.clamp(-2.5, 0.5).exp())
        decisions = data["decisions"].float()
        logprob = motor.log_prob(data["actions"]).sum(-1) + decisions * manager.log_prob(data["skills"])
        entropy = 0.001 * motor.entropy().sum(-1) + 0.01 * decisions * manager.entropy()
        return logprob, self.value(features).squeeze(-1), entropy


def option_clock(skills, remaining, done, period=10):
    remaining = torch.where(remaining == 0, torch.full_like(remaining, period), remaining) - 1
    return torch.where(done, torch.zeros_like(skills), skills), torch.where(
        done, torch.zeros_like(remaining), remaining
    )


def advantages(rewards, values, dones, bootstrap, gamma=0.99, lam=0.95):
    """Finite-horizon episodes: both rule termination and deadline are terminal."""
    result = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        continuation = (~dones[t]).float()
        next_value = bootstrap if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * continuation * next_value - values[t]
        carry = delta + gamma * lam * continuation * carry
        result[t] = carry
    return result, result + values


def ppo_update(policy, optimizer, data, epochs=4, minibatch=128):
    """Joint option/motor likelihood; no fictitious manager decisions between options."""
    count = len(data["actions"])
    data["advantage"] = (data["advantage"] - data["advantage"].mean()) / data["advantage"].std(
        unbiased=False
    ).clamp_min(1e-6)
    totals = []
    for _ in range(epochs):
        for ids in torch.randperm(count, device=data["actions"].device).split(minibatch):
            batch = {key: value[ids] for key, value in data.items()}
            logprob, value, entropy = policy.evaluate(batch)
            ratio = (logprob - batch["logprob"]).exp()
            objective = torch.minimum(ratio * batch["advantage"], ratio.clamp(0.8, 1.2) * batch["advantage"])
            loss = -objective.mean() + 0.5 * (value - batch["return"]).square().mean() - entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite tactical PPO loss")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            totals.append(float(loss.detach()))
    return {"loss": sum(totals) / len(totals), "grad_norm": float(norm)}
