"""Strict, atomic campaign checkpoints, including optimizer and team rotation state.

Resume starts fresh simulator episodes. It is not a bit-exact physics restore.
Only load checkpoints produced locally by this campaign.
"""

import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def strict_restore(runner, directory, optimizers=False, actors_only=False):
    """HARL's default loader swallows errors; explicitly require all requested weights."""
    directory = Path(directory)
    for team, actors in runner.actors.items():
        for key, actor in actors.items():
            actor.actor.load_state_dict(
                torch.load(directory / f"actor_agent_{key}.pt", map_location=runner.device, weights_only=True)
            )
    if not actors_only:
        for team, critic in runner.critics.items():
            critic.critic.load_state_dict(
                torch.load(directory / f"{team}_critic_agent.pt", map_location=runner.device, weights_only=True)
            )
            if runner.value_normalizers is not None:
                runner.value_normalizers[team].load_state_dict(
                    torch.load(directory / f"{team}_value_normalizer.pt", map_location=runner.device, weights_only=True)
                )
    if optimizers:
        state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
        if state["training_mode"] != runner.training_mode:
            raise ValueError("Optimizer resume requires the same adversarial training mode")
        for team, actors in runner.actors.items():
            for key, actor in actors.items():
                actor.actor_optimizer.load_state_dict(state["actors"][team][key])
        for team, critic in runner.critics.items():
            critic.critic_optimizer.load_state_dict(state["critics"][team])
        for key in ("current_team_index", "current_team_train_steps", "training_teams"):
            if key in state:
                setattr(runner, key, state[key])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if runner.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])


def assert_finite(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            assert_finite(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            assert_finite(value)
    elif isinstance(tree, torch.Tensor) and not torch.isfinite(tree).all():
        raise FloatingPointError("Non-finite tensor in rollout or optimizer diagnostics")
    elif isinstance(tree, (float, np.floating)) and not np.isfinite(tree):
        raise FloatingPointError("Non-finite scalar in optimizer diagnostics")


def install_campaign_checkpoints(runner, output, interval=10):
    """Wrap existing runner hooks; no installed HARL source modifications."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original_train, original_insert = runner.train, runner.insert
    progress = {"updates": 0, "steps": 0, "started": time.time()}
    created_snapshots = []
    batch = runner.algo_args["train"]["episode_length"] * runner.algo_args["train"]["n_rollout_threads"]

    def save(final=False):
        name = f"update_{progress['updates']:06d}"
        target = output / name
        if not target.exists():
            temporary = output / (name + ".pending")
            temporary.mkdir(exist_ok=True)
            runner.save(temporary)
            state = {
                "training_mode": runner.training_mode,
                "actors": {
                    team: {key: actor.actor_optimizer.state_dict() for key, actor in actors.items()}
                    for team, actors in runner.actors.items()
                },
                "critics": {team: critic.critic_optimizer.state_dict() for team, critic in runner.critics.items()},
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if runner.device.type == "cuda" else [],
            }
            for key in ("current_team_index", "current_team_train_steps", "training_teams"):
                if hasattr(runner, key):
                    state[key] = getattr(runner, key)
            torch.save(state, temporary / "training_state.pt")
            atomic_json(temporary / "metadata.json", dict(progress, batch=batch))
            os.replace(temporary, target)
            created_snapshots.append(target)
        atomic_json(output / "receipt.json", dict(progress, checkpoint=str(target.resolve()), complete=final))
        # Only prune snapshots CREATED BY THIS PROCESS in this new chunk. Keep
        # the latest three, all previous chunks/finals, and every video target.
        while len(created_snapshots) > 3:
            expired = created_snapshots.pop(0)
            if expired.parent != output or not (expired / "metadata.json").is_file():
                raise RuntimeError("Refusing checkpoint cleanup outside the current chunk")
            shutil.rmtree(expired)

    def insert(data):
        # Validate observations, rewards, values and sampled actions before PPO.
        for index in (0, 1, 2, 6, 7, 8):
            assert_finite(data[index])
        return original_insert(data)

    def train():
        result = original_train()
        assert_finite(result)
        progress["updates"] += 1
        progress["steps"] += batch
        if progress["updates"] == 1 or progress["updates"] % interval == 0:
            save()
        atomic_json(output / "heartbeat.json", dict(progress, updated=time.time(), pid=os.getpid()))
        return result

    runner.insert, runner.train = insert, train
    return lambda: save(final=True)
