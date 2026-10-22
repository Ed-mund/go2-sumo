#!/usr/bin/env python3
"""V3 hierarchical PPO train/evaluation entrypoint. Never stops unrelated GPU processes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import site
import time

from sumo_campaign import preflight, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--opponents", type=Path)
    parser.add_argument("--role", type=int, choices=(0, 1), default=0)
    parser.add_argument("--steps", type=int, required=True, help="Environment transitions")
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--sensor-audit", action="store_true")
    parser.add_argument("--fixed-drill", type=int, default=-1, choices=range(-1, 9))
    parser.add_argument("--view", default="corner")
    parser.add_argument("--variant", type=int, default=0)
    args = parser.parse_args()
    if site.ENABLE_USER_SITE:
        raise RuntimeError(
            "Use PYTHONNOUSERSITE=1 (or the campaign launcher) to avoid unrelated user-site torch packages"
        )
    cfg = json.loads(args.config.read_text())
    phase = next(p for p in cfg["stages"] if p["name"] == args.phase)
    if args.steps <= 0 or args.envs <= 0 or args.steps % args.envs:
        parser.error("Steps must be a positive multiple of environments")
    if not args.evaluate and args.steps % (args.envs * cfg["rollout_steps"]):
        parser.error("Training must contain complete PPO rollouts")
    if args.video and args.envs != 1:
        parser.error("Video requires one environment")
    if (args.video or args.zero_actions or args.sensor_audit) and not args.evaluate:
        parser.error("Video, passive actions and audits are evaluation-only")
    preflight(cfg)  # no Isaac import/CUDA allocation before the free-GPU check

    import numpy as np
    import torch
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
    env = None
    writer = None
    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.direct.adversarial.sumo.sumo_v3 import (
            Go2TacticsLocomotionCfg,
            Go2TacticsPushCfg,
            Go2TacticsFightCfg,
        )
        from tactics_model import TacticsPolicy, CONTRACT, SKILLS, unpack, option_clock, advantages, ppo_update
        from sumo_checkpoint import assert_finite
        from sumo_campaign import VIEWS

        torch.set_num_threads(8)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        task_type = phase["task"]
        env_cfg = {"Locomotion": Go2TacticsLocomotionCfg, "Push": Go2TacticsPushCfg, "Fight": Go2TacticsFightCfg}[
            task_type
        ]()
        env_cfg.scene.num_envs = args.envs
        env_cfg.seed = args.seed
        env_cfg.learner_role = args.role
        env_cfg.drill_fraction = phase["drill_fraction"] if not args.evaluate else 0.0
        if args.phase == "drills" or args.fixed_drill >= 0:
            env_cfg.drill_fraction = 1.0
        env_cfg.drill_mode = env_cfg.drill_fraction > 0
        env_cfg.fixed_drill = args.fixed_drill
        env_cfg.arena_variant_offset = args.variant
        env_cfg.viewer.eye = VIEWS[args.view]
        env = gym.make(
            f"Dog-Sumo-Go2-{task_type}-v3", cfg=env_cfg, debug=True, render_mode="rgb_array" if args.video else None
        ).unwrapped
        teams = len(env.robots)
        if args.role >= teams:
            raise ValueError("Invalid learner role for this task")
        models = [TacticsPolicy().to(env.device) for _ in range(teams)]
        optimizers = [torch.optim.Adam(model.parameters(), lr=phase["lr"]) for model in models]

        def read_checkpoint(path):
            # Local campaign files only: contains NumPy/Python RNG and Adam state.
            saved = torch.load(path, map_location=env.device, weights_only=False)
            if saved["contract"] != CONTRACT:
                raise ValueError("Wrong tactical observation/skill checkpoint contract")
            return saved

        if args.source:
            source = read_checkpoint(args.source)
            for i, model in enumerate(models):
                model.load_state_dict(source["models"][min(i, len(source["models"]) - 1)], strict=True)
                if source["phase"] == args.phase:
                    optimizers[i].load_state_dict(source["optimizers"][min(i, len(source["optimizers"]) - 1)])
                elif not args.evaluate:
                    # New objective: fresh critic head/Adam, preserve visual and motor skills.
                    torch.nn.init.zeros_(model.value.weight)
                    torch.nn.init.zeros_(model.value.bias)
            if not args.evaluate and source["phase"] == args.phase:
                random.setstate(source["python_rng"])
                np.random.set_state(source["numpy_rng"])
                torch.set_rng_state(source["torch_rng"].cpu())
                torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
        elif args.evaluate:
            raise ValueError("Evaluation requires an explicit checkpoint")

        members = json.loads(args.opponents.read_text()) if args.opponents else []
        frozen = []
        for member in members:
            if hashlib.sha256(Path(member["checkpoint"]).read_bytes()).hexdigest() != member["sha256"]:
                raise ValueError("Opponent checkpoint hash mismatch")
            policy = TacticsPolicy().to(env.device)
            saved = read_checkpoint(member["checkpoint"])
            policy.load_state_dict(saved["models"][member["role"]], strict=True)
            policy.eval().requires_grad_(False)
            frozen.append(policy)
        args.output.mkdir(parents=True, exist_ok=True)
        obs, _ = env.reset()
        audit = None
        if args.sensor_audit:
            from sensor_audit import audit_onboard_sensors

            audit = audit_onboard_sensors(env)
            obs, _ = env.reset()
        previous = [torch.zeros(args.envs, dtype=torch.long, device=env.device) for _ in models]
        remaining = [torch.zeros_like(previous[0]) for _ in models]
        assignment = torch.randint(max(1, len(frozen)), (args.envs,), device=env.device)
        video_frames = 0
        if args.video:
            import imageio.v2 as imageio

            writer = imageio.get_writer(args.output / "match.mp4", fps=25, codec="libx264", quality=8)
        stats = {
            "episodes": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "samples": 0,
            "upright": 0,
            "events": {},
            "skill_controls": {s: 0 for s in SKILLS},
            "opponents": {},
            "metrics": {},
        }
        stats["upright_by_robot"] = {f"robot_{i}": 0 for i in range(teams)}
        stats["selected_opponents"] = members
        stats["passive_drill_controls"] = 0
        stats["event_timeline"] = []
        stats["drill_episodes"] = {s: 0 for s in SKILLS}
        stats["drill_successes"] = {s: 0 for s in SKILLS}
        episode_event = torch.zeros(args.envs, dtype=torch.bool, device=env.device)
        event_names = ("sweep_left", "sweep_right", "flank_left", "flank_right", "counter", "escape", "pin", "recovery")
        drill_event = {1: 2, 2: 3, 3: 0, 4: 1, 5: 4, 6: 5, 7: 6, 8: 7}
        steps, updates, started = 0, 0, time.time()
        created = []

        def save(final=False):
            path = args.output / f"update_{updates:06d}.pt"
            if not path.exists():
                temporary = path.with_suffix(".pending")
                torch.save(
                    {
                        "contract": CONTRACT,
                        "phase": args.phase,
                        "role": args.role,
                        "models": [m.state_dict() for m in models],
                        "optimizers": [o.state_dict() for o in optimizers],
                        "steps": steps,
                        "updates": updates,
                        "python_rng": random.getstate(),
                        "numpy_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                    },
                    temporary,
                )
                os.replace(temporary, path)
                created.append(path)
            write_json(
                args.output / "receipt.json", {"checkpoint": str(path.resolve()), "complete": final, "steps": steps}
            )
            # Only snapshots created by this process; all prior chunks/video checkpoints survive.
            while len(created) > 3:
                created.pop(0).unlink()

        while steps < args.steps:
            rollout = []
            count = min(cfg["rollout_steps"], (args.steps - steps) // args.envs)
            for _ in range(count):
                actions = {}
                choices = []
                frame_data = None
                active_drill = env.drill_active.clone()
                passive_drill = env.drill_passive.clone()
                drill_ids = env.drills[:, args.role].clone()
                for i, model in enumerate(models):
                    pixels, vector = unpack(obs[f"team_{i}"][f"robot_{i}"])
                    force = torch.full_like(previous[i], 0 if phase["manager"] is False else -1)
                    if i != args.role and args.phase == "drills":
                        force.zero_()
                    if i == args.role:
                        force = torch.where(active_drill, drill_ids, force)
                    with torch.no_grad():
                        if i != args.role and frozen:
                            output = None
                            for j, opponent in enumerate(frozen):
                                ids = (assignment == j).nonzero().flatten()
                                if len(ids):
                                    fixed = torch.full_like(previous[i][ids], members[j].get("forced_skill", -1))
                                    values = opponent.act(
                                        pixels[ids],
                                        vector[ids],
                                        previous[i][ids],
                                        remaining[i][ids],
                                        fixed,
                                        args.evaluate,
                                    )
                                    if output is None:
                                        output = [
                                            torch.zeros((args.envs, *v.shape[1:]), dtype=v.dtype, device=v.device)
                                            for v in values
                                        ]
                                    for k in range(len(output)):
                                        output[k][ids] = values[k]
                        else:
                            output = model.act(pixels, vector, previous[i], remaining[i], force, args.evaluate)
                    action, skill, decision, logprob, value = output
                    if i != args.role:
                        action[env.drill_passive] = 0
                    if args.zero_actions:
                        action.zero_()
                    assert_finite((pixels, vector, action, value, logprob))
                    actions[f"robot_{i}"] = action
                    choices.append(skill)
                    if i == args.role and not args.evaluate:
                        frame_data = {
                            "pixels": pixels,
                            "vector": vector,
                            "previous": previous[i].clone(),
                            "remaining": remaining[i].clone(),
                            "skills": skill,
                            "decisions": decision,
                            "actions": action,
                            "logprob": logprob,
                            "value": value,
                        }
                env.skills.copy_(torch.stack(choices, 1))
                obs, rewards, terminated, truncated, extras = env.step(actions)
                done = torch.stack(list(terminated.values()) + list(truncated.values())).any(0)
                assert_finite((obs, rewards))
                if frame_data is not None:
                    frame_data.update(reward=rewards[f"team_{args.role}"].clone(), done=done.clone())
                    rollout.append(frame_data)
                info = env.tactical_transition
                stats["samples"] += args.envs
                stats["passive_drill_controls"] += int(passive_drill.sum())
                stats["upright"] += int((info["upright"][:, args.role] > 0.5).sum())
                for i in range(teams):
                    stats["upright_by_robot"][f"robot_{i}"] += int((info["upright"][:, i] > 0.5).sum())
                for skill, name in enumerate(SKILLS):
                    stats["skill_controls"][name] += int((choices[args.role] == skill).sum())
                if "events" in info:
                    if args.video:
                        for robot, event in info["events"][0].nonzero().cpu().tolist():
                            stats["event_timeline"].append(
                                {
                                    "control_step": steps // args.envs + 1,
                                    "robot": robot,
                                    "event": event_names[event],
                                    "skill": SKILLS[int(choices[robot][0])],
                                    "passive_drill": bool(passive_drill[0]),
                                }
                            )
                    for j, name in enumerate(event_names):
                        stats["events"][name] = stats["events"].get(name, 0) + int(
                            info["events"][:, args.role, j].sum()
                        )
                    for skill, event in drill_event.items():
                        episode_event |= active_drill & (drill_ids == skill) & info["events"][:, args.role, event]
                completed = int(done.sum())
                stats["episodes"] += completed
                if completed:
                    for skill, name in enumerate(SKILLS):
                        selected = done & active_drill & (drill_ids == skill)
                        stats["drill_episodes"][name] += int(selected.sum())
                        stats["drill_successes"][name] += int((selected & episode_event).sum())
                    episode_event[done] = False
                    for key, value in extras.get("log", {}).items():
                        stats["metrics"][key] = stats["metrics"].get(key, 0.0) + float(value) * completed
                    outcomes = info.get("outcome", torch.zeros(args.envs, teams, device=env.device))[:, args.role]
                    stats["wins"] += int((done & (outcomes > 0)).sum())
                    stats["losses"] += int((done & (outcomes < 0)).sum())
                    stats["draws"] += int((done & (outcomes == 0)).sum())
                    for j, member in enumerate(members):
                        selected = done & (assignment == j) & ~active_drill
                        row = stats["opponents"].setdefault(member["id"], {"games": 0, "score": 0.0})
                        row["games"] += int(selected.sum())
                        row["score"] += float(((outcomes[selected] + 1) / 2).sum())
                    assignment[done] = torch.randint(max(1, len(frozen)), (completed,), device=env.device)
                for i in range(teams):
                    previous[i], remaining[i] = option_clock(choices[i], remaining[i], done)
                if writer is not None and (steps // args.envs) % 2 == 0:
                    from PIL import Image, ImageDraw

                    frame = Image.fromarray(env.render())
                    panel = env.sensor_preview()
                    frame.paste(panel, (0, frame.height - panel.height))
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, frame.width, 70), fill=(8, 15, 24))
                    draw.text(
                        (10, 5),
                        f"{args.phase} | {args.source.parent.name}/{args.source.name} | seed {args.seed} | {args.view}",
                        fill="white",
                    )
                    draw.text(
                        (10, 25),
                        " | ".join(f"robot_{i}: {SKILLS[int(choices[i][0])]}" for i in range(teams)),
                        fill="white",
                    )
                    draw.text((10, 45), f"Events (this clip): {stats['events']}", fill="white")
                    writer.append_data(np.asarray(frame))
                    video_frames += 1
                steps += args.envs
            if not args.evaluate:
                pixels, vector = unpack(obs[f"team_{args.role}"][f"robot_{args.role}"])
                with torch.no_grad():
                    bootstrap = (
                        models[args.role]
                        .value(models[args.role].features(pixels, vector, previous[args.role], remaining[args.role]))
                        .squeeze(-1)
                    )
                data = {key: torch.stack([row[key] for row in rollout]) for key in rollout[0]}
                advantage, returns = advantages(data.pop("reward"), data.pop("value"), data.pop("done"), bootstrap)
                data.update(advantage=advantage, **{"return": returns})
                flat = {key: value.flatten(0, 1) for key, value in data.items()}
                losses = ppo_update(models[args.role], optimizers[args.role], flat, minibatch=cfg["minibatch"])
                updates += 1
                if updates == 1 or updates % 10 == 0:
                    save()
                write_json(
                    args.output / "heartbeat.json",
                    {"steps": steps, "updates": updates, "time": time.time(), "pid": os.getpid(), **losses},
                )
                print(json.dumps({"steps": steps, "fps": steps / (time.time() - started), **losses}), flush=True)
            else:
                print(f"Evaluation {steps}/{args.steps}", flush=True)
        if not args.evaluate:
            save(final=True)
        stats.update(
            phase=args.phase,
            role=args.role,
            seed=args.seed,
            steps=steps,
            video_frames=video_frames,
            sensor_audit=audit,
            upright_fraction=stats["upright"] / max(stats["samples"], 1),
            checkpoint=str(args.source)
            if args.evaluate
            else json.loads((args.output / "receipt.json").read_text())["checkpoint"],
        )
        stats["metrics"] = {k: v / max(stats["episodes"], 1) for k, v in stats["metrics"].items()}
        stats["upright_by_robot"] = {k: v / max(stats["samples"], 1) for k, v in stats["upright_by_robot"].items()}
        write_json(args.output / "evaluation.json", stats)
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
