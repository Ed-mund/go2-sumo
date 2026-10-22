#!/usr/bin/env python3
"""Prepare/run v3 tactics with isolated smoke tests, role-balanced training and a frozen opponent pool."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

from sumo_campaign import (
    HERE,
    REPO,
    VIEWS,
    child,
    gallery,
    preflight,
    rounded_steps,
    runtime_fingerprint,
    source_fingerprint,
    write_json,
)

from tactics_contract import CONTRACT, SKILLS

PHASES = ("locomotion", "push", "basic_sumo", "drills", "tactics", "league")


def config(path):
    value = json.loads(Path(path).read_text())
    if value.get("python") == "@current":
        value["python"] = sys.executable
    if value.get("version") != 3 or value.get("contract") != CONTRACT:
        raise ValueError("Unsupported v3 contract")
    if tuple(p["name"] for p in value["stages"]) != PHASES:
        raise ValueError("The full six-phase curriculum is required")
    for field in (
        "chunk_steps",
        "rollout_steps",
        "minibatch",
        "evaluation_envs",
        "evaluation_controls",
        "video_controls",
        "minimum_episodes",
    ):
        if not isinstance(value[field], int) or value[field] <= 0:
            raise ValueError(f"Invalid {field}")
    for stage in value["stages"]:
        if stage["envs"] <= 0 or stage["steps"] <= 0 or not math.isfinite(stage["lr"]) or stage["lr"] <= 0:
            raise ValueError("Invalid stage budget/count/rate")
        if not 0 <= stage["drill_fraction"] <= 1:
            raise ValueError("Invalid drill mixture")
    return value


def fingerprints(cfg):
    source = source_fingerprint()
    for path in sorted(HERE.glob("tactics_*.py")):
        source[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"source": source, "runtime": runtime_fingerprint(cfg)}


def command(
    cfg,
    run,
    phase,
    output,
    source,
    steps,
    envs,
    role=0,
    evaluation=False,
    video=False,
    view="corner",
    fixed=-1,
    seed=101,
    audit=False,
):
    cmd = [
        cfg["python"],
        str(HERE / "tactics_train.py"),
        "--config",
        str(run / "config.json"),
        "--output",
        str(output),
        "--phase",
        phase["name"],
        "--steps",
        str(steps),
        "--envs",
        str(envs),
        "--role",
        str(role),
        "--seed",
        str(seed),
        "--view",
        view,
    ]
    if source:
        cmd += ["--source", str(source)]
    if evaluation:
        cmd += ["--evaluate"]
    if video:
        cmd += ["--video"]
    if view == "rough_corner":
        cmd += ["--variant", "1"]
    if fixed >= 0:
        cmd += ["--fixed-drill", str(fixed)]
    if audit:
        cmd += ["--zero-actions"]
        if phase["task"] == "Fight":
            cmd += ["--sensor-audit"]
    return cmd


def quality_errors(report, phase, minimum=16, skill=None):
    errors = []
    if report.get("episodes", 0) < minimum:
        errors.append("Too few completed episodes")
    upright = report.get("upright_fraction", float("nan"))
    if not math.isfinite(upright) or upright < (0.80 if skill == "recover" else 0.90):
        errors.append("Insufficient upright time")
    metrics = report.get("metrics", {})
    required = {"locomotion": {"Metrics/goal_success_percentage": 0.8}, "push": {"Metrics/push_success": 0.8}}.get(
        phase, {}
    )
    for key, minimum_value in required.items():
        value = metrics.get(key, float("nan"))
        if not math.isfinite(value) or value < minimum_value:
            errors.append(f"{key} did not reach {minimum_value}")
    if skill is not None:
        episodes = report.get("drill_episodes", {}).get(skill, 0)
        wins = report.get("drill_successes", {}).get(skill, 0)
        if episodes < minimum or wins / max(episodes, 1) < 0.2:
            errors.append(f"{skill}: fewer than {minimum} trials or less than 20% success")
    elif phase in ("basic_sumo", "tactics", "league"):
        episodes = max(report.get("episodes", 0), 1)
        if report.get("draws", episodes) / episodes > 0.7:
            errors.append("Too many unresolved/double-loss games")
        if report.get("wins", 0) / episodes < 0.35:
            errors.append("Win rate below 35% against sampled historical opponents")
        if phase in ("tactics", "league"):
            usage = report.get("skill_controls", {})
            if sum(v / max(sum(usage.values()), 1) >= 0.02 for v in usage.values()) < 3:
                errors.append("Manager collapsed to fewer than three skill choices")
            events = report.get("events", {})
            families = (
                events.get("sweep_left", 0) + events.get("sweep_right", 0),
                events.get("flank_left", 0) + events.get("flank_right", 0),
                events.get("counter", 0),
                events.get("escape", 0),
                events.get("pin", 0),
            )
            if sum(v > 0 for v in families) < 2:
                errors.append("No demonstrated variety: fewer than two successful tactic families")
    return errors


def invoke(cmd, output, cfg, state, state_path):
    child(cmd, output / "process.log", cfg, state, state_path)
    if "--evaluate" not in cmd:
        receipt = json.loads((output / "receipt.json").read_text())
        expected = int(cmd[cmd.index("--steps") + 1])
        if not receipt.get("complete") or receipt.get("steps") != expected:
            raise RuntimeError("Incomplete tactical training chunk")
        if not Path(receipt["checkpoint"]).is_file():
            raise RuntimeError("Training receipt checkpoint is missing")
    return json.loads((output / "evaluation.json").read_text())


def run_campaign(cfg, run, state, state_path, smoke_only=False):
    preflight(cfg)
    # The controller stays stdlib-only; model work runs in the configured conda child.
    from tactics_league import League

    league = League(run / "league.json")
    if not (run / "smoke/passed.json").exists():
        state["state"] = "validating"
        source = None
        for phase in cfg["stages"]:
            output = run / "smoke" / phase["name"]
            envs = min(32, phase["envs"])
            count = envs * cfg["rollout_steps"]
            invoke(command(cfg, run, phase, output, source, count, envs), output, cfg, state, state_path)
            source = json.loads((output / "receipt.json").read_text())["checkpoint"]
            # Actual optimizer restore and role-1 training are exercised, not just playback.
            resumed = output / "resume"
            role = 0 if phase["task"] == "Locomotion" else 1
            invoke(command(cfg, run, phase, resumed, source, count, envs, role), resumed, cfg, state, state_path)
            source = json.loads((resumed / "receipt.json").read_text())["checkpoint"]
            benchmark = output / "benchmark"
            invoke(
                command(cfg, run, phase, benchmark, source, phase["envs"] * cfg["rollout_steps"], phase["envs"]),
                benchmark,
                cfg,
                state,
                state_path,
            )
            if phase["name"] in ("locomotion", "push", "basic_sumo"):
                clip = run / "videos/smoke" / phase["name"]
                report = invoke(
                    command(
                        cfg,
                        run,
                        phase,
                        clip,
                        source,
                        cfg["video_controls"],
                        1,
                        evaluation=True,
                        video=True,
                        view="low_side",
                        audit=True,
                    ),
                    clip,
                    cfg,
                    state,
                    state_path,
                )
                if not report.get("upright_by_robot") or any(
                    not math.isfinite(v) or v < 0.90 for v in report["upright_by_robot"].values()
                ):
                    raise RuntimeError("Passive standing failed")
                if phase["task"] == "Fight" and not report.get("sensor_audit", {}).get("passed"):
                    raise RuntimeError("Moving-target sensor audit failed")
        # Smoke weights never become the production starting policy.
        write_json(
            run / "smoke/passed.json",
            {
                "time": time.time(),
                "scope": "all phases, optimizer resume, both roles, production benchmarks, passive/sensor video audits",
            },
        )
        gallery(run)
    if smoke_only:
        state["state"] = "smoke_passed"
        return
    if state.get("active_chunk"):
        receipt = Path(state["active_chunk"]) / "receipt.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            state.update(steps=state["chunk_base"] + saved["steps"], checkpoint=saved["checkpoint"])
        state.pop("active_chunk")
    for index in range(state["stage_index"], len(cfg["stages"])):
        phase = cfg["stages"][index]
        state.update(stage=phase["name"], state="training", stage_index=index)
        while state["steps"] < phase["steps"]:
            base = state["steps"]
            role = 0 if phase["task"] == "Locomotion" else state.get("chunks", 0) % 2
            count = rounded_steps(min(cfg["chunk_steps"], phase["steps"] - base), phase["envs"], cfg["rollout_steps"])
            output = run / "checkpoints" / phase["name"] / f"from_{base:012d}_{time.time_ns()}"
            source = state.get("checkpoint") or state.get("previous_checkpoint")
            selected = (
                []
                if phase["task"] != "Fight"
                else league.sample(
                    random.Random(cfg["seed"] + base + index),
                    3,
                    require_style="pusher" if phase["name"] == "drills" else None,
                )
            )
            write_json(output / "opponents.json", selected)
            cmd = command(cfg, run, phase, output, source, count, phase["envs"], role, seed=cfg["seed"])
            cmd += ["--opponents", str(output / "opponents.json")]
            state.update(active_chunk=str(output), chunk_base=base)
            write_json(state_path, state)
            report = invoke(cmd, output, cfg, state, state_path)
            receipt = json.loads((output / "receipt.json").read_text())
            if not receipt["complete"] or receipt["steps"] != count:
                raise RuntimeError("Incomplete tactical training chunk")
            source = receipt["checkpoint"]
            state.update(steps=base + count, checkpoint=source, chunks=state.get("chunks", 0) + 1)
            state.pop("active_chunk")
            write_json(state_path, state)
            league.record(report["opponents"])
            if phase["task"] == "Fight":
                league.add(source, phase["name"], role, -1 if phase["manager"] else 0)
                if phase["name"] == "drills":
                    for skill in range(1, 9):
                        league.add(source, SKILLS[skill], role, skill)
            elif phase["name"] == "push":
                league.add(source, "pusher", role, 0)
            view = list(VIEWS)[state["chunks"] % len(VIEWS)]
            clip = run / "videos" / phase["name"] / f"steps_{state['steps']:012d}" / view
            cmd = command(
                cfg, run, phase, clip, source, cfg["video_controls"], 1, role, evaluation=True, video=True, view=view
            )
            cmd += ["--opponents", str(output / "opponents.json")]
            video_report = invoke(cmd, clip, cfg, state, state_path)
            if video_report["video_frames"] != math.ceil(cfg["video_controls"] / 2):
                raise RuntimeError("Incomplete review clip")
            gallery(run)
        errors = []
        roles = (0,) if phase["task"] == "Locomotion" else (0, 1)
        selections = (
            league.sample(
                random.Random(2026 + index),
                3,
                exclude=(state["checkpoint"],),
                require_style="pusher" if phase["name"] == "drills" else None,
            )
            if phase["task"] == "Fight"
            else []
        )
        opponents = run / "evaluations" / phase["name"] / "opponents.json"
        write_json(opponents, selections)
        skills = range(1, 9) if phase["name"] == "drills" else (-1,)
        for role in roles:
            for skill in skills:
                for seed in (101, 202):
                    folder = run / "evaluations" / phase["name"] / f"role{role}_skill{skill}_seed{seed}"
                    cmd = command(
                        cfg,
                        run,
                        phase,
                        folder,
                        state["checkpoint"],
                        cfg["evaluation_envs"] * cfg["evaluation_controls"],
                        cfg["evaluation_envs"],
                        role,
                        evaluation=True,
                        fixed=skill,
                        seed=seed,
                    )
                    cmd += ["--opponents", str(opponents)]
                    report = invoke(cmd, folder, cfg, state, state_path)
                    errors.extend(
                        f"{folder.name}: {error}"
                        for error in quality_errors(
                            report, phase["name"], cfg["minimum_episodes"], SKILLS[skill] if skill >= 0 else None
                        )
                    )
                if skill >= 0:
                    clip = run / "videos" / phase["name"] / "skill_reviews" / f"role{role}_{SKILLS[skill]}"
                    cmd = command(
                        cfg,
                        run,
                        phase,
                        clip,
                        state["checkpoint"],
                        cfg["video_controls"],
                        1,
                        role,
                        evaluation=True,
                        video=True,
                        view="low_side",
                        fixed=skill,
                    )
                    cmd += ["--opponents", str(opponents)]
                    invoke(cmd, clip, cfg, state, state_path)
        gallery(run)
        write_json(run / "evaluations" / phase["name"] / "gate.json", {"passed": not errors, "errors": errors})
        if errors:
            state["state"] = "quality_gate_failed"
            raise RuntimeError("Phase needs review; " + "; ".join(errors[:6]))
        state.update(previous_checkpoint=state["checkpoint"], checkpoint=None, steps=0, chunks=0, stage_index=index + 1)
        write_json(state_path, state)
    state["state"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Defaults to the saved run config, or the portable v3 template")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, help="Interpreter to save when preparing a new run")
    parser.add_argument("--envs", type=int, help="Override all stage counts when preparing a new test run")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument(
        "--smoke-only", action="store_true", help="Validate all phases and stop before production training"
    )
    modes.add_argument("--status", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    state_path = run / "status.json"
    if args.status:
        state = json.loads(state_path.read_text())
        if state.get("state") in ("training", "validating", "starting"):
            try:
                os.kill(state.get("pid", -999999), 0)
            except (ProcessLookupError, PermissionError):
                state["state"] = "interrupted"
        print(json.dumps(state, indent=2))
        return
    if (args.python or args.envs is not None) and not args.prepare:
        parser.error("--python and --envs are preparation-only; subsequent commands use the saved config")
    if args.envs is not None and args.envs <= 0:
        parser.error("--envs must be positive")
    cfg = config(
        args.config or (run / "config.json" if (run / "config.json").exists() else HERE / "configs/go2_tactics_v3.json")
    )
    if args.python:
        cfg["python"] = str(args.python.expanduser().absolute())
    if args.envs is not None:
        for stage in cfg["stages"]:
            stage["envs"] = args.envs
    run.mkdir(parents=True, exist_ok=True)
    with (run / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, value in (("config.json", cfg), ("fingerprints.json", fingerprints(cfg))):
            path = run / name
            if path.exists() and json.loads(path.read_text()) != value:
                raise RuntimeError("Configuration/source/runtime changed; prepare a new v3 run directory")
            write_json(path, value)
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"state": "prepared", "stage_index": 0, "steps": 0, "chunks": 0, "checkpoint": None}
        )
        if args.prepare:
            write_json(state_path, state)
            gallery(run)
            print(f"Prepared {run}; no simulator, CUDA or training launched.")
            return
        state.update(pid=os.getpid(), state="starting")

        def stop(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        try:
            run_campaign(cfg, run, state, state_path, smoke_only=args.smoke_only)
        except BaseException as exc:
            if state["state"] != "quality_gate_failed":
                state["state"] = "stopped" if isinstance(exc, KeyboardInterrupt) else "failed"
            state["error"] = str(exc)
            raise
        finally:
            state["updated"] = time.time()
            write_json(state_path, state)


if __name__ == "__main__":
    main()
