#!/usr/bin/env python3
"""Prepare, validate and run a resumable dog-sumo curriculum.

Preparation and unit tests never start Isaac or allocate CUDA memory. --run is
explicit, checks GPU availability, then passes a full curriculum smoke before
training. Every chunk ends with an optimizer checkpoint and review video.
"""

import argparse
import fcntl
import hashlib
import html
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_CONFIG = HERE / "configs/go2_arena_v2.json"
VIEWS = {
    "corner": (5.8, -6.0, 4.3),
    "low_side": (0.0, -6.0, 0.95),
    "overhead": (0.01, 0.0, 8.0),
    "rough_corner": (5.8, -6.0, 4.3),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["version"] != 2 or cfg["observation_contract"] != "dog-arena-v2-airpro-1295":
        raise ValueError("Unsupported observation contract")
    if cfg["robot"] != "go2" or cfg["opponent"] != "go2":
        raise ValueError("Only the Go2/Go2 profile is enabled. Audit/register other morphologies before enabling them.")
    for key in (
        "chunk_steps",
        "rollout_steps",
        "video_control_steps",
        "video_stride",
        "evaluation_envs",
        "evaluation_control_steps",
        "minimum_evaluation_episodes",
    ):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if [stage["name"] for stage in cfg["stages"]] != ["locomotion", "push", "fight", "selfplay"]:
        raise ValueError("All four curriculum phases must be present in order")
    for stage in cfg["stages"]:
        if stage["envs"] <= 0 or stage["steps"] <= 0 or stage["lr"] <= 0:
            raise ValueError("Environment counts, steps and learning rates must be positive")
    return cfg


def source_fingerprint():
    files = [
        HERE / name for name in ("train.py", "play.py", "sumo_campaign.py", "sumo_checkpoint.py", "sensor_audit.py")
    ]
    task_root = REPO / "source/isaaclab_tasks/isaaclab_tasks/direct/adversarial"
    files += sorted((task_root / "sumo").rglob("*.py"))
    files += [task_root / "agents/harl_happo_adv_sensor_cfg.yaml"]
    files += [task_root / "__init__.py"]
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def rounded_steps(requested, envs, rollout):
    return math.ceil(requested / (envs * rollout)) * envs * rollout


def runtime_fingerprint(cfg):
    """Record installed versions and adapter patches without importing torch/Isaac."""
    script = """
import hashlib, importlib.metadata as md, importlib.util, json, sys
from pathlib import Path
packages = {}
for name in ("torch", "isaacsim", "isaaclab", "harl", "gymnasium", "numpy"):
    try:
        packages[name] = md.version(name)
    except md.PackageNotFoundError:
        packages[name] = None
spec = importlib.util.find_spec("harl")
root = Path(next(iter(spec.submodule_search_locations))) if spec else None
paths = sorted((root / "runners").glob("*.py")) + sorted((root / "envs/isaaclab").glob("*.py")) if root else []
print(json.dumps({"python": sys.version, "packages": packages,
                 "harl_adapter_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}))
"""
    return json.loads(
        subprocess.check_output([cfg["python"], "-c", script], text=True, env=dict(os.environ, PYTHONNOUSERSITE="1"))
    )


def gpu_sample():
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
    return tuple(int(part.strip()) for part in line.split(","))


def preflight(cfg):
    free, temp, utilization = gpu_sample()
    if free < cfg["minimum_free_vram_mb"] or utilization > 20:
        raise RuntimeError(f"GPU is occupied: {free} MiB free, {utilization}% utilization. No service was stopped.")
    if temp >= cfg["temperature_limit_c"]:
        raise RuntimeError(f"GPU temperature is {temp} C")
    if shutil.disk_usage(REPO).free < cfg["minimum_free_disk_gb"] * 1024**3:
        raise RuntimeError("Insufficient disk headroom")
    if not Path(cfg["python"]).is_file():
        raise RuntimeError("Configured HARL Python interpreter is missing")


def checkpoint_files(directory, teams):
    directory = Path(directory)
    names = [f"actor_agent_robot_{i}.pt" for i in range(teams)]
    names += [f"team_{i}_{suffix}.pt" for i in range(teams) for suffix in ("critic_agent", "value_normalizer")]
    for name in names:
        if not (directory / name).is_file() or (directory / name).stat().st_size < 256:
            raise RuntimeError(f"Missing or incomplete checkpoint: {directory / name}")


def duplicate_actor(source, target):
    """Homogeneous bootstrap only; critics are reinitialized for the new reward task."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    actor = Path(source) / "actor_agent_robot_0.pt"
    if not actor.is_file():
        raise FileNotFoundError(actor)
    for i in range(2):
        shutil.copy2(actor, target / f"actor_agent_robot_{i}.pt")
    return str(target)


def gate_errors(report, stage, cfg):
    errors = []
    if report.get("episodes", 0) < cfg["minimum_evaluation_episodes"]:
        errors.append("Too few completed evaluation episodes")
    for metric, minimum in stage["gate"].items():
        value = report.get("metrics", {}).get(metric)
        if value is None or not math.isfinite(value) or value < minimum:
            errors.append(f"{metric}={value}; required >= {minimum}")
    expected = 1 if stage["name"] == "locomotion" else 2
    posture = report.get("posture", {})
    if len(posture) != expected:
        errors.append("Missing posture metrics")
    for robot, values in posture.items():
        if not math.isfinite(values.get("upright", float("nan"))) or values.get("upright", 0) < 0.90:
            errors.append(f"{robot}: upright fraction below 90%")
    if stage["name"] in ("fight", "selfplay"):
        timeout = report.get("metrics", {}).get("timeout_percentage")
        if timeout is None or not math.isfinite(timeout) or timeout > 0.70:
            errors.append(f"Timeout fraction {timeout}; required <= 0.70")
    if stage["name"] == "locomotion":
        heading = report.get("heading", {}).get("robot_0", {})
        for key, maximum in (("facing_away", 0.35), ("backing_while_moving", 0.50)):
            value = heading.get(key)
            if value is None or not math.isfinite(value) or value > maximum:
                errors.append(f"Locomotion {key}={value}; required <= {maximum}")
    return errors


def train_command(cfg, stage, steps, experiment, output, source=None, resume=False):
    cmd = [
        cfg["python"],
        str(HERE / "train.py"),
        "--algorithm",
        "happo_adv",
        "--task",
        stage["task"],
        "--num_envs",
        str(stage["envs"]),
        "--num_env_steps",
        str(steps),
        "--seed",
        str(cfg["seed"]),
        "--save_interval",
        "10",
        "--log_interval",
        "1",
        "--exp_name",
        experiment,
        "--adversarial_training_mode",
        stage["mode"],
        "--adversarial_training_iterations",
        "2000000",
        "--actor_lr",
        str(stage["lr"]),
        "--critic_lr",
        str(stage["lr"]),
        "--max_grad_norm",
        "1.0",
        "--campaign_output",
        str(output),
        "--device",
        "cuda:0",
        "--headless",
        "--enable_cameras",
        "--rendering_mode",
        "performance",
    ]
    if source:
        cmd += ["--dir", str(source), "--resume_optimizers" if resume else "--transfer_actors_only"]
    return cmd


def play_command(cfg, stage, source, folder, view="corner", metrics=False, zero=False):
    envs = cfg["evaluation_envs"] if metrics else 1
    steps = cfg["evaluation_control_steps"] if metrics else cfg["video_control_steps"]
    cmd = [
        cfg["python"],
        str(HERE / "play.py"),
        "--algorithm",
        "happo_adv",
        "--task",
        stage["task"],
        "--dir",
        str(source),
        "--strict_checkpoint",
        "--deterministic",
        "--debug",
        "--seed",
        "101",
        "--num_envs",
        str(envs),
        "--num_env_steps",
        str(envs * steps),
        "--headless",
        "--enable_cameras",
        "--device",
        "cuda:0",
        "--rendering_mode",
        "performance",
        "--evaluation_json",
        str(folder / "evaluation.json"),
        "--report_heading",
    ]
    if view == "rough_corner":
        cmd += ["--arena_variant", "1"]
    if zero:
        cmd += ["--zero_actions"]
        if stage["name"] in ("fight", "selfplay"):
            cmd += ["--sensor_audit"]
    if not metrics:
        cmd += [
            "--video",
            "--video_length",
            str(steps),
            "--video_stride",
            str(cfg["video_stride"]),
            "--video_dir",
            str(folder),
            "--video_label",
            f"{stage['name']} | {folder.parent.name} | {view}",
            "--camera_eye",
            *map(str, VIEWS[view]),
            "--camera_lookat",
            "0",
            "0",
            "0.35",
        ]
    return cmd


def stop_child(proc):
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=timeout)
            # Kill remaining helpers in this owned process group as well.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
        except subprocess.TimeoutExpired:
            continue


def child(cmd, log, cfg, state, state_path):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", OMNI_KIT_ACCEPT_EULA="YES")
    with log.open("a") as stream:
        stream.write("\nCOMMAND: " + json.dumps(cmd) + "\n")
        stream.flush()
        proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        hot = 0
        try:
            while proc.poll() is None:
                free, temp, utilization = gpu_sample()
                hot = hot + 1 if temp >= cfg["temperature_limit_c"] else 0
                state.update(updated=time.time(), child_pid=proc.pid, log=str(log), gpu_free_mb=free, gpu_temp_c=temp)
                write_json(state_path, state)
                if hot >= 3 or shutil.disk_usage(REPO).free < cfg["minimum_free_disk_gb"] * 1024**3:
                    raise RuntimeError("Temperature or disk watchdog triggered")
                if time.time() - log.stat().st_mtime > cfg["stall_minutes"] * 60:
                    raise RuntimeError("Training/evaluation log stalled")
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            if proc.returncode:
                raise RuntimeError(f"Process exited {proc.returncode}; see {log}")
        except BaseException:
            stop_child(proc)
            raise
        finally:
            state.pop("child_pid", None)


def gallery(run):
    clips = []
    for path in sorted((run / "videos").rglob("*.mp4")):
        rel = path.relative_to(run).as_posix()
        clips.append(
            f"<article><h2>{html.escape(str(path.parent.relative_to(run)))}</h2>"
            f'<video controls preload="none" src="{html.escape(rel)}"></video></article>'
        )
    page = '<!doctype html><meta charset="utf-8"><title>Go2 sumo review</title><style>body{background:#101923;color:#dcebf5;font:16px system-ui;margin:32px}video{width:min(100%,960px)}article{margin:32px 0}h2{font-size:18px}</style><h1>Go2 vs Go2 — policy review</h1>'
    page += "<p>All scheduled clips are retained, including failures. Views use seed 101. Metrics and checkpoints sit beside the clips.</p>"
    (run / "videos.html").write_text(page + "\n".join(clips))


def render(cfg, stage, source, folder, view, state, state_path, zero=False):
    child(play_command(cfg, stage, source, folder, view, zero=zero), folder / "play.log", cfg, state, state_path)
    report = json.loads((folder / "evaluation.json").read_text())
    if report["video_frames"] < math.ceil(cfg["video_control_steps"] / cfg["video_stride"]):
        raise RuntimeError(f"Incomplete review video: {folder}")
    if zero:
        expected = 1 if stage["name"] == "locomotion" else 2
        posture = report.get("posture", {})
        if len(posture) != expected or any(
            not math.isfinite(v.get("upright", float("nan"))) or v["upright"] < 0.90 for v in posture.values()
        ):
            raise RuntimeError(f"Passive standing audit failed: {folder}")
        if stage["name"] in ("fight", "selfplay") and not report.get("sensor_audit", {}).get("passed"):
            raise RuntimeError("Missing moving-target sensor audit")


def smoke(cfg, run, state, state_path):
    source = None
    for i, original in enumerate(cfg["stages"]):
        stage = dict(original, envs=32)
        if i == 1:
            source = duplicate_actor(source, run / "smoke/handoff")
        output = run / "smoke" / stage["name"]
        child(
            train_command(cfg, stage, 6400, run.name + "_smoke_" + stage["name"], output, source),
            output / "train.log",
            cfg,
            state,
            state_path,
        )
        receipt = json.loads((output / "receipt.json").read_text())
        if not receipt["complete"] or receipt["steps"] != 6400:
            raise RuntimeError("Smoke update did not complete")
        source = receipt["checkpoint"]
        checkpoint_files(source, 1 if i == 0 else 2)
        resumed = output / "resume"
        child(
            train_command(cfg, stage, 6400, run.name + "_resume_" + stage["name"], resumed, source, resume=True),
            resumed / "train.log",
            cfg,
            state,
            state_path,
        )
        resumed_receipt = json.loads((resumed / "receipt.json").read_text())
        if not resumed_receipt["complete"] or resumed_receipt["steps"] != 6400:
            raise RuntimeError("Optimizer resume smoke did not complete")
        source = resumed_receipt["checkpoint"]
        render(cfg, stage, source, run / "videos/smoke" / stage["name"], "low_side", state, state_path, zero=True)
        if i < 3:
            # Measure actual production-size allocation/update, including both cameras.
            benchmark = output / "benchmark"
            count = original["envs"] * cfg["rollout_steps"]
            child(
                train_command(cfg, original, count, run.name + "_benchmark_" + stage["name"], benchmark, source),
                benchmark / "train.log",
                cfg,
                state,
                state_path,
            )
            benchmark_receipt = json.loads((benchmark / "receipt.json").read_text())
            if not benchmark_receipt["complete"] or benchmark_receipt["steps"] != count:
                raise RuntimeError("Production-size benchmark did not complete")
    gallery(run)
    write_json(
        run / "smoke/passed.json",
        {
            "time": time.time(),
            "scope": "all four phases, strict transfers, optimizer resume, finite PPO updates, passive videos and production-size benchmarks",
        },
    )


def run_campaign(cfg, run, state, state_path):
    preflight(cfg)
    if not (run / "smoke/passed.json").exists():
        state["state"] = "validating"
        smoke(cfg, run, state, state_path)
    # Recover the last completed atomic snapshot after interruption; unsaved updates are replayed.
    if state.get("active_chunk"):
        receipt_path = Path(state["active_chunk"]) / "receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            state["steps"] = state["chunk_base_steps"] + receipt["steps"]
            state["checkpoint"] = receipt["checkpoint"]
        state.pop("active_chunk")
        write_json(state_path, state)
    for index in range(state.get("stage_index", 0), len(cfg["stages"])):
        stage = cfg["stages"][index]
        if index != state.get("stage_index", 0):
            state.update(stage_index=index, steps=0, checkpoint=None)
        state.update(stage=stage["name"], state="training")
        if not state.get("checkpoint") and index == 1:
            source = duplicate_actor(state["previous_stage_checkpoint"], run / "handoff")
        else:
            source = state.get("checkpoint") or state.get("previous_stage_checkpoint")
        while state.get("steps", 0) < stage["steps"]:
            base = state.get("steps", 0)
            count = rounded_steps(min(cfg["chunk_steps"], stage["steps"] - base), stage["envs"], cfg["rollout_steps"])
            chunk = run / "checkpoints" / stage["name"] / f"from_{base:012d}_{int(time.time())}"
            resume = bool(state.get("checkpoint"))
            state.update(active_chunk=str(chunk), chunk_base_steps=base)
            write_json(state_path, state)
            child(
                train_command(cfg, stage, count, run.name + "_" + stage["name"] + f"_{base}", chunk, source, resume),
                chunk / "train.log",
                cfg,
                state,
                state_path,
            )
            receipt = json.loads((chunk / "receipt.json").read_text())
            if not receipt["complete"] or receipt["steps"] != count:
                raise RuntimeError("Training exited without the expected final checkpoint")
            source = receipt["checkpoint"]
            checkpoint_files(source, 1 if index == 0 else 2)
            state.update(steps=base + count, checkpoint=source)
            state.pop("active_chunk")
            write_json(state_path, state)
            views = list(VIEWS)
            view = views[(state["steps"] // cfg["chunk_steps"]) % len(views)]
            folder = run / "videos" / stage["name"] / f"steps_{state['steps']:012d}" / view
            render(cfg, stage, source, folder, view, state, state_path)
            gallery(run)
        # Multi-environment evaluation is separate from the single-environment showreel.
        reports = []
        for seed in (101, 202):
            folder = run / "evaluations" / stage["name"] / str(seed)
            command = play_command(cfg, stage, source, folder, metrics=True)
            command[command.index("--seed") + 1] = str(seed)
            child(command, folder / "play.log", cfg, state, state_path)
            report = json.loads((folder / "evaluation.json").read_text())
            reports += gate_errors(report, stage, cfg)
        for view in VIEWS:
            render(cfg, stage, source, run / "videos" / stage["name"] / "final" / view, view, state, state_path)
        gallery(run)
        write_json(run / "evaluations" / stage["name"] / "gate.json", {"passed": not reports, "errors": reports})
        if reports:
            state["state"] = "quality_gate_failed"
            raise RuntimeError("Stage needs review before advancing: " + "; ".join(reports))
        state.update(previous_stage_checkpoint=source, stage_index=index + 1, steps=0, checkpoint=None)
        write_json(state_path, state)
    state.update(state="complete", updated=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--status", action="store_true")
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
    cfg = load_config(args.config)
    run.mkdir(parents=True, exist_ok=True)
    with (run / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = run / "config.json"
        if frozen.exists() and json.loads(frozen.read_text()) != cfg:
            raise RuntimeError("Run config is immutable; use a new run directory for changed settings")
        write_json(frozen, cfg)
        sources = run / "source_hashes.json"
        fingerprint = source_fingerprint()
        if sources.exists() and json.loads(sources.read_text()) != fingerprint:
            raise RuntimeError("Campaign source changed; prepare a new run directory and repeat validation")
        write_json(sources, fingerprint)
        runtime = run / "runtime.json"
        installed = runtime_fingerprint(cfg)
        if runtime.exists() and json.loads(runtime.read_text()) != installed:
            raise RuntimeError("Installed training runtime changed; use a new run directory and repeat validation")
        write_json(runtime, installed)
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "state": "prepared",
                "stage_index": 0,
                "steps": 0,
                "checkpoint": None,
                "prepared": time.time(),
                "robot": cfg["robot"],
                "opponent": cfg["opponent"],
            }
        )
        if args.prepare:
            write_json(state_path, state)
            gallery(run)
            print(f"Prepared {run}. No simulator or GPU process launched.")
            return
        state.update(pid=os.getpid(), updated=time.time(), state="starting")
        write_json(state_path, state)

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        try:
            run_campaign(cfg, run, state, state_path)
        except BaseException as exc:
            if state["state"] != "quality_gate_failed":
                state["state"] = "stopped" if isinstance(exc, KeyboardInterrupt) else "failed"
            state.update(error=str(exc), updated=time.time())
            write_json(state_path, state)
            raise
        write_json(state_path, state)


if __name__ == "__main__":
    main()
