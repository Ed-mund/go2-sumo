# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Play an algorithm (supports both coordination + adversarial HARL runners)."""

import argparse
import os
import pprint
import sys
from pathlib import Path

import imageio.v2 as imageio
import torch
from tqdm import tqdm

from huggingface_hub import snapshot_download

from isaaclab.app import AppLauncher
from isaaclab.utils import HF_POLICY_MAP, HF_REPO_ID, policies_summary

parser = argparse.ArgumentParser(
    description="Play an RL agent with HARL.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=policies_summary(HF_POLICY_MAP),
)

parser.add_argument(
    "--algorithm",
    type=str,
    default="happo",
    choices=[
        "happo",
        "hatrpo",
        "haa2c",
        "haddpg",
        "hatd3",
        "hasac",
        "had3qn",
        "maddpg",
        "matd3",
        "mappo",
        "happo_adv",
    ],
    help="Algorithm name.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_env_steps", type=int, default=None, help="Total environment steps to play.")
parser.add_argument("--dir", type=str, default=None, help="Folder with trained models (local path).")
parser.add_argument("--debug", action="store_true", help="Run in debug mode for visualization.")
parser.add_argument(
    "--show_colliders",
    action="store_true",
    help="Show PhysX collision shapes in the rendered viewport for geometry inspection.",
)
parser.add_argument("--video", action="store_true", help="Record playback to an MP4 file.")
parser.add_argument("--video_length", type=int, default=1500, help="Number of simulation steps to record.")
parser.add_argument("--video_dir", type=str, default="videos/play", help="Directory for recorded playback.")
parser.add_argument("--evaluation_json", type=str, default=None, help="Write machine-readable evaluation metrics")
parser.add_argument("--sensor_audit", action="store_true", help="Run controlled Go2 sensor/moving-target checks before playback")
parser.add_argument("--arena_variant", type=int, default=0, help="V2 terrain variant: 0 flat, 1-3 rough")
parser.add_argument("--video_stride", type=int, default=1, help="Record every N control steps at matching playback FPS")
parser.add_argument("--video_label", type=str, default="", help="Burn checkpoint/stage identification into review clips")
parser.add_argument("--strict_checkpoint", action="store_true", help="Abort if any actor fails to load")
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override the viewer camera position in environment coordinates.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override the viewer camera look-at point in environment coordinates.",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Use policy means so repeated seeded camera renders show the same match.",
)
parser.add_argument(
    "--report_heading",
    action="store_true",
    help="Report whether each quadruped faces its goal/opponent and whether it travels backwards.",
)
parser.add_argument(
    "--zero_actions",
    action="store_true",
    help="Hold all joint actions at zero; useful for auditing reset poses and spawn headings.",
)
parser.add_argument(
    "--load_starting_policy",
    action="store_true",
    help="If set, load the starting policy for this env from HuggingFace (if one exists).",
)
parser.add_argument(
    "--load_trained_policy",
    action="store_true",
    help="If set, load the trained policy for this env from HuggingFace (if one exists).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video_stride < 1:
    parser.error("--video_stride must be positive")
if args_cli.arena_variant < 0:
    parser.error("--arena_variant must be nonnegative")
if args_cli.strict_checkpoint and args_cli.algorithm != "happo_adv":
    parser.error("Strict campaign playback currently supports happo_adv only")
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# --------------------------------------------------------------------------------------
# Launch Omniverse
# --------------------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if args_cli.show_colliders:
    import carb

    carb.settings.get_settings().set_int("/persistent/physics/visualizationDisplayColliders", 2)

# --------------------------------------------------------------------------------------
# Imports that require the app
# --------------------------------------------------------------------------------------

from harl.runners import RUNNER_REGISTRY  # noqa: E402

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.direct.adversarial.sumo.heading import (  # noqa: E402
    anatomical_forward_xy,
    resolve_heading_target_position,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"harl_{algorithm}_cfg_entry_point"


def _max_action_dim(action_space) -> int:
    """Recursively find the maximum action dimension across nested dict action spaces."""
    if isinstance(action_space, dict):
        if len(action_space) == 0:
            return 0
        return max(_max_action_dim(v) for v in action_space.values())
    # assume a gymnasium space-like object with .shape
    shape = getattr(action_space, "shape", None)
    if not shape:
        return 0
    return int(shape[0])


def _configure_model_dir(args: dict, algo_args: dict) -> None:
    """Apply HF/local policy loading rules and set algo_args['train']['model_dir']."""
    task_name = args.get("task")
    provided_dir = bool(args.get("dir"))
    load_trained = bool(args.get("load_trained_policy"))
    load_starting = bool(args.get("load_starting_policy"))

    # mutual exclusivity:
    if load_trained and load_starting:
        raise ValueError("Cannot set both --load_trained_policy and --load_starting_policy.")
    if provided_dir and (load_trained or load_starting):
        raise ValueError("Cannot combine --dir with --load_trained_policy/--load_starting_policy.")

    # default: use --dir (if provided)
    if provided_dir:
        algo_args["train"]["model_dir"] = args["dir"]
        return

    # HF load
    if load_trained:
        entry = HF_POLICY_MAP.get(task_name, {})
        policy_location = entry.get("trained")
        if not policy_location:
            print("Sorry, a trained policy doesn't exist for this env.")
            return

        base = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision="main",
            allow_patterns=[f"{policy_location}/**"],
        )
        algo_args["train"]["model_dir"] = os.path.join(base, policy_location)
        return

    if load_starting:
        entry = HF_POLICY_MAP.get(task_name, {})
        policy_location = entry.get("starting")
        if not policy_location:
            print("Sorry, a starting policy doesn't exist for this env.")
            return

        base = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision="main",
            allow_patterns=[f"{policy_location}/**"],
        )
        algo_args["train"]["model_dir"] = os.path.join(base, policy_location)
        return
    return


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    policies_summary(HF_POLICY_MAP)

    args = args_cli.__dict__

    # HARL runner args
    args["env"] = "isaaclab"
    args["algo"] = args["algorithm"]
    args["exp_name"] = "play"

    is_adv = "adv" in str(args["algo"]).lower()

    algo_args = agent_cfg
    algo_args["eval"]["use_eval"] = False
    algo_args["render"]["use_render"] = True

    # Keep policy tensors on the same device as playback. This also makes
    # ``--device cpu`` a useful low-VRAM rendering path while a large GPU
    # training job is active.
    playback_device = str(args.get("device") or env_cfg.sim.device)
    env_cfg.sim.device = playback_device
    algo_args["device"]["cuda"] = playback_device.startswith("cuda")

    # apply model_dir logic (HF and/or local dir)
    algo_args.setdefault("train", {})
    _configure_model_dir(args, algo_args)

    # Env config
    env_args: dict = {}
    if args.get("num_envs") is not None:
        env_cfg.scene.num_envs = args["num_envs"]
    else:
        # ensure downstream uses an int
        args["num_envs"] = int(env_cfg.scene.num_envs)

    if args.get("seed") is not None:
        env_cfg.seed = int(args["seed"])
    if hasattr(env_cfg, "arena_variant_offset"):
        env_cfg.arena_variant_offset = args["arena_variant"]
    if args.get("camera_eye") is not None:
        env_cfg.viewer.eye = tuple(args["camera_eye"])
    if args.get("camera_lookat") is not None:
        env_cfg.viewer.lookat = tuple(args["camera_lookat"])

    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    # Record frames explicitly below. Isaac Sim's Gym RecordVideo/Replicator
    # wrapper can hang while unloading plugins in headless mode.
    env_args["video_settings"] = {"video": False}
    env_args["render_mode"] = "rgb_array" if args["video"] else "human"
    env_args["headless"] = args["headless"]
    env_args["debug"] = args["debug"]

    # create runner
    checkpoint_source = algo_args["train"].get("model_dir")
    if args["strict_checkpoint"]:
        algo_args["train"]["model_dir"] = None
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    if args["strict_checkpoint"]:
        from sumo_checkpoint import strict_restore

        if not checkpoint_source:
            raise ValueError("Strict playback requires --dir")
        strict_restore(runner, checkpoint_source, actors_only=True)

    video_writer = None
    video_path = None
    video_frames = 0
    if args["video"]:
        video_dir = Path(args["video_dir"]).expanduser().resolve()
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / "sumo_trained_match.mp4"
        fps = 1.0 / (env_cfg.sim.dt * env_cfg.decimation * args["video_stride"])
        video_writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)

    obs, _, _ = runner.env.reset()
    sensor_audit = None
    if args["sensor_audit"]:
        from sensor_audit import audit_onboard_sensors
        sensor_audit = audit_onboard_sensors(runner.env.unwrapped)
        obs, _, _ = runner.env.reset()

    # determine max action dim (supports nested dict action spaces for adv)
    max_action_space = 0
    for _, sp in runner.env.action_space.items():
        max_action_space = max(max_action_space, _max_action_dim(sp))

    device = runner.device

    actions = torch.zeros(
        (args["num_envs"], runner.num_agents, max_action_space),
        dtype=torch.float32,
        device=device,
    )
    rnn_states = torch.zeros(
        (args["num_envs"], runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size),
        dtype=torch.float32,
        device=device,
    )
    masks = torch.ones((args["num_envs"], runner.num_agents, 1), dtype=torch.float32, device=device)

    # logging + progress bar (from adv script)
    log_infos: dict[str, float] = {}
    envs_completed = 0
    heading_totals: dict[str, dict[str, float]] = {}
    posture_totals = {}
    played_steps = 0
    pbar = tqdm(total=args["num_env_steps"], desc="Playing")

    while simulation_app.is_running():
        with torch.inference_mode():
            if is_adv:
                # obs: {team: {agent_id: obs_tensor}}
                for team, agents_obs in obs.items():
                    for agent_id, agent_obs in agents_obs.items():
                        agent_num = runner.env.env._agent_map[agent_id]
                        action, _, rnn_state = runner.actors[team][agent_id].get_actions(
                            agent_obs,
                            rnn_states[:, agent_num, :],
                            masks[:, agent_num, :],
                            None,
                            args["deterministic"],
                        )
                        a_dim = action.shape[1]
                        actions[:, agent_num, :a_dim] = action
                        rnn_states[:, agent_num, :] = rnn_state
            else:
                # obs: {agent_name: obs_tensor}  (or similar mapping)
                for agent_num, agent_obs in enumerate(obs.values()):
                    action, _, rnn_state = runner.actor[agent_num].get_actions(
                        agent_obs,
                        rnn_states[:, agent_num, :],
                        masks[:, agent_num, :],
                        None,
                        None,
                    )
                    a_dim = action.shape[1]
                    actions[:, agent_num, :a_dim] = action
                    rnn_states[:, agent_num, :] = rnn_state

        if args["zero_actions"]:
            actions.zero_()

        # Sample posture before auto-reset can hide terminal falls.
        task_env = runner.env.env
        for key, robot in getattr(task_env, "robots", {}).items():
            up = -robot.data.projected_gravity_b[:, 2]
            totals = posture_totals.setdefault(key, {"upright": 0, "inverted": 0, "samples": 0})
            totals["upright"] += int((up > 0.5).sum().item())
            totals["inverted"] += int((up < -0.25).sum().item())
            totals["samples"] += up.numel()
        # step env
        obs, _, rewards, dones, _, _ = runner.env.step(actions)
        if args["strict_checkpoint"]:
            from sumo_checkpoint import assert_finite

            assert_finite((obs, rewards, actions))
        played_steps += 1

        if args["report_heading"]:
            task_env = runner.env.env
            robots = getattr(task_env, "robots", {})
            robot_positions = {robot_id: robot.data.root_pos_w for robot_id, robot in robots.items()}
            blocks = getattr(task_env, "blocks", {})
            block_positions = {block_id: block.data.root_pos_w for block_id, block in blocks.items()}
            for robot_id, robot in robots.items():
                quat = robot.data.root_quat_w
                forward_yaw_offset = getattr(task_env.cfg, "forward_yaw_offsets", {}).get(robot_id, 0.0)
                forward_xy = anatomical_forward_xy(quat, forward_yaw_offset)
                target_position, target_source = resolve_heading_target_position(
                    robot_id,
                    robot_positions,
                    desired_position=getattr(task_env, "_desired_pos", None),
                    block_positions=block_positions,
                )
                if target_position is None:
                    continue
                target_xy = target_position[:, :2] - robot.data.root_pos_w[:, :2]
                target_xy = target_xy / torch.linalg.norm(target_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
                facing = torch.sum(forward_xy * target_xy, dim=-1)
                velocity_xy = robot.data.root_lin_vel_w[:, :2]
                speed = torch.linalg.norm(velocity_xy, dim=-1)
                moving = speed > 0.15
                moving_backward = moving & (torch.sum(velocity_xy * forward_xy, dim=-1) < 0.0)
                values = heading_totals.setdefault(
                    robot_id,
                    {
                        "alignment": 0.0,
                        "reverse_facing": 0.0,
                        "moving": 0.0,
                        "backing": 0.0,
                        "samples": 0.0,
                        "target_source": target_source,
                    },
                )
                values["alignment"] += float(facing.sum().item())
                values["reverse_facing"] += float((facing < 0.0).sum().item())
                values["moving"] += float(moving.sum().item())
                values["backing"] += float(moving_backward.sum().item())
                values["samples"] += float(facing.numel())

        if video_writer is not None and played_steps <= int(args["video_length"]) and (played_steps - 1) % args["video_stride"] == 0:
            frame = runner.env.unwrapped.render()
            if frame is not None:
                preview = getattr(runner.env.unwrapped, "sensor_preview", None)
                if preview is not None:
                    import numpy as np
                    from PIL import Image
                    overlay = Image.fromarray(frame)
                    panel = preview()
                    overlay.paste(panel, (0, overlay.height - panel.height))
                    frame = np.asarray(overlay)
                if args["video_label"]:
                    import numpy as np
                    from PIL import Image, ImageDraw

                    overlay = Image.fromarray(frame)
                    draw = ImageDraw.Draw(overlay)
                    draw.rectangle((0, 0, overlay.width, 48), fill=(8, 15, 24))
                    draw.text((12, 8), args["video_label"], fill=(220, 235, 245))
                    draw.text((12, 28), f"Simulation {played_steps * env_cfg.sim.dt * env_cfg.decimation:.2f}s | seed {args['seed']}", fill=(170, 200, 210))
                    frame = np.asarray(overlay)
                video_writer.append_data(frame)
                video_frames += 1

        # episode completion aggregation
        dones_env = torch.all(dones, dim=1)
        curr_envs_completed = int(dones_env.sum().item())
        envs_completed += curr_envs_completed

        if curr_envs_completed > 0 and hasattr(runner.env, "log_info"):
            for k, v in runner.env.log_info.items():
                if k not in log_infos:
                    log_infos[k] = 0.0
                log_infos[k] += float(v) * curr_envs_completed

        # reset masks/rnn where done
        masks = torch.ones((args["num_envs"], runner.num_agents, 1), dtype=torch.float32, device=device)
        masks[dones_env] = 0.0

        if curr_envs_completed > 0:
            rnn_states[dones_env] = torch.zeros(
                (curr_envs_completed, runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size),
                dtype=torch.float32,
                device=device,
            )

        # update pbar using total env-steps (matches adv script semantics)
        # Count control steps, not internal physics substeps. The previous
        # implementation counted decimation (4) and produced videos one quarter
        # of the requested duration.
        control_steps = pbar.n // int(args["num_envs"]) + 1
        num_steps = int(args["num_envs"]) * control_steps
        pbar.update(max(0, num_steps - pbar.n))

        if num_steps >= int(args["num_env_steps"]):
            break

    if video_writer is not None:
        video_writer.close()
        print(f"Video saved to: {video_path}")
    runner.env.close()
    pbar.close()

    # print averaged log infos (from adv script)
    completed_count = envs_completed
    if envs_completed <= 0:
        envs_completed = 1
    for k in list(log_infos.keys()):
        log_infos[k] /= envs_completed
    if log_infos:
        pprint.pprint(log_infos)
    for robot_id, values in heading_totals.items():
        samples = max(values["samples"], 1.0)
        moving = max(values["moving"], 1.0)
        print(
            f"Heading audit {robot_id} target={values['target_source']}: "
            f"mean_target_alignment={values['alignment'] / samples:.3f}; "
            f"facing_away={100.0 * values['reverse_facing'] / samples:.1f}%; "
            f"backing_while_moving={100.0 * values['backing'] / moving:.1f}%"
        )
    if args["evaluation_json"]:
        from sumo_checkpoint import atomic_json

        atomic_json(args["evaluation_json"], {
            "task": args["task"], "checkpoint": checkpoint_source, "seed": args["seed"],
            "episodes": completed_count, "control_steps": played_steps,
            "video": str(video_path) if video_path else None, "video_frames": video_frames,
            "metrics": log_infos,
            "sensor_audit": sensor_audit,
            "posture": {key: {name: value / max(totals["samples"], 1)
                              for name, value in totals.items() if name != "samples"}
                        for key, totals in posture_totals.items()},
            "heading": {key: {"facing_away": values["reverse_facing"] / max(values["samples"], 1),
                              "backing_while_moving": values["backing"] / max(values["moving"], 1)}
                        for key, values in heading_totals.items()},
        })


if __name__ == "__main__":
    main()
    simulation_app.close()
