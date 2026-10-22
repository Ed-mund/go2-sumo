# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an algorithm."""

import argparse
import sys
import time

from huggingface_hub import snapshot_download

from isaaclab.app import AppLauncher
from isaaclab.utils import HF_POLICY_MAP, HF_REPO_ID, policies_summary

parser = argparse.ArgumentParser(
    description="Train an RL agent with HARL.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=policies_summary(HF_POLICY_MAP),
)

parser.add_argument("--video", action="store_true", help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=20000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=1, help="Seed used for the environment")
parser.add_argument("--save_interval", type=int, default=None, help="How often to save the model")
parser.add_argument("--save_checkpoints", action="store_true", default=False, help="Whether or not to save checkpoints")
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=200,
    help="How often to save a model checkpoint (episodes, episodes = num_envs*episode_length steps)",
)
parser.add_argument("--log_interval", type=int, default=None, help="How often to log outputs")
parser.add_argument("--exp_name", type=str, default="test", help="Name of the Experiment")
parser.add_argument("--num_env_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--dir", type=str, default=None, help="folder with trained models")
parser.add_argument("--campaign_output", type=str, default=None, help="Atomic checkpoints, optimizer state and heartbeat")
parser.add_argument("--resume_optimizers", action="store_true", help="Resume optimizer and opponent rotation state")
parser.add_argument("--transfer_actors_only", action="store_true", help="Start the next stage with fresh critics")
parser.add_argument(
    "--actor_lr",
    type=float,
    default=None,
    help="Optional actor learning-rate override for continuation/fine-tuning runs.",
)
parser.add_argument(
    "--critic_lr",
    type=float,
    default=None,
    help="Optional critic learning-rate override for continuation/fine-tuning runs.",
)
parser.add_argument(
    "--max_grad_norm",
    type=float,
    default=None,
    help="Optional actor/critic gradient-norm clipping override.",
)
parser.add_argument("--debug", action="store_true", help="whether to run in debug mode for visualization")
parser.add_argument(
    "--adversarial_training_mode",
    default="parallel",
    choices=["parallel", "ladder", "leapfrog"],
    help=(
        "the mode type for adversarial training,                     note on ladder training with teams that are"
        " composed of heterogeneous agents, the two teams must place the robots in the same order in their environment "
        "                    for ladder to work"
    ),
)
parser.add_argument(
    "--adversarial_training_iterations",
    default=50_000_000,
    type=int,
    help="the number of iterations to swap training for adversarial modes like ladder and leapfrog",
)

parser.add_argument(
    "--algorithm",
    type=str,
    default="happo",
    choices=["happo", "hatrpo", "haa2c", "mappo", "mappo_unshare", "happo_adv"],
    help="Algorithm name. Choose from: happo, hatrpo, haa2c, mappo, and mappo_unshare.",
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
if args_cli.campaign_output and args_cli.algorithm != "happo_adv":
    parser.error("Campaign checkpoints currently support happo_adv only")
if args_cli.resume_optimizers or args_cli.transfer_actors_only:
    if not args_cli.campaign_output or not args_cli.dir:
        parser.error("Explicit campaign restore requires --campaign_output and --dir")
    if args_cli.resume_optimizers and args_cli.transfer_actors_only:
        parser.error("Choose optimizer resume OR actor-only stage transfer")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from harl.runners import RUNNER_REGISTRY

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = f"harl_{algorithm}_cfg_entry_point"


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

    # If nothing specified, leave whatever hydra config provides (or None).
    # This mirrors the old behavior where user might rely on the default model_dir in config.
    return


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):

    args = args_cli.__dict__

    args["env"] = "isaaclab"

    args["algo"] = args["algorithm"]

    algo_args = agent_cfg

    algo_args["eval"]["use_eval"] = False
    algo_args["train"]["n_rollout_threads"] = args["num_envs"]
    algo_args["train"]["num_env_steps"] = args["num_env_steps"]
    algo_args["train"]["eval_interval"] = args["save_interval"]
    algo_args["train"]["save_checkpoints"] = args["save_checkpoints"]
    algo_args["train"]["checkpoint_interval"] = args["checkpoint_interval"]
    algo_args["train"]["log_interval"] = args["log_interval"]
    algo_args["train"]["model_dir"] = args["dir"]
    algo_args["seed"]["specify_seed"] = True
    algo_args["seed"]["seed"] = args["seed"]
    algo_args["algo"]["adversarial_training_mode"] = args["adversarial_training_mode"]
    algo_args["algo"]["adversarial_training_iterations"] = args["adversarial_training_iterations"]
    env_cfg.sim.device = args["device"]
    env_cfg.seed = args["seed"]
    algo_args["device"]["cuda"] = str(args["device"]).startswith("cuda")
    if args["actor_lr"] is not None:
        algo_args["model"]["lr"] = args["actor_lr"]
    if args["critic_lr"] is not None:
        algo_args["model"]["critic_lr"] = args["critic_lr"]
    if args["max_grad_norm"] is not None:
        algo_args["algo"]["max_grad_norm"] = args["max_grad_norm"]

    algo_args.setdefault("train", {})
    _configure_model_dir(args, algo_args)

    env_args = {}
    env_cfg.scene.num_envs = args["num_envs"]
    env_args["task"] = args["task"]
    env_args["config"] = env_cfg
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    env_args["video_settings"] = {
        "video": bool(args["video"]),
        "video_length": args["video_length"],
        "video_interval": args["video_interval"],
        "log_dir": os.path.join(
            algo_args["logger"]["log_dir"],
            "isaaclab",
            args["task"],
            args["algorithm"],
            args["exp_name"],
            "-".join(["seed-{:0>5}".format(agent_cfg["seed"]["seed"]), hms_time]),
            "videos",
        ),
    }

    env_args["headless"] = args["headless"]
    env_args["debug"] = args["debug"]

    # create runner

    # Campaign loading is strict. Avoid HARL's catch-and-continue restore path.
    checkpoint_source = algo_args["train"]["model_dir"] if args["campaign_output"] else None
    if args["campaign_output"]:
        algo_args["train"]["model_dir"] = None
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    try:
        finalize = None
        if args["campaign_output"]:
            from sumo_checkpoint import install_campaign_checkpoints, strict_restore

            if checkpoint_source:
                strict_restore(runner, checkpoint_source, args["resume_optimizers"], args["transfer_actors_only"])
            finalize = install_campaign_checkpoints(runner, args["campaign_output"])
        runner.run()
        if finalize is not None:
            finalize()
    finally:
        runner.close()


if __name__ == "__main__":
    main()  # type: ignore
    simulation_app.close()
