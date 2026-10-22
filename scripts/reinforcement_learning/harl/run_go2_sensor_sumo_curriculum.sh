#!/usr/bin/env bash

# Go2-vs-Go2 curriculum with deployable actor observations:
#   0. One Go2 learns commanded locomotion with the onboard camera active.
#   1. Duplicate that policy and learn visually guided block pushing.
#   2. Bootstrap direct sensor-only competition in parallel.
#   3. Refine both sides with leapfrog adversarial training.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${SUMO_PYTHON_BIN:-/home/ed/miniforge3/envs/HARL/bin/python}"
RUN_ID="${SUMO_RUN_ID:-$(date +%Y-%m-%d)-go2-sensor-1v1-v1}"
START_PHASE="${SUMO_START_PHASE:-0}"
SEED="${SUMO_SEED:-23}"

STAGE0_TASK="Unitree-Go2-Sensor-Go-To-Point-Sumo-v0"
STAGE1_TASK="Unitree-Go2-Sensor-Sumo-Stage1-Blocks-Push-v0"
STAGE2_TASK="Sumo-Stage2-Unitree-Go2-Sensor-1v1-v0"

STAGE0_ENVS="${SUMO_STAGE0_ENVS:-1024}"
FIGHT_ENVS="${SUMO_FIGHT_ENVS:-640}"
STAGE0_STEPS="${SUMO_STAGE0_STEPS:-100000000}"
STAGE1_STEPS="${SUMO_STAGE1_STEPS:-150000000}"
BOOTSTRAP_STEPS="${SUMO_BOOTSTRAP_STEPS:-200000000}"
LEAGUE_STEPS="${SUMO_LEAGUE_STEPS:-400000000}"
ROTATION_STEPS="${SUMO_LADDER_STEPS:-25000000}"
VIDEO_INTERVAL="${SUMO_VIDEO_INTERVAL:-100000}"
VIDEO_LENGTH="${SUMO_VIDEO_LENGTH:-1000}"

STAGE0_MODEL_DIR="${SUMO_STAGE0_MODEL_DIR:-}"
STAGE1_MODEL_DIR="${SUMO_STAGE1_MODEL_DIR:-}"
BOOTSTRAP_MODEL_DIR="${SUMO_BOOTSTRAP_MODEL_DIR:-}"

ARTIFACT_DIR="${REPO_DIR}/artifacts/go2_sensor_curriculum/${RUN_ID}"
PIPELINE_LOG="${ARTIFACT_DIR}/pipeline.log"
STATUS_FILE="${ARTIFACT_DIR}/status.txt"
HANDOFF_DIR="${ARTIFACT_DIR}/stage0_two_policy_handoff"
RESULTS_DIR="${SCRIPT_DIR}/results/isaaclab"

STAGE0_EXP="${RUN_ID}_stage0_locomotion"
STAGE1_EXP="${RUN_ID}_stage1_visual_push"
BOOTSTRAP_EXP="${RUN_ID}_stage2_parallel"
LEAGUE_EXP="${RUN_ID}_stage2_leapfrog"

mkdir -p "${ARTIFACT_DIR}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

CURRENT_PHASE="startup"
PIPELINE_COMPLETE=0

write_status() {
    local phase="$1" state="$2" detail="${3:-}"
    printf 'run_id=%s\nphase=%s\nstate=%s\ndetail=%s\nupdated=%s\n' \
        "${RUN_ID}" "${phase}" "${state}" "${detail}" "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
}

on_exit() {
    local exit_code=$?
    if (( PIPELINE_COMPLETE == 0 )); then
        local state="failed"
        if (( exit_code == 130 || exit_code == 143 )); then
            state="stopped"
        fi
        write_status "${CURRENT_PHASE}" "${state}" "exit_code=${exit_code}; log=${PIPELINE_LOG}"
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

latest_models() {
    local task="$1" experiment="$2"
    local root="${RESULTS_DIR}/${task}/happo_adv/${experiment}"
    local run
    run="$(find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'seed-*' -print 2>/dev/null | sort | tail -n 1)"
    if [[ -z "${run}" ]]; then
        printf 'No run found below %s\n' "${root}" >&2
        exit 1
    fi
    if [[ -d "${run}/best_model" ]]; then
        printf '%s\n' "${run}/best_model"
    else
        printf '%s\n' "${run}/models"
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Required checkpoint file is missing: %s\n' "$1" >&2
        exit 1
    fi
}

duplicate_stage0_policy() {
    local source_dir="$1"
    require_file "${source_dir}/actor_agent_robot_0.pt"
    require_file "${source_dir}/team_0_critic_agent.pt"
    require_file "${source_dir}/team_0_value_normalizer.pt"
    mkdir -p "${HANDOFF_DIR}"
    cp -f -- "${source_dir}/actor_agent_robot_0.pt" "${HANDOFF_DIR}/actor_agent_robot_0.pt"
    cp -f -- "${source_dir}/actor_agent_robot_0.pt" "${HANDOFF_DIR}/actor_agent_robot_1.pt"
    cp -f -- "${source_dir}/team_0_critic_agent.pt" "${HANDOFF_DIR}/team_0_critic_agent.pt"
    cp -f -- "${source_dir}/team_0_critic_agent.pt" "${HANDOFF_DIR}/team_1_critic_agent.pt"
    cp -f -- "${source_dir}/team_0_value_normalizer.pt" "${HANDOFF_DIR}/team_0_value_normalizer.pt"
    cp -f -- "${source_dir}/team_0_value_normalizer.pt" "${HANDOFF_DIR}/team_1_value_normalizer.pt"
}

run_training() {
    local phase="$1" task="$2" envs="$3" steps="$4" experiment="$5" mode="$6" lr="$7"
    local model_dir="${8:-}"
    local args=(
        train.py --algorithm happo_adv --task "${task}"
        --num_envs "${envs}" --num_env_steps "${steps}" --seed "${SEED}"
        --save_interval 25 --save_checkpoints --checkpoint_interval 50 --log_interval 1
        --exp_name "${experiment}" --adversarial_training_mode "${mode}"
        --adversarial_training_iterations "${ROTATION_STEPS}"
        --actor_lr "${lr}" --critic_lr "${lr}" --max_grad_norm 1.0
        --video --video_length "${VIDEO_LENGTH}" --video_interval "${VIDEO_INTERVAL}"
        --device cuda:0 --headless --enable_cameras --rendering_mode performance
    )
    if [[ -n "${model_dir}" ]]; then
        args+=(--dir "${model_dir}")
    fi

    CURRENT_PHASE="${phase}"
    write_status "${phase}" running "task=${task}; envs=${envs}; steps=${steps}; mode=${mode}"
    printf '\n[%s] %s: task=%s envs=%s steps=%s mode=%s\n' \
        "$(date --iso-8601=seconds)" "${phase}" "${task}" "${envs}" "${steps}" "${mode}" \
        | tee -a "${PIPELINE_LOG}"
    (cd "${SCRIPT_DIR}" && "${PYTHON_BIN}" "${args[@]}") 2>&1 | tee -a "${PIPELINE_LOG}"
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'Python interpreter is not executable: %s\n' "${PYTHON_BIN}" >&2
    exit 1
fi
if ! [[ "${START_PHASE}" =~ ^[0-3]$ ]]; then
    printf 'SUMO_START_PHASE must be 0, 1, 2, or 3.\n' >&2
    exit 2
fi

write_status startup running "artifact_dir=${ARTIFACT_DIR}; start_phase=${START_PHASE}"

if (( START_PHASE <= 0 )); then
    run_training stage0_locomotion "${STAGE0_TASK}" "${STAGE0_ENVS}" "${STAGE0_STEPS}" \
        "${STAGE0_EXP}" parallel 0.0003 "${STAGE0_MODEL_DIR}"
    STAGE0_MODEL_DIR="$(latest_models "${STAGE0_TASK}" "${STAGE0_EXP}")"
fi

if (( START_PHASE <= 1 )); then
    if [[ -z "${STAGE0_MODEL_DIR}" || ! -d "${STAGE0_MODEL_DIR}" ]]; then
        printf 'Stage 0 model directory is required for Stage 1: %s\n' "${STAGE0_MODEL_DIR}" >&2
        exit 1
    fi
    duplicate_stage0_policy "${STAGE0_MODEL_DIR}"
    run_training stage1_visual_push "${STAGE1_TASK}" "${FIGHT_ENVS}" "${STAGE1_STEPS}" \
        "${STAGE1_EXP}" parallel 0.00025 "${HANDOFF_DIR}"
    STAGE1_MODEL_DIR="$(latest_models "${STAGE1_TASK}" "${STAGE1_EXP}")"
fi

if (( START_PHASE <= 2 )); then
    if [[ -z "${STAGE1_MODEL_DIR}" || ! -d "${STAGE1_MODEL_DIR}" ]]; then
        printf 'Stage 1 model directory is required for Stage 2: %s\n' "${STAGE1_MODEL_DIR}" >&2
        exit 1
    fi
    run_training stage2_parallel "${STAGE2_TASK}" "${FIGHT_ENVS}" "${BOOTSTRAP_STEPS}" \
        "${BOOTSTRAP_EXP}" parallel 0.00025 "${STAGE1_MODEL_DIR}"
    BOOTSTRAP_MODEL_DIR="$(latest_models "${STAGE2_TASK}" "${BOOTSTRAP_EXP}")"
fi

if [[ -z "${BOOTSTRAP_MODEL_DIR}" || ! -d "${BOOTSTRAP_MODEL_DIR}" ]]; then
    printf 'Parallel Stage 2 model directory is required for leapfrog training: %s\n' \
        "${BOOTSTRAP_MODEL_DIR}" >&2
    exit 1
fi
run_training stage2_leapfrog "${STAGE2_TASK}" "${FIGHT_ENVS}" "${LEAGUE_STEPS}" \
    "${LEAGUE_EXP}" leapfrog 0.0002 "${BOOTSTRAP_MODEL_DIR}"

FINAL_MODELS="$(latest_models "${STAGE2_TASK}" "${LEAGUE_EXP}")"
CURRENT_PHASE="complete"
PIPELINE_COMPLETE=1
write_status complete complete "final_models=${FINAL_MODELS}"
printf '\n[%s] Sensor-aware Go2 curriculum complete: %s\n' \
    "$(date --iso-8601=seconds)" "${FINAL_MODELS}" | tee -a "${PIPELINE_LOG}"
