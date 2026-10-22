#!/usr/bin/env bash

# Full heterogeneous sumo curriculum:
#   0. Go2 learns locomotion alone.
#   1. Spot learns locomotion alone.
#   2. Their separate policies are merged and learn matched block pushing.
#   3. The mixed policies bootstrap direct competition in parallel.
#   4. Alternating leapfrog training refines the final Go2-vs-Spot match.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${SUMO_PYTHON_BIN:-/home/ed/miniforge3/envs/HARL/bin/python}"
RUN_ID="${SUMO_RUN_ID:-$(date +%Y-%m-%d)-go2-vs-spot-full-v1}"
START_PHASE="${SUMO_START_PHASE:-0}"
SEED="${SUMO_SEED:-1}"

GO2_TASK="Unitree-Go2-Go-To-Point-Sumo-v0"
SPOT_TASK="Spot-Go-To-Point-Sumo-v0"
STAGE1_TASK="Unitree-Go2-vs-Spot-Sumo-Stage1-Blocks-Push-v0"
STAGE2_TASK="Sumo-Stage2-Unitree-Go2-vs-Spot-v0"

GO2_ENVS="${SUMO_GO2_ENVS:-2500}"
SPOT_ENVS="${SUMO_SPOT_ENVS:-2000}"
MIXED_ENVS="${SUMO_MIXED_ENVS:-2000}"
GO2_STEPS="${SUMO_GO2_STEPS:-750000000}"
SPOT_STEPS="${SUMO_SPOT_STEPS:-600000000}"
STAGE1_STEPS="${SUMO_STAGE1_STEPS:-800000000}"
BOOTSTRAP_STEPS="${SUMO_BOOTSTRAP_STEPS:-400000000}"
LEAGUE_STEPS="${SUMO_LEAGUE_STEPS:-1200000000}"
ROTATION_STEPS="${SUMO_LADDER_STEPS:-50000000}"
SAVE_INTERVAL="${SUMO_SAVE_INTERVAL:-25}"
CHECKPOINT_INTERVAL="${SUMO_CHECKPOINT_INTERVAL:-100}"
HEADING_GATE_MAX_PERCENT="${SUMO_HEADING_GATE_MAX_PERCENT:-25.0}"
SAVE_STAGE_VIDEOS="${SUMO_SAVE_STAGE_VIDEOS:-1}"
VIDEO_ENVS="${SUMO_VIDEO_ENVS:-4}"
VIDEO_LENGTH="${SUMO_VIDEO_LENGTH:-1000}"

GO2_MODEL_DIR="${SUMO_GO2_MODEL_DIR:-}"
SPOT_MODEL_DIR="${SUMO_SPOT_MODEL_DIR:-}"
STAGE1_MODEL_DIR="${SUMO_STAGE1_MODEL_DIR:-}"
BOOTSTRAP_MODEL_DIR="${SUMO_BOOTSTRAP_MODEL_DIR:-}"
GO2_START_MODEL_DIR="${SUMO_GO2_START_MODEL_DIR:-}"
SPOT_START_MODEL_DIR="${SUMO_SPOT_START_MODEL_DIR:-}"
STAGE1_START_MODEL_DIR="${SUMO_STAGE1_START_MODEL_DIR:-}"
GO2_STAGE0_LR="${SUMO_GO2_STAGE0_LR:-0.0003}"
SPOT_STAGE0_LR="${SUMO_SPOT_STAGE0_LR:-0.0003}"

ARTIFACT_DIR="${REPO_DIR}/artifacts/go2_vs_spot_curriculum/${RUN_ID}"
PIPELINE_LOG="${ARTIFACT_DIR}/pipeline.log"
STATUS_FILE="${ARTIFACT_DIR}/status.txt"
HANDOFF_DIR="${ARTIFACT_DIR}/go2_spot_stage0_handoff"
RESULTS_DIR="${SCRIPT_DIR}/results/isaaclab"

GO2_EXP="go2_spot_${RUN_ID}_go2_stage0_locomotion"
SPOT_EXP="go2_spot_${RUN_ID}_spot_stage0_locomotion"
STAGE1_EXP="go2_spot_${RUN_ID}_mixed_stage1_push"
BOOTSTRAP_EXP="go2_spot_${RUN_ID}_stage2_bootstrap_parallel"
LEAGUE_EXP="go2_spot_${RUN_ID}_stage2_league_leapfrog"

mkdir -p "${ARTIFACT_DIR}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

PIPELINE_COMPLETE=0
CURRENT_PHASE="startup"

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
        write_status "${CURRENT_PHASE}" "${state}" "exit_code=${exit_code}; inspect=${PIPELINE_LOG}"
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

latest_run_dir() {
    local task="$1" experiment="$2"
    local root="${RESULTS_DIR}/${task}/happo_adv/${experiment}"
    find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'seed-*' -print 2>/dev/null | sort | tail -n 1
}

latest_models() {
    local task="$1" experiment="$2"
    local run
    run="$(latest_run_dir "${task}" "${experiment}")"
    if [[ -z "${run}" ]]; then
        printf 'No result directory found for task=%s experiment=%s\n' "${task}" "${experiment}" >&2
        exit 1
    fi
    if [[ -d "${run}/best_model" ]]; then
        printf '%s\n' "${run}/best_model"
    else
        printf '%s\n' "${run}/models"
    fi
}

require_model_files() {
    local model_dir="$1" actor_name="$2" team_index="$3"
    local filename
    for filename in "${actor_name}" "team_${team_index}_critic_agent.pt" "team_${team_index}_value_normalizer.pt"; do
        if [[ ! -f "${model_dir}/${filename}" ]]; then
            printf 'Checkpoint is missing %s in %s\n' "${filename}" "${model_dir}" >&2
            exit 1
        fi
    done
}

merge_stage0_models() {
    local go2_dir="$1" spot_dir="$2"
    require_model_files "${go2_dir}" actor_agent_robot_0.pt 0
    require_model_files "${spot_dir}" actor_agent_robot_0.pt 0
    mkdir -p "${HANDOFF_DIR}"

    cp -f -- "${go2_dir}/actor_agent_robot_0.pt" "${HANDOFF_DIR}/actor_agent_robot_0.pt"
    cp -f -- "${go2_dir}/team_0_critic_agent.pt" "${HANDOFF_DIR}/team_0_critic_agent.pt"
    cp -f -- "${go2_dir}/team_0_value_normalizer.pt" "${HANDOFF_DIR}/team_0_value_normalizer.pt"

    cp -f -- "${spot_dir}/actor_agent_robot_0.pt" "${HANDOFF_DIR}/actor_agent_robot_1.pt"
    cp -f -- "${spot_dir}/team_0_critic_agent.pt" "${HANDOFF_DIR}/team_1_critic_agent.pt"
    cp -f -- "${spot_dir}/team_0_value_normalizer.pt" "${HANDOFF_DIR}/team_1_value_normalizer.pt"
}

run_heading_gate() {
    local phase="$1" task="$2" model_dir="$3" expected_agents="$4"
    local audit_log="${ARTIFACT_DIR}/${phase}_heading_audit.log"
    CURRENT_PHASE="${phase}_heading_audit"
    write_status "${CURRENT_PHASE}" running "task=${task} model=${model_dir}"
    printf '\n[%s] Auditing anatomical-forward behaviour after %s\n' \
        "$(date --iso-8601=seconds)" "${phase}" | tee -a "${PIPELINE_LOG}"
    (
        cd "${SCRIPT_DIR}"
        "${PYTHON_BIN}" play.py --algorithm happo_adv --task "${task}" \
            --num_envs 64 --num_env_steps 25600 --seed "${SEED}" --dir "${model_dir}" \
            --deterministic --report_heading --device cuda:0 --headless
    ) 2>&1 | tee "${audit_log}" | tee -a "${PIPELINE_LOG}"

    mapfile -t facing_values < <(sed -n 's/.*facing_away=\([0-9.]*\)%.*/\1/p' "${audit_log}")
    mapfile -t backing_values < <(sed -n 's/.*backing_while_moving=\([0-9.]*\)%.*/\1/p' "${audit_log}")
    if (( ${#facing_values[@]} != expected_agents || ${#backing_values[@]} != expected_agents )); then
        printf 'Heading gate could not read all %s agents from %s\n' "${expected_agents}" "${audit_log}" >&2
        exit 1
    fi
    local value
    for value in "${facing_values[@]}" "${backing_values[@]}"; do
        if ! awk -v value="${value}" -v limit="${HEADING_GATE_MAX_PERCENT}" \
            'BEGIN { exit !(value <= limit) }'; then
            printf 'Heading gate failed: %.1f%% exceeds %.1f%% in %s\n' \
                "${value}" "${HEADING_GATE_MAX_PERCENT}" "${audit_log}" >&2
            exit 1
        fi
    done
}

run_policy_video() {
    local phase="$1" task="$2" model_dir="$3"
    if [[ "${SAVE_STAGE_VIDEOS}" != "1" ]]; then
        return 0
    fi
    local video_dir="${ARTIFACT_DIR}/videos/${phase}"
    local video_log="${ARTIFACT_DIR}/${phase}_video.log"
    local video_steps=$((VIDEO_ENVS * VIDEO_LENGTH))
    CURRENT_PHASE="${phase}_video"
    write_status "${CURRENT_PHASE}" running "task=${task} model=${model_dir}"
    printf '\n[%s] Recording deterministic %s policy video to %s\n' \
        "$(date --iso-8601=seconds)" "${phase}" "${video_dir}" | tee -a "${PIPELINE_LOG}"
    if (
        cd "${SCRIPT_DIR}"
        "${PYTHON_BIN}" play.py --algorithm happo_adv --task "${task}" \
            --num_envs "${VIDEO_ENVS}" --num_env_steps "${video_steps}" --seed "${SEED}" \
            --dir "${model_dir}" --deterministic --report_heading \
            --video --video_length "${VIDEO_LENGTH}" --video_dir "${video_dir}" \
            --camera_eye 6.0 6.0 4.5 --camera_lookat 0.0 0.0 0.4 \
            --device cuda:0 --headless --enable_cameras
    ) 2>&1 | tee "${video_log}" | tee -a "${PIPELINE_LOG}"; then
        printf '[%s] Saved %s\n' "$(date --iso-8601=seconds)" \
            "${video_dir}/sumo_trained_match.mp4" | tee -a "${PIPELINE_LOG}"
    else
        local video_exit=$?
        printf '[%s] WARNING: %s video failed with exit code %s; training will continue.\n' \
            "$(date --iso-8601=seconds)" "${phase}" "${video_exit}" | tee -a "${PIPELINE_LOG}"
    fi
}

run_training() {
    local phase="$1" task="$2" envs="$3" steps="$4" experiment="$5" mode="$6" learning_rate="$7"
    local model_dir="${8:-}"
    local args=(
        train.py --algorithm happo_adv --task "${task}"
        --num_envs "${envs}" --num_env_steps "${steps}" --seed "${SEED}"
        --save_interval "${SAVE_INTERVAL}" --save_checkpoints --checkpoint_interval "${CHECKPOINT_INTERVAL}"
        --log_interval 1 --exp_name "${experiment}"
        --adversarial_training_mode "${mode}" --adversarial_training_iterations "${ROTATION_STEPS}"
        --actor_lr "${learning_rate}" --critic_lr "${learning_rate}" --max_grad_norm 1.0
        --device cuda:0 --headless --rendering_mode performance
    )
    if [[ -n "${model_dir}" ]]; then
        args+=(--dir "${model_dir}")
    fi

    CURRENT_PHASE="${phase}"
    write_status "${phase}" running "task=${task} envs=${envs} steps=${steps} mode=${mode}"
    printf '\n[%s] Starting %s: task=%s envs=%s steps=%s mode=%s lr=%s\n' \
        "$(date --iso-8601=seconds)" "${phase}" "${task}" "${envs}" "${steps}" "${mode}" "${learning_rate}" \
        | tee -a "${PIPELINE_LOG}"
    (cd "${SCRIPT_DIR}" && "${PYTHON_BIN}" "${args[@]}") 2>&1 | tee -a "${PIPELINE_LOG}"
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'Python interpreter is not executable: %s\n' "${PYTHON_BIN}" >&2
    exit 1
fi
if ! [[ "${START_PHASE}" =~ ^[0-4]$ ]]; then
    printf 'SUMO_START_PHASE must be 0, 1, 2, 3, or 4 (got %s).\n' "${START_PHASE}" >&2
    exit 2
fi

write_status startup running "artifact_dir=${ARTIFACT_DIR} start_phase=${START_PHASE}"

if (( START_PHASE <= 0 )); then
    run_training go2_stage0_locomotion "${GO2_TASK}" "${GO2_ENVS}" "${GO2_STEPS}" "${GO2_EXP}" \
        parallel "${GO2_STAGE0_LR}" "${GO2_START_MODEL_DIR}"
    GO2_MODEL_DIR="$(latest_models "${GO2_TASK}" "${GO2_EXP}")"
    run_heading_gate go2_stage0 "${GO2_TASK}" "${GO2_MODEL_DIR}" 1
fi

if (( START_PHASE <= 1 )); then
    if [[ -z "${GO2_MODEL_DIR}" || ! -d "${GO2_MODEL_DIR}" ]]; then
        printf 'A Go2 Stage 0 model is required before Spot Stage 0 (got %s).\n' "${GO2_MODEL_DIR}" >&2
        exit 1
    fi
    run_training spot_stage0_locomotion "${SPOT_TASK}" "${SPOT_ENVS}" "${SPOT_STEPS}" "${SPOT_EXP}" \
        parallel "${SPOT_STAGE0_LR}" "${SPOT_START_MODEL_DIR}"
    SPOT_MODEL_DIR="$(latest_models "${SPOT_TASK}" "${SPOT_EXP}")"
    run_heading_gate spot_stage0 "${SPOT_TASK}" "${SPOT_MODEL_DIR}" 1
fi

if (( START_PHASE <= 2 )); then
    if [[ -n "${STAGE1_START_MODEL_DIR}" ]]; then
        if [[ ! -d "${STAGE1_START_MODEL_DIR}" ]]; then
            printf 'Stage 1 starting checkpoint does not exist: %s\n' "${STAGE1_START_MODEL_DIR}" >&2
            exit 1
        fi
        require_model_files "${STAGE1_START_MODEL_DIR}" actor_agent_robot_0.pt 0
        require_model_files "${STAGE1_START_MODEL_DIR}" actor_agent_robot_1.pt 1
        STAGE1_INPUT_DIR="${STAGE1_START_MODEL_DIR}"
    else
        if [[ -z "${GO2_MODEL_DIR}" || ! -d "${GO2_MODEL_DIR}" || -z "${SPOT_MODEL_DIR}" || ! -d "${SPOT_MODEL_DIR}" ]]; then
            printf 'Both separate Stage 0 model directories are required for a fresh mixed Stage 1.\n' >&2
            exit 1
        fi
        merge_stage0_models "${GO2_MODEL_DIR}" "${SPOT_MODEL_DIR}"
        STAGE1_INPUT_DIR="${HANDOFF_DIR}"
    fi
    run_training mixed_stage1_push "${STAGE1_TASK}" "${MIXED_ENVS}" "${STAGE1_STEPS}" "${STAGE1_EXP}" \
        parallel 0.00025 "${STAGE1_INPUT_DIR}"
    STAGE1_MODEL_DIR="$(latest_models "${STAGE1_TASK}" "${STAGE1_EXP}")"
    run_policy_video mixed_stage1 "${STAGE1_TASK}" "${STAGE1_MODEL_DIR}"
    run_heading_gate mixed_stage1 "${STAGE1_TASK}" "${STAGE1_MODEL_DIR}" 2
fi

if (( START_PHASE <= 3 )); then
    if [[ -z "${STAGE1_MODEL_DIR}" || ! -d "${STAGE1_MODEL_DIR}" ]]; then
        printf 'A mixed Stage 1 model directory is required for Stage 2 bootstrap.\n' >&2
        exit 1
    fi
    run_training stage2_bootstrap_parallel "${STAGE2_TASK}" "${MIXED_ENVS}" "${BOOTSTRAP_STEPS}" "${BOOTSTRAP_EXP}" parallel 0.00025 "${STAGE1_MODEL_DIR}"
    BOOTSTRAP_MODEL_DIR="$(latest_models "${STAGE2_TASK}" "${BOOTSTRAP_EXP}")"
    run_policy_video stage2_bootstrap "${STAGE2_TASK}" "${BOOTSTRAP_MODEL_DIR}"
    run_heading_gate stage2_bootstrap "${STAGE2_TASK}" "${BOOTSTRAP_MODEL_DIR}" 2
fi

if [[ -z "${BOOTSTRAP_MODEL_DIR}" || ! -d "${BOOTSTRAP_MODEL_DIR}" ]]; then
    printf 'A Stage 2 bootstrap model directory is required for leapfrog refinement.\n' >&2
    exit 1
fi
run_training stage2_league_leapfrog "${STAGE2_TASK}" "${MIXED_ENVS}" "${LEAGUE_STEPS}" "${LEAGUE_EXP}" leapfrog 0.0002 "${BOOTSTRAP_MODEL_DIR}"

FINAL_MODELS="$(latest_models "${STAGE2_TASK}" "${LEAGUE_EXP}")"
run_policy_video stage2_final "${STAGE2_TASK}" "${FINAL_MODELS}"
run_heading_gate stage2_final "${STAGE2_TASK}" "${FINAL_MODELS}" 2
CURRENT_PHASE=complete
PIPELINE_COMPLETE=1
write_status complete complete "final_models=${FINAL_MODELS}"
printf '\n[%s] Go2-vs-Spot curriculum complete: %s\n' "$(date --iso-8601=seconds)" "${FINAL_MODELS}" | tee -a "${PIPELINE_LOG}"
