#!/usr/bin/env bash

# Train the complete ANYmal-C sumo curriculum locally:
#   Stage 0: locomotion / go-to-point
#   Stage 1: push a movable block out of the ring
#   Stage 2: symmetric ANYmal-vs-ANYmal adversarial sumo
#
# The default budgets are expressed as an exact number of policy updates:
#   Stage 0: 2,048 envs * 200 rollout steps * 1,500 updates =   614.4M steps
#   Stage 1: 2,048 envs * 200 rollout steps * 2,000 updates =   819.2M steps
#   Stage 2: 3,000 envs * 200 rollout steps * 3,000 updates = 1,800.0M steps
#
# Every setting can be overridden with a SUMO_* environment variable. The
# script uses local checkpoints between stages, so the whole curriculum is
# reproducible and does not fall back to the HuggingFace starting policies.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${SUMO_PYTHON_BIN:-/home/ed/miniforge3/envs/HARL/bin/python}"
RUN_ID="${SUMO_RUN_ID:-$(date +%Y-%m-%d-%H-%M-%S)}"
START_STAGE="${SUMO_START_STAGE:-0}"
STAGE0_MODEL_DIR="${SUMO_STAGE0_MODEL_DIR:-}"
STAGE1_MODEL_DIR="${SUMO_STAGE1_MODEL_DIR:-}"
STAGE2_START_MODEL_DIR="${SUMO_STAGE2_START_MODEL_DIR:-}"
STAGE0_TASK="${SUMO_STAGE0_TASK:-Anymal-C-Go-To-Point-Sumo}"
STAGE1_TASK="${SUMO_STAGE1_TASK:-Anymal-C-Sumo-Stage1-Blocks-Push-v0}"
STAGE2_TASK="${SUMO_STAGE2_TASK:-Sumo-Stage2-Anymal-1v1-v0}"

STAGE0_ENVS="${SUMO_STAGE0_ENVS:-2048}"
STAGE0_STEPS="${SUMO_STAGE0_STEPS:-614400000}"
STAGE1_ENVS="${SUMO_STAGE1_ENVS:-2048}"
STAGE1_STEPS="${SUMO_STAGE1_STEPS:-819200000}"
STAGE2_ENVS="${SUMO_STAGE2_ENVS:-3000}"
STAGE2_STEPS="${SUMO_STAGE2_STEPS:-1800000000}"

SEED="${SUMO_SEED:-1}"
SAVE_INTERVAL="${SUMO_SAVE_INTERVAL:-25}"
CHECKPOINT_INTERVAL="${SUMO_CHECKPOINT_INTERVAL:-100}"
LADDER_STEPS="${SUMO_LADDER_STEPS:-100000000}"
ACTOR_LR="${SUMO_ACTOR_LR:-}"
CRITIC_LR="${SUMO_CRITIC_LR:-}"
MAX_GRAD_NORM="${SUMO_MAX_GRAD_NORM:-}"
HEADING_GATE_MAX_PERCENT="${SUMO_HEADING_GATE_MAX_PERCENT:-25.0}"

ARTIFACT_DIR="${REPO_DIR}/artifacts/anymal_sumo_curriculum/${RUN_ID}"
PIPELINE_LOG="${ARTIFACT_DIR}/pipeline.log"
STATUS_FILE="${ARTIFACT_DIR}/status.txt"
HANDOFF0_DIR="${ARTIFACT_DIR}/stage0_to_stage1"

RESULTS_DIR="${SCRIPT_DIR}/results/isaaclab"
STAGE0_EXP="curriculum_${RUN_ID}_stage0_locomotion"
STAGE1_EXP="curriculum_${RUN_ID}_stage1_push"
STAGE2_EXP="curriculum_${RUN_ID}_stage2_1v1"

mkdir -p "${ARTIFACT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'Python interpreter is not executable: %s\n' "${PYTHON_BIN}" >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

write_status() {
    local stage="$1"
    local state="$2"
    local detail="${3:-}"
    printf 'run_id=%s\nstage=%s\nstate=%s\ndetail=%s\nupdated=%s\n' \
        "${RUN_ID}" "${stage}" "${state}" "${detail}" "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
}

latest_run_dir() {
    local task="$1"
    local experiment="$2"
    local root="${RESULTS_DIR}/${task}/happo_adv/${experiment}"
    find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'seed-*' -print 2>/dev/null | sort | tail -n 1
}

run_training() {
    local stage="$1"
    local task="$2"
    local envs="$3"
    local steps="$4"
    local experiment="$5"
    local mode="$6"
    local model_dir="${7:-}"

    local args=(
        train.py
        --algorithm happo_adv
        --task "${task}"
        --num_envs "${envs}"
        --num_env_steps "${steps}"
        --seed "${SEED}"
        --save_interval "${SAVE_INTERVAL}"
        --save_checkpoints
        --checkpoint_interval "${CHECKPOINT_INTERVAL}"
        --log_interval 1
        --exp_name "${experiment}"
        --adversarial_training_mode "${mode}"
        --adversarial_training_iterations "${LADDER_STEPS}"
        --headless
        --rendering_mode performance
    )

    if [[ -n "${model_dir}" ]]; then
        args+=(--dir "${model_dir}")
    fi
    if [[ -n "${ACTOR_LR}" ]]; then
        args+=(--actor_lr "${ACTOR_LR}")
    fi
    if [[ -n "${CRITIC_LR}" ]]; then
        args+=(--critic_lr "${CRITIC_LR}")
    fi
    if [[ -n "${MAX_GRAD_NORM}" ]]; then
        args+=(--max_grad_norm "${MAX_GRAD_NORM}")
    fi

    write_status "${stage}" running "task=${task} envs=${envs} steps=${steps}"
    printf '\n[%s] Starting %s: task=%s envs=%s steps=%s\n' \
        "$(date --iso-8601=seconds)" "${stage}" "${task}" "${envs}" "${steps}" | tee -a "${PIPELINE_LOG}"
    (
        cd "${SCRIPT_DIR}"
        "${PYTHON_BIN}" "${args[@]}"
    ) 2>&1 | tee -a "${PIPELINE_LOG}"
}

prepare_stage0_handoff() {
    local source_dir="$1"
    mkdir -p "${HANDOFF0_DIR}"

    for filename in actor_agent_robot_0.pt team_0_critic_agent.pt team_0_value_normalizer.pt; do
        if [[ ! -f "${source_dir}/${filename}" ]]; then
            printf 'Stage 0 checkpoint is missing %s in %s\n' "${filename}" "${source_dir}" >&2
            exit 1
        fi
        cp -f -- "${source_dir}/${filename}" "${HANDOFF0_DIR}/${filename}"
    done

    # Stage 0 has one robot/team. Stage 1 has two symmetric robots/teams, so
    # seed both sides with the same learned locomotion actor and critic state.
    cp -f -- "${source_dir}/actor_agent_robot_0.pt" "${HANDOFF0_DIR}/actor_agent_robot_1.pt"
    cp -f -- "${source_dir}/team_0_critic_agent.pt" "${HANDOFF0_DIR}/team_1_critic_agent.pt"
    cp -f -- "${source_dir}/team_0_value_normalizer.pt" "${HANDOFF0_DIR}/team_1_value_normalizer.pt"
}

run_heading_gate() {
    local stage="$1" task="$2" model_dir="$3" expected_agents="$4"
    local audit_log="${ARTIFACT_DIR}/${stage}_heading_audit.log"
    printf '\n[%s] Auditing anatomical-forward behaviour after %s\n' \
        "$(date --iso-8601=seconds)" "${stage}" | tee -a "${PIPELINE_LOG}"
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

write_status pipeline starting "artifact_dir=${ARTIFACT_DIR} start_stage=${START_STAGE}"

if (( START_STAGE <= 0 )); then
    run_training \
        stage0_locomotion \
        "${STAGE0_TASK}" \
        "${STAGE0_ENVS}" \
        "${STAGE0_STEPS}" \
        "${STAGE0_EXP}" \
        parallel

    STAGE0_RUN_DIR="$(latest_run_dir "${STAGE0_TASK}" "${STAGE0_EXP}")"
    STAGE0_MODEL_DIR="${STAGE0_RUN_DIR}/best_model"
    run_heading_gate stage0 "${STAGE0_TASK}" "${STAGE0_MODEL_DIR}" 1
fi

if (( START_STAGE <= 1 )); then
    if [[ -z "${STAGE0_MODEL_DIR}" || ! -d "${STAGE0_MODEL_DIR}" ]]; then
        printf 'A valid Stage 0 model directory is required (got: %s).\n' "${STAGE0_MODEL_DIR}" >&2
        exit 1
    fi
    prepare_stage0_handoff "${STAGE0_MODEL_DIR}"

    run_training \
        stage1_push \
        "${STAGE1_TASK}" \
        "${STAGE1_ENVS}" \
        "${STAGE1_STEPS}" \
        "${STAGE1_EXP}" \
        parallel \
        "${HANDOFF0_DIR}"

    STAGE1_RUN_DIR="$(latest_run_dir "${STAGE1_TASK}" "${STAGE1_EXP}")"
    STAGE1_MODEL_DIR="${STAGE1_RUN_DIR}/best_model"
    run_heading_gate stage1 "${STAGE1_TASK}" "${STAGE1_MODEL_DIR}" 2
fi

if [[ -z "${STAGE1_MODEL_DIR}" || ! -d "${STAGE1_MODEL_DIR}" ]]; then
    printf 'A valid Stage 1 model directory is required (got: %s).\n' "${STAGE1_MODEL_DIR}" >&2
    exit 1
fi

if [[ -z "${STAGE2_START_MODEL_DIR}" ]]; then
    STAGE2_START_MODEL_DIR="${STAGE1_MODEL_DIR}"
fi

if [[ ! -d "${STAGE2_START_MODEL_DIR}" ]]; then
    printf 'A valid Stage 2 starting model directory is required (got: %s).\n' "${STAGE2_START_MODEL_DIR}" >&2
    exit 1
fi

run_training \
    stage2_1v1 \
    "${STAGE2_TASK}" \
    "${STAGE2_ENVS}" \
    "${STAGE2_STEPS}" \
    "${STAGE2_EXP}" \
    leapfrog \
    "${STAGE2_START_MODEL_DIR}"

STAGE2_RUN_DIR="$(latest_run_dir "${STAGE2_TASK}" "${STAGE2_EXP}")"
run_heading_gate stage2 "${STAGE2_TASK}" "${STAGE2_RUN_DIR}/best_model" 2
write_status complete complete "stage2_run=${STAGE2_RUN_DIR}"
printf '\n[%s] Curriculum complete. Final run: %s\n' \
    "$(date --iso-8601=seconds)" "${STAGE2_RUN_DIR}" | tee -a "${PIPELINE_LOG}"
