#!/usr/bin/env bash

# Round-robin morphology league for ANYmal-C, Spot, and Unitree Go2.
# Each species owns one actor/critic/value-normalizer bank. Six ordered matches
# per round put every species in both team slots; no actor is shared across bodies.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${SUMO_PYTHON_BIN:-/home/ed/miniforge3/envs/HARL/bin/python}"
RUN_ID="${SUMO_LEAGUE_RUN_ID:-$(date +%Y-%m-%d)-three-species-league-v1}"
SEED="${SUMO_SEED:-1}"
ENVS="${SUMO_LEAGUE_ENVS:-1600}"
MATCH_STEPS="${SUMO_LEAGUE_MATCH_STEPS:-100000000}"
ROUNDS="${SUMO_LEAGUE_ROUNDS:-2}"
CHECKPOINT_INTERVAL="${SUMO_CHECKPOINT_INTERVAL:-50}"
ANYMAL_START="${SUMO_ANYMAL_START_MODEL_DIR:?Set SUMO_ANYMAL_START_MODEL_DIR}"
GO2_SPOT_START="${SUMO_GO2_SPOT_START_MODEL_DIR:?Set SUMO_GO2_SPOT_START_MODEL_DIR}"

ARTIFACT_DIR="${REPO_DIR}/artifacts/three_species_sumo_league/${RUN_ID}"
PIPELINE_LOG="${ARTIFACT_DIR}/pipeline.log"
STATUS_FILE="${ARTIFACT_DIR}/status.txt"
BANK_DIR="${ARTIFACT_DIR}/policy_bank"
RESULTS_DIR="${SCRIPT_DIR}/results/isaaclab"
mkdir -p "${ARTIFACT_DIR}" "${BANK_DIR}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

write_status() {
    printf 'run_id=%s\nphase=%s\nstate=%s\ndetail=%s\nupdated=%s\n' \
        "${RUN_ID}" "$1" "$2" "$3" "$(date --iso-8601=seconds)" > "${STATUS_FILE}"
}

on_exit() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        write_status "${CURRENT_PHASE:-startup}" stopped "exit_code=${exit_code}; log=${PIPELINE_LOG}"
    fi
}
trap on_exit EXIT

require_file() {
    [[ -f "$1" ]] || { printf 'Required policy file is missing: %s\n' "$1" >&2; exit 1; }
}

init_species() {
    local species="$1" source="$2" actor_slot="$3" team_slot="$4"
    local target="${BANK_DIR}/${species}"
    mkdir -p "${target}"
    require_file "${source}/actor_agent_robot_${actor_slot}.pt"
    require_file "${source}/team_${team_slot}_critic_agent.pt"
    require_file "${source}/team_${team_slot}_value_normalizer.pt"
    cp -f -- "${source}/actor_agent_robot_${actor_slot}.pt" "${target}/actor.pt"
    cp -f -- "${source}/team_${team_slot}_critic_agent.pt" "${target}/critic.pt"
    cp -f -- "${source}/team_${team_slot}_value_normalizer.pt" "${target}/value_normalizer.pt"
}

prepare_match() {
    local first="$1" second="$2" handoff="$3"
    mkdir -p "${handoff}"
    cp -f -- "${BANK_DIR}/${first}/actor.pt" "${handoff}/actor_agent_robot_0.pt"
    cp -f -- "${BANK_DIR}/${first}/critic.pt" "${handoff}/team_0_critic_agent.pt"
    cp -f -- "${BANK_DIR}/${first}/value_normalizer.pt" "${handoff}/team_0_value_normalizer.pt"
    cp -f -- "${BANK_DIR}/${second}/actor.pt" "${handoff}/actor_agent_robot_1.pt"
    cp -f -- "${BANK_DIR}/${second}/critic.pt" "${handoff}/team_1_critic_agent.pt"
    cp -f -- "${BANK_DIR}/${second}/value_normalizer.pt" "${handoff}/team_1_value_normalizer.pt"
}

latest_run_dir() {
    local task="$1" experiment="$2"
    find "${RESULTS_DIR}/${task}/happo_adv/${experiment}" -mindepth 1 -maxdepth 1 \
        -type d -name 'seed-*' -print 2>/dev/null | sort | tail -n 1
}

harvest_match() {
    local first="$1" second="$2" models="$3"
    for file in actor_agent_robot_0.pt actor_agent_robot_1.pt team_0_critic_agent.pt \
        team_1_critic_agent.pt team_0_value_normalizer.pt team_1_value_normalizer.pt; do
        require_file "${models}/${file}"
    done
    cp -f -- "${models}/actor_agent_robot_0.pt" "${BANK_DIR}/${first}/actor.pt"
    cp -f -- "${models}/team_0_critic_agent.pt" "${BANK_DIR}/${first}/critic.pt"
    cp -f -- "${models}/team_0_value_normalizer.pt" "${BANK_DIR}/${first}/value_normalizer.pt"
    cp -f -- "${models}/actor_agent_robot_1.pt" "${BANK_DIR}/${second}/actor.pt"
    cp -f -- "${models}/team_1_critic_agent.pt" "${BANK_DIR}/${second}/critic.pt"
    cp -f -- "${models}/team_1_value_normalizer.pt" "${BANK_DIR}/${second}/value_normalizer.pt"
}

run_video() {
    local label="$1" task="$2" models="$3"
    local video_dir="${ARTIFACT_DIR}/videos/${label}"
    if ! (cd "${SCRIPT_DIR}" && "${PYTHON_BIN}" play.py --algorithm happo_adv \
        --task "${task}" --num_envs 4 --num_env_steps 4000 --seed "${SEED}" \
        --dir "${models}" --deterministic --report_heading --video --video_length 1000 \
        --video_dir "${video_dir}" --camera_eye 6 6 4.5 --camera_lookat 0 0 0.4 \
        --device cuda:0 --headless --enable_cameras) 2>&1 | tee -a "${PIPELINE_LOG}"; then
        printf '[%s] WARNING: video failed for %s; continuing league.\n' \
            "$(date --iso-8601=seconds)" "${label}" | tee -a "${PIPELINE_LOG}"
    fi
}

run_match() {
    local round="$1" index="$2" first="$3" second="$4" task="$5"
    local label="round${round}_match${index}_${first}_vs_${second}"
    local experiment="league_${RUN_ID}_${label}"
    local handoff="${ARTIFACT_DIR}/handoffs/${label}"
    prepare_match "${first}" "${second}" "${handoff}"
    CURRENT_PHASE="${label}"
    write_status "${label}" running "task=${task} envs=${ENVS} steps=${MATCH_STEPS}"
    printf '\n[%s] Starting %s with independent %s and %s policies\n' \
        "$(date --iso-8601=seconds)" "${label}" "${first}" "${second}" | tee -a "${PIPELINE_LOG}"
    (cd "${SCRIPT_DIR}" && "${PYTHON_BIN}" train.py --algorithm happo_adv --task "${task}" \
        --num_envs "${ENVS}" --num_env_steps "${MATCH_STEPS}" --seed "${SEED}" \
        --save_interval 25 --save_checkpoints --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
        --log_interval 1 --exp_name "${experiment}" --adversarial_training_mode leapfrog \
        --adversarial_training_iterations 25000000 --actor_lr 0.0002 --critic_lr 0.0002 \
        --max_grad_norm 1.0 --device cuda:0 --headless --rendering_mode performance \
        --dir "${handoff}") 2>&1 | tee -a "${PIPELINE_LOG}"
    local run_dir
    run_dir="$(latest_run_dir "${task}" "${experiment}")"
    harvest_match "${first}" "${second}" "${run_dir}/models"
    run_video "${label}" "${task}" "${run_dir}/models"
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'Python interpreter is not executable: %s\n' "${PYTHON_BIN}" >&2
    exit 1
fi

# The completed Go2-vs-Spot curriculum has Go2 in slot 0 and Spot in slot 1.
# The validated symmetric ANYmal policy supplies its canonical slot-0 actor.
init_species go2 "${GO2_SPOT_START}" 0 0
init_species spot "${GO2_SPOT_START}" 1 1
init_species anymal "${ANYMAL_START}" 0 0

for round in $(seq 1 "${ROUNDS}"); do
    run_match "${round}" 1 go2 spot "Sumo-Stage2-Unitree-Go2-vs-Spot-v0"
    run_match "${round}" 2 spot anymal "Sumo-Stage2-Spot-vs-Anymal-v0"
    run_match "${round}" 3 anymal go2 "Sumo-Stage2-Anymal-vs-Unitree-Go2-v0"
    run_match "${round}" 4 spot go2 "Sumo-Stage2-Spot-vs-Unitree-Go2-v0"
    run_match "${round}" 5 anymal spot "Sumo-Stage2-Anymal-vs-Spot-v0"
    run_match "${round}" 6 go2 anymal "Sumo-Stage2-Unitree-Go2-vs-Anymal-v0"
done

CURRENT_PHASE=complete
write_status complete complete "policy_bank=${BANK_DIR}"
printf '\n[%s] Three-species league complete: %s\n' \
    "$(date --iso-8601=seconds)" "${BANK_DIR}" | tee -a "${PIPELINE_LOG}"
