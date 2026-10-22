#!/usr/bin/env bash

# Wait for a validated Go2-vs-Spot pipeline to finish, discover its final model,
# then run the ordered ANYmal/Spot/Go2 morphology league under its own watchdog.

set -euo pipefail

if (( $# != 3 )); then
    printf 'Usage: %s GO2_SPOT_PID GO2_SPOT_RUN_ID GO2_SPOT_STATUS_FILE\n' "$0" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WAIT_PID="$1"
GO2_SPOT_RUN_ID="$2"
GO2_SPOT_STATUS_FILE="$3"
QUEUE_ID="${SUMO_QUEUE_ID:-$(date +%Y-%m-%d)-three-species-after-go2-spot-v1}"
QUEUE_DIR="${REPO_DIR}/artifacts/three_species_sumo_queue/${QUEUE_ID}"
QUEUE_LOG="${QUEUE_DIR}/queue.log"
QUEUE_STATUS="${QUEUE_DIR}/status.txt"
ANYMAL_START="${SUMO_ANYMAL_START_MODEL_DIR:?Set SUMO_ANYMAL_START_MODEL_DIR}"
RESULTS_ROOT="${SCRIPT_DIR}/results/isaaclab/Sumo-Stage2-Unitree-Go2-vs-Spot-v0/happo_adv"
WATCHDOG_PID=""
mkdir -p "${QUEUE_DIR}"

write_status() {
    printf 'queue_id=%s\nphase=%s\nstate=%s\ndetail=%s\nupdated=%s\n' \
        "${QUEUE_ID}" "$1" "$2" "$3" "$(date --iso-8601=seconds)" > "${QUEUE_STATUS}"
}

cleanup() {
    if [[ -n "${WATCHDOG_PID}" ]]; then
        kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

write_status waiting waiting "pid=${WAIT_PID}; run=${GO2_SPOT_RUN_ID}"
while [[ -r "/proc/${WAIT_PID}/cmdline" ]] && \
    tr '\0' ' ' < "/proc/${WAIT_PID}/cmdline" | grep -q 'run_go2_vs_spot_full_curriculum.sh'; do
    printf '[%s] Waiting for Go2-vs-Spot process %s\n' \
        "$(date --iso-8601=seconds)" "${WAIT_PID}" >> "${QUEUE_LOG}"
    sleep 60
done

if [[ ! -f "${GO2_SPOT_STATUS_FILE}" ]] || \
    ! grep -q '^state=complete$' "${GO2_SPOT_STATUS_FILE}"; then
    write_status waiting blocked "Go2-vs-Spot did not finish cleanly; inspect ${GO2_SPOT_STATUS_FILE}"
    exit 1
fi

experiment="go2_spot_${GO2_SPOT_RUN_ID}_stage2_league_leapfrog"
run_dir="$(find "${RESULTS_ROOT}/${experiment}" -mindepth 1 -maxdepth 1 \
    -type d -name 'seed-*' -print 2>/dev/null | sort | tail -n 1)"
GO2_SPOT_MODELS="${run_dir}/best_model"
for file in actor_agent_robot_0.pt actor_agent_robot_1.pt team_0_critic_agent.pt \
    team_1_critic_agent.pt team_0_value_normalizer.pt team_1_value_normalizer.pt; do
    if [[ ! -f "${GO2_SPOT_MODELS}/${file}" ]]; then
        write_status startup blocked "missing ${GO2_SPOT_MODELS}/${file}"
        exit 1
    fi
done

queue_pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
setsid env SUMO_WATCHDOG_TEMP_C=83 SUMO_WATCHDOG_TEMP_SAMPLES=3 \
    SUMO_WATCHDOG_MIN_DISK_GB=20 SUMO_WATCHDOG_STALL_MINUTES=20 \
    SUMO_WATCHDOG_POLL_SECONDS=60 \
    "${SCRIPT_DIR}/monitor_sumo_unattended.sh" "${queue_pgid}" "${QUEUE_LOG}" \
    "${QUEUE_DIR}/watchdog.log" "${REPO_DIR}" \
    > "${QUEUE_DIR}/watchdog-launcher.log" 2>&1 </dev/null &
WATCHDOG_PID=$!
printf '%s\n' "${WATCHDOG_PID}" > "${QUEUE_DIR}/watchdog.pid"

write_status league running "go2_spot=${GO2_SPOT_MODELS}; anymal=${ANYMAL_START}"
SUMO_GO2_SPOT_START_MODEL_DIR="${GO2_SPOT_MODELS}" \
SUMO_ANYMAL_START_MODEL_DIR="${ANYMAL_START}" \
SUMO_LEAGUE_RUN_ID="${QUEUE_ID}" \
    bash "${SCRIPT_DIR}/run_three_species_sumo_league.sh" 2>&1 | tee -a "${QUEUE_LOG}"
write_status complete complete "league=${QUEUE_ID}"
