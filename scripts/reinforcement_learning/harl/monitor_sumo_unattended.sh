#!/usr/bin/env bash

# Stop an unattended sumo process group if the GPU remains too hot, disk space
# becomes unsafe, or the pipeline stops producing logs. This monitor never
# restarts training: the periodic HARL checkpoints remain the recovery point.

set -euo pipefail

if (( $# != 4 )); then
    printf 'Usage: %s PGID PIPELINE_LOG WATCHDOG_LOG DISK_PATH\n' "$0" >&2
    exit 2
fi

TARGET_PGID="$1"
PIPELINE_LOG="$2"
WATCHDOG_LOG="$3"
DISK_PATH="$4"
TEMP_LIMIT_C="${SUMO_WATCHDOG_TEMP_C:-83}"
TEMP_SAMPLES="${SUMO_WATCHDOG_TEMP_SAMPLES:-3}"
MIN_DISK_GB="${SUMO_WATCHDOG_MIN_DISK_GB:-20}"
STALL_MINUTES="${SUMO_WATCHDOG_STALL_MINUTES:-20}"
POLL_SECONDS="${SUMO_WATCHDOG_POLL_SECONDS:-60}"

mkdir -p "$(dirname -- "${WATCHDOG_LOG}")"

log_message() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" >> "${WATCHDOG_LOG}"
}

group_is_running() {
    ps -g "${TARGET_PGID}" -o pid= 2>/dev/null | grep -q '[0-9]'
}

stop_group() {
    local reason="$1"
    log_message "SAFETY STOP: ${reason}; sending TERM to process group ${TARGET_PGID}"
    kill -TERM -- "-${TARGET_PGID}" 2>/dev/null || true
    for _ in $(seq 1 60); do
        group_is_running || return 0
        sleep 1
    done
    log_message "Process group still running after 60 seconds; sending INT"
    kill -INT -- "-${TARGET_PGID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
        group_is_running || return 0
        sleep 1
    done
    log_message "Process group still running after graceful signals; sending KILL"
    kill -KILL -- "-${TARGET_PGID}" 2>/dev/null || true
}

hot_samples=0
log_message "Monitoring process group ${TARGET_PGID}: temp=${TEMP_LIMIT_C}C/${TEMP_SAMPLES} samples, disk=${MIN_DISK_GB}GB, stall=${STALL_MINUTES}m"

while group_is_running; do
    now_epoch="$(date +%s)"
    free_kb="$(df --output=avail -k "${DISK_PATH}" | tail -n 1 | tr -d ' ')"
    free_gb=$((free_kb / 1024 / 1024))
    if (( free_gb < MIN_DISK_GB )); then
        stop_group "free disk ${free_gb}GB is below ${MIN_DISK_GB}GB"
        exit 20
    fi

    if [[ -f "${PIPELINE_LOG}" ]]; then
        log_epoch="$(stat -c %Y "${PIPELINE_LOG}")"
        log_age_minutes=$(((now_epoch - log_epoch) / 60))
        if (( log_age_minutes >= STALL_MINUTES )); then
            stop_group "pipeline log has not changed for ${log_age_minutes} minutes"
            exit 21
        fi
    fi

    gpu_sample="$(nvidia-smi --query-gpu=temperature.gpu,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
    gpu_temp="$(printf '%s' "${gpu_sample}" | cut -d, -f1 | tr -d ' ')"
    if [[ "${gpu_temp}" =~ ^[0-9]+$ ]] && (( gpu_temp >= TEMP_LIMIT_C )); then
        hot_samples=$((hot_samples + 1))
    else
        hot_samples=0
    fi
    log_message "healthy: gpu=${gpu_sample:-unavailable}; free_disk=${free_gb}GB; hot_samples=${hot_samples}/${TEMP_SAMPLES}"
    if (( hot_samples >= TEMP_SAMPLES )); then
        stop_group "GPU held at or above ${TEMP_LIMIT_C}C for ${hot_samples} samples"
        exit 22
    fi
    sleep "${POLL_SECONDS}"
done

log_message "Training process group exited; watchdog complete"
