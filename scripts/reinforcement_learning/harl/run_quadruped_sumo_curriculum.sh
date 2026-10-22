#!/usr/bin/env bash

# Run the complete locomotion -> block push -> 1v1 curriculum for one audited
# quadruped. The underlying runner and checkpoint handoff are shared with ANYmal.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROBOT="${SUMO_ROBOT:-spot}"

case "${ROBOT}" in
    spot)
        model_name="Spot"
        default_envs=2000
        ;;
    a1)
        model_name="Unitree-A1"
        default_envs=2500
        ;;
    go1)
        model_name="Unitree-Go1"
        default_envs=2500
        ;;
    go2)
        model_name="Unitree-Go2"
        default_envs=2500
        ;;
    *)
        printf 'Unknown SUMO_ROBOT=%s (expected spot, a1, go1, or go2).\n' "${ROBOT}" >&2
        exit 2
        ;;
esac

export SUMO_RUN_ID="${SUMO_RUN_ID:-$(date +%Y-%m-%d)-${ROBOT}-full-curriculum-v1}"
export SUMO_STAGE0_TASK="${model_name}-Go-To-Point-Sumo-v0"
export SUMO_STAGE1_TASK="${model_name}-Sumo-Stage1-Blocks-Push-v0"
export SUMO_STAGE2_TASK="Sumo-Stage2-${model_name}-1v1-v0"
export SUMO_STAGE0_ENVS="${SUMO_STAGE0_ENVS:-${default_envs}}"
export SUMO_STAGE1_ENVS="${SUMO_STAGE1_ENVS:-${default_envs}}"
export SUMO_STAGE2_ENVS="${SUMO_STAGE2_ENVS:-${default_envs}}"

# Long, conservative budgets. Every value is an exact number of policy updates.
export SUMO_STAGE0_STEPS="${SUMO_STAGE0_STEPS:-$((SUMO_STAGE0_ENVS * 200 * 1500))}"
export SUMO_STAGE1_STEPS="${SUMO_STAGE1_STEPS:-$((SUMO_STAGE1_ENVS * 200 * 2000))}"
export SUMO_STAGE2_STEPS="${SUMO_STAGE2_STEPS:-$((SUMO_STAGE2_ENVS * 200 * 3000))}"
export SUMO_LADDER_STEPS="${SUMO_LADDER_STEPS:-50000000}"
export SUMO_ACTOR_LR="${SUMO_ACTOR_LR:-0.0003}"
export SUMO_CRITIC_LR="${SUMO_CRITIC_LR:-0.0003}"
export SUMO_MAX_GRAD_NORM="${SUMO_MAX_GRAD_NORM:-1.0}"

exec bash "${SCRIPT_DIR}/run_anymal_sumo_curriculum.sh"
