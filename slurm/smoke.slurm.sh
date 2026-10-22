#!/usr/bin/env bash
# Bounded v3 smoke test. Runs every phase once, writes diagnostic mp4 clips,
# and stops before the production curriculum.
#
#   sbatch slurm/smoke.slurm.sh
#
#SBATCH --job-name=go2-smoke
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/projects/b6ac/go2-sumo/.logs/%x-%j.out
#SBATCH --error=/projects/b6ac/go2-sumo/.logs/%x-%j.err

set -euo pipefail
ulimit -c 0

REPO_DIR="${PROJECTDIR:-/projects/b6ac}/go2-sumo"
# shellcheck disable=SC1091
source "${REPO_DIR}/slurm/env.sh"
cd "${REPO_DIR}"

nvidia-smi --list-gpus || echo "nvidia-smi probe failed; continuing"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

RUN_DIR="${REPO_DIR}/artifacts/isambard-smoke"
python scripts/reinforcement_learning/harl/tactics_campaign.py \
  --prepare \
  --config "${REPO_DIR}/slurm/go2_tactics_v3.isambard.json" \
  --python "${CONDA_ENV}/bin/python" \
  --envs 4 \
  --run-dir "${RUN_DIR}"
python scripts/reinforcement_learning/harl/tactics_campaign.py \
  --smoke-only \
  --run-dir "${RUN_DIR}"
python scripts/reinforcement_learning/harl/tactics_campaign.py \
  --status \
  --run-dir "${RUN_DIR}"

echo "Smoke clips: ${RUN_DIR}/videos/smoke"
find "${RUN_DIR}/videos" -name '*.mp4' -printf '%p %s\n' || true
