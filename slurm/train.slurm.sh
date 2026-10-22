#!/usr/bin/env bash
# Production v3 curriculum. Smoke-gates the run, then trains and writes a review
# clip after each chunk. Resubmit the same script to continue a stopped run;
# the campaign keeps the latest checkpoint receipt.
#
#   sbatch slurm/train.slurm.sh
#
#SBATCH --job-name=go2-train
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=18:00:00
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

RUN_DIR="${REPO_DIR}/artifacts/isambard-train"
if [[ ! -f "${RUN_DIR}/config.json" ]]; then
  python scripts/reinforcement_learning/harl/tactics_campaign.py \
    --prepare \
    --config "${REPO_DIR}/slurm/go2_tactics_v3.isambard.json" \
    --python "${CONDA_ENV}/bin/python" \
    --run-dir "${RUN_DIR}"
fi
python scripts/reinforcement_learning/harl/tactics_campaign.py \
  --run \
  --run-dir "${RUN_DIR}"
python scripts/reinforcement_learning/harl/tactics_campaign.py \
  --status \
  --run-dir "${RUN_DIR}"

echo "Review page: ${RUN_DIR}/videos.html"
find "${RUN_DIR}/videos" -name '*.mp4' -printf '%p %s\n' || true
