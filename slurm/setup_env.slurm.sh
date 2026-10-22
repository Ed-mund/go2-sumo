#!/usr/bin/env bash
# Create go2-sumo/env for Isambard-AI (aarch64 GH200) and launch Isaac Lab once.
# The documented stack is Isaac Sim 4.5 / Python 3.10, which has no aarch64 wheel.
# This job installs Isaac Sim 5.1 / Python 3.11 / Torch 2.9+cu130 instead.
#
#   sbatch slurm/setup_env.slurm.sh
#
#SBATCH --job-name=go2-setup
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=/projects/b6ac/go2-sumo/.logs/%x-%j.out
#SBATCH --error=/projects/b6ac/go2-sumo/.logs/%x-%j.err

set -euo pipefail
ulimit -c 0

REPO_DIR="${PROJECTDIR:-/projects/b6ac}/go2-sumo"
CONDA_ENV="${REPO_DIR}/env"
CONDA_BASE="${HOME}/miniforge3"

# shellcheck disable=SC1091
source "${REPO_DIR}/slurm/env.sh"
cd "${REPO_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

nvidia-smi --list-gpus || die "No GPU visible"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
export CONDA_PKGS_DIRS PIP_CACHE_DIR TMPDIR

if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
  log "Creating conda env at ${CONDA_ENV}"
  conda create --prefix "${CONDA_ENV}" python=3.11 pip -y
fi
conda activate "${CONDA_ENV}"
export PATH="${CONDA_ENV}/bin:${PATH}"

log "Installing ffmpeg from conda-forge"
conda install --prefix "${CONDA_ENV}" -y -c conda-forge ffmpeg
export IMAGEIO_FFMPEG_EXE="${CONDA_ENV}/bin/ffmpeg"

python -m pip install --upgrade pip
python -m pip install --cache-dir "${PIP_CACHE_DIR}" setuptools==69.5.1 wheel toml

if ! python -c "import isaacsim" >/dev/null 2>&1; then
  log "Installing Isaac Sim 5.1 aarch64"
  python -m pip install --cache-dir "${PIP_CACHE_DIR}" \
    "isaacsim[all,extscache]==5.1.0.0" --extra-index-url https://pypi.nvidia.com \
    || python -m pip install --cache-dir "${PIP_CACHE_DIR}" \
      "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
fi

if ! python -c "import torch; assert torch.__version__.split('+')[0] == '2.9.0'" >/dev/null 2>&1; then
  log "Installing torch 2.9.0+cu130"
  python -m pip install --cache-dir "${PIP_CACHE_DIR}" \
    torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130
fi

log "Installing this checkout editable, without letting setup.py replace Torch"
python -m pip install --cache-dir "${PIP_CACHE_DIR}" --no-deps \
  -e "${REPO_DIR}/source/isaaclab" \
  -e "${REPO_DIR}/source/isaaclab_assets" \
  -e "${REPO_DIR}/source/isaaclab_tasks" \
  -e "${REPO_DIR}/source/isaaclab_rl"

python - <<'PY'
import importlib
import subprocess
import sys

packages = {
    "gymnasium": "gymnasium==1.2.3",
    "imageio": "imageio==2.37.3",
    "imageio_ffmpeg": "imageio-ffmpeg==0.6.0",
    "flatdict": "flatdict==4.0.1",
    "warp": "warp-lang==1.5.0",
    "rsl_rl": "rsl-rl-lib==2.3.1",
    "hydra": "hydra-core==1.3.2",
    "prettytable": "prettytable==3.3.0",
    "einops": "einops",
    "moviepy": "moviepy==2.2.1",
    "h5py": "h5py",
    "tensorboard": "tensorboard",
    "onnx": "onnx==1.16.1",
    "PIL": "pillow==11.0.0",
    "transformers": "transformers==4.41.2",
    "huggingface_hub": "huggingface-hub==0.23.4",
    "toml": "toml",
    "numpy": "numpy==1.26.4",
    "packaging": "packaging",
}
missing = []
for module, spec in packages.items():
    try:
        importlib.import_module(module)
    except Exception as exc:
        print(f"need {spec}: {exc.__class__.__name__}: {exc}", flush=True)
        missing.append(spec)
cache = __import__("os").environ["PIP_CACHE_DIR"]
if "flatdict==4.0.1" in missing:
    # flatdict 4.0.1 imports pkg_resources at build time. Pip's isolated build
    # env uses a setuptools release that no longer provides that module.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--cache-dir", cache, "setuptools==69.5.1"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--cache-dir", cache, "--no-build-isolation", "flatdict==4.0.1"]
    )
    missing.remove("flatdict==4.0.1")
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--cache-dir", cache, *missing])
PY

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), flush=True)
assert torch.cuda.is_available(), "CUDA is not available to Torch"
PY

log "Launching Isaac Lab headless with cameras"
launch_log="${TMPDIR}/isaaclab-launch.log"
python - <<'PY' >"${launch_log}" 2>&1 || true
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app
print("ISAACLAB_APP_OK", flush=True)
app.close()
PY
cat "${launch_log}"
grep -q "ISAACLAB_APP_OK" "${launch_log}" || die "Isaac Lab did not start"
if grep -q "Failed to startup python extension" "${launch_log}"; then
  die "An Isaac Lab extension failed to load"
fi

log "Doctor"
PYTHONNOUSERSITE=1 python scripts/reinforcement_learning/harl/sumo_doctor.py || true
python -m pip check || true
log "Setup complete"
