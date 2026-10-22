# Shared environment for go2-sumo Slurm jobs. Source after slurm/setup_env.slurm.sh.
# Isaac Sim and the PyTorch wheel ship their own CUDA. Do not module-load cuda/12.6:
# that puts the older toolkit ahead of the CUDA 13 libraries this stack needs.
REPO_DIR="${REPO_DIR:-${PROJECTDIR:-/projects/b6ac}/go2-sumo}"
CONDA_ENV="${CONDA_ENV:-${REPO_DIR}/env}"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniforge3}"
CACHE_DIR="${REPO_DIR}/.cache"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export CONDA_PKGS_DIRS="${CACHE_DIR}/conda_pkgs"
export TMPDIR="${CACHE_DIR}/tmp"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export XDG_DATA_HOME="${CACHE_DIR}/xdg-data"
export XDG_CONFIG_HOME="${CACHE_DIR}/xdg-config"
export TORCH_HOME="${CACHE_DIR}/torch"
export HF_HOME="${CACHE_DIR}/hf"
export CUDA_CACHE_PATH="${CACHE_DIR}/cuda"
export __GL_SHADER_DISK_CACHE=1
export __GL_SHADER_DISK_CACHE_PATH="${CACHE_DIR}/gl_shaders"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.aarch64.json}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"

# Isaac Sim wants the system OpenMP runtime, not the copy bundled in the Torch wheel.
if [[ -f /usr/lib64/libgomp.so.1 && "${LD_PRELOAD:-}" != *"/usr/lib64/libgomp.so.1"* ]]; then
  export LD_PRELOAD="/usr/lib64/libgomp.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

mkdir -p "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}" "${TMPDIR}" "${XDG_CACHE_HOME}/ov" \
  "${XDG_DATA_HOME}/ov" "${XDG_CONFIG_HOME}" "${TORCH_HOME}" "${HF_HOME}" \
  "${CUDA_CACHE_PATH}" "${__GL_SHADER_DISK_CACHE_PATH}" "${REPO_DIR}/.logs" \
  "${HOME}/.cache" "${HOME}/.local/share" "${HOME}/.nvidia-omniverse"

# Isaac Sim writes Nucleus and Kit caches under the home directory. Home is quota-limited;
# keep those trees on the project filesystem.
if [[ ! -e "${HOME}/.cache/ov" && ! -L "${HOME}/.cache/ov" ]]; then
  ln -s "${XDG_CACHE_HOME}/ov" "${HOME}/.cache/ov"
fi
if [[ ! -e "${HOME}/.local/share/ov" && ! -L "${HOME}/.local/share/ov" ]]; then
  ln -s "${XDG_DATA_HOME}/ov" "${HOME}/.local/share/ov"
fi

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

if [[ -x "${CONDA_ENV}/bin/python" ]]; then
  conda activate "${CONDA_ENV}"
  export PATH="${CONDA_ENV}/bin:${PATH}"
  if [[ -x "${CONDA_ENV}/bin/ffmpeg" ]]; then
    export IMAGEIO_FFMPEG_EXE="${CONDA_ENV}/bin/ffmpeg"
  fi
fi
