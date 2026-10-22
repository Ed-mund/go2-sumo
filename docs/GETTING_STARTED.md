# Testing Go2 Sumo on another computer

## Status and target machine

This is a source distribution, not a pre-trained demo. CPU tests are independent
of Isaac. Full environment validation must run on the destination GPU.

The supported launcher is **Linux x86-64**, with Python **3.10**, Isaac Sim
**4.5.0.0** and CUDA-enabled Torch **2.5.1 / CUDA 12.4**. Use an RTX-capable NVIDIA
GPU and a compatible installed driver. The original machine has an RTX 3090
24 GB; this is not a benchmark guarantee. The launch guard currently requires
14,000 MiB free VRAM and 25 GiB free disk. Allow substantially more disk for the
simulator, asset/shader caches, checkpoints and videos (100+ GB is prudent).

Ubuntu 22.04 is a reasonable starting point. NVIDIA's pip route requires
GLIBC 2.34+ and Python 3.10 for Isaac Sim 4.5; see the
[versioned Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/setup/installation/pip_installation.html).
Follow this repository's Torch pins instead of newer examples in that guide.
Native Windows, macOS/CPU-only simulation, and newer GPU architectures needing
a newer CUDA stack are not validated by this project.

## 1. Clone and create an isolated environment

```bash
git clone https://github.com/edmundman/go2-sumo.git
cd go2-sumo
conda create -n go2-sumo python=3.10 pip -y
conda activate go2-sumo
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
python -m pip install setuptools==69.5.1 wheel toml
```

Keep `PYTHONNOUSERSITE=1` in each shell. It prevents unrelated user-installed
Torch packages from overriding the conda environment. Do not use `sudo pip`.

## 2. Install the pinned simulation stack

Review NVIDIA's license terms before installing/using Isaac Sim. Dependencies
and assets are not redistributed here, and acceptance is not automated.

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/sumo.txt --extra-index-url https://pypi.nvidia.com
python -m pip check
python scripts/reinforcement_learning/harl/sumo_doctor.py
```

Run from the repository root so editable source paths resolve. This installs
the bundled Isaac Lab fork, not another checkout. Do not use the old
`isaaclab.sh -i` all-framework installer for this path. V3 uses its own PPO
trainer and does **not** require HARL or local HARL patches; RSL-RL is installed
only because upstream task registration imports its configuration wrappers.

Pins are selected compatibility constraints, not a full transitive lockfile.
The clean installation recipe still needs a destination-machine GPU test. Stop
and report resolver or `pip check` failures rather than ignoring dependencies.
The public snapshot corrects an inherited Hugging Face dependency conflict by
pinning hub 0.23.4 with transformers 4.41.2; no packages on the source machine
were altered while packaging.

`sumo_doctor.py` checks metadata, editable paths and `nvidia-smi` without importing
Isaac/Torch or allocating GPU memory. It is not a simulation acceptance test.
The first simulator launch may prompt for license acceptance and spend minutes
downloading extensions, NVIDIA Go2 USD assets and shader caches. Internet access
is needed unless these have already been cached; no Git LFS download is needed.

## 3. CPU tests (no simulator or GPU)

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s scripts/reinforcement_learning/harl -p 'test_*.py' -v
```

For CPU-only development, use a separate Python 3.10 environment and install only
`torch==2.5.1` from `https://download.pytorch.org/whl/cpu` plus
`requirements/cpu-test.txt`. Then run the command above and
`python scripts/reinforcement_learning/harl/sumo_doctor.py --cpu-only`.

## 4. A small, bounded GPU test

```bash
python scripts/reinforcement_learning/harl/tactics_campaign.py --prepare --envs 4 --run-dir artifacts/first-test
python scripts/reinforcement_learning/harl/tactics_campaign.py --smoke-only --run-dir artifacts/first-test
python scripts/reinforcement_learning/harl/tactics_campaign.py --status --run-dir artifacts/first-test
```

The first command just saves configuration/fingerprints. The second tests all
six phases, checkpoint transfer, opposite-side updates, optimizer resume,
configured-count benchmarks, passive standing and moving-target sensor checks.
It **does not enter production training**. Shader/asset warmup can make the first
test slow; the watchdog logs failures instead of proceeding silently.

Look at `artifacts/first-test/videos.html`, `videos/smoke/`, and the per-phase
`smoke/` logs/reports. These clips are diagnostics, not trained fighting footage.
Review feet/body collisions, terrain stability, sensor self-occlusion and pin
measurements. A passing automated test does not replace this review.

`--envs` and `--python /path/to/python` are preparation-only overrides. Later
commands reuse the saved configuration. A new source/config/runtime requires a
new run directory. Do not move prepared manifests between machines: prepare
again on the destination, because interpreter/checkpoint paths are local.

## 5. Start the real curriculum only after validation

```bash
# Defaults: 64 locomotion environments; 32 in the other phases.
python scripts/reinforcement_learning/harl/tactics_campaign.py --prepare --run-dir artifacts/training-v3
python scripts/reinforcement_learning/harl/tactics_campaign.py --run --run-dir artifacts/training-v3
```

Use a persistent terminal session for long runs. `--run` validates that new
configuration before starting fresh locomotion. Default budgets total 1.15B
transitions. To change them, copy the template JSON to an ignored
`.local-config/` directory, edit it, then pass `--config` during preparation.
Do not change the `@current` interpreter marker unless deliberately selecting
another environment. Safety thresholds are not hardware sizing guarantees.

To stop, Ctrl-C the controller. It stops only its own child process group and
keeps the most recent completed checkpoint receipt; in-progress rollout work
since that checkpoint may be lost. Repeat `--run` on the same machine/checkout
to resume. Simulator episode state is not restored bit-for-bit.

Do not load checkpoints from untrusted sources: optimizer/RNG snapshots use
Python pickle through Torch. The source-only public repo includes no checkpoints,
videos, local campaign manifests or private experiment history. Older v2 actors
are incompatible with v3. Cross-machine trained-checkpoint migration is a
separate explicit export/rebase task, not supported by copying `status.json`.

## Troubleshooting / reporting

- Wrong Torch/version: confirm `which python`, conda activation and
  `PYTHONNOUSERSITE=1`; do not upgrade the stack opportunistically.
- Wrong editable checkout: reinstall `requirements/sumo.txt` from this root.
- Occupied GPU / overheating / low disk: release resources yourself and retry;
  the launcher never kills another service.
- Missing assets or long first launch: inspect the process log and network/cache
  access; this repo deliberately does not bundle NVIDIA USDs.
- Filtered-contact shape errors: retain the log and stop; the reward code fails
  closed rather than awarding unqualified contacts.
- Bad behavior: retain failure videos and stage reports, not just highlights.

When opening an issue, include commit SHA, OS/GPU/driver, doctor output,
`pip check`, failing command and relevant log excerpt. Redact personal paths,
tokens, usernames and private URLs before posting.
