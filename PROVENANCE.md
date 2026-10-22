# Provenance and distribution scope

This repository is a source-only snapshot of the Go2 sumo development tree at
commit `278b5301`, followed by portable-publication changes. Its upstream lineage
is some45bucks/IsaacLab-HARL → DIRECTLab/IsaacLab-HARL → NVIDIA Isaac Lab.
The full original history is intentionally not imported. This does not claim
authorship of upstream code; existing copyright/SPDX headers, license files,
dependency notices and the upstream README/citation are retained.

The new public repository is independent of any existing private repository.
No private repository visibility was changed. The source development tree and
its saved checkpoints remain untouched.

Included: tracked simulator/task/training source, robot collision profiles,
sensor/terrain code, earlier robot environment implementations, documentation
and CPU tests. New portable launch controls, dependency constraints, metadata
diagnostics and public CPU CI were added when packaging.

Excluded: training weights (`.pt`/`.pth`), experiment outputs, videos, local
manifests, conda environments, credentials, old Git history, upstream LFS GIFs,
paper-result plotting artifacts and host-specific Docker `.env` files. These
omissions mean some legacy demos and documentation images are unavailable.
The old cluster launch scripts and upstream README are historical references,
not the supported v3 installation path. Use `docs/GETTING_STARTED.md` instead.

The source retains its upstream BSD-3-Clause license (`LICENSE`) and the separate
Mimic license (`LICENSE-mimic`) where applicable. Go2 sumo additions are provided
under BSD-3-Clause. Third-party license notices remain under `docs/licenses/`.
NVIDIA Isaac Sim and robot asset licenses are separate; their binaries/assets
are not included in this Git repository. No real-hardware deployment is claimed.

Known validation boundary: unit tests and mock-simulator integration can validate
PPO updates/checkpoint handling/event logic, not physical realism, calibrated
sensors, stability on a different GPU, or successful learned tactics. The setup
recipe is pinned but not a hermetic environment image; report dependency drift.
