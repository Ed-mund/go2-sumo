# Go2 Sumo

Sensor-driven robot-dog sumo in Isaac Lab: learn to walk, push, recover, flank,
sweep and pin through a six-phase curriculum and a historical-opponent league.

[![Sumo CPU tests](https://github.com/edmundman/go2-sumo/actions/workflows/sumo-cpu.yml/badge.svg)](https://github.com/edmundman/go2-sumo/actions/workflows/sumo-cpu.yml)

**Experimental research code, simulation only.** The new v3 trainer has CPU unit
tests and mock-simulator integration tests. Native Isaac physics, GPU throughput,
sensor calibration and learned tactical quality still need validation. There is
no trained v3 checkpoint bundled, and skill names are intentions—not a claim
that a robot has already learned those moves.

## Start here

```bash
git clone https://github.com/edmundman/go2-sumo.git
cd go2-sumo
```

Follow **[setup and testing on another machine](docs/GETTING_STARTED.md)** for the
isolated Python 3.10 / Isaac Sim 4.5 / Torch 2.5.1 environment. Do not install the
latest Isaac Sim or copy an existing machine's conda folder.

Once installed:

```bash
export PYTHONNOUSERSITE=1
python scripts/reinforcement_learning/harl/sumo_doctor.py
python -m unittest discover -s scripts/reinforcement_learning/harl -p 'test_*.py' -v
python scripts/reinforcement_learning/harl/tactics_campaign.py --prepare --envs 4 --run-dir artifacts/first-test
# This uses the GPU, saves diagnostic clips, and stops before long training:
python scripts/reinforcement_learning/harl/tactics_campaign.py --smoke-only --run-dir artifacts/first-test
```

Preparation does not launch Isaac or CUDA. GPU commands refuse an occupied or
overheated GPU; they never stop unrelated processes. Review the smoke clips and
contact/sensor checks before starting the full campaign.

## What is included

- Separate policies for both Go2s, a visual encoder and a high-level skill selector.
- Locomotion → pushing → basic sumo → skill drills → tactics → opponent league.
- AIR/PRO-inspired RGB, range and inertial sensing, with timing/noise/history;
  no opponent ground-truth pose in actor observations.
- Physical uneven surfaces, varied appearance, motor limits and collision proxies.
- Contact-qualified sweeps/pins, bounded bonuses, fall and impact costs.
- Atomic checkpoints, both-side evaluation, quality gates and regular multi-angle
  videos with sensor previews and skill/event overlays.
- Prior ANYmal/Spot/Unitree environment code retained for reference. The new
  tactical contact mapping is Go2-specific; swapping robots requires validation.

The default full budget is **1.15 billion environment transitions**, not a promise
to finish in five days. Counts and budgets are configurable. Start small.

## Documentation

- [Installation, CPU checks, GPU smoke tests and training](docs/GETTING_STARTED.md)
- [Tactical architecture, rewards, curriculum and limitations](docs/source/how-to/go2_tactics_v3.md)
- [AIR/PRO sensor and terrain design](docs/source/how-to/go2_air_pro_sumo_v2.md)
- [Provenance, licenses and omitted artifacts](PROVENANCE.md)
- [Contributing and reporting failures](CONTRIBUTING.md)
- [Original IsaacLab-HARL README and citation](README.upstream.md)

## Credits and license

Derived from [some45bucks/IsaacLab-HARL](https://github.com/some45bucks/IsaacLab-HARL),
[DIRECTLab/IsaacLab-HARL](https://github.com/DIRECTLab/IsaacLab-HARL), and
[NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab). Upstream notices and
[BSD-3-Clause license](LICENSE) are retained; bundled Mimic code has its separate
[license](LICENSE-mimic). Isaac Sim and robot assets have their own terms and are
downloaded separately. This is not an official Unitree or NVIDIA project.
