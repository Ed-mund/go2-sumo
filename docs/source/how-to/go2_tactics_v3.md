# Go2 tactical sumo v3

Prepared 2026-10-22. This is **new training code**, not a trained fighting policy.
CPU algorithm tests and mock-simulator integration are not physics validation.
For portable installation and bounded GPU tests, start with
[GETTING_STARTED](../../GETTING_STARTED.md).

## Architecture and scope

V3 retains the AIR/PRO reference sensors, physical floors, repaired Go2 collision
profile, motor limits and timing from v2. The hardware caveats in
`go2_air_pro_sumo_v2.md` still apply: published mounts, uncalibrated camera/LiDAR
approximations, no real robot driver or verified AIR/PRO low-level control access.

New versioned tasks:

- `Dog-Sumo-Go2-Locomotion-v3`
- `Dog-Sumo-Go2-Push-v3`
- `Dog-Sumo-Go2-Fight-v3`

The new standalone PyTorch PPO trainer is separate from HARL's flat joint-action
actor. It imports the existing environments but does not patch installed packages.
V2 tasks/actors remain available; **v2 weights cannot be loaded into v3**.

Each robot has its own network and Adam optimizer:

- A trainable convolutional image encoder consumes four 64×36 RGB frames, sampled
  at approximately 10 Hz from the 25 Hz camera. This retains substantially more
  leg detail than v2's 16×9 pooled RGB. Ordered history spans roughly 0.3 seconds;
  it is not an RNN or a long-term opponent belief model.
- A 5 Hz categorical manager chooses push, circle left/right, sweep left/right,
  brace/counter, disengage, pin, or recover.
- A skill-conditioned motor policy updates the 12 joints at 50 Hz using current
  onboard observations. There are no scripted punches, canned joint trajectories
  or kinematic animation overrides during matches.
- PPO stores raw Gaussian motor samples and the actual skill decision mask.
  Manager likelihood/entropy is applied only when a decision was made, never on
  a held option or a forced drill label. Option clocks reset per environment.
- Both actor and critic remain sensor-limited: no contact labels, opponent pose,
  true velocity, privileged detector output or world position is fed to either.

Observation contract: `go2-tactics-v3-rgb4x36x64-options9`.
45 proprio + 27,648 RGB-history values + 384 range-history values + 2 sensor ages
= 28,079 environment features. The policy separately receives its own previous
skill and option countdown. Rollout images are stored as uint8 to reduce memory.

## Curriculum

| Phase | Environment transitions | Purpose |
| --- | ---: | --- |
| Locomotion | 200M | Stable movement toward a visible beacon |
| Push | 150M | Transfer locomotion to contact work; separate actors |
| Basic sumo | 80M | Establish a competent pushing/standing baseline |
| Drills | 120M | Train each skill using forced labels and tailored starting poses |
| Tactics | 200M | Free skill selection, retaining 25% drill episodes |
| League | 400M | Historical opponents, retaining 10% drill episodes |

Budgets are configurable upper training targets, not a guarantee of skill mastery
or a five-day completion promise. Current counts are 64 locomotion / 32 elsewhere,
pending actual GPU throughput and VRAM benchmarks. Full phases stop for review if
quality gates fail; they do not silently skip a weak locomotion or drill stage.

The learner side alternates across chunks. Both sides therefore receive training;
the opposing side is frozen during each chunk. Phase transitions preserve visual,
manager and motor weights but reset critic heads/Adam for the new objective.
Within-phase checkpoints retain both actors, both Adam states and RNG.

In drills, targets for sweeps, pins and recovery are passive in a random half of
eligible episodes, independent of terrain index. Other episodes use learned
opponents; the pool retains a predictable pusher. Recovery begins from a leaned
pose; pin practice begins with the defender on its side. These are **drill reset
conditions only**, not teleporting assistance during fighting. Brace/disengage
practice starts close. Full matches retain ordinary face-to-face starts.

## Contact-qualified scoring and limits

Opponent-filtered sensors are installed separately on the base and each foot;
IsaacLab requires one source body per filtered sensor. They distinguish opponent
body/leg contacts from ground contacts. Filtered force magnitudes are accumulated
over the four physics substeps so brief touches are not automatically lost.
These channels are training/evaluation instruments, not fictitious AIR/PRO sensors.

- Sweep: a moving, non-grounded front foot contacts an opponent lower leg/foot while
  the attacker has at least two other supports and both dogs were upright. Within
  a short window the opponent must tilt, while the attacker stays stable for eight
  controls. A touch alone earns nothing; an unrelated much later fall earns nothing.
- Pin: attacker upright, defender tipped, close separation, height advantage,
  real opponent contact and the defender's base contacting the floor, held for
  0.5 seconds. It ends the game. Nearby upside-down robots are insufficient.
- Counter: actual incoming contact followed by outward progress of the opponent,
  with the defender remaining upright. Continuous contact does not reset the
  original progress baseline every frame.
- Flank and disengage: measured position/bearing transitions, not just selecting
  the corresponding skill ID. Escape requires a prior close engagement.
- Recovery: measured and reported everywhere, but rewarded only in drills to
  avoid paying for deliberate falls in ordinary combat.

Every event is one-shot per robot per episode; total positive tactic shaping,
including small drill exploration bonuses, is capped at 3. Winning remains worth
20. Contact magnitude is not a positive reward. Costs discourage excessive impacts,
mechanical power and uncontrolled falls; the old blanket flat-orientation penalty
is removed for v3 combat to permit controlled lean and three-legged support.

These detectors are explicit, testable **heuristics**, not proof of causality or
perfect semantic recognition of martial-arts techniques. E.g. close temporal
contact attribution can still credit an incidental imbalance. Inspect failure
clips, contact overlays and forces before adjusting thresholds or increasing scale.
No motion-capture prior or learned dynamics calibration has been added.

## Opponent pool and evaluation

The pool keeps immutable checkpoint paths plus SHA-256 hashes, with a bounded
24-member roster. It retains early style baselines and recent opponents, samples
up to three for a chunk with a uniform floor and priority for difficult matchups,
and assigns them per environment until reset. Old files are never overwritten
by league sampling. Skill specialists are the trained conditional policy held
at a particular skill, not separately claimed-to-be-successful networks.

Passive drill outcomes are excluded from historical-opponent scores. Evaluation
excludes the exact current checkpoint path and tests both sides on seeds 101/202.
This is a small checkpoint league, not an Elo tournament or a full exploitability
analysis. Network roles, forced skills, selected opponents and seeds are recorded.

Gates include completed episode count, upright time, locomotion/pushing success,
win/draw rates, and separate trials for **each of eight non-push drills** on both
sides. Later phases also require at least three used skill choices and successful
events from at least two tactic families. Merely switching skill labels does not
satisfy the behavioral-diversity gate.

Every ~2M transitions, retain a complete 24-second clip cycling corner, low side,
overhead and rough-floor views. Overlays show selected skills, detected event
counts, raw onboard RGB and LiDAR bins. Stage-end drill clips cover every skill on
both sides. Failures are kept, not cherry-picked away. JSON reports and
`videos.html` accompany the clips. A skill label is the requested intention, not
proof that the robot executed it successfully.

## Preparing and launching later

From the cloned repository root, with the installed environment activated:

```bash
export PYTHONNOUSERSITE=1
python scripts/reinforcement_learning/harl/tactics_campaign.py --prepare --envs 4 --run-dir artifacts/first-test
python scripts/reinforcement_learning/harl/tactics_campaign.py --status --run-dir artifacts/first-test
# Only after the GPU becomes available; no production training:
python scripts/reinforcement_learning/harl/tactics_campaign.py --smoke-only --run-dir artifacts/first-test
```

Preparation uses no simulator/CUDA. Explicit launch refuses an occupied/hot GPU
or low disk; the existing watchdog controls only owned child processes. Before
long training it performs isolated optimizer updates for all six phases, strict
checkpoint transfers, resume/other-role updates, production-size benchmarks, passive
standing clips and the controlled moving-target sensor audit. Smoke weights are
discarded as production initializers: locomotion starts from scratch.

Checkpoints are atomic, every ten PPO updates plus first/final. Only the last three
snapshots created inside the current chunk are retained; all chunk-final/video
checkpoints survive. Interrupted chunks recover the latest receipt and restart
fresh simulator episodes, not bit-exact physics state. Source/config/runtime
fingerprints require a new run directory after changes. Prepared manifests are
machine-specific; prepare new ones on each destination machine.

Runtime isolation matters: unrelated user-site packages can override the pinned
**Torch 2.5.1 / CUDA 12.4** environment. Child launches and runtime fingerprinting
use `PYTHONNOUSERSITE=1`. V3 does not need HARL installed; absent HARL metadata is
recorded as null rather than preventing preparation.

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s scripts/reinforcement_learning/harl -p 'test_*.py' -v
```

Remaining mandatory GPU review: correct filtered-contact grouping across clones,
brief contact capture, base-to-floor pin measurements, passive stability on the
rough meshes, sensor self-occlusion, finite PPO at production count, multi-angle
collision playback and actual learned tactic quality. The code cannot establish
these by CPU tests. Keep everything simulation-only until hardware calibration,
supported control access and supervised safety checks are separately complete.
