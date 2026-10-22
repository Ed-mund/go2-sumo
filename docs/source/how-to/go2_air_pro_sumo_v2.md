# Go2 AIR/PRO sumo v2 — preparation and validation notes

Prepared 2026-10-22. **CPU tests only; simulator, sensor calibration, collision
playback and training remain unvalidated.** The user's unrelated vLLM service
must remain running. Nothing in this workflow stops other GPU services.

## What is different

- Fresh locomotion → pushing → parallel sumo → alternating-team self-play.
  Planned budgets: 200M / 150M / 200M / 400M environment transitions. These are
  budgets, not promises of successful learning or completion within five days.
- Both dogs start with the same locomotion weights, then maintain separate actors
  and critics. There is no merged fighting policy. New stages get fresh critics.
- Published Go2 camera/IMU/radar mounting transforms replace the arbitrary depth
  camera mount. No stock AIR/PRO depth camera or foot-force measurements are assumed.
- Actors receive RGB, range points, IMU-derived acceleration/gyro/estimated gravity,
  joint encoders and previous sampled commands. No exact position, body velocity,
  opponent state, target vector, ring-center vector, semantic labels or force sensors.
  Contact sensors remain in the simulator for rewards/diagnostics only.
- Locomotion uses an observable cyan beacon (not an invisible goal vector); sumo
  uses the painted ring and actual robot appearance. Recognition must be learned;
  there is no pretrained robot detector or demonstrated generalization yet.
- Same repaired Go2 collision profile in every stage. Preserve the collision audit
  procedure in `audit_quadruped_collisions.rst`; LiDAR response is not a collision
  penetration test or proof that all visual meshes match colliders.
- Full-inversion cost, modest stationary-engagement cost after 2 seconds, motor
  target limits and occasional one-control-step action delay. Pushing while both
  dogs translate together is not penalized as a stall. Inversion victory has a
  recovery window. The inherited flip bonus is a geometric proxy, not a verified
  contact-force-based pin detector.

## Hardware correspondence — what is and is not real

[Unitree's specification table](https://www.unitree.com/go2/) lists RGB camera and
LiDAR on AIR/PRO, but not depth camera, foot-end force sensors or standard secondary
development. This simulation does not connect to a robot or provide a deployment
driver. Access to low-level control and sensors on the actual AIR/PRO firmware must
be established before any physical deployment; do not assume an EDU SDK path works.

Mount references come from [Unitree's Go2 URDF at a pinned commit](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/go2_description/urdf/go2_description.urdf).
Coordinates are meters relative to `base`, RPY in radians:

| Sensor | Translation | RPY |
| --- | --- | --- |
| Front RGB | (0.32715, -0.00003, 0.04297) | (0, 0, 0) |
| IMU | (-0.02557, 0, 0.04232) | (0, 0, 0) |
| Radar/LiDAR | (0.28945, 0, -0.046825) | (0, 2.8782, 0) |

AIR/PRO does not uniquely identify the LiDAR revision. The public product page now
advertises L2; older Go2s used L1. Default here is a **reference +Z hemisphere** at
the published radar transform, not a claim of a calibrated L1 or L2 replica.

- RGB: reference 90° pinhole, 128×72 at 25 Hz; actor uses 16×9 RGB and 2-frame history.
  This is downsampled simulation, not the native hardware stream. Exact intrinsics,
  distortion, exposure, timing and camera-to-base calibration are pending.
- LiDAR: PhysX dynamic collision-scene queries at the radar mount, regular angular
  sampling over a sphere, retaining the +Z hemisphere. Actor uses 3×32 pooled bins,
  4-frame history and a 10 Hz processed-cloud rate. Range is deliberately cropped to
  0.05–6 m for this arena. Ray grid, instantaneous scans and noise are approximations,
  not the proprietary scan trajectory, motion distortion or material reflectivity.
  It can see moving articulated collision geometry; do not replace it with the
  installed IsaacLab static-mesh RayCaster for opponent sensing.
- IMU is recomputed every physics tick (200 Hz); a simple complementary filter
  estimates gravity using only accelerometer/gyro. No simulator attitude enters the
  actor. Bias/noise distributions are reference assumptions, not measured data.
- 50 Hz policy, joint encoder noise, camera brightness variation and frame loss,
  range noise/dropout, explicit RGB/LiDAR frame ages. Reset histories cannot expose
  the previous episode. Hardware latency and packet loss still need measurement.

Observation contract `dog-arena-v2-airpro-1295`:
45 proprio + 2×432 RGB + 4×96 LiDAR + 2 frame ages. Old sensor/omniscient checkpoints
are incompatible and must not be silently loaded. HARL's critic is also sensor-limited.

## Terrain and appearance

The rendered and colliding floor are the **same triangle mesh**. 10×10 m patches,
12.5 cm grid, mild smooth variations up to 4 cm for locomotion and 2.5 cm for combat,
with a flat central reset pad. Robots/blocks spawn above the highest local floor
point and settle; no initial floor penetration is intended. One in four arenas is
flat. There are no stairs, loose gravel, deformable mats or moving obstacles yet.

Seeded per-arena colors and procedural checker/stripe/grain-like patterns, surface
friction variation and seeded shared lighting expose the RGB policy to different
appearances. These are procedural surface patterns, not photorealistic textures.
Terrain is invariant under a 180° turn for fair sides. Physics geometry is fixed
throughout a run, not recooked on every reset. Evaluation seeds 101/202 differ from
training seed 23. A white band follows the surface at a fixed 3 m sumo radius.

## Run management

Configuration: `scripts/reinforcement_learning/harl/configs/go2_arena_v2.json`.
Initial counts are 128 locomotion / 64 combat environments: conservative **unmeasured**
counts for RGB + PhysX LiDAR. Unique meshes disable physics replication, and the
native range backend uses USD transform synchronization. This costs throughput;
increase counts only after measuring. No five-day completion estimate until then.

From the repository root:

```bash
python scripts/reinforcement_learning/harl/sumo_campaign.py --prepare --run-dir artifacts/go2_air_pro_v2_20261022
python scripts/reinforcement_learning/harl/sumo_campaign.py --status --run-dir artifacts/go2_air_pro_v2_20261022
# Only once the GPU is available and the runtime checks can be reviewed:
python scripts/reinforcement_learning/harl/sumo_campaign.py --run --run-dir artifacts/go2_air_pro_v2_20261022
```

Preparation does not import Isaac or allocate CUDA. Run refuses occupied GPU,
insufficient free disk, hot GPU, changed source/config or an existing campaign lock.
`runtime.json` also records installed versions and HARL adapter hashes; changing
those requires a fresh validation run. A later isolation check found that Torch
2.11.0+cu130 comes from the user's site-packages, not this conda environment.
Training uses `PYTHONNOUSERSITE=1`, selecting conda Torch 2.5.1+cu124. Runtime
fingerprinting now uses the same isolation. Compatibility still needs GPU testing.
Watchdog stops only its own process group on sustained ≥83°C, <25 GiB free disk or
20 minutes without log progress. It does not automatically retry a failed audit.

Before the long run: all four phases get small finite PPO-update/strict checkpoint
handoff tests, optimizer-resume tests, passive-standing videos and production-size
one-update benchmarks. Combat passive videos first run controlled moving-opponent
RGB/LiDAR and sensor-following checks; missing data, nonfinite outputs or failure
halts the campaign. **These tests have been implemented, not run on the GPU yet.**

After every approximately 2M transitions, a separate process makes a 24-second
review clip. Views cycle corner, low side, overhead and rough-floor corner. Flat
views share seed/terrain; rough view is explicitly different. Camera and LiDAR-bin
panels show the onboard feeds. All scheduled clips, including bad matches, remain
in `videos/`; `videos.html` is a local review gallery. Stage-end evaluation uses
32 environments and two seeds, with posture, success, heading and timeout gates.
No automatic advancement if a stage fails its quality gate.

Atomic snapshots include models, Adam state, team rotation and RNG, every 10 PPO
updates plus first/final. Keep latest 3 intermediate snapshots **created by the
current chunk**, and retain every chunk-final/video checkpoint. Resume recovers
the latest receipt and starts fresh simulation episodes; it is not a bit-exact
physics continuation. Progress since the last atomic snapshot can be replayed.

CPU checks:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' /home/ed/miniforge3/envs/HARL/bin/python -m unittest discover -s scripts/reinforcement_learning/harl -p test_sumo_campaign.py -v
```

## Adding another dog / hardware calibration

1. Add a morphology/mount entry in `robot_profiles.py`; validate actuator limits,
   joint order, anatomical forward and all links against the resolved USD.
2. Run the full collision audit, passive posture and real optimizer update. Inspect
   low-side/overhead collider-overlay contact videos. Never infer collisions from RGB.
3. Match sensor frame names/transforms and actual intrinsics/range/timing; record raw
   stationary and moving-target hardware data. Identify L1 vs L2 explicitly.
4. Implement the hardware stream adapter using the same pixel/point preprocessing,
   timestamps, encoder ordering and gravity estimator. No hidden simulator fallback.
5. Register versioned tasks and a new campaign config. Do not enable another profile
   in the controller until those audits pass. Reusing task rules is supported;
   swapping a body does not automatically make the old policy safe or compatible.

Before real hardware: verify supported control access, actuator/latency calibration,
hard joint/current limits, emergency stop and a supervised non-contact test plan.
This checkpoint must not be treated as ready for physical robot combat.
