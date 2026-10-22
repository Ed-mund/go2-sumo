Train quadruped sumo with onboard sensors
==========================================

.. note::

   This page describes the legacy forward-depth curriculum. It is not a stock
   Go2 AIR/PRO hardware specification. The prepared RGB + IMU + reference LiDAR
   successor and its calibration limitations are documented in
   ``go2_air_pro_sumo_v2.md``. Neither simulated sensor policy is hardware-certified.

The deployable policy must never receive simulator ground-truth opponent poses.
Keep ground truth inside rewards, termination, logging, and an explicitly
privileged critic only.  The current HARL Isaac Lab adapter does not expose a
separate privileged critic state, so the Go2 sensor curriculum keeps both actor
and critic sensor-limited.

Observation contract
--------------------

Every curriculum stage must use the same observation dimension and semantics:

* body velocity, angular velocity, gravity projection, joint position and
  velocity, and previous command;
* a deployable navigation vector (goal in Stage 0, mapped ring centre later);
* forward depth proximity and its one-step change;
* mapped distance from ring centre and ring radius.

The camera is attached to the anatomical-forward link before environments are
cloned.  Audit it in a nose-to-nose reset before training::

   ./isaaclab.sh -p scripts/tools/audit_go2_onboard_depth.py \
       --headless --enable_cameras --device cuda:0

The saved depth images must show the opponent in the centre.  Test collisions
separately; a correct image does not prove correct contact geometry.

Curriculum and checkpoint transfer
----------------------------------

Train locomotion first, duplicate the homogeneous Stage 0 policy for both
sides, then train block pushing, parallel sumo, and leapfrog sumo.  Do not load
old omniscient checkpoints into a sensor policy.  They have a different input
contract even if a future compatibility shim happens to make tensor dimensions
match.

The Go2 pipeline saves checkpoints, status, logs, and periodic policy videos::

   scripts/reinforcement_learning/harl/run_go2_sensor_sumo_curriculum.sh

Sensor checks before a long run
-------------------------------

Verify finite observations, correct camera direction, target visibility,
occlusion, sensor range, update rate, noise, reset history, and GPU headroom.
Start with a one-update benchmark at the intended environment count.  RTX
sensors substantially reduce throughput compared with state-only training, so
do not reuse a state-only environment count without measuring it.
