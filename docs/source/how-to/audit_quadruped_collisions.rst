.. _audit-quadruped-collisions:

Auditing Quadruped Collision Geometry
=====================================

Do this audit before training a contact-heavy task with a new quadruped asset.
Rendered geometry is not evidence that collision geometry exists: a robot can look
complete while its torso ends, hip links, or lower legs have no collider at all.
Training against that asset teaches policies to exploit the holes and makes old
checkpoints incompatible with a later geometry repair.

The short rule is: inspect the resolved USD, measure every link, render the collision
shapes, and pass a real optimizer update before starting the curriculum.

For adversarial quadruped tasks, also audit posture rewards against the repaired
geometry. An XY-only projected-gravity penalty is zero both when upright and when
perfectly upside down. Use a separate inverted-back penalty and, where the game rules
permit it, terminate only after inversion persists for a short recovery window. A
height-only fall detector can miss an inverted robot because a complete torso collider
holds its centre of mass well above the ground.

Pre-training checklist
----------------------

#. Resolve the asset in a live one-environment scene. Do not inspect only the source
   URDF or an unresolved USD reference.
#. For every torso and leg link, record:

   * visual bounds in link-local coordinates;
   * collision shape type, dimensions, axis, translation, and rotation;
   * parent/child joint local positions;
   * the rigid body that owns each shape.

#. Compare the visual and collision coverage for the torso centre, both torso ends,
   hip/abduction links, upper legs, lower legs, and feet.
#. Add simple analytic proxies only where coverage is missing. Capsules, spheres, and
   boxes are preferable to complex triangle meshes for high-throughput training.
#. Put a proxy on the rigid link whose local coordinates were measured. Do not attach
   one long shape across two adjacent rigid bodies.
#. Leave clearance near joints. A capsule whose rounded tip overlaps the knee or foot
   collider on an adjacent link can create permanent self-contact and explosive
   velocities at reset.
#. Use a modest contact offset. Contact offset starts contact generation; it does not
   repair missing physical volume.
#. Enable CCD and adequate solver iterations for fast leg contacts, but treat these as
   supplements to correct geometry.
#. Validate passive standing, policy playback, and an actual PPO update before a long
   run. All observations, rewards, actions, and gradients must remain finite.
#. Test upright, side, transient-inversion, and sustained-back poses numerically. Verify
   that upright receives zero back penalty, transient inversion can recover, and a
   sustained inversion produces the intended loss for the correct robot.

Recommended validation gates
----------------------------

Use the same checkpoint and seed for every visual comparison so only geometry changes.

``Gate 1: reset and passive stand``
   Run one environment for several episodes. Look for launch impulses, jitter,
   tunnelling, extreme velocities, or a robot falling before actions are applied.

``Gate 2: deterministic contact playback``
   Record low side, opposite side, corner, and overhead views. Inspect actual contact
   frames rather than a single favourable angle. A leg crossing another leg in a 2-D
   image is not automatically penetration; use a collision-shape overlay to resolve
   depth ambiguity.

``Gate 3: optimizer smoke test``
   Run at least one complete rollout and optimizer update with 32 or more environments.
   Playback can remain finite even when a bad contact produces observations large
   enough to corrupt PPO during the update.

``Gate 4: scaled throughput test``
   Increase to the intended environment count and confirm that FPS, GPU memory, reset
   behaviour, and reward distributions remain reasonable.

``Gate 5: anatomical-forward audit``
   Do not infer the nose from a visually ambiguous torso panel. Record the front hip
   joint locations in the resolved asset frame and define a per-asset yaw offset from
   the USD root axis to anatomical forward. Locomotion resets must face the sampled
   goal, block-push resets must face the assigned block, and opposing robots must face
   one another. Run ``play.py --zero_actions --report_heading`` to verify reset
   alignment, then audit a trained checkpoint with ``--deterministic --report_heading``.
   Reject a checkpoint if either ``facing_away`` or ``backing_while_moving`` exceeds
   the curriculum threshold. High task reward alone is insufficient: an earlier Go2
   locomotion policy reached 97% of goals while backing up 91% of the time.

   Reversing can still be trained deliberately. The locomotion stage samples 15% of
   episodes with the goal directly behind the robot and disables the forward-facing
   penalty only for those episodes. Keep this fraction below the heading-gate limit and
   log ``Metrics/reverse_spawn_percentage`` so a useful reverse gait cannot silently
   turn back into a universal tail-first exploit.

ANYmal-C sumo case study
------------------------

The ANYmal-C sumo audit found several independent gaps:

* The stock torso collider was a cylinder approximately 0.75 m long and 0.15 m wide,
  while the rendered chassis and bumpers extended to roughly 1.05 m by 0.32 m.
* The rendered front and rear faces extended about 14 cm beyond the earlier proxy.
* The stock thigh collider covered mainly the distal knee region.
* The visible lower-leg rod belonged to the foot visual hierarchy, while the shank had
  only a thin 0.20 m cylinder and the foot had a small sphere.
* The hip links needed explicit exterior coverage.

The stable repair uses a central torso capsule, separate measured front/rear boxes,
upper-leg capsules, hip spheres, and lower-leg capsules aligned from each shank origin
towards its measured fixed-foot joint. The lower-leg capsule has asymmetric joint
clearances so its rounded tips do not overlap the adjacent knee and foot shapes.

Two attempted repairs were rejected during isolation testing:

* Attaching a full lower-leg proxy to the foot link caused overlapping shapes across
  fixed-together rigid bodies.
* A full-width central chassis box matched the visual shell but overlapped articulated
  hip/leg geometry in the standing pose. It caused extreme physics values and NaNs.
  The measured end boxes were stable, so the central box remains disabled.

The implementation and diagnostic switches are in
``source/isaaclab_tasks/isaaclab_tasks/direct/adversarial/sumo/anymal_c/sumo_stage_2_anymal_1v1.py``.
Stage 1 reuses the same repair method.

Useful diagnostic switches
--------------------------

The switches below allow collider groups to be isolated without maintaining divergent
asset variants:

.. code-block:: bash

   ISAACLAB_SUMO_FULL_TORSO_PROXY=1  # measured front/rear boxes
   ISAACLAB_SUMO_CHASSIS_PROXY=0     # rejected central full-width box
   ISAACLAB_SUMO_SHANK_PROXY=1       # measured lower-leg capsules
   ISAACLAB_SUMO_HIP_PROXY=1         # hip spheres
   ISAACLAB_SUMO_SHOW_PROXY_GEOMETRY=1  # cyan/yellow overlay in recorded RGB
   ISAACLAB_SUMO_COLLIDER_DIAGNOSTICS=1 # resolved USD geometry/joint dump

For a one-update Stage-2 stability check:

.. code-block:: bash

   cd scripts/reinforcement_learning/harl
   PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
   "${CONDA_PREFIX}/bin/python" train.py \
     --algorithm happo_adv \
     --task Sumo-Stage2-Anymal-1v1-v0 \
     --num_envs 32 \
     --num_env_steps 6400 \
     --seed 71 \
     --exp_name quadruped_collider_smoke \
     --adversarial_training_mode parallel \
     --adversarial_training_iterations 100000000 \
     --headless --rendering_mode performance \
     --dir /absolute/path/to/checkpoint

For a deterministic collision-overlay video, add
``ISAACLAB_SUMO_SHOW_PROXY_GEOMETRY=1`` to the environment and run ``play.py`` with
``--deterministic --video --debug --headless --enable_cameras``. Use a low camera and
the same seed/checkpoint as the normal render.

Starting a real curriculum
--------------------------

Once all gates pass, retrain every curriculum stage from the beginning. Locomotion,
push-object, and one-versus-one policies all experience the robot's collision geometry,
so resuming a policy trained with missing colliders can preserve exploits and obscure
whether the repair works. Keep the old checkpoint only as a deterministic comparison
policy for the audit renders.

Apply the same restart rule after changing spawn headings or forward-facing rewards.
The shared sumo stages use an explicit ``forward_yaw_offset`` per asset and penalize
facing away from the goal, block, or opponent. The full Go2-vs-Spot launcher runs a
heading gate after both locomotion stages, block push, competitive bootstrap, and final
league training, preventing a backward policy from silently entering the next stage.
Stage-0 runs may be seeded from validated locomotion checkpoints with
``SUMO_GO2_START_MODEL_DIR`` or ``SUMO_SPOT_START_MODEL_DIR``; this is appropriate for
reward shaping changes that retain the same collision geometry and observation/action
spaces.

For an interrupted mixed push stage, set ``SUMO_START_PHASE=2`` and
``SUMO_STAGE1_START_MODEL_DIR`` to a checkpoint containing both actors, both critics,
and both value normalizers. Set ``SUMO_STAGE1_STEPS`` to the remaining, rather than the
original, step budget. For unattended runs, launch the pipeline under
``systemd-inhibit`` and monitor its process group with
``monitor_sumo_unattended.sh``. The monitor safely stops the group after sustained high
temperature, low disk space, or a stale pipeline log; it deliberately does not restart
from an unknown checkpoint automatically.

The full heterogeneous launcher records deterministic evaluation videos after mixed
push training, competitive bootstrap, and final league training. Videos are written
under the run artifact directory and video failure is non-fatal, so an encoder or camera
problem cannot discard an otherwise healthy multi-day training run. Set
``SUMO_SAVE_STAGE_VIDEOS=0`` to disable them, or adjust ``SUMO_VIDEO_LENGTH`` and
``SUMO_VIDEO_ENVS`` for shorter or broader evaluations.

Multi-species leagues must keep one actor, critic, and value-normalizer bank per robot
morphology. Do not treat a shared 12-joint action width as policy compatibility. The
three-species launcher rotates ANYmal-C, Spot, and Go2 through all six ordered pairings,
harvesting each trained actor back into only its own species bank after every match.
This places every morphology in both team slots and prevents spawn-side bias from being
mistaken for fighting skill. Each ordered match produces a deterministic evaluation
video. A1 and Go1 should join only after completing their own locomotion and push
prerequisites; copying a mature Go2 actor into them would invalidate the league.

For each new quadruped, save the measurements and validation result beside its task
configuration. At minimum record the asset revision, proxy dimensions and owning links,
joint clearances, contact/solver settings, smoke-test command, FPS, and the paths to one
normal and one collider-overlay video.

Spot and Unitree audit (2026-08-11)
-----------------------------------

The live resolved-USD audit is implemented in
``scripts/tools/audit_quadruped_colliders.py``. Reports for Spot, A1, Go1, and Go2 are
written under ``artifacts/quadruped_collision_audits/2026-08-11``. The audit found that
none of the stock assets was suitable for inter-robot contact training without repair:

================  ====================  ==============================================
Robot             Stock torso coverage  Main gaps
================  ====================  ==============================================
Spot              83% / 20% / 13%      No hip colliders; radially thin leg capsules
Unitree A1         58% / 13% / 11%      No hip colliders; thigh/calf joint markers only
Unitree Go1        52% / 9% / 10%       Partial hip/thigh coverage; calf nearly absent
Unitree Go2        31% / 5% / 7%        Severely undersized torso and thigh collision
================  ====================  ==============================================

Percentages are collision-to-visual axis spans in the resolved default pose. They are a
screening signal, not a license to add one full visual bounding box: adjacent articulated
links occupy part of those bounds.

The measured repairs are stored in
``source/isaaclab_tasks/isaaclab_tasks/direct/adversarial/sumo/quadruped_colliders.py``.
Each profile uses separate central/end torso boxes, a sphere owned by every hip link,
and capsules owned by upper/lower leg links. Capsules stop short of both joints, feet
that already had adequate sphere colliders are unchanged, CCD is enabled per rigid link,
and contact offset remains a modest 12 mm.

Validation completed on 2026-08-11:

* Spot passed full PPO updates in locomotion, block-push, and 1v1 tasks.
* Unitree Go2 passed full PPO updates in locomotion, block-push, and 1v1 tasks.
* Unitree A1 and Go1 passed full 1v1 PPO updates through the shared staged code path.
* All smoke tests had finite rewards and optimizer updates; no sustained-back losses
  occurred in the Spot and Go2 validation rollouts.

The reusable launcher is ``run_quadruped_sumo_curriculum.sh``. Select a model with
``SUMO_ROBOT=spot``, ``a1``, ``go1``, or ``go2``. Every run begins at locomotion and
hands its own local checkpoints through block push into 1v1; policies are never copied
between robot models merely because they share a 12-joint action space.
