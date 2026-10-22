# Contributing

Use the environment and tests in [GETTING_STARTED](docs/GETTING_STARTED.md).
Keep changes scoped, version observation/action contracts, and add CPU tests for
reward/event logic and checkpoint handling. Never claim a GPU test from a mocked
simulator test. Include failure clips and per-stage metrics when changing physics.

Do not commit weights, videos, machine-specific run manifests, credentials or
large binary assets. Do not add arbitrary robot meshes without checking their
license. New morphologies need measured collider coverage, facing/spawn audits,
joint/action limits, sensor mounts, contact-body mapping and full curriculum
validation; changing a robot-name string is insufficient.

Open issues for reproducible defects with commit, doctor output, versions, command
and a redacted log excerpt. Security-sensitive issues should not contain live
credentials or private checkpoint links. Checkpoints are executable pickle data:
load only trusted local campaign outputs.
