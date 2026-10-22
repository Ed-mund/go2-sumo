"""Run with HARL Python -m unittest discover -s this_directory -p test_sumo_campaign.py.

No Isaac application, CUDA allocation, network request or training subprocess.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np

import sumo_campaign as campaign
import sumo_checkpoint as checkpoints

HELPERS = campaign.REPO / "source/isaaclab_tasks/isaaclab_tasks/direct/adversarial/sumo/sensor_history.py"
spec = importlib.util.spec_from_file_location("sumo_sensor_test_helpers", HELPERS)
sensors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensors)
spec = importlib.util.spec_from_file_location("sumo_surface_test_helpers", HELPERS.with_name("arena_surface.py"))
surfaces = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surfaces)


class CampaignTests(unittest.TestCase):
    def test_batch_rounding_never_silently_drops_steps(self):
        self.assertEqual(campaign.rounded_steps(2_000_000, 1024, 200), 2_048_000)
        self.assertEqual(campaign.rounded_steps(2_048_000, 1024, 200), 2_048_000)

    def test_busy_gpu_blocks_before_any_training(self):
        cfg = campaign.load_config(campaign.DEFAULT_CONFIG)
        with patch.object(campaign, "gpu_sample", return_value=(1700, 70, 100)):
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                campaign.preflight(cfg)

    def test_missing_or_small_checkpoints_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                campaign.checkpoint_files(folder, 1)
            (Path(folder) / "actor_agent_robot_0.pt").write_text("bad")
            with self.assertRaises(RuntimeError):
                campaign.checkpoint_files(folder, 1)

    def test_gates_require_episodes_metrics_and_posture(self):
        cfg = campaign.load_config(campaign.DEFAULT_CONFIG)
        stage = cfg["stages"][0]
        self.assertTrue(campaign.gate_errors({}, stage, cfg))
        report = {
            "episodes": 64,
            "metrics": {"Metrics/goal_success_percentage": 0.9},
            "posture": {"robot_0": {"upright": 0.97}},
            "heading": {"robot_0": {"facing_away": 0.15, "backing_while_moving": 0.2}},
        }
        self.assertFalse(campaign.gate_errors(report, stage, cfg))
        report["metrics"]["Metrics/goal_success_percentage"] = float("nan")
        self.assertTrue(campaign.gate_errors(report, stage, cfg))

    def test_fighting_that_never_resolves_fails(self):
        cfg = campaign.load_config(campaign.DEFAULT_CONFIG)
        report = {
            "episodes": 64,
            "metrics": {"timeout_percentage": 1.0},
            "posture": {"robot_0": {"upright": 1.0}, "robot_1": {"upright": 1.0}},
        }
        self.assertTrue(campaign.gate_errors(report, cfg["stages"][2], cfg))

    def test_no_training_recording_and_explicit_transfer(self):
        cfg = campaign.load_config(campaign.DEFAULT_CONFIG)
        cmd = campaign.train_command(cfg, cfg["stages"][0], 204800, "test", Path("out"), "source")
        self.assertNotIn("--video", cmd)
        self.assertIn("--transfer_actors_only", cmd)
        cmd = campaign.train_command(cfg, cfg["stages"][0], 204800, "test", Path("out"), "source", True)
        self.assertIn("--resume_optimizers", cmd)

    def test_bad_posture_and_timeout_nan_fail_closed(self):
        cfg = campaign.load_config(campaign.DEFAULT_CONFIG)
        report = {
            "episodes": 64,
            "metrics": {"timeout_percentage": float("nan")},
            "posture": {"robot_0": {"upright": float("nan")}, "robot_1": {"upright": 1.0}},
        }
        self.assertGreaterEqual(len(campaign.gate_errors(report, cfg["stages"][2], cfg)), 2)

    def test_smoke_requires_dynamic_sensor_audit_and_rough_clip(self):
        cfg = campaign.load_config(campaign.DEFAULT_CONFIG)
        cmd = campaign.play_command(cfg, cfg["stages"][2], "checkpoint", Path("out"), zero=True)
        self.assertIn("--sensor_audit", cmd)
        cmd = campaign.play_command(cfg, cfg["stages"][2], "checkpoint", Path("out"), view="rough_corner")
        self.assertIn("--arena_variant", cmd)


class SensorTests(unittest.TestCase):
    def test_rgb_is_derived_only_from_pixels(self):
        image = torch.zeros(2, 72, 128, 4, dtype=torch.uint8)
        image[0, ..., 0] = 255
        image[1, ..., 1] = 255
        actual = sensors.rgb_features(image)
        self.assertEqual(actual.shape, (2, 432))
        self.assertTrue(torch.all(actual[0, :144] == 1))
        self.assertTrue(torch.all(actual[1, 144:288] == 1))
        self.assertTrue(torch.all(actual[:, 288:] == 0))

    def test_lidar_hemisphere_and_approaching_returns(self):
        points = torch.tensor([[[2.0, 0.0, 0.1], [1.0, 0.0, -0.1], [float("nan"), 0.0, 1.0]]])
        far = sensors.lidar_features(points)
        self.assertEqual(int((far > 0).sum()), 1)
        points[0, 0, 0] = 1.0
        near = sensors.lidar_features(points)
        self.assertGreater(near.max().item(), far.max().item())
        self.assertTrue(torch.isfinite(near).all())

    def test_complementary_filter_uses_only_imu_and_rejects_impact_tilt(self):
        g = torch.tensor([[0.0, 0.0, -1.0]])
        gyro = torch.zeros_like(g)
        self.assertTrue(torch.allclose(sensors.gravity_estimate(g, gyro, -g * 9.81, 0.02), g))
        impact = torch.tensor([[50.0, 0.0, 9.81]])
        self.assertTrue(torch.allclose(sensors.gravity_estimate(g, gyro, impact, 0.02), g))
        self.assertFalse(torch.allclose(sensors.gravity_estimate(g, torch.tensor([[1.0, 0.0, 0.0]]), impact, 0.02), g))

    def test_terrain_is_physical_bounded_symmetric_and_reproducible(self):
        flat, _, _, flat_max = surfaces.surface_mesh(23, 0, 0)
        self.assertAlmostEqual(flat_max, 0.001)
        self.assertTrue(np.allclose(flat[:, 2], 0.001))
        for stage in (0, 1, 2):
            vertices, triangles, colors, maximum = surfaces.surface_mesh(23, 1, stage)
            z = vertices[:, 2].reshape(81, 81)
            self.assertTrue(np.allclose(z, z[::-1, ::-1]))
            self.assertGreater(maximum, 0.008)
            self.assertLessEqual(maximum, 0.041 if stage == 0 else 0.026)
            self.assertEqual(len(colors), len(triangles))
            self.assertGreaterEqual(colors.min(), 0)
            self.assertLessEqual(colors.max(), 1)
            self.assertTrue(np.array_equal(vertices, surfaces.surface_mesh(23, 1, stage)[0]))
            self.assertFalse(np.array_equal(colors, surfaces.surface_mesh(24, 1, stage)[2]))

    def test_partial_reset_does_not_leak_old_episode_depth(self):
        history = sensors.DepthHistory(2, features=2, frames=4)
        eligible = torch.tensor([True, True])
        history.update(torch.tensor([[0.2, 0.3], [0.4, 0.5]]), 2, eligible)
        history.reset(torch.tensor([0]))
        self.assertEqual(history.values[0].sum().item(), 0)
        self.assertGreater(history.values[1].sum().item(), 0)
        history.update(torch.ones(2, 2), 3, torch.tensor([False, True]))
        self.assertEqual(history.values[0].sum().item(), 0)
        history.update(torch.ones(2, 2) * 0.8, 4, eligible)
        self.assertTrue(torch.allclose(history.values[0], torch.full((4, 2), 0.8)))

    def test_repeated_read_does_not_advance_history(self):
        history = sensors.DepthHistory(1, features=2, frames=4)
        history.update(torch.ones(1, 2), 2, torch.tensor([True]))
        history.update(torch.zeros(1, 2), 2, torch.tensor([True]))
        self.assertTrue(torch.all(history.values == 1))
        history.update(torch.zeros(1, 2), 4, torch.tensor([True]))
        self.assertTrue(torch.all(history.values[:, 0] == 0))
        self.assertTrue(torch.all(history.values[:, 1:] == 1))

    def test_ranges_and_joint_bounds(self):
        depth = torch.tensor([[[float("nan"), float("inf"), 0.6, 6.0]]])
        actual = sensors.depth_proximity(depth, shape=(1, 4))
        self.assertTrue(torch.allclose(actual, torch.tensor([[0.0, 0.0, 0.9, 0.0]])))
        limits = torch.tensor([[[-0.5, 0.5], [-0.2, 0.2]]])
        target = sensors.joint_targets(torch.tensor([[100.0, -100.0]]), torch.zeros(1, 2), limits, 0.25)
        self.assertTrue(torch.allclose(target, torch.tensor([[0.5, -0.2]])))

    def test_bracing_grace_then_stall_and_recovery(self):
        counter = torch.zeros(1, dtype=torch.long)
        for _ in range(99):
            counter, stalled = sensors.stall_update(counter, torch.tensor([True]), torch.tensor([0.0]), 0.02)
        self.assertFalse(stalled.item())
        counter, stalled = sensors.stall_update(counter, torch.tensor([True]), torch.tensor([0.0]), 0.02)
        self.assertTrue(stalled.item())
        counter, stalled = sensors.stall_update(counter, torch.tensor([True]), torch.tensor([0.2]), 0.02)
        self.assertFalse(stalled.item())
        self.assertEqual(counter.item(), 0)


def fake_runner():
    actor = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(2, 1)
    aopt, copt = torch.optim.Adam(actor.parameters()), torch.optim.Adam(critic.parameters())
    normalizer = torch.nn.Linear(1, 1)
    runner = SimpleNamespace(
        actors={"team_0": {"robot_0": SimpleNamespace(actor=actor, actor_optimizer=aopt)}},
        critics={"team_0": SimpleNamespace(critic=critic, critic_optimizer=copt)},
        value_normalizers={"team_0": normalizer},
        device=torch.device("cpu"),
        training_mode="leapfrog",
        current_team_index=1,
        current_team_train_steps=1234,
        training_teams=["team_1"],
        algo_args={"train": {"episode_length": 2, "n_rollout_threads": 2}},
    )

    def train():
        for model, optimizer in ((actor, aopt), (critic, copt)):
            optimizer.zero_grad()
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
        return {"loss": torch.tensor(1.0)}, {}

    def save(folder):
        torch.save(actor.state_dict(), folder / "actor_agent_robot_0.pt")
        torch.save(critic.state_dict(), folder / "team_0_critic_agent.pt")
        torch.save(normalizer.state_dict(), folder / "team_0_value_normalizer.pt")

    runner.train, runner.save, runner.insert = train, save, lambda data: None
    return runner


class CheckpointTests(unittest.TestCase):
    def test_retention_keeps_three_new_snapshots_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as folder:
            unrelated = Path(folder) / "keep_me.txt"
            unrelated.write_text("user checkpoint note")
            runner = fake_runner()
            finalize = checkpoints.install_campaign_checkpoints(runner, folder, interval=1)
            for _ in range(6):
                runner.train()
            finalize()
            self.assertEqual(len(list(Path(folder).glob("update_*"))), 3)
            self.assertTrue(unrelated.exists())
            self.assertTrue((Path(folder) / "update_000006/training_state.pt").exists())

    def test_roundtrip_optimizer_weights_rotation_and_atomic_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = fake_runner()
            finalize = checkpoints.install_campaign_checkpoints(runner, folder)
            runner.train()
            finalize()
            receipt = json.loads((Path(folder) / "receipt.json").read_text())
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["steps"], 4)
            target = fake_runner()
            target.current_team_index = 0
            checkpoints.strict_restore(target, receipt["checkpoint"], optimizers=True)
            self.assertEqual(target.current_team_index, 1)
            self.assertEqual(target.current_team_train_steps, 1234)
            for a, b in zip(
                runner.actors["team_0"]["robot_0"].actor.parameters(),
                target.actors["team_0"]["robot_0"].actor.parameters(),
            ):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(target.actors["team_0"]["robot_0"].actor_optimizer.state)

    def test_restore_missing_weights_and_nonfinite_rollout_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                checkpoints.strict_restore(fake_runner(), folder)
        with self.assertRaises(FloatingPointError):
            checkpoints.assert_finite({"obs": torch.tensor([float("nan")])})


if __name__ == "__main__":
    unittest.main()
