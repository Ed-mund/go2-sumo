"""CPU-only tests: PPO decisions/gradients, credit assignment, contact events and league orchestration."""

import importlib.util
import json
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

import tactics_model as model
import tactics_campaign as campaign
from tactics_league import League
import tactics_train

torch.set_num_threads(2)
path = campaign.REPO / "source/isaaclab_tasks/isaaclab_tasks/direct/adversarial/sumo/tactics_events.py"
spec = importlib.util.spec_from_file_location("event_helpers", path)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def inputs(count=2):
    return (
        torch.randint(0, 256, (count, 12, 36, 64), dtype=torch.uint8),
        torch.randn(count, model.VECTOR),
        torch.zeros(count, dtype=torch.long),
        torch.zeros(count, dtype=torch.long),
    )


def event_inputs(count=1):
    return dict(
        up=torch.ones(count, 2),
        gap=torch.ones(count) * 0.7,
        height=torch.tensor([[0.5, 0.25]]).repeat(count, 1),
        leg_hit=torch.zeros(count, 2, 2, dtype=torch.bool),
        foot_moving=torch.ones(count, 2, 2, dtype=torch.bool),
        stance=torch.ones(count, 2, dtype=torch.long) * 3,
        body_contact=torch.zeros(count, 2, dtype=torch.bool),
        grounded=torch.ones(count, 2, dtype=torch.bool),
        flank=torch.zeros(count, 2, dtype=torch.bool),
        flank_side=torch.ones(count, 2),
        radius=torch.ones(count, 2),
        incoming=torch.zeros(count, 2, dtype=torch.bool),
        skills=torch.zeros(count, 2, dtype=torch.long),
    )


class ModelTests(unittest.TestCase):
    def test_pack_dimensions_and_pixel_quantization(self):
        raw = torch.rand(2, 45 + model.PIXELS + 386)
        pixels, vector = model.unpack(raw)
        self.assertEqual(pixels.shape, (2, 12, 36, 64))
        self.assertEqual(vector.shape, (2, model.VECTOR))
        self.assertEqual(pixels.dtype, torch.uint8)
        self.assertTrue(torch.equal(vector[:, :45], raw[:, :45]))

    def test_exact_behavior_likelihood_and_real_cnn_gradient(self):
        policy = model.TacticsPolicy()
        pixels, vector, previous, remaining = inputs(4)
        with torch.no_grad():
            actions, skills, decisions, logprob, value = policy.act(pixels, vector, previous, remaining)
        data = dict(
            pixels=pixels,
            vector=vector,
            previous=previous,
            remaining=remaining,
            actions=actions,
            skills=skills,
            decisions=decisions,
            logprob=logprob,
            advantage=torch.tensor([-1.0, -0.3, 0.3, 1.0]),
            **{"return": value + 1},
        )
        actual, _, _ = policy.evaluate(data)
        self.assertTrue(torch.allclose(logprob, actual))
        before = policy.vision[0].weight.detach().clone()
        result = model.ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=0.001), data, epochs=1, minibatch=2)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertFalse(torch.equal(before, policy.vision[0].weight))

    def test_manager_only_receives_decision_gradients(self):
        policy = model.TacticsPolicy()
        pixels, vector, previous, remaining = inputs()
        forced = torch.tensor([3, 4])
        actions, skills, decisions, logprob, _ = policy.act(pixels, vector, previous, remaining, forced)
        self.assertTrue(torch.equal(skills, forced))
        self.assertFalse(decisions.any())
        logprob.sum().backward()
        self.assertEqual(policy.manager.weight.grad.abs().sum().item(), 0)
        remaining[:] = 4
        previous[:] = 6
        output = policy.act(pixels, vector, previous, remaining)
        self.assertTrue(torch.all(output[1] == 6))
        self.assertFalse(output[2].any())

    def test_mixed_drill_and_free_decisions(self):
        policy = model.TacticsPolicy()
        output = policy.act(*inputs(), forced=torch.tensor([4, -1]))
        self.assertEqual(output[1][0], 4)
        self.assertTrue(torch.equal(output[2], torch.tensor([False, True])))

    def test_option_persistence_and_episode_reset(self):
        skill = torch.tensor([3, 4])
        remaining = torch.tensor([0, 5])
        skill, remaining = model.option_clock(skill, remaining, torch.tensor([False, True]))
        self.assertTrue(torch.equal(remaining, torch.tensor([9, 0])))
        self.assertTrue(torch.equal(skill, torch.tensor([3, 0])))

    def test_gae_does_not_leak_across_resets(self):
        reward = torch.tensor([[1.0], [100.0]])
        adv, returns = model.advantages(
            reward, torch.zeros_like(reward), torch.tensor([[True], [False]]), torch.tensor([2.0]), gamma=1, lam=1
        )
        self.assertEqual(adv[0].item(), 1.0)
        self.assertEqual(returns[1].item(), 102.0)

    def test_policy_checkpoint_strict_roundtrip(self):
        a, b = model.TacticsPolicy(), model.TacticsPolicy()
        b.load_state_dict(a.state_dict(), strict=True)
        data = inputs()
        self.assertTrue(torch.equal(a.act(*data, deterministic=True)[0], b.act(*data, deterministic=True)[0]))
        with self.assertRaises(RuntimeError):
            b.load_state_dict({"old_actor": torch.zeros(1)}, strict=True)


class EventTests(unittest.TestCase):
    def test_nearby_upside_down_robot_is_not_a_pin(self):
        tracker = helpers.TacticsEvents(1)
        data = event_inputs()
        data["up"][0, 1] = -1
        for _ in range(30):
            events, reward = tracker.update(**data)
        self.assertFalse(events.any())
        self.assertEqual(reward.sum(), 0)

    def test_pin_requires_contact_grounding_duration_and_no_repeated_bonus(self):
        tracker = helpers.TacticsEvents(1)
        data = event_inputs()
        data["up"][0, 1] = -1
        data["body_contact"][0, 0] = True
        data["grounded"][0, 1] = False
        for _ in range(30):
            tracker.update(**data)
        self.assertEqual(tracker.pin_hold.sum(), 0)
        data["grounded"][0, 1] = True
        for _ in range(24):
            events, _ = tracker.update(**data)
            self.assertFalse(events[..., 6].any())
        events, reward = tracker.update(**data)
        self.assertTrue(events[0, 0, 6])
        self.assertEqual(reward[0, 0], 2)
        self.assertEqual(tracker.update(**data)[1].sum(), 0)

    def test_leg_contact_alone_never_rewards_and_unrelated_fall_expires(self):
        tracker = helpers.TacticsEvents(1)
        data = event_inputs()
        data["leg_hit"][0, 0, 0] = True
        self.assertEqual(tracker.update(**data)[1].sum(), 0)
        data["leg_hit"].zero_()
        for _ in range(31):
            tracker.update(**data)
        data["up"][0, 1] = 0.4
        for _ in range(10):
            events, reward = tracker.update(**data)
        self.assertFalse(events[0, 0, 0])

    def test_sweep_contact_then_imbalance_while_attacker_stable(self):
        tracker = helpers.TacticsEvents(1)
        data = event_inputs()
        data["leg_hit"][0, 0, 0] = True
        tracker.update(**data)
        data["leg_hit"].zero_()
        data["up"][0, 1] = 0.4
        for _ in range(8):
            events, reward = tracker.update(**data)
        self.assertTrue(events[0, 0, 0])
        self.assertGreater(reward[0, 0], 0)

    def test_counter_does_not_reset_baseline_during_continuous_contact(self):
        tracker = helpers.TacticsEvents(1)
        data = event_inputs()
        data["incoming"][0, 0] = True
        tracker.update(**data)
        data["radius"][0, 1] = 1.3
        events, _ = tracker.update(**data)
        self.assertTrue(events[0, 0, 4])

    def test_drill_credit_and_partial_reset_are_isolated(self):
        tracker = helpers.TacticsEvents(2)
        data = event_inputs(2)
        data["flank"][:, 0] = True
        data["skills"][:, 0] = 1
        events, reward = tracker.update(**data, drills=torch.tensor([True, False]))
        self.assertEqual(reward[0, 0], 2)
        self.assertAlmostEqual(reward[1, 0].item(), 0.2, places=6)
        tracker.reset(torch.tensor([0]))
        self.assertFalse(tracker.awarded[0].any())
        self.assertTrue(tracker.awarded[1].any())

    def test_deliberate_fall_recovery_is_not_paid_in_combat(self):
        tracker = helpers.TacticsEvents(1)
        data = event_inputs()
        data["up"][0, 0] = 0.3
        tracker.update(**data)
        data["up"][0, 0] = 1.0
        for _ in range(10):
            events, reward = tracker.update(**data)
        self.assertTrue(events[0, 0, 7])
        self.assertEqual(reward[0, 0], 0)


class CampaignTests(unittest.TestCase):
    def test_runtime_fingerprint_works_without_requiring_harl(self):
        import sys

        result = campaign.runtime_fingerprint({"python": sys.executable})
        self.assertIn("torch", result["packages"])
        self.assertIsInstance(result["harl_adapter_hashes"], dict)

    def test_smoke_only_never_enters_production(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(campaign, "preflight"):
            run = Path(folder)
            campaign.write_json(run / "smoke/passed.json", {"passed": True})
            state = {"state": "starting", "stage_index": 0, "steps": 0}
            with patch.object(campaign, "invoke", side_effect=AssertionError("Production must not launch")):
                campaign.run_campaign({}, run, state, run / "status.json", smoke_only=True)
            self.assertEqual(state["state"], "smoke_passed")
            self.assertEqual(state["steps"], 0)

    def test_portable_interpreter_and_saved_environment_override(self):
        import sys

        self.assertEqual(campaign.config(campaign.HERE / "configs/go2_tactics_v3.json")["python"], sys.executable)
        with tempfile.TemporaryDirectory() as folder, patch.object(campaign, "fingerprints", return_value={}):
            with patch("sys.argv", ["campaign", "--prepare", "--envs", "4", "--run-dir", folder]):
                campaign.main()
            stored = json.loads((Path(folder) / "config.json").read_text())
            self.assertTrue(all(stage["envs"] == 4 for stage in stored["stages"]))
            with patch("sys.argv", ["campaign", "--smoke-only", "--run-dir", folder]), patch.object(
                campaign, "run_campaign"
            ) as run:
                campaign.main()
            self.assertEqual(run.call_args.args[0], stored)
            self.assertTrue(run.call_args.kwargs["smoke_only"])

    def test_training_receipt_must_be_complete_exact_and_present(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(campaign, "child"):
            root = Path(folder)
            checkpoint = root / "policy.pt"
            checkpoint.write_bytes(b"test checkpoint")
            campaign.write_json(root / "evaluation.json", {"episodes": 2})
            cmd = ["python", "trainer", "--steps", "128"]
            for complete, steps in ((False, 128), (True, 64), (True, 128)):
                campaign.write_json(
                    root / "receipt.json", {"complete": complete, "steps": steps, "checkpoint": str(checkpoint)}
                )
                if complete and steps == 128:
                    self.assertEqual(campaign.invoke(cmd, root, {}, {}, root / "status.json"), {"episodes": 2})
                else:
                    with self.assertRaises(RuntimeError):
                        campaign.invoke(cmd, root, {}, {}, root / "status.json")
            checkpoint.unlink()
            with self.assertRaises(RuntimeError):
                campaign.invoke(cmd, root, {}, {}, root / "status.json")

    def test_all_phases_and_no_gpu_on_prepare(self):
        cfg = campaign.config(campaign.HERE / "configs/go2_tactics_v3.json")
        self.assertEqual(tuple(p["name"] for p in cfg["stages"]), campaign.PHASES)
        self.assertGreater(cfg["stages"][4]["drill_fraction"], 0)
        with tempfile.TemporaryDirectory() as folder:
            with patch("sys.argv", ["campaign", "--prepare", "--run-dir", folder]), patch.object(
                campaign, "fingerprints", return_value={}
            ), patch.object(campaign, "run_campaign") as run:
                campaign.main()
                run.assert_not_called()
            self.assertEqual(json.loads((Path(folder) / "status.json").read_text())["state"], "prepared")

    def test_fake_diversity_and_missing_drill_results_fail(self):
        report = {
            "episodes": 32,
            "upright_fraction": 0.95,
            "wins": 20,
            "draws": 2,
            "skill_controls": {"push": 50, "circle_left": 30, "sweep_left": 20},
            "events": {},
        }
        self.assertTrue(campaign.quality_errors(report, "tactics"))
        report["events"] = {"sweep_left": 2, "pin": 1}
        self.assertFalse(campaign.quality_errors(report, "tactics"))
        self.assertTrue(campaign.quality_errors(report, "drills", skill="sweep_left"))
        report["upright_fraction"] = float("nan")
        self.assertTrue(campaign.quality_errors(report, "tactics"))

    def test_league_freezes_hashes_preserves_styles_and_excludes_current(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            league = League(root / "league.json")
            for i in range(30):
                checkpoint = root / f"policy{i}.pt"
                checkpoint.write_bytes(f"model{i}".encode())
                league.add(checkpoint, "pusher" if i == 0 else "league", 0)
            self.assertLessEqual(len(league.data["members"]), 24)
            chosen = league.sample(random.Random(3), 3, exclude=(str(root / "policy29.pt"),), require_style="pusher")
            self.assertEqual(chosen[0]["style"], "pusher")
            self.assertNotIn(str(root / "policy29.pt"), [m["checkpoint"] for m in chosen])
            league.record({chosen[0]["id"]: {"games": 10, "score": 2.0}})
            self.assertEqual(league.data["members"][0]["games"], 10)
            (root / "policy0.pt").write_bytes(b"corruption")
            with self.assertRaises(ValueError):
                league.sample(random.Random(3), 3, require_style="pusher")


class MockSimulatorIntegration(unittest.TestCase):
    def test_train_transfer_role_switch_resume_and_frozen_evaluation(self):
        """Exercise the real trainer/Adam/checkpoint code; explicitly NOT a physics test."""
        # Load Torch's lazy optimizer dependencies before patch.dict restores sys.modules.
        import torch._dynamo

        class Config:
            def __init__(self):
                self.scene = types.SimpleNamespace(num_envs=2)
                self.viewer = types.SimpleNamespace()

        class Env:
            def __init__(self, task, cfg):
                self.cfg = cfg
                self.num_envs = cfg.scene.num_envs
                self.device = torch.device("cpu")
                self.teams = 1 if "Locomotion" in task else 2
                self.robots = {f"robot_{i}": None for i in range(self.teams)}
                self.skills = torch.zeros(self.num_envs, self.teams, dtype=torch.long)
                self.drills = torch.zeros_like(self.skills)
                self.drill_active = torch.zeros(self.num_envs, dtype=torch.bool)
                self.drill_passive = self.drill_active.clone()
                self.unwrapped = self
                self.tick = 0

            def observations(self):
                row = torch.zeros(self.num_envs, 45 + model.PIXELS + 386)
                row[:, :3] = 0.1
                return {f"team_{i}": {f"robot_{i}": row.clone()} for i in range(self.teams)}

            def reset(self):
                self.tick = 0
                return self.observations(), {}

            def step(self, actions):
                self.tick += 1
                self.assertions(actions)
                done = torch.full((self.num_envs,), self.tick % 3 == 0, dtype=torch.bool)
                self.tactical_transition = {
                    "upright": torch.ones(self.num_envs, self.teams),
                    "outcome": torch.ones(self.num_envs, self.teams),
                }
                rewards = {f"team_{i}": torch.ones(self.num_envs) for i in range(self.teams)}
                terminated = {k: done.clone() for k in rewards}
                truncated = {k: torch.zeros_like(done) for k in rewards}
                return (
                    self.observations(),
                    rewards,
                    terminated,
                    truncated,
                    {"log": {"Metrics/goal_success_percentage": 1.0}},
                )

            def assertions(self, actions):
                assert all(value.shape == (self.num_envs, 12) for value in actions.values())

            def close(self):
                pass

        fake_tasks = types.ModuleType("isaaclab_tasks")
        fake_cfg = types.ModuleType("isaaclab_tasks.direct.adversarial.sumo.sumo_v3")
        for name in ("Go2TacticsLocomotionCfg", "Go2TacticsPushCfg", "Go2TacticsFightCfg"):
            setattr(fake_cfg, name, Config)
        fake_app = types.ModuleType("isaaclab.app")
        fake_app.AppLauncher = lambda **kw: types.SimpleNamespace(app=types.SimpleNamespace(close=lambda: None))
        fake_gym = types.ModuleType("gymnasium")
        fake_gym.make = lambda task, cfg, **kw: Env(task, cfg)
        modules = {
            "isaaclab": types.ModuleType("isaaclab"),
            "isaaclab.app": fake_app,
            "isaaclab_tasks": fake_tasks,
            "isaaclab_tasks.direct.adversarial.sumo.sumo_v3": fake_cfg,
            "gymnasium": fake_gym,
        }
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", modules), patch.object(
            tactics_train, "preflight"
        ):
            root = Path(folder)
            cfg = campaign.config(campaign.HERE / "configs/go2_tactics_v3.json")
            cfg.update(rollout_steps=4, minibatch=4)
            path = root / "config.json"
            path.write_text(json.dumps(cfg))

            def run(name, phase, source=None, role=0, evaluate=False, opponents=None):
                output = root / name
                argv = [
                    "train",
                    "--config",
                    str(path),
                    "--output",
                    str(output),
                    "--phase",
                    phase,
                    "--steps",
                    "8",
                    "--envs",
                    "2",
                    "--role",
                    str(role),
                ]
                if source:
                    argv += ["--source", str(source)]
                if evaluate:
                    argv += ["--evaluate"]
                if opponents:
                    argv += ["--opponents", str(opponents)]
                with patch("sys.argv", argv):
                    tactics_train.main()
                if evaluate:
                    return json.loads((output / "evaluation.json").read_text())
                return json.loads((output / "receipt.json").read_text())["checkpoint"]

            first = run("loco", "locomotion")
            second = run("push0", "push", first)
            third = run("push1", "push", second, role=1)
            before = torch.load(second, weights_only=False, map_location="cpu")
            after = torch.load(third, weights_only=False, map_location="cpu")
            self.assertTrue(
                all(torch.equal(value, after["models"][0][key]) for key, value in before["models"][0].items())
            )
            self.assertTrue(
                any(not torch.equal(value, after["models"][1][key]) for key, value in before["models"][1].items())
            )
            self.assertTrue(after["optimizers"][1]["state"])
            league = League(root / "league.json")
            league.add(second, "pusher", 0, 0)
            opponents = root / "opponents.json"
            opponents.write_text(json.dumps(league.sample(random.Random(1))))
            report = run("eval", "push", third, role=1, evaluate=True, opponents=opponents)
            self.assertEqual(report["steps"], 8)
            self.assertEqual(report["episodes"], 2)
            self.assertEqual(report["upright_fraction"], 1.0)
        torch.set_num_threads(2)


if __name__ == "__main__":
    unittest.main()
