"""Offline boundary tests for UR5e MuJoCo simulation and sim rollout glue.

No model weights, no GPU compute, no hardware access, no network access.
"""

from __future__ import annotations

import contextlib
import io
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lerobot.envs.configs import EnvConfig
from lerobot.inference import RTCInferenceConfig, SyncInferenceConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.robots.sim_robot import SimRobot, SimRobotConfig
from lerobot.rollout.configs import EpisodicStrategyConfig
from lerobot.rollout.strategies.episodic import EpisodicStrategy as NativeStrategy
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config

from lerobot_robot_ur5e import UR5eSimEnvConfig, UR5eSimTeacherConfig
from lerobot_robot_ur5e.ur5e import GRIPPER, JOINTS
from scripts import rollout as hw_rollout
from scripts import sim_rollout


class SimulationContractTests(unittest.TestCase):
    def test_plugin_registers_sim_env_and_teacher_configs(self):
        self.assertIn("ur5e_sim", EnvConfig.get_known_choices())
        self.assertIn("ur5e_sim_teacher", TeleoperatorConfig.get_known_choices())
        with self.assertRaisesRegex(ValueError, "fps=15"):
            UR5eSimEnvConfig(fps=30)

    def test_sim_robot_descriptor_observation_hygiene_and_stepping(self):
        cfg = UR5eSimEnvConfig(
            observation_width=160,
            observation_height=120,
            episode_length=30,
            env_id="unit_test_contract",
        )
        robot = SimRobot(SimRobotConfig(env=cfg))
        teacher = make_teleoperator_from_config(
            UR5eSimTeacherConfig(env_id="unit_test_contract")
        )
        try:
            self.assertEqual(robot.descriptor.fps, 15)
            self.assertEqual(robot.robot_type, "ur5e_sim")
            expected_names = (*JOINTS, GRIPPER)
            self.assertEqual(robot.descriptor.action_feature.names, expected_names)
            self.assertEqual(robot.descriptor.action_feature.shape, (7,))
            self.assertEqual(list(robot.action_features), list(expected_names))
            self.assertEqual(
                robot.cameras,
                {
                    "front": "observation.images.front",
                    "wrist": "observation.images.wrist",
                },
            )
            self.assertEqual(
                set(robot.observation_features),
                {*expected_names, "front", "wrist"},
            )

            teacher.connect()
            robot.connect()
            obs0 = robot.get_observation()
            # Strictly learned-policy keys only: no ground-truth cube/tray/contact/success fields.
            self.assertEqual(set(obs0), {*expected_names, "front", "wrist"})
            self.assertEqual(obs0["front"].shape, (120, 160, 3))
            self.assertEqual(obs0["wrist"].shape, (120, 160, 3))
            self.assertEqual(obs0["front"].dtype, np.uint8)
            self.assertEqual(obs0["wrist"].dtype, np.uint8)
            self.assertGreater(float(obs0["front"].std()), 5.0)
            self.assertGreater(float(obs0["wrist"].std()), 5.0)
            self.assertLess(obs0[GRIPPER], 0.1)

            # Drive arm + close gripper (> 0.5 threshold) for several steps; confirm motion + RGB change.
            for _ in range(10):
                action = teacher.get_action()
                action[GRIPPER] = 0.8  # >= 0.5 -> closed
                robot.send_action(action)
            obs1 = robot.get_observation()
            self.assertGreater(obs1[GRIPPER], 0.4)
            self.assertGreater(
                float(np.abs(obs1["front"].astype(float) - obs0["front"].astype(float)).mean()),
                0.1,
            )
            self.assertGreater(
                float(np.abs(obs1["wrist"].astype(float) - obs0["wrist"].astype(float)).mean()),
                0.1,
            )

            # Fixed-layout reset restores initial open state between episodes.
            robot.start_episode()
            obs_reset = robot.get_observation()
            self.assertLess(obs_reset[GRIPPER], 0.1)
            np.testing.assert_allclose(
                [obs_reset[j] for j in JOINTS],
                [obs0[j] for j in JOINTS],
                atol=1e-2,
            )
        finally:
            teacher.disconnect()
            robot.disconnect()

    def test_sim_robot_honors_short_and_long_episode_length_via_native_timelimit(self):
        # Short control (5 steps) and long (>240 default, 245 steps) both obey native TimeLimit.
        for length, env_id in ((5, "unit_test_short_limit"), (245, "unit_test_long_limit")):
            with self.subTest(episode_length=length):
                cfg = UR5eSimEnvConfig(
                    observation_width=64,
                    observation_height=48,
                    episode_length=length,
                    env_id=env_id,
                )
                robot = SimRobot(SimRobotConfig(env=cfg))
                try:
                    self.assertEqual(robot.descriptor.max_episode_steps, length)
                    robot.connect()
                    obs0 = robot.get_observation()
                    hold = {j: obs0[j] for j in JOINTS}
                    hold[GRIPPER] = 0.0
                    for step in range(1, length):
                        robot.send_action(hold)
                        self.assertFalse(
                            robot.episode_ended,
                            f"episode truncated early at step {step} (expected {length})",
                        )
                    robot.send_action(hold)
                    self.assertTrue(robot.episode_ended)
                    self.assertTrue(bool(robot.latest.truncated[0]))
                    self.assertEqual(int(robot.latest.elapsed_steps[0]), length)
                finally:
                    robot.disconnect()


class LaserTraceTests(unittest.TestCase):
    def test_laser_stays_between_pads_and_spot_traces_the_line(self):
        from lerobot_robot_ur5e.sim.laser_trace import (
            DEFAULT_LAYOUT,
            LINE_HALF_WIDTH_M,
            LaserTrace,
            trace_targets,
        )

        sim = LaserTrace()
        sim.reset()
        start, end = DEFAULT_LAYOUT.endpoints
        spot_positions = []
        for index, target in enumerate(trace_targets()):
            sim.step(target)
            spot = sim.data.site("laser_dot").xpos.copy()
            spot_positions.append(spot)
            with self.subTest(frame=index):
                self.assertTrue(sim.laser_gripped())
                self.assertGreater(spot[2], 0)
                self.assertLessEqual(abs(spot[0] - start[0]), LINE_HALF_WIDTH_M)

        spot_xy = np.array(spot_positions)[:, :2]
        np.testing.assert_allclose(spot_xy[0], start, atol=0.008)
        np.testing.assert_allclose(spot_xy[-1], end, atol=0.008)
        step_distance = np.linalg.norm(np.diff(spot_xy, axis=0), axis=1)
        self.assertLessEqual(float(step_distance.max()), 0.01)

    def test_varied_layout_is_framed_and_a_cropped_view_is_rejected(self):
        import mujoco

        from lerobot_robot_ur5e.sim.laser_trace import CAMERAS, LaserTrace, sample_layouts
        from scripts.sim_laser_record import HEIGHT, WIDTH, TraceVisibility

        layouts = list(sample_layouts(10, seed=42))
        for attribute in ("center_xy", "length_m", "angle_deg"):
            self.assertEqual(len({getattr(layout, attribute) for layout in layouts}), 10)
        sim = LaserTrace(layouts[0], width=WIDTH, height=HEIGHT)
        sim.reset()
        visibility = TraceVisibility(sim)
        with mujoco.Renderer(sim.model, height=HEIGHT, width=WIDTH) as renderer:
            for camera in CAMERAS:
                image, margin = visibility.frame(renderer, camera)
                self.assertEqual(image.shape, (HEIGHT, WIDTH, 3))
                self.assertGreater(margin, 2)
            sim.data.camera("wrist").xpos[1] += 2
            with self.assertRaisesRegex(ValueError, "wrist crops"):
                visibility.frame(renderer, "wrist")


class SimRolloutGlueTests(unittest.TestCase):
    def _base_cfg(self):
        return SimpleNamespace(
            robot=SimpleNamespace(type="sim", env=SimpleNamespace(type="ur5e_sim")),
            teleop=None,
            interactive=False,
            return_to_initial_position=True,
            strategy=EpisodicStrategyConfig(reset_to_initial_position=True),
            inference=RTCInferenceConfig(rtc=RTCConfig(enabled=True, execution_horizon=10)),
            dataset=SimpleNamespace(push_to_hub=False),
            fps=15,
        )

    def test_validation_rejects_hardware_teleop_hub_push_and_silent_sync_fallback(self):
        cfg = self._base_cfg()
        with patch.object(sim_rollout.upstream.rollout, "__wrapped__") as native:
            cfg.robot = SimpleNamespace(type="ur5e")
            with self.assertRaisesRegex(ValueError, "robot.type=sim"):
                sim_rollout.run_sim_rollout(cfg)

            cfg = self._base_cfg()
            cfg.teleop = object()
            with self.assertRaisesRegex(ValueError, "never accepts --teleop"):
                sim_rollout.run_sim_rollout(cfg)

            cfg = self._base_cfg()
            cfg.dataset.push_to_hub = True
            with self.assertRaisesRegex(ValueError, "push_to_hub"):
                sim_rollout.run_sim_rollout(cfg)

            cfg = self._base_cfg()
            cfg.inference = SyncInferenceConfig()
            with self.assertRaisesRegex(ValueError, "inference.type=rtc"):
                sim_rollout.run_sim_rollout(cfg)

            cfg = self._base_cfg()
            cfg.inference = RTCInferenceConfig(rtc=RTCConfig(enabled=False))
            with self.assertRaisesRegex(ValueError, "silently fall back to sync"):
                sim_rollout.run_sim_rollout(cfg)

            native.assert_not_called()

    def test_run_sim_rollout_installs_and_restores_strict_loader_and_anchor_hook(self):
        cfg = self._base_cfg()
        orig_factory = sim_rollout.upstream.create_strategy
        orig_loader = sim_rollout.rollout_context._load_pretrained_policy
        orig_bind = sim_rollout.rollout_context.bind_relative_anchor
        observed = {}

        def fake_native(passed_cfg):
            observed["reset_to_init"] = passed_cfg.strategy.reset_to_initial_position
            observed["return_to_init"] = passed_cfg.return_to_initial_position
            observed["bind"] = sim_rollout.rollout_context.bind_relative_anchor
            observed["strategy"] = sim_rollout.upstream.create_strategy(
                passed_cfg.strategy
            )
            raise RuntimeError("stop before model load")

        with (
            patch.object(
                sim_rollout.upstream.rollout, "__wrapped__", side_effect=fake_native
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop before model load"):
                sim_rollout.run_sim_rollout(cfg)

        self.assertFalse(observed["reset_to_init"])
        self.assertFalse(observed["return_to_init"])
        self.assertIs(observed["bind"], hw_rollout.bind_relative_anchor)
        self.assertIsInstance(observed["strategy"], sim_rollout.SimEpisodicStrategy)
        self.assertIs(sim_rollout.upstream.create_strategy, orig_factory)
        self.assertIs(sim_rollout.rollout_context._load_pretrained_policy, orig_loader)
        self.assertIs(sim_rollout.rollout_context.bind_relative_anchor, orig_bind)

    def test_sim_episodic_strategy_resets_via_sim_robot_and_tracks_all_attempts(self):
        starts = []

        class FakeSimRobot(SimRobot):
            def __init__(self):
                self.latest = None

            def start_episode(self, seed=None):
                starts.append(len(starts) + 1)
                self.latest = SimpleNamespace(
                    terminated=np.array([False]),
                    truncated=np.array([False]),
                    is_success=np.array([False]),
                    elapsed_steps=np.array([0]),
                )

        sim_robot = FakeSimRobot()
        dataset = SimpleNamespace(num_episodes=0, frames=0)

        def save():
            if not dataset.frames:
                raise ValueError("empty")
            dataset.num_episodes += 1
            dataset.frames = 0

        dataset.save_episode = save
        dataset.clear_episode_buffer = lambda: setattr(dataset, "frames", 0)

        engine = SimpleNamespace(
            reset=lambda: None,
            resume=lambda: None,
            pause=lambda: None,
            stop=lambda: None,
        )
        strategy = sim_rollout.SimEpisodicStrategy(
            EpisodicStrategyConfig(reset_to_initial_position=False)
        )
        strategy._engine = engine
        strategy._interpolator = SimpleNamespace(reset=lambda: None, multiplier=1)
        strategy._events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
        }

        cfg = SimpleNamespace(
            dataset=SimpleNamespace(
                num_episodes=2,
                episode_time_s=5,
                reset_time_s=0,
                single_task="pick",
            ),
            fps=15,
            play_sounds=False,
            display_data=False,
            display_ip=None,
            display_port=None,
            display_compressed_images=False,
            display_mode="rerun",
        )
        ctx = SimpleNamespace(
            runtime=SimpleNamespace(cfg=cfg, shutdown_event=Event()),
            hardware=SimpleNamespace(
                robot_wrapper=SimpleNamespace(inner=sim_robot),
                teleop=None,
            ),
            processors=None,
            data=SimpleNamespace(dataset=dataset, dataset_features={}),
        )

        outcomes = [
            # Attempt 1: discarded (rerecord_episode)
            (False, True, False, 12, True),
            # Attempt 2: saved, predicate failure (truncated)
            (False, True, False, 20, False),
            # Attempt 3: saved, predicate success (terminated)
            (True, False, True, 18, False),
        ]

        def fake_native_policy_loop(self_strategy, **kwargs):
            term, trunc, succ, steps, discard = outcomes.pop(0)
            sim_robot.latest = SimpleNamespace(
                terminated=np.array([term]),
                truncated=np.array([trunc]),
                is_success=np.array([succ]),
                elapsed_steps=np.array([steps]),
            )
            dataset.frames = steps
            if discard:
                kwargs["events"]["rerecord_episode"] = True

        with (
            patch.object(NativeStrategy, "_policy_loop", new=fake_native_policy_loop),
            patch.dict(
                NativeStrategy.run.__globals__,
                VideoEncodingManager=lambda _: contextlib.nullcontext(),
            ),
        ):
            strategy.run(ctx)

        # Every started attempt (including the discarded retry) triggered a fresh SimRobot reset and was logged.
        self.assertEqual(starts, [1, 2, 3])
        self.assertEqual(dataset.num_episodes, 2)
        self.assertEqual(len(strategy.attempts), 3)
        self.assertTrue(strategy.attempts[0]["discarded"])
        self.assertFalse(strategy.attempts[1]["is_success"])
        self.assertTrue(strategy.attempts[2]["is_success"])

    def test_cli_help_and_missing_local_checkpoint_fail_closed(self):
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        help_res = subprocess.run(
            [sys.executable, "-m", "scripts.sim_rollout", "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(help_res.returncode, 0, help_res.stderr)
        self.assertIn("--robot.type", help_res.stdout)
        self.assertIn("--inference.type", help_res.stdout)

        missing_res = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.sim_rollout",
                "--robot.type=sim",
                "--robot.env.type=ur5e_sim",
                "--policy.path=outputs/missing_checkpoint",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertNotEqual(missing_res.returncode, 0)
        self.assertIn("checkpoint directory is missing", missing_res.stderr)


if __name__ == "__main__":
    unittest.main()
