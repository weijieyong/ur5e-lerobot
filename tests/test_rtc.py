"""Stock installed RTC scheduling/queue regressions on CPU; no model or hardware."""

import unittest
from types import SimpleNamespace

import torch

from lerobot.inference.contracts import (
    ActionChunk,
    ActionProvenance,
    ExecutionMode,
    ObservationSnapshot,
)
from lerobot.inference.execution import ChunkRuntime
from lerobot.inference.prediction import predict_chunk
from lerobot.policies.chunk import ChunkPolicySpec


class RTCQueueTests(unittest.TestCase):
    def test_native_prediction_pads_prefix_with_last_action_and_unwraps_output(self):
        calls = []
        actions = torch.ones(1, 4, 2)

        def predict(batch, **kwargs):
            calls.append(kwargs)
            return {"action": actions, "prediction": {}}

        policy = SimpleNamespace(predict_action_chunk=predict)
        kwargs = dict(
            spec=ChunkPolicySpec(4, 4),
            mode=ExecutionMode.RTC_GUIDED,
            canonical_action_dim=2,
            model_action_dim=2,
            device="cpu",
            rtc_horizon=4,
        )
        first = predict_chunk(policy, lambda value: value + 100, {}, **kwargs)
        self.assertIsNone(calls[-1]["prev_chunk_left_over"])
        self.assertEqual(calls[-1]["inference_delay"], 0)
        torch.testing.assert_close(first.model_actions, actions[0])
        torch.testing.assert_close(first.canonical_actions, actions[0] + 100)
        prefix = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        predict_chunk(
            policy,
            lambda value: value,
            {},
            model_continuation=prefix,
            inference_delay=1,
            **kwargs,
        )
        torch.testing.assert_close(
            calls[-1]["prev_chunk_left_over"],
            torch.tensor([[2.0, 3.0], [4.0, 5.0], [4.0, 5.0], [4.0, 5.0]]),
        )

    def test_native_runtime_cold_start_and_continuation_prefix(self):
        now = [100.0]
        runtime = ChunkRuntime(
            mode=ExecutionMode.RTC_GUIDED,
            action_interval=1 / 15,
            refill_seconds=10,
            max_observation_age_s=5,
            action_timeout_s=5,
            startup_timeout_s=5,
            clock=lambda: now[0],
        )
        runtime.activate()
        first = runtime.begin(ObservationSnapshot({}, now[0], "pick"))
        self.assertIsNotNone(first)
        self.assertEqual(first.delay, 0)
        self.assertIsNone(first.continuation.model_actions)
        actions = torch.arange(20, dtype=torch.float32).reshape(20, 1)
        provenance = ActionProvenance(now[0], "pick")
        now[0] += 0.3
        self.assertTrue(
            runtime.accept(
                first,
                ActionChunk(actions, actions + 100, provenance, 20),
                task_version=0,
            )
        )
        torch.testing.assert_close(runtime.queue.get_left_over(), actions)
        self.assertEqual(runtime.queue.qsize(), len(actions))
        torch.testing.assert_close(runtime.queue.get_processed_left_over(), actions + 100)
        torch.testing.assert_close(runtime.queue.get(), actions[0] + 100)
        second = runtime.begin(ObservationSnapshot({}, now[0], "pick"))
        self.assertIsNotNone(second)
        self.assertGreater(second.delay, 0)
        torch.testing.assert_close(second.continuation.model_actions, actions[1:])
        torch.testing.assert_close(
            second.continuation.canonical_actions, actions[1:] + 100
        )
        runtime.queue.get()
        now[0] += 0.3
        self.assertTrue(
            runtime.accept(
                second,
                ActionChunk(actions, actions + 200, provenance, 20),
                task_version=0,
            )
        )
        self.assertEqual(runtime.last_accept["trimmed_actions"], 1)
        self.assertEqual(runtime.queue.qsize(), len(actions) - 1)
        torch.testing.assert_close(runtime.queue.get_left_over(), actions[1:])
        torch.testing.assert_close(
            runtime.queue.get_processed_left_over(), actions[1:] + 200
        )
        torch.testing.assert_close(runtime.queue.get(), actions[1] + 200)

    def test_rtc_engine_enables_starvation_grace_with_ur5e_command_hold(self):
        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.inference.rtc import RTCInferenceEngine
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.rollout.context import ThreadSafeRobot
        from lerobot_robot_ur5e.config_ur5e import UR5eConfig
        from lerobot_robot_ur5e.ur5e import UR5e

        robot = UR5e(UR5eConfig(cameras={}, timing_log_dir=""))
        wrapper = ThreadSafeRobot(robot)
        policy = SimpleNamespace(
            config=SimpleNamespace(
                n_action_steps=50,
                chunk_size=50,
                output_features={"action": SimpleNamespace(shape=(7,))},
            ),
            chunk_inference_spec=lambda: ChunkPolicySpec(
                50, 50, modes=(ExecutionMode.RTC_GUIDED,)
            ),
            predict_action_chunk=lambda batch, **kwargs: None,
            supports_text_generation=lambda: False,
        )
        engine = RTCInferenceEngine(
            policy=policy,
            preprocessor=SimpleNamespace(steps=[]),
            postprocessor=SimpleNamespace(steps=[]),
            robot_wrapper=wrapper,
            rtc_config=RTCConfig(
                enabled=True,
                execution_horizon=10,
                prefix_attention_schedule=RTCAttentionSchedule.EXP,
            ),
            dataset_features={},
            task="pick",
            fps=15,
            device="cpu",
            action_starvation_grace_s=1.0,
        )
        self.assertTrue(wrapper.supports_hold)
        self.assertEqual(engine._runtime.starvation_grace, 1.0)


if __name__ == "__main__":
    unittest.main()
