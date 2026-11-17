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
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig


class RTCQueueTests(unittest.TestCase):
    def test_startup_keeps_first_action_despite_elapsed_inference_time(self):
        self.check_merge(consumed=0)

    def test_elapsed_estimate_cannot_skip_beyond_consumed_actions(self):
        self.check_merge(consumed=2)

    def check_merge(self, consumed):
        for snapshot_api in (False, True):
            with self.subTest(snapshot_api=snapshot_api, consumed=consumed):
                queue = ActionQueue(RTCConfig(enabled=True))
                original = torch.arange(10, dtype=torch.float32).reshape(10, 1)
                processed = original + 100
                if consumed:
                    queue.merge(original, processed, real_delay=0)
                snapshot = queue.snapshot()
                index = queue.get_action_index()
                for step in range(consumed):
                    torch.testing.assert_close(queue.get(), processed[step])
                kwargs = (
                    {"snapshot": snapshot}
                    if snapshot_api
                    else {"action_index_before_inference": index}
                )
                self.assertTrue(
                    queue.merge(original, processed, real_delay=5, **kwargs)
                )
                self.assertEqual(queue.qsize(), 10 - consumed)
                torch.testing.assert_close(queue.get_left_over(), original[consumed:])
                torch.testing.assert_close(
                    queue.get_processed_left_over(), processed[consumed:]
                )
                torch.testing.assert_close(queue.get(), processed[consumed])

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
        now[0] += 0.3  # Five elapsed action intervals, zero actions actually consumed.
        self.assertTrue(
            runtime.accept(
                first,
                ActionChunk(actions, actions + 100, provenance, 20),
                task_version=0,
            )
        )
        torch.testing.assert_close(runtime.queue.get_left_over(), actions)
        torch.testing.assert_close(runtime.queue.get(), actions[0] + 100)
        second = runtime.begin(ObservationSnapshot({}, now[0], "pick"))
        self.assertIsNotNone(second)
        self.assertGreater(second.delay, 0)
        torch.testing.assert_close(second.continuation.model_actions, actions[1:])
        torch.testing.assert_close(
            second.continuation.canonical_actions, actions[1:] + 100
        )
        runtime.queue.get()  # Only one action consumed while the successor is being inferred.
        now[0] += 0.3
        self.assertTrue(
            runtime.accept(
                second,
                ActionChunk(actions, actions + 200, provenance, 20),
                task_version=0,
            )
        )
        self.assertEqual(runtime.last_accept["trimmed_actions"], 1)
        torch.testing.assert_close(runtime.queue.get(), actions[1] + 200)


if __name__ == "__main__":
    unittest.main()
