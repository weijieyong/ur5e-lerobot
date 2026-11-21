"""Manual rollout transitions through the real pinned episodic strategy; no hardware."""

import contextlib
import io
from pathlib import Path
import tempfile
from threading import Event, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from lerobot.rollout.configs import EpisodicStrategyConfig
from lerobot.rollout.strategies.episodic import EpisodicStrategy as NativeStrategy
from scripts.rollout import (
    ManualEpisodicStrategy,
    bind_relative_anchor,
    joint_vector,
    run_rollout,
    start_tolerance,
)
from scripts.record import RecordingFeedback
from scripts.rollout_input import RolloutInput
from lerobot_robot_ur5e.config_ur5e import SpaceMouseConfig
from lerobot_robot_ur5e.spacemouse import SpaceMouse
from lerobot_robot_ur5e.ur5e import CONNECTED, JOINTS, GRIPPER


class FakeEngine:
    active = False

    def __init__(self):
        self.calls = []
        self.task = "pick and place"
        self.starts = 0
        self.latency = SimpleNamespace(configure=lambda _: None)

    def stop(self):
        self.active = False

    def start(self):
        self.starts += 1
        if self.starts > 1:
            raise AssertionError("0.7 engines cannot restart within a session")

    def pause(self):
        self.active = False

    def resume(self):
        self.calls.append(("resume", self.task))
        self.active = True

    def set_task(self, task):
        self.calls.append(("task", task))
        self.task = task

    def reset(self):
        self.calls.append(("reset", self.task))


class RolloutTests(unittest.TestCase):
    def test_preprocessor_copy_shares_the_live_policy_queue(self):
        # RTC deep-copies the preprocessor; that copy must not duplicate the policy on the GPU.
        from copy import deepcopy
        from lerobot.processor import RelativeActionsProcessorStep

        class Policy:
            queued, copies = 3, 0

            def count_queued_actions(self):
                return self.queued

            def __deepcopy__(self, memo):
                Policy.copies += 1
                return Policy()

        policy, step = Policy(), RelativeActionsProcessorStep(enabled=True)
        pipeline = SimpleNamespace(steps=[step])
        self.assertIs(bind_relative_anchor(policy, pipeline), step)
        copied = deepcopy(pipeline).steps[0]
        self.assertEqual(Policy.copies, 0)
        policy.queued = 0
        self.assertEqual(copied._count_queued_actions(), 0)

    def test_outside_home_requires_fresh_start_after_explicit_home(self):
        attempts, homes = [], []
        q = np.ones(6)

        def ready(strategy, **kwargs):
            attempts.append(True)
            if len(attempts) == 2:
                strategy.control.key("h")

        def policy(strategy, **kwargs):
            dataset.frames = 1

        with self.session(ready, policy, episodes=1) as (strategy, ctx, dataset):
            ctx.hardware.robot_wrapper.inner.measured_joints = lambda: q

            def home():
                homes.append(True)
                q[:] = 0

            ctx.hardware.robot_wrapper.inner.go_home = home
            strategy.run(ctx)
        self.assertEqual(len(homes), 1)
        self.assertGreaterEqual(len(attempts), 3)

    def test_invalid_measurement_and_motion_fail_without_resume(self):
        for kind in ("measurement", "home"):
            with self.subTest(kind=kind):

                def ready(strategy, **kwargs):
                    if kind == "home":
                        strategy.control.key("h")

                with self.session(ready, lambda *a, **k: self.fail("policy ran")) as (
                    strategy,
                    ctx,
                    _,
                ):
                    if kind == "measurement":
                        ctx.hardware.robot_wrapper.inner.measured_joints = lambda: (
                            [float("nan")] * 6
                        )
                    else:

                        def broken():
                            raise RuntimeError("motion failed")

                        ctx.hardware.robot_wrapper.inner.go_home = broken
                    with self.assertRaises((ValueError, RuntimeError)):
                        strategy.run(ctx)
                    self.assertFalse(strategy._engine.active)

    def test_task_persists_across_saved_and_discarded_trials(self):
        tasks, ready_count = [], []

        def ready(strategy, **kwargs):
            ready_count.append(True)
            if len(ready_count) == 2:
                for key in ["t", *"new task", "enter"]:
                    strategy.control.key("space" if key == " " else key)

        def policy(strategy, **kwargs):
            tasks.append(strategy._engine.task)
            # Busy editing/home is ignored; save/discard keys remain effective.
            strategy.control.key("t")
            strategy.control.key("h")
            self.assertEqual(strategy.control.phase, "RUNNING")
            dataset.frames = 1
            if len(tasks) == 2:
                strategy.control.key("r")

        with self.session(ready, policy, episodes=3) as (strategy, ctx, dataset):
            engine = strategy._engine
            strategy.run(ctx)
            self.assertEqual(
                engine.calls,
                [
                    entry
                    for task in tasks
                    for entry in [("task", task), ("reset", task), ("resume", task)]
                ],
            )
        self.assertEqual(tasks, ["pick and place", "new task", "new task", "new task"])
        self.assertEqual(dataset.num_episodes, 8)

    def test_actual_native_policy_loop_records_dispatched_task(self):
        labels = []

        def ready(strategy, **kwargs):
            pass

        with self.session(ready, NativeStrategy._policy_loop, episodes=1) as (
            strategy,
            ctx,
            dataset,
        ):
            strategy._interpolator.needs_new_action = lambda: True
            strategy._interpolator.emitted_policy_action = True
            strategy._engine.dispatched_task = "native action provenance"
            dataset.add_frame = lambda frame: (
                labels.append(frame["task"]),
                setattr(dataset, "frames", 1),
            )
            ctx.processors = object()
            ctx.runtime.cfg.use_torch_compile = False
            with (
                patch.object(
                    strategy, "_process_observation_and_notify", return_value={}
                ),
                patch.object(strategy, "_handle_warmup", return_value=False),
                patch.object(strategy, "_log_telemetry"),
                patch.dict(
                    NativeStrategy._policy_loop.__globals__,
                    send_next_action=lambda *a: object(),
                    transition_to_robot_action=lambda _: {},
                    build_dataset_frame=lambda *a, **k: {},
                    episode_ended=lambda _: True,
                ),
            ):
                strategy.run(ctx)
        self.assertEqual(labels, ["native action provenance"])

    def test_start_rechecks_stop_discard_shutdown_and_reader_failure(self):
        for cancel in ("n", "r", "q", "shutdown", "reader"):
            with self.subTest(cancel=cancel):

                def ready(strategy, **kwargs):
                    pass

                with self.session(
                    ready, lambda *a, **k: self.fail("policy resumed")
                ) as (strategy, ctx, _):

                    def measured():
                        if cancel == "shutdown":
                            ctx.runtime.shutdown_event.set()
                        elif cancel == "reader":
                            strategy.control.fail(EOFError("lost terminal"))
                        else:
                            strategy.control.key(cancel)
                        return np.zeros(6)

                    ctx.hardware.robot_wrapper.inner.measured_joints = measured
                    # Refused START returns READY, then exit on the next poll.
                    observe = ctx.hardware.robot_wrapper.get_observation
                    polls = []

                    def poll():
                        polls.append(True)
                        if len(polls) > 1:
                            strategy.control.key("q")
                        return observe()

                    ctx.hardware.robot_wrapper.get_observation = poll
                    if cancel == "reader":
                        with self.assertRaisesRegex(
                            RuntimeError, "terminal input failed"
                        ):
                            strategy.run(ctx)
                    else:
                        strategy.run(ctx)
                    self.assertFalse(strategy._engine.active)

    def test_home_is_queued_on_input_thread_and_runs_on_main(self):
        import threading

        main = get_ident()
        homes = []

        def ready(strategy, **kwargs):
            if not homes:
                worker = threading.Thread(target=lambda: strategy.control.key("h"))
                worker.start()
                worker.join()
                self.assertFalse(homes)
            else:
                strategy.control.key("q")

        with self.session(ready, lambda *a, **k: self.fail("auto started")) as (
            strategy,
            ctx,
            _,
        ):
            ctx.hardware.robot_wrapper.inner.go_home = lambda: homes.append(get_ident())
            strategy.run(ctx)
        self.assertEqual(homes, [main])

    def test_nominal_tolerance_direct_coordinates_and_validation(self):
        import os

        for invalid in (None, [], [0] * 5, [float("nan")] * 6, [[0]] * 6):
            with self.assertRaisesRegex(ValueError, "six finite"):
                joint_vector(invalid, "home")
        for invalid in ("0", "-1", "nan", "inf", "bad"):
            with patch.dict(os.environ, ROLLOUT_START_TOLERANCE_DEG=invalid):
                with self.assertRaises(ValueError):
                    start_tolerance()
        for degrees, accepted in ((5.0, True), (5.001, False), (360.0, False)):
            calls = []

            def ready(strategy, **kwargs):
                calls.append(True)
                if len(calls) > 1:
                    strategy.control.key("q")

            def policy(strategy, **kwargs):
                dataset.frames = 1

            with (
                self.session(ready, policy, episodes=1) as (strategy, ctx, dataset),
                patch.dict(os.environ, ROLLOUT_START_TOLERANCE_DEG="5"),
            ):
                ctx.hardware.robot_wrapper.inner.measured_joints = lambda: np.deg2rad(
                    [degrees] * 6
                )
                strategy.run(ctx)
                self.assertEqual(dataset.num_episodes, 6 if accepted else 5)

    def test_listener_factory_restored_and_one_engine_setup_on_success_or_failure(self):
        import sys
        from unittest.mock import Mock

        module = sys.modules[NativeStrategy.__module__]
        original = module.init_keyboard_listener
        for failure in (False, True):
            cfg = SimpleNamespace(
                dataset=SimpleNamespace(single_task="configured"), duration=0
            )
            ctx = SimpleNamespace(runtime=SimpleNamespace(cfg=cfg))
            strategy = ManualEpisodicStrategy(
                EpisodicStrategyConfig(), SimpleNamespace(show=lambda _: None)
            )
            listener = Mock()
            if failure:
                listener.start.side_effect = RuntimeError("listener setup failed")
            engine = FakeEngine()

            def init(ctx):
                strategy._engine = engine

            with (
                patch.object(strategy, "_init_engine", side_effect=init) as initialize,
                patch("scripts.rollout.RolloutTerminal", return_value=listener),
            ):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "listener setup failed"):
                        strategy.setup(ctx)
                    listener.stop.assert_called_once()
                else:
                    strategy.setup(ctx)
                    self.assertIs(strategy._listener, listener)
                    self.assertIs(strategy._events, strategy.control.events)
                initialize.assert_called_once_with(ctx)
                listener.start.assert_called_once()
                self.assertIs(module.init_keyboard_listener, original)

    def test_wrapper_rejects_teleop_interactive_missing_home_and_tty_before_context(
        self,
    ):
        from scripts import rollout as wrapper

        cfg = SimpleNamespace(
            teleop=None,
            interactive=False,
            strategy=EpisodicStrategyConfig(),
            robot=SimpleNamespace(home_joints_deg=[0.0] * 6),
            inference=SimpleNamespace(type="rtc"),
            fps=15,
        )
        with (
            patch.object(wrapper.upstream.rollout, "__wrapped__") as native,
            patch.object(
                wrapper, "preflight", side_effect=RuntimeError("no terminal")
            ) as tty,
        ):
            cfg.teleop = object()
            with self.assertRaisesRegex(ValueError, "does not accept --teleop"):
                run_rollout(cfg)
            cfg.teleop, cfg.interactive = None, True
            with self.assertRaisesRegex(ValueError, "interactive=false"):
                run_rollout(cfg)
            cfg.interactive = False
            cfg.robot.home_joints_deg = None
            with self.assertRaisesRegex(ValueError, "six finite"):
                run_rollout(cfg)
            cfg.robot.home_joints_deg = [0.0] * 6
            with self.assertRaisesRegex(RuntimeError, "no terminal"):
                run_rollout(cfg)
            native.assert_not_called()
            tty.assert_called_once()

    def test_wrapper_disables_native_auto_home_and_restores_factories_on_error(self):
        from scripts import rollout as wrapper

        cfg = SimpleNamespace(
            teleop=None,
            interactive=False,
            return_to_initial_position=True,
            strategy=EpisodicStrategyConfig(reset_to_initial_position=True),
            robot=SimpleNamespace(home_joints_deg=[0.0] * 6),
            inference=SimpleNamespace(type="rtc"),
            fps=15,
        )
        original = wrapper.upstream.create_strategy

        def native(config):
            self.assertFalse(config.strategy.reset_to_initial_position)
            self.assertFalse(config.return_to_initial_position)
            self.assertIsNone(config.teleop)
            raise RuntimeError("primary failure")

        with (
            patch.object(wrapper.upstream.rollout, "__wrapped__", side_effect=native),
            patch.object(wrapper, "preflight"),
            patch.object(wrapper, "RecordingFeedback"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "primary failure"):
                run_rollout(cfg)
        self.assertIs(wrapper.upstream.create_strategy, original)

    def test_teardown_preserves_primary_error(self):
        strategy = ManualEpisodicStrategy(EpisodicStrategyConfig(), None)
        with patch.object(
            NativeStrategy, "teardown", side_effect=RuntimeError("cleanup failed")
        ):
            with self.assertRaisesRegex(ValueError, "primary") as raised:
                try:
                    raise ValueError("primary")
                finally:
                    strategy.teardown(object())
            self.assertIn("cleanup failed", raised.exception.__notes__[0])
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                strategy.teardown(object())

    def test_requests_during_reset_are_not_replayed_and_do_not_resume(self):
        from threading import Thread

        for request in ("n", "r", "q", "shutdown", "reader"):
            polls = []

            def ready(strategy, **kwargs):
                polls.append(True)
                if len(polls) > 1:
                    strategy.control.key("q")

            with self.session(ready, lambda *a, **k: self.fail("resumed")) as (
                strategy,
                ctx,
                _,
            ):

                def reset():
                    def callback():
                        strategy.control.key("t")
                        strategy.control.key("h")
                        if request == "reader":
                            strategy.control.fail(EOFError("lost input during reset"))
                        elif request == "shutdown":
                            ctx.runtime.shutdown_event.set()
                        else:
                            strategy.control.key(request)

                    worker = Thread(target=callback)
                    worker.start()
                    worker.join(0.5)
                    self.assertFalse(worker.is_alive())

                strategy._engine.reset = reset
                if request == "reader":
                    with self.assertRaisesRegex(
                        RuntimeError, "lost input during reset"
                    ):
                        strategy.run(ctx)
                else:
                    strategy.run(ctx)
                self.assertFalse(strategy._engine.active)
                self.assertEqual(strategy.control.current, "pick and place")
                self.assertFalse(strategy.control.home)

    def test_stop_during_delayed_resume_reaches_native_episode_outcome(self):
        from threading import Thread

        for key, discard in (
            ("n", False),
            ("right", False),
            ("r", True),
            ("left", True),
        ):
            with self.subTest(key=key):
                resume_entered, callback_done = Event(), Event()
                observed = []
                callback_errors = []

                def policy(strategy, **kwargs):
                    events = kwargs["events"]
                    observed.append((events["exit_early"], events["rerecord_episode"]))
                    # Supply a buffered frame so the fake dataset supports either
                    # native save or discard; the native run loop owns that decision.
                    dataset.frames = 1
                    if events["exit_early"]:
                        events["exit_early"] = False

                with self.session(lambda *a, **k: None, policy, episodes=1) as (
                    strategy,
                    ctx,
                    dataset,
                ):
                    original_resume = strategy._engine.resume

                    def delayed_resume():
                        if not resume_entered.is_set():
                            resume_entered.set()
                            if not callback_done.wait(1):
                                raise AssertionError(
                                    "input callback did not finish during resume"
                                )
                        original_resume()

                    def callback():
                        try:
                            if not resume_entered.wait(1):
                                raise AssertionError("resume was not reached")
                            if (
                                strategy.control.phase != "RUNNING"
                                or not strategy.control.lock.locked()
                            ):
                                raise AssertionError(
                                    "probe missed RUNNING admission-lock window"
                                )
                            strategy.control.key(key)
                            self.assertFalse(strategy.control.events["stop_recording"])
                        except BaseException as error:
                            callback_errors.append(error)
                        finally:
                            callback_done.set()

                    strategy._engine.resume = delayed_resume
                    worker = Thread(target=callback, daemon=True)
                    worker.start()
                    try:
                        strategy.run(ctx)
                    finally:
                        # Always unblock resume and bound thread cleanup, including assertions.
                        resume_entered.set()
                        callback_done.set()
                        worker.join(2)
                    self.assertFalse(worker.is_alive(), "input probe thread leaked")
                    self.assertEqual(callback_errors, [])
                    self.assertEqual(
                        observed[0], (True, discard), f"{key} lost during resume"
                    )
                    self.assertEqual(len(observed), 2 if discard else 1)
                    self.assertEqual(dataset.num_episodes, 6)

    def test_listener_failure_before_run_is_not_silently_treated_as_exit(self):
        with self.session(lambda *a, **k: None, lambda *a, **k: self.fail("ran")) as (
            strategy,
            ctx,
            _,
        ):
            strategy.control.fail(EOFError("terminal lost before run"))
            with self.assertRaisesRegex(RuntimeError, "terminal lost before run"):
                strategy.run(ctx)

    def test_terminal_cleanup_error_still_runs_native_teardown(self):
        from unittest.mock import Mock

        strategy = ManualEpisodicStrategy(EpisodicStrategyConfig(), None)
        strategy._listener = Mock()
        strategy._listener.stop.side_effect = RuntimeError("restore failed")
        with patch.object(NativeStrategy, "teardown") as native:
            with self.assertRaisesRegex(RuntimeError, "restore failed"):
                strategy.teardown(object())
            native.assert_called_once()
            self.assertIsNone(strategy._listener)

    def test_home_success_alone_never_accepts_start(self):
        homes, polls = [], []

        def ready(strategy, **kwargs):
            polls.append(True)
            if not homes:
                strategy.control.key("h")
            elif len(polls) > 3:
                strategy.control.key("q")

        with self.session(
            ready, lambda *a, **k: self.fail("home call bypassed measurement")
        ) as (strategy, ctx, _):
            ctx.hardware.robot_wrapper.inner.go_home = lambda: homes.append(True)
            ctx.hardware.robot_wrapper.inner.measured_joints = lambda: np.ones(6)
            strategy.run(ctx)
        self.assertEqual(homes, [True])
        self.assertGreater(len(polls), 3)

    def test_native_engine_initialization_once_for_multiple_trials(self):
        engines = []

        def policy(strategy, **kwargs):
            engines.append(strategy._engine)
            dataset.frames = 1

        with self.session(
            lambda *a, **k: None, policy, episodes=3, initialize=True
        ) as (strategy, ctx, dataset):
            engine = ctx.policy.inference
            strategy.run(ctx)
            self.assertEqual(engine.starts, 1)
            self.assertTrue(all(item is engine for item in engines))
            self.assertEqual(len(engines), 3)
            self.assertFalse(engine.active)

    def test_handover_keeps_close_command_when_object_stops_fingers_partway(self):
        with tempfile.TemporaryDirectory() as directory:
            mouse = SpaceMouse(
                SpaceMouseConfig(axes="full", calibration_dir=Path(directory))
            )
            mouse.hid = SimpleNamespace(get_state=lambda: (np.zeros(6), 0))
            robot = SimpleNamespace(
                measured_joints=lambda: np.zeros(6),
                rtde_r=SimpleNamespace(getActualTCPPose=lambda: [0.0] * 6),
                gripper=SimpleNamespace(position=lambda: 80 / 255),
                _gripper_closed=True,
            )
            with patch.dict(CONNECTED, {mouse.config.robot_ip: robot}, clear=True):
                mouse.reset_reference()
                action = mouse.get_action()
            self.assertEqual(action[GRIPPER], 1.0)

    def test_reset_reference_holds_measured_gripper_and_drops_queued_home(self):
        with tempfile.TemporaryDirectory() as directory:
            mouse = SpaceMouse(
                SpaceMouseConfig(axes="full", calibration_dir=Path(directory))
            )
            mouse.hid = SimpleNamespace(get_state=lambda: (np.zeros(6), 0))
            mouse._gripper_cmd = (
                0.0  # Prior READY phase was open; policy has since closed it.
            )
            mouse._home_requested.set()
            q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            robot = SimpleNamespace(
                measured_joints=lambda: q,
                rtde_r=SimpleNamespace(getActualTCPPose=lambda: [0.0] * 6),
                gripper=SimpleNamespace(position=lambda: 1.0),
                go_home=lambda: self.fail("Queued policy-phase home was executed"),
            )
            mouse.reset_reference()
            with patch.dict(CONNECTED, {mouse.config.robot_ip: robot}, clear=True):
                action = mouse.get_action()
            self.assertEqual(action[GRIPPER], 1.0)
            np.testing.assert_array_equal([action[k] for k in JOINTS], q)

    @contextlib.contextmanager
    def session(self, ready_loop, policy_loop, episodes=2, initialize=False):
        dataset = SimpleNamespace(num_episodes=5, frames=0)

        def save():
            if not dataset.frames:
                raise ValueError("Empty episode")
            dataset.num_episodes += 1
            dataset.frames = 0

        dataset.save_episode = save
        dataset.clear_episode_buffer = lambda: setattr(dataset, "frames", 0)
        cfg = SimpleNamespace(
            dataset=SimpleNamespace(
                num_episodes=episodes,
                episode_time_s=5,
                reset_time_s=60,
                single_task="pick and place",
            ),
            robot=SimpleNamespace(home_joints_deg=[0.0] * 6),
            fps=15,
            task="pick and place",
            play_sounds=False,
            display_data=False,
            interactive=False,
            duration=0,
            interpolation_multiplier=1,
            status_interval_s=0,
            device="cpu",
            display_ip=None,
            display_port=None,
            display_compressed_images=False,
            display_mode="rerun",
        )
        ctx = SimpleNamespace(
            runtime=SimpleNamespace(cfg=cfg, shutdown_event=Event()),
            hardware=SimpleNamespace(robot_wrapper=object(), teleop=object()),
            data=SimpleNamespace(dataset=dataset, dataset_features={}),
        )
        strategy = ManualEpisodicStrategy(
            EpisodicStrategyConfig(
                smooth_handover=False, reset_to_initial_position=False
            ),
            RecordingFeedback(cfg),
        )
        strategy.control = RolloutInput(cfg.dataset.single_task, strategy.feedback.show)
        strategy._events = strategy.control.events
        strategy._engine = FakeEngine()
        strategy._interpolator = SimpleNamespace(reset=lambda: None, multiplier=1)
        ctx.hardware.teleop = None

        def observe():
            if strategy.control.phase == "READY":
                ready_loop(strategy, events=strategy._events)
                if strategy.control.phase == "READY":
                    strategy.control.key("n")
            return {}

        ctx.hardware.robot_wrapper = SimpleNamespace(
            inner=SimpleNamespace(measured_joints=lambda: np.zeros(6)),
            get_observation=observe,
        )
        with (
            patch.object(NativeStrategy, "_policy_loop", new=policy_loop),
            patch("scripts.rollout.time.sleep"),
            patch.dict(
                NativeStrategy.run.__globals__,
                VideoEncodingManager=lambda _: contextlib.nullcontext(),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            if initialize:
                ctx.policy = SimpleNamespace(inference=strategy._engine)
                with patch("scripts.rollout.RolloutTerminal"):
                    strategy.setup(ctx)
            yield strategy, ctx, dataset

    def test_policy_never_runs_before_start_and_save_finishes_before_next_ready(self):
        phases = []

        def ready(strategy, **kwargs):
            self.assertFalse(strategy._engine.active)
            phases.append(("ready", dataset.num_episodes))

        def policy(strategy, **kwargs):
            self.assertTrue(strategy._engine.active)
            self.assertEqual(kwargs["control_time_s"], 5)
            # Both the first START and subsequent unedited trials use the configured task.
            self.assertEqual(strategy._engine.task, ctx.runtime.cfg.dataset.single_task)
            phases.append(("policy", dataset.num_episodes))
            dataset.frames += 1

        with self.session(ready, policy) as (strategy, ctx, dataset):
            strategy.run(ctx)
        self.assertEqual(
            phases, [("ready", 5), ("policy", 5), ("ready", 6), ("policy", 6)]
        )
        self.assertEqual(dataset.num_episodes, 7)
        self.assertFalse(strategy._engine.active)

    def test_exit_while_ready_sends_no_policy_action_and_saves_no_empty_episode(self):
        def ready(strategy, **kwargs):
            kwargs["events"]["stop_recording"] = True

        def policy(strategy, **kwargs):
            self.fail("Policy ran before operator start")

        with self.session(ready, policy) as (strategy, ctx, dataset):
            strategy.run(ctx)
            self.assertEqual(dataset.num_episodes, 5)

    def test_discard_waits_for_new_start_and_reuses_episode_number(self):
        attempts = []

        def ready(strategy, **kwargs):
            self.assertFalse(strategy._engine.active)
            self.assertEqual(dataset.frames, 0)

        def policy(strategy, **kwargs):
            attempts.append(dataset.num_episodes)
            dataset.frames = 1
            if len(attempts) == 1:
                kwargs["events"]["rerecord_episode"] = True

        with self.session(ready, policy, episodes=1) as (strategy, ctx, dataset):
            strategy.run(ctx)
            self.assertEqual(dataset.num_episodes, 6)
        self.assertEqual(attempts, [5, 5])

    def test_policy_loop_holds_robot_on_episode_end_and_preserves_primary_error(self):
        holds = []

        def ready(strategy, **kwargs):
            pass

        def policy(strategy, **kwargs):
            kwargs["dataset"].frames = 1

        with self.session(ready, policy, episodes=1) as (strategy, ctx, _dataset):
            ctx.hardware.robot_wrapper.supports_hold = True
            ctx.hardware.robot_wrapper.hold = lambda: holds.append("hold")
            strategy.run(ctx)
        self.assertEqual(holds, ["hold"])

        def failing_policy(strategy, **kwargs):
            raise ValueError("primary loop failure")

        with self.session(ready, failing_policy, episodes=1) as (strategy, ctx, _dataset):
            ctx.hardware.robot_wrapper.supports_hold = True
            ctx.hardware.robot_wrapper.hold = lambda: (_ for _ in ()).throw(
                RuntimeError("hold failed")
            )
            with self.assertRaisesRegex(ValueError, "primary loop failure") as caught:
                strategy.run(ctx)
        self.assertIn("hold failed", caught.exception.__notes__[0])


if __name__ == "__main__":
    unittest.main()
