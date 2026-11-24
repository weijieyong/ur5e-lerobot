"""Manual READY/start around LeRobot's episodic rollout; no installed-library edits."""

import sys
import contextlib
import os
import math
from dataclasses import dataclass
from types import SimpleNamespace
import time

from lerobot.configs import parser
from lerobot.scripts import lerobot_rollout as upstream
from lerobot.rollout.strategies.episodic import EpisodicStrategy as NativeStrategy
from lerobot.rollout import context as rollout_context
from lerobot.processor import bind_relative_anchor as native_bind_relative_anchor
from scripts.policy_loading import load_rollout_policy, local_checkpoint_path
from scripts.record import RecordingFeedback, StoppedBeforeStart
from scripts.rollout_input import RolloutInput, RolloutTerminal, preflight


@dataclass
class RolloutConfig(upstream.RolloutConfig):
    def __post_init__(self):
        path = parser.get_path_arg("policy")
        if path is None and self.policy is not None:
            path = self.policy.pretrained_path
        if path is not None:
            local_checkpoint_path(path)
        super().__post_init__()


class ManualEpisodicStrategy(NativeStrategy):
    def __init__(self, config, feedback):
        super().__init__(config)
        self.feedback = feedback

    def setup(self, ctx):
        self.control = RolloutInput(
            ctx.runtime.cfg.dataset.single_task,
            self.feedback.show,
            preview=lambda text: print(text, end="", flush=True),
        )
        module = sys.modules[NativeStrategy.__module__]
        original = module.init_keyboard_listener

        def listener_factory():
            self._listener = RolloutTerminal(self.control)
            self._listener.start()
            return self._listener, self.control.events

        module.init_keyboard_listener = listener_factory
        try:
            super().setup(ctx)
            self._engine.pause()
        except BaseException:
            if self._listener is not None:
                with contextlib.suppress(Exception):
                    self._listener.stop()
            raise
        finally:
            module.init_keyboard_listener = original

    def run(self, ctx):
        try:
            super().run(ctx)
        except StoppedBeforeStart:
            pass
        self.control.check()

    def _start_episode(self, robot, timer):
        self._engine.pause()

    def _reset_loop(self, **kwargs):
        return

    def _wait_ready(self, ctx, robot, timer):
        control = self.control
        control.ready()
        while True:
            control.check()
            if getattr(self._engine, "failed", False):
                raise RuntimeError(
                    f"Rollout inference failed while paused: {self._engine.failure_traceback}"
                )
            events = control.events
            if events["stop_recording"] or ctx.runtime.shutdown_event.is_set():
                raise StoppedBeforeStart
            if control.phase == "HOME_PENDING":
                with control.lock:
                    control.home = False
                    control.phase = "BUSY"
                    control.check()
                    if events["stop_recording"] or ctx.runtime.shutdown_event.is_set():
                        raise StoppedBeforeStart
                robot.inner.go_home()
                control.ready()
            elif control.phase == "START_PENDING":
                measured = joint_vector(
                    robot.inner.measured_joints(), "measured_joints (radians)"
                )
                home = joint_vector(
                    ctx.runtime.cfg.robot.home_joints_deg, "home_joints_deg"
                )
                tolerance = start_tolerance()
                errors = [
                    abs(math.degrees(q) - target)
                    for q, target in zip(measured, home, strict=True)
                ]
                with control.lock:
                    control.check()
                    if events["stop_recording"] or ctx.runtime.shutdown_event.is_set():
                        raise StoppedBeforeStart
                    if (
                        events["exit_early"]
                        or events["rerecord_episode"]
                        or control.phase != "START_PENDING"
                    ):
                        accepted = False
                    else:
                        accepted = max(errors) <= tolerance
                    if accepted:
                        self.active_task = control.current
                        self._engine.set_task(self.active_task)
                        self._engine.reset()
                        self._interpolator.reset()
                        timer.restart()
                        control.check()
                        if (
                            events["stop_recording"]
                            or ctx.runtime.shutdown_event.is_set()
                        ):
                            raise StoppedBeforeStart
                        if not events["exit_early"] and not events["rerecord_episode"]:
                            control.phase = "RUNNING"
                            self._engine.resume()
                            return
                if max(errors) > tolerance:
                    self.feedback.show(
                        f"RECOVERY — POLICY PAUSED: joint errors (deg) {errors}; tolerance {tolerance:g}.\n"
                        "Use explicit h with a clear path or reset on pendant, then issue a fresh START."
                    )
                control.ready()
            robot.get_observation()
            time.sleep(1 / ctx.runtime.cfg.fps)

    def _policy_loop(self, **kwargs):
        ctx, events = kwargs["ctx"], kwargs["events"]
        self._wait_ready(ctx, kwargs["robot"], kwargs["timer"])
        self.feedback.show(
            f"POLICY RUNNING | dataset episode {ctx.data.dataset.num_episodes}\n"
            f"Active task: {self.active_task}\n"
            "n/Right: STOP + SAVE | r/Left: STOP + DISCARD | q/Esc/Ctrl+C: exit"
        )
        try:
            result = super()._policy_loop(**kwargs)
            self.control.check()
        finally:
            primary = sys.exception()
            with self.control.lock:
                self.control.phase = "BUSY"
                self.control.home = False
            self._engine.pause()
            robot = kwargs["robot"]
            try:
                if getattr(robot, "supports_hold", False):
                    robot.hold()
                elif callable(getattr(getattr(robot, "inner", None), "hold", None)):
                    robot.inner.hold()
            except Exception as error:
                if primary is None:
                    raise
                primary.add_note(f"Robot hold on episode stop also failed: {error!r}")
        self.feedback.show(
            "DISCARDING — POLICY PAUSED"
            if events["rerecord_episode"]
            else "SAVING — POLICY PAUSED\nWait for READY (or session completion)."
        )
        return result

    def teardown(self, ctx):
        if hasattr(self, "control"):
            with self.control.lock:
                self.control.phase = "CLOSED"
        primary = sys.exception()
        cleanup_error = None
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception as error:
                cleanup_error = error
            self._listener = None
        try:
            super().teardown(ctx)
        except Exception as error:
            if cleanup_error is not None:
                error.add_note(f"Terminal cleanup also failed: {cleanup_error!r}")
            if primary is None:
                raise
            primary.add_note(f"Rollout cleanup also failed: {error!r}")
        if cleanup_error is not None:
            if primary is None:
                raise cleanup_error
            primary.add_note(f"Terminal cleanup also failed: {cleanup_error!r}")


def joint_vector(value, name):
    try:
        values = list(value)
        if len(values) != 6 or any(not math.isfinite(float(v)) for v in values):
            raise ValueError
        if any(getattr(v, "ndim", 0) != 0 for v in values):
            raise ValueError
        return [float(v) for v in values]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain six finite joint values") from error


def start_tolerance():
    value = float(os.environ.get("ROLLOUT_START_TOLERANCE_DEG", "5"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("ROLLOUT_START_TOLERANCE_DEG must be finite and > 0")
    return value


def run_rollout(cfg):
    if cfg.teleop is not None:
        raise ValueError(
            "Policy-only rollout does not accept --teleop; use record/teleop separately"
        )
    if cfg.interactive or cfg.strategy.type != "episodic":
        raise ValueError(
            "Rollout requires episodic strategy and --interactive=false; terminal input is wrapper-owned"
        )
    joint_vector(getattr(cfg.robot, "home_joints_deg", None), "robot.home_joints_deg")
    start_tolerance()
    preflight()
    cfg.strategy.reset_to_initial_position = False
    cfg.return_to_initial_position = False
    print(
        "Wrapper settings: reset_to_initial_position=false return_to_initial_position=false; explicit h only",
        flush=True,
    )
    rtc = getattr(cfg.inference, "rtc", None)
    print(
        f"Rollout settings: max_joint_step_rad={getattr(cfg.robot, 'max_joint_step_rad', 'n/a')} "
        f"fps={cfg.fps:g} inference={cfg.inference.type} "
        f"rtc_enabled={getattr(rtc, 'enabled', 'n/a')} rtc_mode={getattr(rtc, 'mode', 'n/a')} "
        f"rtc_execution_horizon={getattr(rtc, 'execution_horizon', 'n/a')}",
        flush=True,
    )
    feedback = RecordingFeedback(cfg)
    original_init, original_factory = (
        upstream.init_visualization,
        upstream.create_strategy,
    )
    original_loader = rollout_context._load_pretrained_policy
    original_bind = rollout_context.bind_relative_anchor
    strategy_module = sys.modules[NativeStrategy.__module__]
    original_log_say = strategy_module.log_say

    def init_visualization(mode, **kwargs):
        if mode == "rerun" and kwargs.get("ip") is None and kwargs.get("port") is None:
            feedback.init_viewer("rollout")
        else:
            original_init(mode, **kwargs)

    def create_strategy(config):
        return (
            ManualEpisodicStrategy(config, feedback)
            if config.type == "episodic"
            else original_factory(config)
        )

    def log_say(text, *args, **kwargs):
        if text.startswith("Recording episode ") or text in {
            "Reset the environment",
            "Re-record episode",
        }:
            return
        original_log_say(text, *args, **kwargs)

    upstream.init_visualization, upstream.create_strategy = (
        init_visualization,
        create_strategy,
    )
    rollout_context._load_pretrained_policy = lambda config: load_rollout_policy(
        config, original_loader
    )
    rollout_context.bind_relative_anchor = bind_relative_anchor
    strategy_module.log_say = log_say
    try:
        return upstream.rollout.__wrapped__(cfg)
    finally:
        primary = sys.exception()
        upstream.init_visualization, upstream.create_strategy = (
            original_init,
            original_factory,
        )
        rollout_context._load_pretrained_policy = original_loader
        rollout_context.bind_relative_anchor = original_bind
        strategy_module.log_say = original_log_say
        try:
            feedback.close_viewer()
        except Exception as error:
            if primary is None:
                raise
            primary.add_note(f"Viewer cleanup also failed: {error!r}")


def bind_relative_anchor(policy, pipeline):
    """Bind the relative-action step to the policy's queue without a reference deepcopy can follow.

    RTC deep-copies the preprocessor; a bound method there deep-copies the whole policy onto the
    GPU (a second ~9 GB, which runs a 10 GB card out of memory). Functions are not deep-copied,
    so the copy keeps counting the live policy's queue.
    """

    def count_queued_actions():
        return policy.count_queued_actions()

    return native_bind_relative_anchor(
        SimpleNamespace(count_queued_actions=count_queued_actions), pipeline
    )


@parser.wrap()
def rollout(cfg: RolloutConfig):
    return run_rollout(cfg)


if __name__ == "__main__":
    upstream.register_third_party_plugins()
    rollout()
