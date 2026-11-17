"""Recoverable tilt warnings around the pinned LeRobot teleoperation command."""

import time

from lerobot.configs import parser
from lerobot.scripts import lerobot_teleoperate as upstream
from lerobot_robot_ur5e.spacemouse import TiltLimitError
from scripts.record import RecordingFeedback


def run_teleoperation(cfg):
    feedback = RecordingFeedback(cfg)
    original_loop, original_init = upstream.teleop_loop, upstream.init_visualization
    active = (
        "TELEOP ACTIVE | NOT RECORDING\n"
        "Drive with the SpaceMouse. Press h to home with a clear path. Esc / Ctrl+C exits."
    )

    def init_visualization(mode, **kwargs):
        if mode == "rerun" and kwargs.get("ip") is None and kwargs.get("port") is None:
            feedback.init_viewer("teleoperation")
        else:
            original_init(mode, **kwargs)

    def teleop_loop(**kwargs):
        duration = kwargs.get("duration")
        deadline = None if duration is None else time.perf_counter() + duration
        recovering = False
        feedback.show(active)
        while True:
            remaining = None if deadline is None else deadline - time.perf_counter()
            if remaining is not None and remaining <= 0:
                return
            try:
                result = original_loop(
                    **{**kwargs, "duration": 0 if recovering else remaining}
                )
            except TiltLimitError:
                if not recovering:
                    feedback.show(
                        "RECOVERY | NOT RECORDING\n"
                        "Current TCP orientation is outside the bounded tilt range; puck motion is held.\n"
                        "Press h to home with a clear path, or restore posture on the pendant.\n"
                        "Teleoperation resumes automatically once tilt is back in range. Esc / Ctrl+C exits.\n"
                        "Home does not redefine the saved neutral."
                    )
                    recovering = True
                delay = 1 / kwargs["fps"]
                if deadline is not None:
                    delay = min(delay, max(0.0, deadline - time.perf_counter()))
                time.sleep(delay)
                continue
            if not recovering:
                return result
            feedback.show(active)
            recovering = False

    upstream.teleop_loop, upstream.init_visualization = teleop_loop, init_visualization
    try:
        return upstream.teleoperate.__wrapped__(cfg)
    finally:
        upstream.teleop_loop, upstream.init_visualization = original_loop, original_init
        feedback.close_viewer()


@parser.wrap()
def teleoperate(cfg: upstream.TeleoperateConfig):
    return run_teleoperation(cfg)


if __name__ == "__main__":
    upstream.register_third_party_plugins()
    teleoperate()
