"""Small UI/lifecycle wrapper around the pinned LeRobot 0.7.0 recorder.

Use manual start/stop with LeRobot's recording, keyboard, retry and writer.
Hooks below are scoped to this command; no installed package is modified.
"""

import os
import signal
import socket
import time

from lerobot.configs import parser
from lerobot.scripts import lerobot_record as upstream
from lerobot_robot_ur5e.spacemouse import SpaceMouse, TiltLimitError


class StoppedBeforeStart(Exception):
    """Exit through upstream cleanup without trying to save an empty episode."""


class RecordingFeedback:
    def __init__(self, cfg):
        self.cfg = cfg
        self.first_episode = None
        self.episode = None
        self.viewer = None
        self.rr = None
        self.dataset = None
        self._native_build_blueprint = None

    def show(self, text):
        print(f"\n{'=' * 72}\n{text}\n{'=' * 72}\n", flush=True)
        if self.rr is not None:
            from PIL import Image, ImageDraw, ImageFont

            title, _, body = text.partition("\n")
            phase = title.split()[0]
            color = {
                "READY": (52, 152, 219),
                "RECORDING": (255, 165, 0),
                "POLICY": (255, 165, 0),
                "SAVING": (46, 204, 113),
                "FINISHED": (46, 204, 113),
                "DISCARDING": (231, 76, 60),
                "RECOVERY": (231, 76, 60),
            }.get(phase, (149, 165, 166))
            banner = Image.new("RGB", (640, 180), color)
            draw = ImageDraw.Draw(banner)
            draw.text(
                (24, 28), phase, font=ImageFont.load_default(size=56), fill="black"
            )
            draw.text(
                (24, 108),
                "NOT RECORDING"
                if "NOT RECORDING" in title
                else title.partition(" ")[2],
                font=ImageFont.load_default(size=24),
                fill="black",
            )
            self.rr.log("recording_status/banner", self.rr.Image(banner), static=True)
            markdown = f"# {title}\n\n" + body.replace("\n", "\n\n")
            self.rr.log(
                "recording_status",
                self.rr.TextDocument(markdown, media_type="text/markdown"),
                static=True,
            )

    def init_viewer(self, session_name="recording"):
        import rerun as rr
        import rerun.blueprint as rrb
        from rerun.experimental import ViewerClient
        from lerobot.utils import rerun_visualization

        # A separate viewer/port belongs only to this command, never an existing viewer.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.setdefault("RERUN_FLUSH_NUM_BYTES", "8000")
        rr.init(session_name)
        self.viewer = ViewerClient.spawn(
            port=port,
            detach_process=False,
            hide_welcome_screen=True,
            memory_limit=os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%"),
        )
        rr.connect_grpc(self.viewer.url)
        self.rr = rr
        native_build = rerun_visualization._build_blueprint

        def build_with_status(*paths):
            # Keep LeRobot's default grid (and its growth); only add the phase/controls strip on top.
            return rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(
                            origin="/recording_status/banner", name="Phase"
                        ),
                        rrb.TextDocumentView(
                            origin="/recording_status", name="Controls"
                        ),
                        column_shares=[2, 5],
                    ),
                    native_build(*paths).root_container,
                    row_shares=[1, 4],
                )
            )

        # LeRobot sends its own layout on the first frame; wrap its builder while this viewer lives.
        self._native_build_blueprint = native_build
        rerun_visualization._build_blueprint = build_with_status
        rerun_visualization.log_rerun_data.blueprint = (
            None  # As init_rerun: first frame builds the layout.
        )
        rerun_visualization.log_rerun_data.blueprint_paths = None
        rr.send_blueprint(build_with_status(set(), set(), set()))
        self.show("CONNECTING — NOT RECORDING\nWait for the robot to connect.")

    def close_viewer(self):
        if self._native_build_blueprint is not None:
            from lerobot.utils import rerun_visualization

            rerun_visualization._build_blueprint = self._native_build_blueprint
            self._native_build_blueprint = None
        if self.viewer is not None:
            self.viewer.close()

    def loop(self, original_loop, **kwargs):
        dataset = kwargs.get("dataset")
        events = kwargs["events"]
        if dataset is None:
            # Skip upstream's pre-save reset. Reset now happens in READY after save/discard.
            return
        self.dataset = dataset
        self.episode = dataset.num_episodes
        if self.first_episode is None:
            self.first_episode = self.episode
        # Ignore next/retry inputs queued while saving. READY needs fresh operator input.
        events["exit_early"] = False
        events["rerecord_episode"] = False
        self.show(
            f"READY — NOT RECORDING\n{dataset.num_episodes} saved episodes; next dataset episode {self.episode}.\n"
            "Live preview/teleop are active. Return home (h) and reset the scene.\n"
            "Keyboard Right arrow / n: START recording | Esc / q / Ctrl+C: exit\n"
            "Waiting for the operator; no automatic start."
        )
        while True:
            self.control_loop(
                original_loop, **{**kwargs, "dataset": None, "timer": None}
            )
            if events["stop_recording"]:
                raise StoppedBeforeStart
            if not events["rerecord_episode"]:
                break
            events["rerecord_episode"] = False
            events["exit_early"] = False
        ordinal = self.episode - self.first_episode + 1
        self.show(
            f"RECORDING {ordinal}/{self.cfg.dataset.num_episodes} | dataset episode {self.episode}\n"
            "Keyboard Right arrow / n: STOP + SAVE | keyboard Left arrow / r: STOP + DISCARD | Esc / q / Ctrl+C: save + exit\n"
            "After saving/discarding, wait in READY to reset and start manually. Do not home while recording."
        )
        upstream.log_say("Recording started", self.cfg.play_sounds)
        if kwargs.get("timer") is not None:
            kwargs["timer"].restart()  # READY is outside the recording cadence window.
        try:
            result = self.control_loop(original_loop, **kwargs)
        except TiltLimitError:
            # The native loop's exception cleanup stopped the asynchronous image writer.
            writer = getattr(dataset, "writer", None)
            image_writer = getattr(writer, "image_writer", None)
            if image_writer is not None:
                writer.start_image_writer(
                    num_processes=image_writer.num_processes,
                    num_threads=image_writer.num_threads,
                )
            events["rerecord_episode"] = True
            self.show(
                "DISCARDING — NOT RECORDING\n"
                "Current TCP orientation is outside the bounded tilt range.\n"
                "This unsaved attempt is discarded. Return to READY and press h to home with a clear path."
            )
            return
        if events["rerecord_episode"]:
            self.show(
                "DISCARDING — NOT RECORDING\nThis attempt will not be saved. Reset in READY and retry this episode."
            )
        else:
            self.show(
                "SAVING EPISODE — NOT RECORDING\nPlease wait for READY (or FINISHED at the requested count).\nThe next recording will not start automatically."
            )
        return result

    def control_loop(self, original_loop, **kwargs):
        # The operator controls all phase boundaries; ignore upstream episode/reset timers.
        kwargs["control_time_s"] = float("inf")
        events = kwargs["events"]

        def request_stop(signum, frame):
            # Match Esc instead of interrupting the writer and losing the buffered episode.
            # A second Ctrl+C can still interrupt a stuck control call.
            if events["stop_recording"]:
                raise KeyboardInterrupt
            events["stop_recording"] = True
            events["exit_early"] = True

        previous_handler = signal.signal(signal.SIGINT, request_stop)
        try:
            recovering = False
            while True:
                try:
                    result = original_loop(**kwargs)
                    if (
                        kwargs.get("dataset") is None
                        and not events["stop_recording"]
                        and not events["rerecord_episode"]
                        and isinstance(kwargs.get("teleop"), SpaceMouse)
                    ):
                        kwargs["teleop"].check_tilt_range()
                    return result
                except TiltLimitError:
                    if kwargs.get("dataset") is not None:
                        raise
                    if events["stop_recording"]:
                        raise StoppedBeforeStart
                    events["rerecord_episode"] = events["exit_early"] = False
                    if not recovering:
                        self.show(
                            "RECOVERY — NOT RECORDING\n"
                            "Current TCP orientation is outside the bounded tilt range; puck motion is held.\n"
                            "Press h to home with a clear path, or restore posture on the pendant.\n"
                            "Then keyboard Right arrow / n starts only once measured tilt is back in range. Esc exits.\n"
                            "Home does not redefine the saved neutral."
                        )
                        recovering = True
                    time.sleep(1 / kwargs["fps"])
        finally:
            signal.signal(signal.SIGINT, previous_handler)


def run_recording(cfg):
    feedback = RecordingFeedback(cfg)
    original_loop = upstream.record_loop
    original_init = upstream.init_visualization
    original_log_say = upstream.log_say

    def init_visualization(mode, **kwargs):
        if mode == "rerun" and kwargs.get("ip") is None and kwargs.get("port") is None:
            feedback.init_viewer()
        else:
            original_init(mode, **kwargs)

    def log_say(text, *args, **kwargs):
        # Upstream announces an episode before our manual start gate; avoid a false cue.
        if text.startswith("Recording episode ") or text in {
            "Reset the environment",
            "Re-record episode",
        }:
            return
        original_log_say(text, *args, **kwargs)

    upstream.record_loop = lambda **kwargs: feedback.loop(original_loop, **kwargs)
    upstream.init_visualization = init_visualization
    upstream.log_say = log_say
    try:
        try:
            dataset = upstream.record.__wrapped__(cfg)
        except StoppedBeforeStart:
            dataset = feedback.dataset
        feedback.rr = None  # Upstream has already shut down the SDK during cleanup.
        feedback.show(
            f"FINISHED — {dataset.num_episodes} saved episodes in the dataset."
        )
        return dataset
    except BaseException:
        print(
            "Recording stopped before normal completion. Review the last episode before resuming.",
            flush=True,
        )
        raise
    finally:
        upstream.record_loop = original_loop
        upstream.init_visualization = original_init
        upstream.log_say = original_log_say
        feedback.close_viewer()


@parser.wrap()
def record(cfg: upstream.RecordConfig):
    return run_recording(cfg)


if __name__ == "__main__":
    upstream.register_third_party_plugins()
    try:
        record()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
