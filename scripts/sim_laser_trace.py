"""Render the scripted MuJoCo laser trace to a local MP4."""

import argparse
from datetime import date
from pathlib import Path

import av
import mujoco

from lerobot_robot_ur5e.sim.laser_trace import FPS, LaserTrace, trace_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        help="path for the MP4 (defaults to a dated directory under outputs/)",
    )
    output_arg = parser.parse_args().output
    output = Path(output_arg or f"outputs/laser-trace-{date.today()}/laser-trace.mp4")

    sim = LaserTrace()
    sim.reset()
    width = sim.model.vis.global_.offwidth
    height = sim.model.vis.global_.offheight
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        mujoco.Renderer(sim.model, height=height, width=width) as renderer,
        av.open(str(output), mode="w") as video,
    ):
        stream = video.add_stream("libx264", rate=FPS)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for target in trace_targets():
            sim.step(target)
            renderer.update_scene(sim.data, camera="trace")
            frame = av.VideoFrame.from_ndarray(renderer.render(), format="rgb24")
            video.mux(stream.encode(frame))
        video.mux(stream.encode())

    print(output)


if __name__ == "__main__":
    main()
