"""Collect local LeRobot episodes of the varied MuJoCo laser trace."""

import argparse
from dataclasses import asdict
from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_robot_ur5e.sim.config_sim import (
    SIM_DEFAULT_HEIGHT,
    SIM_DEFAULT_WIDTH,
    SIM_STATE_ACTION_NAMES,
)
from lerobot_robot_ur5e.sim.laser_trace import (
    CAMERAS,
    FPS,
    LINE_HALF_WIDTH_M,
    MARKER_NAMES,
    TASK_INSTRUCTION,
    LaserTrace,
    sample_layouts,
    trace_targets,
)

WIDTH = SIM_DEFAULT_WIDTH
HEIGHT = SIM_DEFAULT_HEIGHT


def dataset_features() -> dict:
    joints = {"dtype": "float32", "shape": (7,), "names": list(SIM_STATE_ACTION_NAMES)}
    video = {"dtype": "video", "shape": (HEIGHT, WIDTH, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": joints.copy(),
        "action": joints.copy(),
        **{f"observation.images.{camera}": video.copy() for camera in CAMERAS},
    }


class TraceVisibility:
    """Check framing and occlusion against the markings in each MuJoCo camera."""

    def __init__(self, sim: LaserTrace) -> None:
        self.sim = sim
        start, end = sim.layout.endpoints
        line = sim.data.site("trace_line")
        line_z = line.xpos[2] + sim.model.site(line.id).size[2]
        center_xy = np.linspace(start, end, num=int(np.ceil(sim.layout.length_m / 0.003)) + 1)
        self.centerline = np.column_stack((center_xy, np.full(len(center_xy), line_z)))
        line_corners = np.array([start, start, end, end]) + np.outer(
            [-1, 1, -1, 1], LINE_HALF_WIDTH_M * sim.layout.transverse
        )
        outline = [[*xy, line_z] for xy in line_corners]
        self.markers = []
        unit_corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        for name in MARKER_NAMES:
            site = sim.data.site(name)
            radius, half_height, _ = sim.model.site(name).size
            center = site.xpos + [0, 0, half_height]
            self.markers.append((name, site.id, center))
            outline.extend([*xy, center[2]] for xy in center[:2] + radius * unit_corners)
        self.outline = np.array(outline)
        self.line_ids = [
            line.id, sim.model.site("laser_dot").id, *(site_id for _, site_id, _ in self.markers)
        ]

    def _project(self, camera: str, points: np.ndarray) -> np.ndarray:
        camera_pose = self.sim.data.camera(camera)
        local = (points - camera_pose.xpos) @ camera_pose.xmat.reshape(3, 3)
        if np.any(local[:, 2] >= 0):
            raise ValueError(f"markings behind {camera} camera")
        fovy = np.deg2rad(self.sim.model.camera(camera).fovy[0])
        focal = HEIGHT / (2 * np.tan(fovy / 2))
        return local[:, :2] / -local[:, 2:3] * [focal, -focal] + [
            (WIDTH - 1) / 2, (HEIGHT - 1) / 2
        ]

    def frame(self, renderer: mujoco.Renderer, camera: str) -> tuple[np.ndarray, float]:
        bounds = self._project(camera, self.outline)
        margin = float(np.minimum(bounds, [WIDTH - 1, HEIGHT - 1] - bounds).min())
        if margin < 2:
            raise ValueError(f"{camera} crops the trace markings (margin {margin:.1f} px)")

        renderer.update_scene(self.sim.data, camera=camera)
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render()
        renderer.disable_segmentation_rendering()

        site_pixels = np.where(
            segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_SITE, segmentation[:, :, 0], -1
        )
        x, y = self._project(camera, self.centerline).round().astype(int).T
        hidden = ~np.isin(site_pixels[y, x], self.line_ids)
        if np.any(hidden):
            raise ValueError(f"{camera} loses {int(hidden.sum())} sampled points on the black line")
        for name, site_id, center in self.markers:
            x, y = self._project(camera, center[None, :]).round().astype(int)[0]
            neighborhood = site_pixels[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
            if site_id not in neighborhood:
                raise ValueError(f"{camera} loses the {name} dot")

        return renderer.render(), margin


def collect_episode(dataset: LeRobotDataset, sim: LaserTrace, episode: int) -> dict:
    sim.reset()
    visibility = TraceVisibility(sim)
    start, end = sim.layout.endpoints
    transverse = sim.layout.transverse
    max_cross_track = 0.0
    framing = dict.fromkeys(CAMERAS, float("inf"))
    with mujoco.Renderer(sim.model, height=HEIGHT, width=WIDTH) as renderer:
        for frame, target in enumerate(trace_targets(sim.layout)):
            context = f"Episode {episode}, frame {frame}"
            action = sim.command(target)
            spot = sim.data.site("laser_dot").xpos.copy()
            if not sim.laser_gripped() or spot[2] <= 0:
                raise ValueError(f"{context} lost the laser grasp or tabletop spot")
            cross_track = abs(float((spot[:2] - start) @ transverse))
            if cross_track > LINE_HALF_WIDTH_M:
                raise ValueError(f"{context} went {cross_track * 1000:.1f} mm off the line center")
            max_cross_track = max(max_cross_track, cross_track)
            if frame == 0:
                start_error = float(np.linalg.norm(spot[:2] - start))
            row = {"observation.state": sim.measured_state(), "action": action, "task": TASK_INSTRUCTION}
            for camera in CAMERAS:
                try:
                    image, margin = visibility.frame(renderer, camera)
                except ValueError as exc:
                    raise ValueError(f"{context}, layout {sim.layout}: {exc}") from exc
                row[f"observation.images.{camera}"] = image
                framing[camera] = min(framing[camera], margin)
            dataset.add_frame(row)
            sim.advance()

    end_error = float(np.linalg.norm(spot[:2] - end))
    if max(start_error, end_error) > 0.008:
        raise ValueError(f"Episode {episode} missed an endpoint by more than 8 mm")
    dataset.save_episode(parallel_encoding=False)
    return {
        "episode_index": episode,
        **asdict(sim.layout),
        "start_xy_m": start.tolist(),
        "end_xy_m": end.tolist(),
        "frames": frame + 1,
        "start_error_mm": start_error * 1000,
        "finish_error_mm": end_error * 1000,
        "max_cross_track_mm": max_cross_track * 1000,
        "minimum_camera_margin_px": framing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes <= 0 or args.seed < 0:
        parser.error("--episodes must be positive and --seed must be nonnegative")
    root = args.dataset_root.resolve()
    if root.exists():
        parser.error(f"dataset root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(f"outputs/laser-trace-{date.today()}/{root.name}-collection.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    reports = []
    with TemporaryDirectory(prefix=f".{root.name}-", dir=root.parent) as temp:
        staging = Path(temp) / "dataset"
        dataset = LeRobotDataset.create(
            args.repo_id,
            root=staging,
            fps=FPS,
            features=dataset_features(),
            robot_type="ur5e_sim_laser_trace",
            video_backend="pyav",
            rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="fast", crf=17),
            encoder_threads=2,
        )
        for index, layout in enumerate(sample_layouts(args.episodes, args.seed)):
            sim = LaserTrace(layout, width=WIDTH, height=HEIGHT)
            report = collect_episode(dataset, sim, index)
            reports.append(report)
            print(
                f"Saved episode {index + 1}/{args.episodes}: {report['frames']} frames, "
                f"{layout.length_m * 100:.1f} cm, {layout.angle_deg:+.1f}°, both views verified",
                flush=True,
            )
        dataset.finalize()
        staging.rename(root)

    with report_path.open("w") as report_file:
        json.dump({"seed": args.seed, "repo_id": args.repo_id, "episodes": reports}, report_file, indent=2)
        report_file.write("\n")
    print(f"Dataset: {root}\nCollection report: {report_path}")


if __name__ == "__main__":
    main()
