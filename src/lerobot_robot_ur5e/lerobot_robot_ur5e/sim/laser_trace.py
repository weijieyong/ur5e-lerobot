"""MuJoCo tabletop scene and scripted motion for tracing a line with a gripped laser."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain, repeat

import mujoco
import numpy as np

from .ik import DOWNWARD_PINCH_ROT, PinchIK
from .scene import (
    GRIPPER_ACTUATOR_NAME,
    GRIPPER_DRIVER_JOINTS,
    GRIPPER_DRIVER_MAX_RAD,
    GRIPPER_PREFIX,
    HOME_QPOS,
    TABLE_TOP_Z,
    _camera_xyaxes,
    build_tabletop_spec,
)

LINE_HALF_WIDTH_M = 0.006
TRACE_PINCH_HEIGHT_M = 0.16
LASER_HALF_LENGTH_M = 0.05
FPS = 15
CAMERAS = ("front", "wrist")
MARKER_NAMES = tuple(f"trace_{end}_{index}" for end in ("start", "finish") for index in range(3))
TASK_INSTRUCTION = "Trace the black line with the laser from the yellow dots to the blue dots"


@dataclass(frozen=True)
class TraceLayout:
    center_xy: tuple[float, float] = (0.46, 0.0)
    length_m: float = 0.32
    angle_deg: float = 0.0

    @property
    def direction(self) -> np.ndarray:
        angle = np.deg2rad(self.angle_deg)
        return np.array([np.sin(angle), np.cos(angle)])

    @property
    def transverse(self) -> np.ndarray:
        return self.direction[::-1] * [1, -1]

    @property
    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        half_line = self.direction * self.length_m / 2
        return self.center_xy - half_line, self.center_xy + half_line


DEFAULT_LAYOUT = TraceLayout()


def sample_layouts(count: int, seed: int) -> Iterator[TraceLayout]:
    """Spread the episodes across each layout range, with independently shuffled values."""
    rng = np.random.default_rng(seed)
    bounds = np.array([[0.40, 0.52], [-0.08, 0.08], [0.15, 0.29], [-40, 40]])
    strata = (np.arange(count)[:, None] + rng.random((count, len(bounds)))) / count
    for column in strata.T:
        rng.shuffle(column)
    for x, y, length, angle in bounds[:, 0] + strata * np.diff(bounds, axis=1).ravel():
        yield TraceLayout((float(x), float(y)), float(length), float(angle))


def trace_targets(layout: TraceLayout = DEFAULT_LAYOUT) -> Iterator[np.ndarray]:
    """Yield points on the line, pausing briefly at both endpoints."""
    start, end = layout.endpoints
    steps = round(5 * FPS * layout.length_m / DEFAULT_LAYOUT.length_m)
    for progress in chain(repeat(0.0, FPS), np.linspace(0, 1, steps + 1), repeat(1.0, FPS)):
        eased = progress**2 * (3 - 2 * progress)
        yield start + (end - start) * eased


def _add_wrist_camera_housing(camera: mujoco.MjsCamera, xyaxes: list[float]) -> None:
    """Mount a visual camera and bracket on the gripper at the optical frame."""
    parent = camera.parent
    frame = parent.add_frame(pos=camera.pos, xyaxes=xyaxes)
    right, up = np.reshape(xyaxes, (2, 3))
    rear = camera.pos + 0.025 * np.cross(right, up)
    parent.add_site(
        name="wrist_camera_bracket",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.035, 0.0, -0.014, *rear],
        size=[0.009, 0, 0],
        rgba=[0.15, 0.18, 0.21, 1],
    )
    # The camera looks along local -Z. Keep the face and lenses behind the optical origin.
    for name, position, half_size, color in (
        ("housing", [0, 0, 0.015], [0.024, 0.019, 0.012], [0.68, 0.74, 0.79, 1]),
        ("face", [0, 0, 0.003], [0.022, 0.017, 0.0015], [0.09, 0.12, 0.15, 1]),
        ("rear_plate", [0, 0, 0.027], [0.019, 0.014, 0.0015], [0.13, 0.27, 0.36, 1]),
    ):
        parent.add_site(
            name=f"wrist_camera_{name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=position,
            size=half_size,
            rgba=color,
        ).set_frame(frame)
    for side, offset in (("left", -0.012), ("right", 0.012)):
        parent.add_site(
            name=f"wrist_camera_{side}_lens",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[offset, 0, 0.0015],
            size=[0.0065, 0.001, 0],
            rgba=[0.18, 0.42, 0.49, 1],
        ).set_frame(frame)


def build_laser_trace_model(
    width: int = 960, height: int = 720, layout: TraceLayout = DEFAULT_LAYOUT,
) -> mujoco.MjModel:
    """Add the tracing line, endpoint dots, free laser tool, and an external camera."""
    spec = build_tabletop_spec(width, height)
    # Blending samples at pixel edges makes object IDs ambiguous in segmentation renders.
    spec.visual.quality.offsamples = 0
    world = spec.worldbody
    table = spec.body("table")
    mark_z = TABLE_TOP_Z - table.pos[2] + 0.0005
    line_center = layout.center_xy - table.pos[:2]
    start, end = layout.endpoints
    transverse = layout.transverse
    half_angle = np.deg2rad(layout.angle_deg) / 2
    table.add_site(
        name="trace_line",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[*line_center, mark_z],
        size=[LINE_HALF_WIDTH_M, layout.length_m / 2, 0.0003],
        quat=[np.cos(half_angle), 0.0, 0.0, -np.sin(half_angle)],
        rgba=[0.013, 0.013, 0.016, 1.0],
    )
    for name, endpoint, color in (
        ("start", start, [1.0, 0.85, 0.015, 1.0]),
        ("finish", end, [0.02, 0.32, 1.0, 1.0]),
    ):
        for index, offset in enumerate((-0.032, 0.0, 0.032)):
            table.add_site(
                name=f"trace_{name}_{index}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[*(endpoint - table.pos[:2] + offset * transverse), mark_z + 0.0008],
                size=[0.014 if index == 1 else 0.008, 0.0003, 0.0],
                rgba=color,
            )

    laser = world.add_body(name="laser", pos=[0.70, 0.40, 0.013], quat=[1, 0, 1, 0])
    laser.add_freejoint(name="laser_joint")
    laser.add_geom(
        name="laser_housing",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.013, 0.021, LASER_HALF_LENGTH_M],
        mass=0.035,
        rgba=[0.80, 0.09, 0.06, 1.0],
        condim=4,
        friction=[1.2, 0.02, 0.002],
    )
    laser.add_site(
        name="laser_cap",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, LASER_HALF_LENGTH_M - 0.006],
        size=[0.0133, 0.0213, 0.006],
        rgba=[0.12, 0.13, 0.15, 1.0],
    )
    laser.add_site(
        name="laser_aperture",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, LASER_HALF_LENGTH_M + 0.0005],
        size=[0.008, 0.0005, 0.0],
        rgba=[0.48, 0.025, 0.025, 1.0],
    )

    spec.add_material(name="laser_red", rgba=[1.0, 0.015, 0.005, 1.0], emission=1.0)
    spot = world.add_body(name="laser_spot", mocap=True, pos=[0.0, 0.0, -1.0])
    spot.add_site(
        name="laser_dot",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        material="laser_red",
        size=[0.0045, 0.0003, 0.0],
    )

    # Face the line from the side so the gripper stays clear of both sightlines.
    front = spec.camera("front")
    front_pos = np.array([1.15, 0.0, 0.74])
    front.pos = front_pos
    front.alt.xyaxes = _camera_xyaxes(front_pos, np.array([0.46, 0.0, 0.12]), np.array([0, 0, 1]))
    front.fovy = 46.0

    wrist = spec.camera("wrist")
    wrist_pos = np.array([0.19, 0.0, -0.045])
    wrist.pos = wrist_pos
    wrist_axes = _camera_xyaxes(
        wrist_pos, np.array([0.0, 0.0, 0.305]), np.array([-1, 0, 0])
    )
    wrist.alt.xyaxes = wrist_axes
    wrist.fovy = 82.0
    _add_wrist_camera_housing(wrist, wrist_axes)

    camera_pos = np.array([1.12, -0.49, 0.84])
    camera_target = np.array([0.38, -0.025, 0.30])
    world.add_camera(
        name="trace",
        pos=camera_pos,
        xyaxes=_camera_xyaxes(camera_pos, camera_target, np.array([0, 0, 1])),
        fovy=48.0,
    )
    return spec.compile()


class LaserTrace:
    """Start with the laser between the pads and sweep the tabletop at 15 Hz."""

    def __init__(
        self, layout: TraceLayout = DEFAULT_LAYOUT, *, width: int = 960, height: int = 720,
    ) -> None:
        self.layout = layout
        self.model = build_laser_trace_model(width, height, layout)
        self.data = mujoco.MjData(self.model)
        self.ik = PinchIK(self.model)
        self._gripper_actuator = self.model.actuator(GRIPPER_ACTUATOR_NAME).id
        self._gripper_qpos = [self.model.joint(name).qposadr[0] for name in GRIPPER_DRIVER_JOINTS]
        self._laser_geom = self.model.geom("laser_housing").id
        self._table_geom = self.model.geom("table_top").id
        self._pad_geoms = [
            {self.model.geom(f"{GRIPPER_PREFIX}{side}_pad{index}").id for index in (1, 2)}
            for side in ("left", "right")
        ]
        self._aperture = self.model.site("laser_aperture").id
        self._spot_mocap = self.model.body("laser_spot").mocapid[0]
        self._substeps = round(1 / (FPS * self.model.opt.timestep))

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.ik.arm_qpos] = HOME_QPOS
        start, _ = self.layout.endpoints
        start_joints = self._target_joints(start)
        self.data.qpos[self.ik.arm_qpos] = start_joints
        self.data.ctrl[self.ik.arm_actuators] = start_joints

        # Pre-position the pads just outside the housing before placing the tool at reset.
        self.data.ctrl[self._gripper_actuator] = 110
        mujoco.mj_step(self.model, self.data, nstep=300)
        mujoco.mj_kinematics(self.model, self.data)
        laser = self.model.joint("laser_joint")
        laser_pos = laser.qposadr[0]
        laser_dof = laser.dofadr[0]
        self.data.qpos[laser_pos : laser_pos + 3] = self.data.site_xpos[self.ik.site_id]
        mujoco.mju_mat2Quat(
            self.data.qpos[laser_pos + 3 : laser_pos + 7],
            self.data.site_xmat[self.ik.site_id],
        )
        self.data.qvel[laser_dof : laser_dof + 6] = 0.0
        self.data.ctrl[self._gripper_actuator] = 255
        mujoco.mj_step(self.model, self.data, nstep=200)
        self._update_spot()

        self._pinch_target_xy = start
        self._spot_correction = np.zeros(2)
        for _ in range(FPS):
            self.step(start)

    def command(self, target_xy: np.ndarray) -> np.ndarray:
        """Set actuator targets and return the LeRobot action for the current observation."""
        # Correct for compliance and tool tilt using the actual projected spot.
        spot = self.data.mocap_pos[self._spot_mocap]
        if spot[2] > TABLE_TOP_Z:
            correction = self._pinch_target_xy - spot[:2]
            self._spot_correction += 0.25 * (correction - self._spot_correction)
        self._pinch_target_xy = target_xy + self._spot_correction
        joints = self._target_joints(self._pinch_target_xy)
        self.data.ctrl[self.ik.arm_actuators] = joints
        return np.array([*joints, 1.0], dtype=np.float32)

    def advance(self) -> None:
        mujoco.mj_step(self.model, self.data, nstep=self._substeps)
        self._update_spot()

    def step(self, target_xy: np.ndarray) -> None:
        self.command(target_xy)
        self.advance()

    def measured_state(self) -> np.ndarray:
        closure = np.clip(self.data.qpos[self._gripper_qpos].mean() / GRIPPER_DRIVER_MAX_RAD, 0, 1)
        return np.array([*self.data.qpos[self.ik.arm_qpos], closure], dtype=np.float32)

    def laser_gripped(self) -> bool:
        contacts = set()
        for contact in self.data.contact:
            if self._laser_geom in contact.geom:
                contacts.update(contact.geom)
        return self._table_geom not in contacts and all(contacts & pad for pad in self._pad_geoms)

    def _target_joints(self, target_xy: np.ndarray) -> np.ndarray:
        return self.ik.solve(
            self.data.qpos, np.array([*target_xy, TRACE_PINCH_HEIGHT_M]), DOWNWARD_PINCH_ROT
        )

    def _update_spot(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        origin = self.data.site_xpos[self._aperture]
        direction = self.data.site_xmat[self._aperture].reshape(3, 3)[:, 2]
        spot = self.data.mocap_pos[self._spot_mocap]
        spot[:] = [0.0, 0.0, -1.0]
        if direction[2] < -1e-6 and origin[2] > TABLE_TOP_Z:
            distance = (TABLE_TOP_Z - origin[2]) / direction[2]
            hit = origin + distance * direction
            table = self.data.geom("table_top")
            table_half_size = self.model.geom("table_top").size[:2]
            if np.all(np.abs(hit[:2] - table.xpos[:2]) <= table_half_size):
                spot[:] = [*hit[:2], TABLE_TOP_Z + 0.0025]
        mujoco.mj_kinematics(self.model, self.data)
