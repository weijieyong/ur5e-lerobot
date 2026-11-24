"""Compose the UR5e + Robotiq 2F-85 MuJoCo pick-and-place scene from pinned Menagerie assets."""

from pathlib import Path

import mujoco
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
UR5E_XML = ASSETS_DIR / "universal_robots_ur5e" / "ur5e.xml"
GRIPPER_XML = ASSETS_DIR / "robotiq_2f85" / "2f85.xml"

ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_ACTUATOR_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
GRIPPER_PREFIX = "gripper/"
GRIPPER_ACTUATOR_NAME = f"{GRIPPER_PREFIX}fingers_actuator"
GRIPPER_DRIVER_JOINTS: tuple[str, str] = (
    f"{GRIPPER_PREFIX}right_driver_joint",
    f"{GRIPPER_PREFIX}left_driver_joint",
)
GRIPPER_DRIVER_MAX_RAD = 0.8
PINCH_SITE_NAME = f"{GRIPPER_PREFIX}pinch"

# Fixed tabletop layout (table top surface lies at z = 0.0 m).
TABLE_TOP_Z = 0.0
CUBE_HALF_SIZE = 0.025  # 5 cm red cube; fits cleanly inside 85 mm 2F-85 jaw opening.
CUBE_START_POS = np.array([0.46, -0.14, TABLE_TOP_Z + CUBE_HALF_SIZE], dtype=np.float64)

TRAY_CENTER_XY = np.array([0.46, 0.16], dtype=np.float64)
TRAY_FLOOR_THICKNESS = 0.008
TRAY_TOP_Z = TABLE_TOP_Z + TRAY_FLOOR_THICKNESS
TRAY_INNER_HALF_XY = np.array([0.085, 0.085], dtype=np.float64)
TRAY_WALL_THICKNESS = 0.010
TRAY_WALL_HEIGHT = 0.018

# Fixed episode start arm pose (radians, base -> wrist_3): gripper over center workspace pointing down.
HOME_QPOS = np.array(
    [2.90, -1.35, 1.65, -1.8708, -1.5708, 0.0],
    dtype=np.float64,
)


def _camera_xyaxes(pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> list[float]:
    """Compute MuJoCo camera xyaxes (right_x, up_y) looking from `pos` toward `target`."""
    forward = target - pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    cam_up = cam_up / np.linalg.norm(cam_up)
    return [*right.tolist(), *cam_up.tolist()]


def build_pick_place_model(width: int = 640, height: int = 480) -> mujoco.MjModel:
    """Attach unmodified Menagerie UR5e and Robotiq 2F-85 assets onto a fixed tabletop scene."""
    if not UR5E_XML.is_file() or not GRIPPER_XML.is_file():
        raise FileNotFoundError(
            f"Pinned Menagerie assets missing under {ASSETS_DIR}"
        )

    spec = mujoco.MjSpec.from_file(str(UR5E_XML))
    gripper_spec = mujoco.MjSpec.from_file(str(GRIPPER_XML))

    # Physics options: elliptic friction cone + implicitfast for stable rigid-body grasps.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.timestep = 0.002
    spec.option.noslip_iterations = 3
    spec.option.iterations = 50

    # Ensure offscreen framebuffer fits requested camera dimensions.
    spec.visual.global_.offwidth = max(int(width), 640)
    spec.visual.global_.offheight = max(int(height), 480)
    spec.visual.headlight.ambient = [0.22, 0.22, 0.22]
    spec.visual.headlight.diffuse = [0.42, 0.42, 0.42]
    spec.visual.headlight.specular = [0.15, 0.15, 0.15]

    # Attach Robotiq 2F-85 unmodified onto UR5e's wrist_3 attachment_site.
    attach_site = spec.site("attachment_site")
    if attach_site is None:
        raise RuntimeError("UR5e Menagerie XML is missing attachment_site")
    frame = attach_site.parent.add_frame(
        pos=list(attach_site.pos),
        quat=list(attach_site.quat),
    )
    frame.attach_body(gripper_spec.worldbody, GRIPPER_PREFIX, "")

    # Exclude adjacent mount/flange contacts so wrist_3 does not collide with gripper base.
    spec.add_exclude(bodyname1="wrist_3_link", bodyname2=f"{GRIPPER_PREFIX}base_mount")
    spec.add_exclude(bodyname1="wrist_3_link", bodyname2=f"{GRIPPER_PREFIX}base")

    # Mount the wrist camera on the gripper base looking down toward the fingertips.
    gripper_base = spec.body(f"{GRIPPER_PREFIX}base")
    if gripper_base is None:
        raise RuntimeError("Attached Robotiq 2F-85 is missing gripper/base body")
    wrist_pos = np.array([0.078, 0.0, 0.045], dtype=np.float64)
    wrist_target = np.array([0.0, 0.0, 0.200], dtype=np.float64)
    wrist_up = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    gripper_base.add_camera(
        name="wrist",
        pos=wrist_pos.tolist(),
        xyaxes=_camera_xyaxes(wrist_pos, wrist_target, wrist_up),
        fovy=64.0,
    )

    world = spec.worldbody

    # Overhead studio lighting.
    world.add_light(
        name="top_key",
        pos=[0.5, -0.3, 1.6],
        dir=[-0.1, 0.15, -1.0],
        diffuse=[0.65, 0.65, 0.65],
        specular=[0.20, 0.20, 0.20],
        castshadow=True,
    )
    world.add_light(
        name="fill_light",
        pos=[1.2, 0.6, 1.1],
        dir=[-0.6, -0.3, -0.7],
        diffuse=[0.30, 0.30, 0.30],
        specular=[0.08, 0.08, 0.08],
        castshadow=False,
    )

    # Fixed external front camera viewing the full table area, arm, red cube, and blue tray.
    front_pos = np.array([1.28, 0.0, 0.68], dtype=np.float64)
    front_target = np.array([0.34, 0.0, 0.18], dtype=np.float64)
    world.add_camera(
        name="front",
        pos=front_pos.tolist(),
        xyaxes=_camera_xyaxes(front_pos, front_target, np.array([0.0, 0.0, 1.0])),
        fovy=48.0,
    )

    # Floor and tabletop (table top surface at z = 0.0).
    world.add_geom(
        name="room_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, -0.75],
        size=[2.5, 2.5, 0.1],
        rgba=[0.72, 0.74, 0.78, 1.0],
        contype=0,
        conaffinity=0,
    )
    table = world.add_body(name="table", pos=[0.35, 0.0, -0.025])
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.65, 0.65, 0.025],
        rgba=[0.52, 0.48, 0.43, 1.0],
        friction=[0.8, 0.01, 0.001],
        solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        solref=[0.004, 1.0],
    )

    # Blue tray with flat floor and low retaining rim.
    tray = world.add_body(
        name="blue_tray",
        pos=[float(TRAY_CENTER_XY[0]), float(TRAY_CENTER_XY[1]), TABLE_TOP_Z],
    )
    blue_rgba = [0.14, 0.36, 0.78, 1.0]
    outer_hx = float(TRAY_INNER_HALF_XY[0] + TRAY_WALL_THICKNESS)
    outer_hy = float(TRAY_INNER_HALF_XY[1] + TRAY_WALL_THICKNESS)
    tray.add_geom(
        name="tray_floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, TRAY_FLOOR_THICKNESS / 2.0],
        size=[outer_hx, outer_hy, TRAY_FLOOR_THICKNESS / 2.0],
        rgba=blue_rgba,
        friction=[0.9, 0.01, 0.001],
        solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        solref=[0.004, 1.0],
    )
    wall_z = TRAY_FLOOR_THICKNESS + TRAY_WALL_HEIGHT / 2.0
    rim_rgba = [0.10, 0.28, 0.68, 1.0]
    for suffix, pos, size in (
        (
            "north",
            [outer_hx - TRAY_WALL_THICKNESS / 2.0, 0.0, wall_z],
            [TRAY_WALL_THICKNESS / 2.0, outer_hy, TRAY_WALL_HEIGHT / 2.0],
        ),
        (
            "south",
            [-(outer_hx - TRAY_WALL_THICKNESS / 2.0), 0.0, wall_z],
            [TRAY_WALL_THICKNESS / 2.0, outer_hy, TRAY_WALL_HEIGHT / 2.0],
        ),
        (
            "east",
            [0.0, outer_hy - TRAY_WALL_THICKNESS / 2.0, wall_z],
            [float(TRAY_INNER_HALF_XY[0]), TRAY_WALL_THICKNESS / 2.0, TRAY_WALL_HEIGHT / 2.0],
        ),
        (
            "west",
            [0.0, -(outer_hy - TRAY_WALL_THICKNESS / 2.0), wall_z],
            [float(TRAY_INNER_HALF_XY[0]), TRAY_WALL_THICKNESS / 2.0, TRAY_WALL_HEIGHT / 2.0],
        ),
    ):
        tray.add_geom(
            name=f"tray_wall_{suffix}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=rim_rgba,
            friction=[0.8, 0.01, 0.001],
        )

    # Free-body red cube with genuine rigid-body mass and contact friction.
    cube = world.add_body(name="red_cube", pos=CUBE_START_POS.tolist())
    cube.add_freejoint(name="red_cube_freejoint")
    cube.add_geom(
        name="red_cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE],
        mass=0.06,
        rgba=[0.85, 0.12, 0.12, 1.0],
        friction=[1.2, 0.02, 0.002],
        condim=4,
        solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        solref=[0.004, 1.0],
    )

    # Remove the UR5e-only keyframe whose qpos length does not include gripper/cube DOFs.
    for key in list(spec.keys):
        spec.delete(key)

    return spec.compile()
