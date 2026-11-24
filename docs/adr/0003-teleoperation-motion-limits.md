# 0003: Motion limits and SpaceMouse control semantics

Status: accepted (2026-11-24)

## Context

The arm is driven by servoJ streaming from a SpaceMouse (teleop, recording) or a policy (rollout).
Limits must hold whatever the input, and must not cause snaps when control changes hands.

## Decision

### Control cadence and step limit (`config_ur5e.py`)

- `control_period_s` must equal 1 / recording fps (`CONTROL_FPS`). servoJ streams at that period.
- `max_joint_step_rad` caps how far any joint *target* may be from the *measured* angle per control
  step. 0.05 rad at 15 Hz is about 0.75 rad/s of target increments. It is not a guarantee on
  measured velocity.

### SpaceMouse input (`spacemouse.py`, `SpaceMouseConfig`)

- Raw HID axes are `x, y, z, roll, pitch, yaw`. Older devices send translation in report 1 (channels
  0-2) and rotation in report 2 (3-5); newer devices send all six in report 1.
- `axis_map` entry *i* is `"<sign><raw channel>"`, mapping raw channels to operator-frame
  `(vx, vy, vz, wx, wy, wz)`.
- `deadband` is a fraction of full deflection; `full_scale` is the raw HID value at full deflection.
- `operator_yaw_deg` rotates the operator's view about base z. At 0, pushing the puck "forward" moves
  along base +x. Set it at bring-up so "forward" is away from the operator.
- `axes`: `xyz_yaw` translates and rotates about vertical, keeping the gripper orientation otherwise;
  `bounded` allows all six axes with roll/pitch limited about a saved neutral; `full` is unrestricted.
- Rotations apply about base axes. The target pose is `[x, y, z, rx, ry, rz]` in the base frame,
  using UR axis-angle rotation vectors. Converting a rotation near 180° takes the axis from the
  matrix's symmetric part.
- Buttons latch a gripper command, initialized from the measured opening. `close_button` and
  `open_button` select the button (0 = left, 1 = right).
- `stop_on_escape` defaults off. The teleop task enables it; record and rollout own their Esc handling.

### Targets and limits

- When idle (and on the first call) the target re-anchors to the measured TCP pose, so nothing winds up.
- `max_lead_m` bounds how far the commanded position may run ahead of the measured one.
- IK solutions further than `max_ik_jump_rad` from the measured joints (any joint) are rejected.
- TCP position bounds are in the base frame, in metres. An empty bound is unbounded; there is no ceiling.
- Inside the box, outward motion is clamped and sliding along a face is allowed. Outside a saved
  boundary, motion inward is allowed gradually, without a snap to the face.

### Teaching boundaries (`scripts/set_boundaries.py`)

- Each face's sign is its inward margin direction. The two captures per side axis are sorted into
  min/max, so left/right and front/back order does not depend on how the base is mounted.
- Input and tilt are checked before a measurement is accepted. `h` recovery stays available.
- Teaching may replace an old box, so it disables only the position bounds. Tilt and joint guards stay on.

## Consequences

Changing the control period, step size or bounds changes recorded data. Recordings and rollouts must
use the same `CONTROL_FPS`.
