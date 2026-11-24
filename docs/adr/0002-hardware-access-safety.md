# 0002: Hardware access never moves or powers anything unasked

Status: accepted (2026-11-24)

## Context

The UR5e, its gripper and the cameras are shared physical devices. Read-only commands
(`pixi run hw-check`, `set-home`, `set-floor`, `set-neutral`) must be safe to run at any time, and
the grippers have failure modes that latch until power-cycled.

## Decision

### Read-only checks (`scripts/hw_check.py`)

- The robot check uses only the dashboard server's `is in remote control` query. It never powers on,
  releases brakes, or loads or plays a program. The first dashboard line is a greeting and is discarded.
- The gripper check constructs the client but never calls `activate()`, which may move the fingers.
- Each device check catches its own failure and continues, so one missing device does not hide the rest.

### One UR control connection (`ur5e.py`)

- The controller accepts one control connection. `CONNECTED` maps robot IP to the connected `UR5e`, and
  the SpaceMouse teleoperator borrows that robot's connection for IK and homing. So
  `SpaceMouseConfig.robot_ip` must equal the robot's IP.
- Creating `RTDEControlInterface` uploads the ur_rtde control script. The pendant must be in Remote Control.

### Gripper clients (`ur5e.py`, `config_ur5e.py`)

- Every client provides `activate()` (make ready; may move the fingers), `check()` (read-only status, raises
  if not ready), `position()` (0 open .. 1 closed), `grip(closed)` and `close()`.
- `GRIPPER_TYPE`: `robotiq` is the 2F-85 through the Robotiq URCap socket (port 63352). `onrobot_rg2`
  is the RG2 on the tool flange through the OnRobot URCap XML-RPC (port 41414). `none` uses only the six
  arm joints. `gripper_port=None` selects the default for the type.
- Force is 0-255, shared by teleop, recording and rollout. The RG2 scales it linearly to 0-40 N. Unset
  or empty `GRIPPER_FORCE` takes `DEFAULT_GRIPPER_FORCE`: 64 for Robotiq, 128 (about 20 N) for RG2.
  Speed (0-255) applies to the Robotiq only; the RG2 has no speed input.
- RG2: "closed" means `gripper_close_width_mm`, not 0 mm. Closing to 0 mm on nothing trips the S2 safety
  switch and latches the gripper until power-cycled.
- RG2: the XML-RPC URL keeps its trailing slash. Without it, `xmlrpc.client` posts to `/RPC2` and the
  URCap returns 404. Each call opens its own HTTP connection, so `close()` has nothing to release.
- RG2: activation needs no motion. `activate()` only runs `check()`, so a client never starts while
  unpowered or latched.
- RG2: the URCap ignores `rg_grip` while the fingers move. A busy gripper is stopped first, so a
  reversal takes effect at once.
- Robotiq: an object can stop the fingers below `POS=127`. After a policy hands control back,
  teleop keeps the robot's last close *intent* (`_gripper_closed`) rather than re-deriving it from position.

### Homing

- `go_home` releases any held object before moving, so every trial starts at home with the gripper open.
  It first waits for the gripper to reach `HOME_OPEN_POSITION`, and gives up without moving if it does not.
- After `go_home`, teleop sets its gripper command to open, matching the gripper.

### Disconnect

- `SpaceMouse.disconnect` marks itself disconnected even when input cleanup raises, so LeRobot's
  following `robot.disconnect()` still runs.

## Consequences

Adding a read-only command means checking it against this list. A new gripper type must provide the
five-method client interface and state its own latch and activation behaviour here.
