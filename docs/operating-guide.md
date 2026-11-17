# Operating guide

Start at the [README](../README.md). This guide owns setup, calibration, collection and supervised rollout.

## Setup

### Requirements outside Pixi

> [!NOTE]
> Pixi installs the whole software stack (Python, PyTorch, CUDA libraries, LeRobot, the
> local plugin); no host Python, ROS, Conda or CUDA toolkit is needed. This section lists
> only what Pixi cannot provide: OS, NVIDIA driver, network and storage.

Tested on **Ubuntu 22.04 x86-64** (robot PC and training machine: full workflow) with
**Pixi 0.83.0 and a working NVIDIA GPU/driver**. `pixi run smoke-docker` additionally
checks a clean install and the offline tests in an Ubuntu 24.04 container. With the source
checkout and network/storage below, Pixi installs the locked software environment.
Other Linux distributions, ARM, macOS and Windows/WSL have not been validated.

| Requirement | What to provide on the host |
| --- | --- |
| OS | Ubuntu 22.04 x86-64 (robot PC and training machine: full workflow); Ubuntu 24.04 install and offline tests in the clean container. |
| Pixi and source | Pixi 0.83.0 on PATH and a checkout including `pixi.lock`. Git is needed to clone, not to run the tasks. |
| NVIDIA GPU/driver | A driver that supports the GPU and the locked CUDA 12.9 runtime. The clean-container test passed on RTX 5080 / driver 595.99.02; this is a tested combination, not a minimum driver version. |
| GPU capacity | pi0.5 training/inference uses bfloat16 and needs enough VRAM for the chosen workload. The small CUDA test establishes neither policy fit nor a minimum VRAM/RAM requirement. |
| Network and certificates | Internet access with working HTTPS trust to download Pixi and packages on first setup. Model downloads are separate; an installed environment with cached assets can be used offline. |
| Storage | Writable native Linux storage for the checkout and caches. The installed environment was about 13 GB; budget 30 GB+ for dependencies/cache, plus model weights, datasets and outputs. This is a planning allowance, not a measured minimum. |

Docker and NVIDIA Container Toolkit are needed only to repeat the container smoke
test, not for normal native `pixi run` use.

From the project directory (the first command also installs the environment):

```sh
pixi run --locked setup
pixi run --locked test
```

### Additional requirements for live use

- **Training/inference:** the intended datasets, base model/checkpoint, processor
  files and tokenizer assets must be downloaded or copied separately. The example
  config defaults to `HF_HUB_OFFLINE=1`; an empty cache cannot load a model.
- **Interactive robot commands:** an X11 desktop session for previews and keyboard
  input, reachable UR5e controller in Remote Control with the intended TCP/tool
  and safety settings, and the URCap for the configured `GRIPPER_TYPE`: `robotiq`
  (Robotiq URCap socket, port 63352) or `onrobot_rg2` (OnRobot URCap XML-RPC, port
  41414; the RG2 is powered by the tool flange, so the arm must be powered on).
  The RG2 has no speed input (a full stroke takes about a second); its URCap ignores a
  new grip while the fingers move, so the plugin stops a moving gripper before reversing.
  `GRIPPER_FORCE` (0–255, RG2 0–40 N; empty means 128, about 20 N) applies to teleop,
  recording and rollout alike; the former `ROLLOUT_GRIPPER_FORCE` is still read if it is unset.
- **USB devices:** D455/D405 cameras on suitable USB 3 connections and SpaceMouse
  access as the logged-in user. Host udev/device permissions are outside Pixi;
  use the SpaceMouse rule linked below and check camera access with `hw-check`.
- **Machine configuration:** correct IPs, camera serials, paths and measured
  home/floor/boundary/neutral references in `config/local/machine.env`. Example
  values do not establish a calibrated or safe robot setup.

> [!CAUTION]
> Closing the RG2 to 0 mm latches its safety switch, which only a Power cycle in the
> OnRobot toolbar clears. `GRIPPER_CLOSE_WIDTH_MM` (default 8) keeps closes above 0 mm.

`pixi run smoke-docker` ([details](../docker/smoke/README.md)) checks a locked
install, config creation, CUDA float32/bfloat16 calculations and the offline tests in a
clean container, on a GPU host with Docker and NVIDIA Container Toolkit.
It shares the host kernel/NVIDIA driver; it does not test policy loading, training,
GUI or USB/robot operation. Follow the sections below for those task-specific
preparations.

If Pixi is absent from PATH on the robot PC, use `~/.pixi/bin/pixi`.
`setup` creates `config/local/machine.env` from
[the example](../config/machine.env.example), or adds missing defaults while preserving
existing values/comments. Edit that private file for the machine; never replace
an existing calibration with example values. Camera colour controls and the TCP
floor in the example are placeholders and need measurement/review before live use.
Restart camera/robot commands after configuration changes.

[Pixi tasks](../pixi.toml) are the command definitions; [pixi.lock](../pixi.lock) pins the
runtime. Activation loads private settings and empties inherited `PYTHONPATH`,
`PYTHONHOME` and `LD_LIBRARY_PATH`, disabling user-site packages. No ROS/catkin,
host Conda/venv or local shell customization is required. Keep the pinned stack:
PyPI pyrealsense2 supplies distribution metadata required by LeRobot; the local
camera factory handles D405 RGB-only sensor selection. SpaceMouse uses hidraw;
see [the udev rule](../config/udev/99-3dconnexion.rules) for device permissions.

Use `pixi install --locked` to reject lock drift, `pixi run --frozen <task>` for
an existing lock, and `pixi run --as-is --dry-run <task>` to inspect a command.
`pixi run test` runs offline regressions with hardware stand-ins, including the
native recording loop and environment isolation; it does not validate physical safety.
Do not run `setup` in a linked worktree: it changes shared Git hook configuration.
`pixi run setup` also points git at `.githooks/`, whose pre-commit hook runs
`pixi run lint` and `pixi run test`.

For source transfer, choose the destination host/path explicitly for your installation.
Exclude `.pixi`, `.cache`, `.git`, `config/local`, `hf`, `data` and `outputs`;
preserve private calibration and transfer complete datasets/models separately with
hash manifests. Use native Linux storage and enough free space for caches/temp files.
Run locked installation and setup only as part of an authorized source update.

## Robot preparation and teleoperation

Run interactive commands from the robot PC's **X11 desktop**, with an operator
at the robot and the pendant accessible. Connect the controller network, Robotiq
through its URCap socket setup, both USB cameras and SpaceMouse. Confirm robot IP,
camera identities, task and output dataset in private config. On the pendant,
confirm the active TCP/tool definition, power/brakes and controller safety settings;
select Remote Control before software motion. Keep fingers clear for gripper activation.
Never run teleop, recording, rollout or boundary teaching concurrently against the arm.

### Read-only checks

```sh
pixi run hw-check
pixi run hw-check --only cameras --preview --camera-seconds 10
pixi run hw-check --only spacemouse --spacemouse-seconds 20
```

`hw-check` queries arm/gripper, cameras and input; it does not upload an arm script,
activate or command the gripper. Failures return nonzero.
The SpaceMouse-only check never connects to the robot. Camera preview closes with
q/Esc; inspect front/wrist identity, view, orientation, lighting and fixed controls.

Move one puck axis at a time. Mapped `v` is robot-base x/y/z in m/s; `w` is rotation
in rad/s. With operator yaw zero, +vx means base +X and +vz means up. The default
HID map (for a SpaceMouse Compact) is `+0 -1 -2 +3 -4 -5`; verify physical directions for the actual setup.
Adjust `SPACEMOUSE_AXIS_MAP` (six distinct signed channels),
`SPACEMOUSE_OPERATOR_YAW_DEG` and close/open button assignments as needed, then
recheck. Left defaults to close, right to open during teleoperation.

### TCP, floor, home and neutral

Stop software motion sessions before capturing references with the pendant:

- Position the arm at the intended episode home, then `pixi run set-home` saves
  six measured joint angles **in degrees, base to wrist 3**. Preserve the chosen
  home unless deliberately replacing it; verify its approach path before using h.
  In teleop, record and rollout, h first opens the gripper where the arm is
  (releasing anything held), waits up to 3 s for it to open, then moves home; if it
  does not open the arm stays put and the command stops with an error.
- Confirm the active TCP, place fingertips just above the table in the intended
  orientation, then `pixi run set-floor` saves measured **base-frame TCP z in metres**
  plus `UR5E_FLOOR_MARGIN_M` (default 0.01 m). Return to home with the pendant.
  Retain an existing floor only if tool/TCP, orientation and table reference remain valid.
- At the intended neutral posture, `pixi run set-neutral` saves the UR rotation
  vector **in radians, not Euler angles**. `SPACEMOUSE_NEUTRAL_ROTVEC` stays fixed
  across sessions/episodes; home does not redefine it.

These commands read measurements and update local configuration without commanding
motion. Restart teleop/record/rollout after reference changes.

### Motion and recovery

```sh
pixi run teleop
```

Connecting uploads the arm control script and may activate the gripper. Start
with low speeds, for example 0.03 m/s and 0.15 rad/s in
`SPACEMOUSE_LINEAR_SPEED_M_S` / `SPACEMOUSE_ANGULAR_SPEED_RAD_S`. These maxima
apply at full puck deflection to teleop and recording. Preserve deliberate local
settings; restart to apply changes. Keep `CONTROL_FPS=15`.
Gently check x/y/z, yaw, then gripper buttons with empty fingers in clear space.
Test roll/pitch well above the table. Stop and diagnose wrong direction, jerkiness,
rejected commands or cleanup errors before collection.

`SPACEMOUSE_AXES=bounded` clamps neutral-relative roll and pitch separately
(`SPACEMOUSE_TILT_LIMIT_DEG`, initially 30 degrees), with free base-Z yaw:
`R = Rz(yaw) Ry(pitch) Rx(roll) R_neutral`. This is not a total-tilt cone.
`xyz_yaw` locks roll/pitch; `full` removes tilt limits.
`pixi run hw-check --only robot` reports neutral-relative angles without motion.

An outside-range posture holds puck targets instead of snapping to neutral.
RECOVERY permits h with a clear path or pendant repositioning. Teleop resumes
when in range; recording/rollout still require Right. A recording tilt error
discards the unsaved attempt only. Other hardware/input errors propagate.
Esc/Ctrl+C stops normal streaming.

> [!WARNING]
> Homing is a blocking joint move: only the pendant stop/e-stop interrupts it. Use h only
> while READY, never inside a recorded demonstration or while the policy runs.

### Teach floor and side boundaries

```sh
pixi run set-boundaries
```

Stop other control sessions. Enter enables SpaceMouse teaching, temporarily
disabling saved floor/side guards; tilt and joint guards remain. Teaching speeds
are capped by `BOUNDARY_LINEAR_SPEED_M_S` (0.03 m/s) and
`BOUNDARY_ANGULAR_SPEED_RAD_S` (0.15 rad/s), never exceeding normal SpaceMouse speeds.
Move to each face, release the puck, then Enter: **floor, left, right, front, back**.
Each X or Y pair is sorted into min/max, so either side order works whichever way the
base is mounted. Esc/Ctrl+C
cancels without saving. There is no ceiling.

After the fifth capture the robot disconnects. Review the proposed bounds; y saves
with a config backup and concurrent-edit protection. `UR5E_WORKSPACE_MARGIN_M`
(default 0.01 m) insets all five faces, separately from the `set-floor` clearance.
Saved keys are `UR5E_X_MIN_M`, `UR5E_X_MAX_M`, `UR5E_Y_MIN_M`, `UR5E_Y_MAX_M` and
`UR5E_Z_MIN_M`; empty sides are unbounded. Boxes too narrow for the clearance are rejected.
Restart to apply. Outward targets clamp; tangential/inward motion and gradual
recovery from an outside start remain possible.

> [!WARNING]
> Floor, side and tilt guards constrain SpaceMouse TCP targets only. They do not
> constrain joint-space homing or policy actions, the whole arm/tool or swept paths.
> Tilting can lower the gripper body while TCP height stays fixed. Controller safety,
> collision clearance, home-path safety and emergency-stop response need physical
> validation; working collection and offline tests do not establish them.

### Camera colour adjustment

Close other camera users, then run `pixi run calibrate-cameras` on the X11 desktop.
Put white paper in each view, select a patch and press Enter; Esc cancels selection.
The helper recommends exposure/gain/white balance and shows both final tuned images.
Default is no write; explicit y/yes saves six controls, preserves other settings,
backs up config and refuses concurrent edits. Review brightness/clipping warnings
before accepting; acceptance is not independent calibration validation.
Original live controls are restored and cameras close on exit; restart commands
to apply saved settings. Reports/images remain under `outputs/camera-calibration/`.
This adjusts colour, not camera geometry or hand-eye calibration.

## Collection and review

Stop teleop, put the robot at home and reset the scene. For a new dataset:

```sh
pixi run record 3
```

| Phase/input | Result |
|---|---|
| READY | Live preview and teleop; no frames saved. Reset/home, then Right/n starts. |
| RECORDING: Right/n | Stop and save; wait for READY before resetting. |
| RECORDING: Left/r | Discard attempt; READY waits at the same episode number. |
| RECORDING: Esc/q or first Ctrl+C | Save buffered attempt and exit; review a partial last episode. |
| READY: Esc/q/Ctrl+C | Exit without an empty episode. |
| SAVING | Wait for finalization; queued inputs are cleared before READY. |

There are no automatic starts or manual-recording phase timers.
`RECORD_EPISODE_TIME_S` and `RESET_TIME_S` do not time the manual recording phases. The terminal/Rerun show phase and episode; banner colours
are blue READY, orange RECORDING, green SAVING and red DISCARDING. After the
requested saved count the command finalizes/exits. Its owned viewer closes on
exit; older independent viewer windows can be closed manually. A second Ctrl+C
can force interruption, so allow normal saving/cleanup to finish.

`pixi run record` defaults to `RECORD_EPISODES`. Existing datasets require
`pixi run record 1 --resume=true`; the count is **additional** demonstrations.
Use a fresh dataset name for a fresh collection. Review/reject unusable episodes
with native LeRobot tools and replace as needed; a saved attempt is not automatically
an accepted demonstration.

```sh
pixi run timing
pixi run info
pixi run view 0
```

Inspect both videos, task completion, gripper response and state/action curves.
Expect 15 fps metadata, seven state/action values and front/wrist images. LeRobot
v3 shares MP4 files across episodes; metadata preserves reader boundaries.
State is measured joint radians plus gripper opening (0 open, 1 closed; Robotiq POS/255,
RG2 1 − width/open width, about 0.9 on an empty close, lower when holding an object); action is six absolute joint
targets plus binary gripper intent. Actions follow that row's observation, and
measured motion may lag. The plugin records clipped targets actually dispatched,
relying on the pinned default processors preserving the action dictionary; rerun
the recording regression if processors/LeRobot change.

`TIMING_LOG_DIR` holds one JSONL per robot session (empty disables logging).
`pixi run timing /path/to/log.jsonl` selects a log; otherwise it reads the latest.
It reports intervals, command duration, camera buffer age/skew, reuse and clipping.
`CAMERA_MAX_AGE_MS` defaults to 100 ms. Logs include resets/home/save pauses and
lack dataset frame mapping. Dataset timestamps are synthetic `frame_index / fps`;
buffer timestamps do not measure exposure synchronization. Recheck timing after
camera/control changes. A recovered D405 warmup retry was seen during initial setup;
repeated startup failures need investigation.

## Supervised RTC rollout

Deploy using [training and deployment](training.md) first.

Stop teleop/recording and use a focused POSIX terminal with usable stdin/termios.
Place the robot at the demonstrated home joint configuration with a clear
workspace and pendant available. Rollout requires no SpaceMouse or global keyboard
listener; an explicit `--teleop` or `--interactive=true` is rejected before setup.

```sh
pixi run rollout 1
```

The number means additional **saved rollout episodes**, not a checkpoint selector.
Use a separate dataset name starting with `rollout_` in `ROLLOUT_DATASET_REPO_ID`;
choose a fresh name per session or `pixi run rollout 1 --resume=true` to append.
Model loading typically takes 60–70 seconds once per command; episodes reuse it.

GPU memory: OpenPI states pi0.5 inference needs more than 8 GB, with an RTX 4090 (24 GB)
as its example GPU; LeRobot's pi0.5 page gives no figure. Use a 24 GB-class card. Measured
here, rollout peaks at about 8.9 GiB, so a 10 GB RTX 3080 runs it with little headroom: the
live Rerun viewer (about 0.3 GiB) is enough to run out of memory. Rollout therefore runs
without the live viewer and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
review episodes afterwards with `pixi run view`. With spare GPU memory, opt in with
`pixi run rollout 1 --display_data=true`. A command that fails after creating its dataset leaves a stub folder, and
the next run stops with `FileExistsError`: delete it only after its `meta/info.json` shows
`total_episodes` 0, or choose a fresh name.

READY — POLICY PAUSED displays the full current instruction, initially the resolved
`dataset.single_task` (`TASK`). Press n/Right to START immediately; editing is optional.
Every unedited START reuses the selected instruction, including after save/discard.
While RUNNING, n/Right stops and saves, r/Left stops and discards. q/Esc or Ctrl+C
exits; wait for saving and teardown. These software controls are not physical
emergency stops. No policy or teleop actions are sent in READY. The model stays
resident across trials and the full active instruction is shown while RUNNING.

In READY, t opens a fresh replacement draft. Type printable text (including normal
composed UTF-8), spaces and Backspace; Backspace removes one code point. Enter
accepts a nonempty valid draft. Empty Enter or Esc cancels and keeps the previous
instruction. Enter never starts or homes. h/n/r/q are literal text in the editor;
arrows never issue episode commands. Invalid bytes, controls, unsupported escapes
or more than 1024 code points reject the draft and preserve the previous task.
There is no cursor editing, history, completion or clipboard feature. Bracketed
paste frames are dropped in every phase, including embedded command keys/newlines;
a paste into a draft rejects that draft. A lone Escape currently uses a 50 ms
recognition timeout: if a framed paste pauses immediately after its first Escape,
that byte can cancel the editor or request exit before the marker is recognized.
This pathological paste timing is unsupported and may cancel/exit. Late markers
and their payload are still consumed: they cannot home, START, save, discard or
replace the current task. Split markers after the recognized introducer remain
buffered and discarded; arbitrary-delay paste protection is outside this typing-only workflow.
**Unbracketed paste is unsupported**:
slow pasted characters cannot be distinguished from typing. Keep terminal paste
framing enabled. Terminal mode/framing is restored on exit; loss of the input
reader ends the session visibly rather than leaving an apparently usable READY.
t/h during running, saving or discarding are ignored. Editing never retargets an
active trial; native recording labels use the engine's dispatched task.

Immediately before each accepted START, rollout reads the underlying UR5e joint
coordinates in radians and compares each directly against the six finite configured
`robot.home_joints_deg` values. The nominal `ROLLOUT_START_TOLERANCE_DEG` defaults
to 5 degrees per joint (finite and greater than zero). Equivalent angles differing
by a full turn do not count as the same configuration. Outside this tolerance,
policy stays paused: use explicit h with a clear path or reset on the pendant,
then issue a fresh START. Invalid measurements and read/motion failures terminate
with an error. The tolerance is an adjustable application guard, not a safety-rated
threshold; it and the defaults require operator lab validation.

Only h in READY queues the existing slow `go_home` motion (gripper opens first),
executed by the main control thread. Home returns to READY and never auto-starts; successful motion
alone does not establish the next START check. The wrapper prints and forces both
native automatic reset-to-initial and exit-return settings false. Starting, saving,
discarding and exiting never automatically home. After the final trial the process
exits at its final pose. To home afterward, start teleop and press h with a clear
path. Treat stop/cleanup warnings as faults; a logged attempt does not establish
completed shutdown. Record/teleop/boundary teaching retain their existing controls.

Rollout-only settings live in private config; these are **example defaults**, not
verified current machine values:

| Setting | Example | Meaning |
|---|---:|---|
| `ROLLOUT_MAX_JOINT_STEP_RAD` | 0.01 | Target distance from each measured joint per send |
| `ROLLOUT_GRIPPER_SPEED` | 128 | 0–255; Robotiq only (RG2 has no speed input) |
| `GRIPPER_FORCE` | empty | 0–255, shared with teleop and recording; empty means 64 (Robotiq) or 128 (RG2, about 20 N of 0–40 N) |
| `ROLLOUT_HOME_SPEED_RAD_S` | 0.1 | Explicit h move in READY |
| `ROLLOUT_START_TOLERANCE_DEG` | 5 | Nominal per-joint START guard; environment override |
| `ROLLOUT_EPISODE_TIME_S` | 30 | Time after Right/START, including initial queue fill |
| `ROLLOUT_EPISODES` | 10 | Additional saved episodes if no count is passed |
| `RTC_EXECUTION_HORIZON` | 10 | RTC prefix/overlap setting |
| `ROLLOUT_DENOISE_STEPS` | 10 | Diffusion compute/quality tradeoff |

Collection defaults to a 0.05-rad joint target cap; rollout defaults to 0.01 rad.
These are offsets from freshly measured joints, not velocity limits or a fixed
physical-speed ratio. Clipping can alter tracking/grasp timing. Keep control/data
FPS at 15 and interpolation multiplier 1; do not change FPS to slow playback.
Servo gain/lookahead affect controller response. Chunk size, non-RTC action steps
and RTC execution horizon have different meanings; unequal values are not a bug.
SpaceMouse floor/side/tilt guards do not constrain policy joint actions.

Run short supervised attempts. Success means the object rests on the tray within
30 seconds. Label outcomes manually, retaining failed attempts in success counts;
discard only invalid recordings/setup mistakes. Record attempt conditions and
interventions; no success rate is currently established. Offline loss and command
execution are not physical success measurements.
