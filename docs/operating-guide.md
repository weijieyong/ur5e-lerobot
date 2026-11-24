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
| Pixi and source | Pixi 0.83.0 on PATH and a checkout including `pixi.lock`. Git is needed to clone and run `setup`. |
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

Before hardware checks, confirm the [pendant prerequisite](#pendant-prerequisite)
and complete the configuration step in [First-time bring-up order](#first-time-bring-up-order).

- **Training/inference:** the intended datasets, base model/checkpoint, processor
  files and tokenizer assets must be downloaded or copied separately. The example
  config defaults to `HF_HUB_OFFLINE=1`; an empty cache cannot load a model.
- **Interactive robot commands:** an X11 desktop session for previews and keyboard
  input, reachable UR5e controller in Remote Control with the intended TCP/tool
  and safety settings. Use the URCap for the configured gripper, `robotiq`
  (Robotiq URCap socket, port 63352) or `onrobot_rg2` (OnRobot URCap XML-RPC, port
  41414; the RG2 is powered by the tool flange, so the arm must be powered on).
  Set `GRIPPER_TYPE=none` for six-joint collection.
  The RG2 has no speed input (a full stroke takes about a second); its URCap ignores a
  new grip while the fingers move, so the plugin stops a moving gripper before reversing.
  `GRIPPER_FORCE` (0–255, RG2 0–40 N; empty means 128, about 20 N) applies to teleop,
  recording and rollout alike; the former `ROLLOUT_GRIPPER_FORCE` is still read if it is unset.
- **USB devices:** D455 or D435i (front) and D405 (wrist) cameras on suitable USB 3 connections and SpaceMouse
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
existing values/comments. Pulling and rerunning setup does not migrate existing values:
for example, an old 30-degree tilt limit stays 30 until you edit it to 10.
Edit that private file for the machine; never replace
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

### First-time bring-up order

Follow these steps in order on a new machine, after `pixi run --locked setup`. The sections
below describe each command in detail. Steps 7 to 10 save a pose or move the arm, so keep the
pendant and the emergency stop within reach.

1. Connect the hardware. Plug both cameras into USB 3 ports, plug in the SpaceMouse and
   connect the PC to the robot controller network. The gripper has no cable to the PC. The
   PC reaches it through the URCap on the robot: port 63352 for a Robotiq, port 41414 for an
   RG2.
2. Confirm the [pendant prerequisite](#pendant-prerequisite) and select Remote Control.
3. Find the camera serials with `pixi run lerobot-find-cameras realsense`. See
   [Camera serial numbers](#camera-serial-numbers).
4. Edit `config/local/machine.env`. Set `UR5E_IP`, `GRIPPER_TYPE`, `FRONT_CAMERA_SERIAL`
   (the external camera), `WRIST_CAMERA_SERIAL` (the camera on the gripper), `DATASET_REPO_ID`
   and `TASK`. Leave the home, neutral, floor and side keys alone. Steps 7 to 9 save measured
   values for them.
5. Check the devices. These commands only read from the hardware. They do not command the arm
   or the gripper:

   ```sh
   pixi run hw-check --only spacemouse
   pixi run hw-check --only cameras --preview
   pixi run hw-check
   ```

   Confirm that the front and wrist images come from the right cameras and that each
   SpaceMouse direction matches the description in [Read-only checks](#read-only-checks).
6. Adjust the camera colour with `pixi run calibrate-cameras`. Hold white paper in front of
   each camera. The command asks for y or n before it writes to `machine.env`.
7. Save the home and neutral poses. Move the arm to each pose with the pendant, then run the
   command. Both only read the robot and write to `machine.env`:

   ```sh
   pixi run set-home
   pixi run set-neutral
   ```

   Nothing runs `set-neutral` for you, and it does not change `SPACEMOUSE_AXES`. The default
   `xyz_yaw` locks roll and pitch. To allow limited tilt, run `pixi run set-neutral`
   yourself, then set `SPACEMOUSE_AXES=bounded` in `machine.env`. The `bounded` mode refuses
   to start without a saved neutral.
8. Save the floor with `pixi run set-floor` only if you will skip step 9. Hold the
   fingertips just above the table first, then return the arm to home.
9. Teach the workspace boundaries with `pixi run set-boundaries`. This moves the arm with the
   SpaceMouse and captures the floor, left, right, front and back faces. It replaces
   `UR5E_Z_MIN_M`, so the floor from step 8 does not apply once you save here.
10. Check teleoperation with `pixi run teleop`. Connecting uploads the control script and
   may activate the gripper. Set the low speeds listed under
   [Motion and recovery](#motion-and-recovery) before you start.
11. Record with `pixi run record`, then review the data with `pixi run info` and
    `pixi run view 0`. Training and rollout are in [training and deployment](training.md).

Restart any running command after you change `machine.env` or save a pose.

## Robot preparation and teleoperation

Run interactive commands from the robot PC's **X11 desktop**, with an operator
at the robot and the pendant accessible. Connect the controller network, Robotiq
through its URCap socket setup, both USB cameras and SpaceMouse. Confirm robot IP,
camera identities, task and output dataset in private config. Confirm the
[pendant prerequisite](#pendant-prerequisite), power/brakes and controller safety settings;
select Remote Control before software motion. Keep fingers clear for gripper activation.
Never run teleop, recording, rollout or boundary teaching concurrently against the arm.

### Pendant prerequisite

The robot must already be activated, with the intended TCP, payload, tool and vendor
URCap configured per the UR and gripper-vendor instructions. Recheck floor/workspace
references after changing the TCP or mounted tool.

### Camera serial numbers

With both cameras plugged in and no other program using them, list them:

```sh
pixi run lerobot-find-cameras realsense
```

Each camera prints its model (`Name`) and serial number (`Id`). Put the front camera's (D455 or D435i) serial in
`FRONT_CAMERA_SERIAL` and the D405's in `WRIST_CAMERA_SERIAL` in `config/local/machine.env`.
The command also saves one frame per camera to `outputs/captured_images/realsense_<serial>.png`;
use those to tell the cameras apart if both are the same model.

### Read-only checks

```sh
pixi run hw-check
pixi run hw-check --only cameras --preview --camera-seconds 10
pixi run hw-check --only spacemouse --spacemouse-seconds 20
```

`hw-check` queries the arm, any enabled gripper, cameras and input. It does not
upload an arm script, activate or command the gripper. Failures return nonzero.
The SpaceMouse-only check never connects to the robot. Camera preview closes with
q/Esc; inspect front/wrist identity, view, orientation, lighting and fixed controls.

Move one puck axis at a time. Mapped `v` is robot-base x/y/z in m/s; `w` is rotation
in rad/s. With operator yaw zero, +vx means base +X and +vz means up. The default
HID map (for a SpaceMouse Compact) is `+0 -1 -2 +3 -4 -5`; verify physical directions for the actual setup.
Adjust `SPACEMOUSE_AXIS_MAP` (six distinct signed channels),
`SPACEMOUSE_OPERATOR_YAW_DEG` and close/open button assignments as needed, then
recheck. The **SpaceMouse buttons** default to left = close, right = open during teleoperation.
Recording START/save/discard uses the **keyboard arrow keys**, not these buttons.

### TCP, floor, home and neutral

Stop software motion sessions before capturing references with the pendant:

- Position the arm at the intended episode home, then `pixi run set-home` saves
  six measured joint angles **in degrees, base to wrist 3**. Preserve the chosen
  home unless deliberately replacing it; verify its approach path before using h.
  With a gripper enabled, h first opens it where the arm is
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

Connecting uploads the arm control script and may activate an enabled gripper. Start
with low speeds, for example 0.03 m/s and 0.15 rad/s in
`SPACEMOUSE_LINEAR_SPEED_M_S` / `SPACEMOUSE_ANGULAR_SPEED_RAD_S`. These maxima
apply at full puck deflection to teleop and recording. Preserve deliberate local
settings; restart to apply changes. Keep `CONTROL_FPS=15`.
Gently check x/y/z, yaw, then gripper buttons with empty fingers in clear space.
Test roll/pitch well above the table. Stop and diagnose wrong direction, jerkiness,
rejected commands or cleanup errors before collection.

`SPACEMOUSE_AXES=bounded` clamps neutral-relative roll and pitch separately
(`SPACEMOUSE_TILT_LIMIT_DEG`, initially 10 degrees), with free base-Z yaw:
`R = Rz(yaw) Ry(pitch) Rx(roll) R_neutral`. This is not a total-tilt cone.
`xyz_yaw` locks roll/pitch; `full` removes tilt limits.
`pixi run hw-check --only robot` reports neutral-relative angles without motion.
In bounded mode, check those angles at the saved home pose: its roll and pitch must
fit the configured limit. Home and neutral are separate references.

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
Exposure ceilings default to the current front/wrist exposure settings in `machine.env`
(or the live value if empty). Override them with `--front-max-exposure` and
`--wrist-max-exposure`, for example `pixi run calibrate-cameras --front-max-exposure VALUE`.
Use each sensor's reported range; native units vary by sensor. No calibrated ceiling is supplied.
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

For six-joint collection, set `GRIPPER_TYPE=none` and a fresh `DATASET_REPO_ID`
in `config/local/machine.env`. Restart `pixi run teleop` or `pixi run record` to
apply the setting. These sessions skip all gripper communication, ignore the
SpaceMouse gripper buttons, and move only the arm when you press h. The dataset
stores six measured joint angles in `observation.state` and six absolute joint
targets in `action`, all in radians. Configured cameras still record. The
existing pi0.5 training recipe expects seven values including `gripper.pos`.

| Phase/input | Result |
|---|---|
| READY | Live preview and teleop; no frames saved. Reset/home, then keyboard Right arrow/n starts. |
| RECORDING: keyboard Right arrow/n | Stop and save; wait for READY before resetting. |
| RECORDING: keyboard Left arrow/r | Discard attempt; READY waits at the same episode number. |
| RECORDING: Esc/q or first Ctrl+C | Save buffered attempt and exit; review a partial last episode. |
| READY: Esc/q/Ctrl+C | Exit without an empty episode. |
| SAVING | Wait for finalization; queued inputs are cleared before READY. |

There are no automatic starts or manual-recording phase timers.
READY waits indefinitely in recording and rollout; `RESET_TIME_S` is unused by these wrappers.
The terminal/Rerun show phase and episode; banner colours
are blue READY, orange RECORDING, green SAVING and red DISCARDING. After the
requested saved count the command finalizes/exits. Its owned viewer closes on
exit; older independent viewer windows can be closed manually. A second Ctrl+C
can force interruption, so allow normal saving/cleanup to finish.

`pixi run record` defaults to `RECORD_EPISODES`. Existing datasets require
`pixi run record 1 --resume=true`; the count is **additional** demonstrations.
Use a fresh dataset name for a fresh collection. Review/reject unusable episodes
with the [filtered-copy workflow](training.md#filter-saved-episodes) and replace as needed.
A saved attempt is not automatically an accepted demonstration.

```sh
pixi run timing
pixi run info
pixi run view 0
```

Inspect both videos, task completion, state/action curves and gripper response
if enabled. Expect 15 fps metadata, seven state/action values with the gripper
enabled or six with `GRIPPER_TYPE=none`, and front/wrist images. LeRobot
v3 shares MP4 files across episodes; metadata preserves reader boundaries.
State contains measured joint angles in radians, and action contains six absolute
joint targets. With the gripper enabled, state also contains gripper opening
(0 open, 1 closed) and action also contains the binary gripper command. Robotiq
reports POS/255, and RG2 reports 1 − width/open width (about 0.9 on an empty close,
lower when holding an object). Actions follow that row's observation, and
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

Deploy using [training and deployment](training.md) first. Before motion, compare the
training dataset's `meta/info.json` `fps` with `CONTROL_FPS` (used for control and saved rollout FPS):
all must be 15 for this recipe. Keep `--interpolation_multiplier=1`. Training checks dataset
FPS; rollout does not infer or enforce the checkpoint's collection rate.
With `GRIPPER_TYPE=none`, use a checkpoint trained with six-joint state and action vectors.

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
review episodes afterwards with `pixi run rollout-info` and `pixi run rollout-view 0`.
With spare GPU memory, opt in with
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

Only h in READY queues the slow `go_home` motion on the main control thread.
An enabled gripper opens before the arm moves. Home returns to READY and never
auto-starts. Successful motion alone does not establish the next START check. The wrapper prints and forces both
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
FPS at 15 and `--interpolation_multiplier=1`; do not change FPS to slow playback.
Servo gain/lookahead affect controller response. Chunk size, non-RTC action steps
and RTC execution horizon have different meanings; unequal values are not a bug.
SpaceMouse floor/side/tilt guards do not constrain policy joint actions.

Run short supervised attempts. Success means the object rests on the tray within
30 seconds. Label outcomes manually, retaining failed attempts in success counts;
discard only invalid recordings/setup mistakes. Record attempt conditions and
interventions; no success rate is currently established. Offline loss and command
execution are not physical success measurements.
