# Glossary

Terms used across the docs and code. A supervised, real-hardware workflow: teleoperate a UR5e
(Robotiq 2F-85 or OnRobot RG2 gripper) to collect demonstrations of one pick-and-place task,
fine-tune a policy on them (currently pi0.5), and run that policy on the same robot.

## Task

**Task instruction**:
The natural-language sentence describing the task, given to the policy with every observation.
Set by `TASK`; a rollout may edit it while READY.
_Avoid_: prompt, command, goal

**Object**:
The single item that is picked up.
_Avoid_: target, item, cube (unless it literally is one)

**Tray**:
The flat surface the object is placed on.
_Avoid_: bin, container, bowl, goal

**Table area**:
The marked region of the table within which the object and tray are placed at the start of an episode.
_Avoid_: workspace (that is the TCP box below), scene, cell

**Home pose**:
The single fixed arm configuration every episode starts from, measured with `pixi run set-home`.
_Avoid_: start pose, ready pose, rest position

**Reset phase**:
The unrecorded interval between episodes in which the operator returns the arm home and rearranges the object and tray.
_Avoid_: reset time, break

## Safety references

**Floor**:
The minimum TCP height in the robot base frame: the measured TCP z plus a clearance margin
(`pixi run set-floor`).
_Avoid_: table height, z limit

**Workspace**:
The box the TCP must stay inside: the floor plus four optional side bounds, taught with
`pixi run set-boundaries`. There is no ceiling.
_Avoid_: boundaries (that is the teaching step), table area

**Neutral**:
The fixed TCP orientation that SpaceMouse roll and pitch limits are measured from
(`pixi run set-neutral`).
_Avoid_: zero orientation, home orientation

## Data

**Demonstration**:
One teleoperated, recorded attempt at the task, from the home pose to the object resting on the tray.
_Avoid_: trajectory, demo episode, sample

**Rollout**:
One attempt at the task driven by the policy on the real robot.
_Avoid_: evaluation, inference run, deployment run

**Episode**:
The recorded unit of either a demonstration or a rollout.
_Avoid_: trial, run

**Robot state**:
The measured arm joint angles plus the measured gripper opening at one instant.
_Avoid_: proprio, qpos, joint state

**Action**:
What the robot is told to do for one control step: an absolute target angle for each of the six arm joints plus a gripper command.
_Avoid_: delta, twist, command

**Gripper command**:
The desired gripper state, either open or closed, that is part of an action.
_Avoid_: gripper action, gripper target

**Gripper opening**:
The gripper's measured finger position, as reported by the gripper itself.
_Avoid_: gripper state, gripper width

**Action chunk**:
A sequence of consecutive actions predicted by the policy in one inference call.
_Avoid_: horizon, trajectory

**Training statistics**:
State and relative-arm/absolute-gripper normalization fitted automatically from
training episodes at startup. Raw recording bytes stay unchanged.
See [training](docs/training.md).

## Recording and rollout states

**READY**:
Live and idle: the arm holds still or is teleoperated, nothing is saved, and in a rollout no policy actions are sent.
Home (h) and task-instruction edits happen here.
_Avoid_: idle, paused (except as shown on screen: READY — POLICY PAUSED)

**START**:
The operator action (Right arrow or n) that leaves READY and begins recording a demonstration or running the policy.
_Avoid_: go, play

**RTC**:
Real-time chunking: while the robot executes the current action chunk, the policy computes the next one,
conditioned on the actions it has already committed to, so motion stays continuous across inference calls.
`RTC_EXECUTION_HORIZON` sets that committed prefix.
_Avoid_: async inference, streaming

## Machines

**Robot PC**:
The machine wired to the robot, cameras and SpaceMouse that runs teleoperation, recording and rollouts.
_Avoid_: control PC, client, host

**Training machine**:
The machine that runs fine-tuning. It can be the robot PC if that has enough GPU memory.
_Avoid_: server, cluster, cloud
