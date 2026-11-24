# 0006: Camera calibration waits for firmware and tunes on a midtone

Status: accepted (2026-11-24)

## Decision (`scripts/calibrate_cameras.py`)

- After each control write, several frames are discarded. RealSense firmware controls and queued
  frames need time to catch up, and an early frame shows the old settings.
- Brightness is tuned to a midtone target (at most 160) on the white-paper patch. That leaves headroom
  to compare colour temperatures without clipping.

## Consequences

Calibration takes a few seconds per setting. Raising the brightness target above 160 makes white-balance
comparison unreliable.
