# 0007: Tests fake hardware only and use independent oracles

Status: accepted (2026-11-24)

## Decision (`tests/`)

- Test doubles stand in for external hardware and writers only. Recording and record-feedback tests
  use the real plugin and LeRobot's actual record loop, processors and episode state machine.
- Expected values come from an independent construction, not from the code under test. Examples:
  orientation tests build known ZYX rotations from degrees; training-stats tests enumerate complete
  windows separately (no cross-episode or padded starts).
- Fixtures are shaped so a plausible bug cannot pass. Boundary captures use different values on every
  axis, so a swapped capture axis is caught. Stats fixtures push the sixth arm joint beyond [0, 1], so
  an accidental gripper range check is caught.
- Tests use 127.0.0.1 and the RFC 5737 documentation range 192.0.2.x. RFC 1918 addresses are treated as
  machine-private and fail `tests/test_public_hygiene.py`.
- The CPU-only environment test reuses the parent's selected Pixi workspace target, for hosts where CUDA
  virtual-package detection is unavailable.
