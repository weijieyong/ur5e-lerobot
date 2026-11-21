"""Report host-clock timings; these logs include teleoperation and reset phases."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def report(label, values):
    if values:
        median, p95 = np.percentile(values, [50, 95])
        print(
            f"{label}: median={median:.1f} ms  p95={p95:.1f} ms  max={max(values):.1f} ms"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="JSONL log; default: newest in TIMING_LOG_DIR",
    )
    args = parser.parse_args()
    path = args.path
    if path is None:
        directory = os.environ.get("TIMING_LOG_DIR")
        paths = list(Path(directory).glob("ur5e-*.jsonl")) if directory else []
        if not paths:
            parser.error(
                "No timing logs found; enable TIMING_LOG_DIR and run teleop/record first."
            )
        path = max(paths, key=lambda item: item.stat().st_mtime_ns)
    events = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    samples = [
        row
        for row in events
        if row["event"] == "sample" and "observation_start_s" in row
    ]
    if not samples:
        parser.error(f"No observation/action samples in {path}")
    target_ms = samples[0]["control_period_s"] * 1000
    intervals = (
        np.diff([row["observation_start_s"] for row in samples]) * 1000
    ).tolist()
    print(f"{path}\nSamples: {len(samples)}; configured interval: {target_ms:.1f} ms")
    report("Observation-to-observation", intervals)
    report(
        "Observation read",
        [(r["observation_end_s"] - r["observation_start_s"]) * 1000 for r in samples],
    )
    report(
        "Observation start to command start",
        [(r["send_start_s"] - r["observation_start_s"]) * 1000 for r in samples],
    )
    report(
        "Command call", [(r["send_end_s"] - r["send_start_s"]) * 1000 for r in samples]
    )
    print(
        f"Intervals >120% of target: {sum(dt > target_ms * 1.2 for dt in intervals)}/{len(intervals)}"
    )
    print(
        f"Clipped actions: {sum(r['clipped'] for r in samples)}/{len(samples)}; home moves: {sum(r['event'] == 'home' for r in events)}"
    )
    ws_clamped = sum(bool(r.get("workspace_clamped")) for r in samples)
    ws_held = sum(bool(r.get("workspace_held")) for r in samples)
    if ws_clamped or ws_held:
        print(
            f"Workspace clamped: {ws_clamped}/{len(samples)} (held fallback: {ws_held})"
        )
    cameras = sorted({key for row in samples for key in row["camera_ready_s"]})
    for camera in cameras:
        stamps = [r["camera_ready_s"].get(camera) for r in samples]
        ages = [
            (r["observation_end_s"] - stamp) * 1000
            for r, stamp in zip(samples, stamps)
            if stamp is not None
        ]
        report(f"{camera} buffer age at observation end", ages)
        repeated = sum(a is not None and a == b for a, b in zip(stamps, stamps[1:]))
        print(
            f"  Reused consecutive buffers: {repeated}; unavailable timestamps: {stamps.count(None)}"
        )
    report(
        "Camera buffer timestamp skew",
        [
            (max(stamps) - min(stamps)) * 1000
            for row in samples
            if len(
                stamps := [s for s in row["camera_ready_s"].values() if s is not None]
            )
            >= 2
        ],
    )
    print(
        "Includes resets and pauses; gaps are reported, not corrected. Camera times are host buffer-ready times."
    )
    print(
        "This does not measure exposure synchronization or identify dataset episode/frame numbers."
    )


if __name__ == "__main__":
    main()
