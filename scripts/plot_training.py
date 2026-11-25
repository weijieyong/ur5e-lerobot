#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly>=6,<7",
# ]
# ///
"""Build a self-contained HTML report for selecting saved LeRobot checkpoints.

Usage:
    pixi run plot-training RUN_DIR [RUN_DIR ...] -o REPORT.html

Reads run.json, status.json, train.out, and native/checkpoints/ metadata
without loading or requiring model weights.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from plotly.offline import get_plotlyjs

PALETTE = ("#2563eb", "#ea580c", "#0d9488", "#7c3aed", "#dc2626", "#65a30d")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LOG_EVENT_RE = re.compile(
    r"(?P<tqdm>Training:\s*[\d.]+%\|[^|\r\n]*\|\s*(?P<t_step>\d+)/(?P<t_total>\d+)\b)"
    r"|(?P<info>INFO\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+(?P<i_body>[^\r\n]+))"
)
EVAL_RE = re.compile(r"^step\s+(?P<step>\d+):\s*eval_loss=(?P<loss>\S+)")
SPLIT_RE = re.compile(
    r"^Train/eval split:\s*(?P<train_ep>\d+)\s+train,\s*(?P<eval_ep>\d+)\s+eval"
    r"\s*\(eval_split=(?P<eval_split>[\d.]+)"
)
KV_RE = re.compile(r"([A-Za-z0-9_/]+):([^\s]+)")
ROUNDED_STEP_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)(?P<suffix>[KMB])$", re.IGNORECASE)
STAMP_RE = re.compile(r"^(?P<head>\d{8}T\d{6})\.\d+(?P<tail>Z.*)$")


def parse_float(token: str | None) -> float | None:
    if token is None:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def rounded_step_matches(step_token: str, candidate: int) -> bool:
    if step_token.isdigit():
        return int(step_token) == candidate
    match = ROUNDED_STEP_RE.match(step_token)
    if not match:
        return False
    num_str = match.group("num")
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[match.group("suffix").upper()]
    decimals = len(num_str.split(".", 1)[1]) if "." in num_str else 0
    tol = 0.5 * mult / (10**decimals) + 1e-9
    return abs(candidate - float(num_str) * mult) <= tol


def read_json_file(path: Path, warnings: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        warnings.append(f"Missing {label}.")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Unreadable {label}: {exc}")
        return {}


def parse_train_log(log_path: Path, warnings: list[str]) -> dict[str, Any]:
    """Parse train.out using preceding TQDM counts for exact steps on rounded INFO lines."""
    out: dict[str, Any] = {
        "train_points": [],
        "eval_points": [],
        "split_text": None,
        "max_tqdm_step": None,
        "tqdm_total": None,
    }
    if not log_path.is_file():
        warnings.append("Missing train.out; no loss or LR curves available.")
        return out

    with log_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        text = ANSI_RE.sub("", fh.read())

    train_by_step: dict[int, dict[str, Any]] = {}
    eval_by_step: dict[int, dict[str, Any]] = {}
    last_tqdm: int | None = None
    tqdm_fresh = False
    unresolved = 0

    for m in LOG_EVENT_RE.finditer(text):
        if m.group("tqdm") is not None:
            t_step = int(m.group("t_step"))
            last_tqdm = t_step
            tqdm_fresh = True
            if out["max_tqdm_step"] is None or t_step > out["max_tqdm_step"]:
                out["max_tqdm_step"] = t_step
            out["tqdm_total"] = int(m.group("t_total"))
            continue

        body = m.group("i_body").strip()
        ev = EVAL_RE.match(body)
        if ev:
            eloss = parse_float(ev.group("loss"))
            if eloss is not None:
                step = int(ev.group("step"))
                eval_by_step[step] = {"step": step, "eval_loss": eloss}
            continue

        if body.startswith("step:") and " loss:" in body:
            kv = dict(KV_RE.findall(body))
            step_tok = kv.get("step", "")
            if step_tok.isdigit():
                exact_step = int(step_tok)
                if last_tqdm == exact_step:
                    tqdm_fresh = False
            elif last_tqdm is not None and tqdm_fresh and rounded_step_matches(step_tok, last_tqdm):
                exact_step = last_tqdm
                tqdm_fresh = False
            else:
                unresolved += 1
                continue

            loss_val = parse_float(kv.get("loss"))
            lr_val = parse_float(kv.get("lr"))
            if loss_val is not None or lr_val is not None:
                train_by_step[exact_step] = {"step": exact_step, "loss": loss_val, "lr": lr_val}
            continue

        sp = SPLIT_RE.match(body)
        if sp:
            out["split_text"] = (
                f"{sp.group('train_ep')} train / {sp.group('eval_ep')} eval "
                f"(eval_split={sp.group('eval_split')})"
            )

    out["train_points"] = [train_by_step[s] for s in sorted(train_by_step)]
    out["eval_points"] = [eval_by_step[s] for s in sorted(eval_by_step)]
    if unresolved:
        warnings.append(f"Skipped {unresolved} rounded INFO record(s) without a matching TQDM step.")
    if not out["train_points"]:
        warnings.append("No training metrics parsed from train.out.")
    if not out["eval_points"]:
        warnings.append("No eval_loss entries parsed from train.out.")
    return out


def scan_checkpoints(
    run_path: Path,
    origin_run_dir: str | None,
    train_points: list[dict[str, Any]],
    eval_points: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Inventory saved checkpoints, deduplicating aliases such as last."""
    ckpt_root = run_path / "native" / "checkpoints"
    if not ckpt_root.is_dir():
        warnings.append("No native/checkpoints directory found.")
        return []

    by_step: dict[int, dict[str, Any]] = {}
    for entry in sorted(ckpt_root.iterdir(), key=lambda p: (not p.name.isdigit(), p.name)):
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue

        step_val: int | None = None
        step_file = entry / "training_state" / "training_step.json"
        if step_file.is_file():
            try:
                loaded = json.loads(step_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("step"), int):
                    step_val = loaded["step"]
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append(f"Unreadable {entry.name}/training_state/training_step.json: {exc}")
        if step_val is None:
            warnings.append(f"Skipped incomplete checkpoint entry {entry.name!r}: no readable training_step.json.")
            continue

        is_numeric = entry.name.isdigit()
        folder = entry.name
        if not is_numeric and entry.is_symlink():
            try:
                target_name = entry.resolve().name
                if target_name.isdigit():
                    folder = target_name
            except OSError:
                pass

        existing = by_step.get(step_val)
        if existing is not None:
            if is_numeric and not existing["is_numeric"]:
                if existing["folder"] != entry.name:
                    existing["aliases"].add(existing["folder"])
                existing["folder"] = entry.name
                existing["is_numeric"] = True
            elif entry.name != existing["folder"]:
                existing["aliases"].add(entry.name)
        else:
            aliases = {entry.name} if folder != entry.name else set()
            by_step[step_val] = {"folder": folder, "is_numeric": is_numeric, "aliases": aliases}

    train_by_step = {p["step"]: p for p in train_points}
    eval_by_step = {p["step"]: p for p in eval_points}
    checkpoints: list[dict[str, Any]] = []

    for step in sorted(by_step):
        folder = by_step[step]["folder"]
        if origin_run_dir:
            deploy_path = f"{origin_run_dir.rstrip('/')}/native/checkpoints/{folder}/pretrained_model"
            path_source = "run.json run_dir"
        else:
            deploy_path = str(run_path.resolve() / "native" / "checkpoints" / folder / "pretrained_model")
            path_source = "local input path (run.json run_dir missing)"

        ev = eval_by_step.get(step)
        tr = train_by_step.get(step)
        tr_exact = tr is not None
        if tr is None and train_points:
            tr = min(train_points, key=lambda p: (abs(p["step"] - step), 0 if p["step"] <= step else 1))

        checkpoints.append(
            {
                "step": step,
                "aliases": sorted(by_step[step]["aliases"]),
                "deploy_path": deploy_path,
                "path_source": path_source,
                "eval_loss": None if ev is None else ev["eval_loss"],
                "train_loss": None if tr is None else tr["loss"],
                "train_loss_step": None if tr is None else tr["step"],
                "train_loss_exact": tr_exact,
                "lr": None if tr is None else tr["lr"],
                "lr_step": None if tr is None else tr["step"],
                "lr_exact": tr_exact,
            }
        )

    if not checkpoints:
        warnings.append("No saved checkpoints found under native/checkpoints/.")
    return checkpoints


def default_label(preset: str | None, dir_name: str) -> str:
    if not preset or dir_name == preset:
        return preset or dir_name
    if dir_name.startswith(f"{preset}-"):
        suffix = dir_name[len(preset) + 1 :]
        m = STAMP_RE.match(suffix)
        short = f"{m.group('head')}{m.group('tail')}" if m else suffix
        return f"{preset} ({short})"
    return f"{preset} ({dir_name})"


def load_run(run_path: Path, color: str) -> dict[str, Any]:
    warnings: list[str] = []
    run_meta = read_json_file(run_path / "run.json", warnings, "run.json")
    status_meta = read_json_file(run_path / "status.json", warnings, "status.json")

    origin_run_dir = run_meta.get("run_dir") if isinstance(run_meta.get("run_dir"), str) else None
    dir_name = Path(origin_run_dir).name if origin_run_dir else run_path.resolve().name
    preset = run_meta.get("preset") if isinstance(run_meta.get("preset"), str) else None

    eff = run_meta.get("effective") or {}
    assets = run_meta.get("assets") or {}
    guards = (run_meta.get("preset_content") or {}).get("guards") or {}
    init_adapter = assets.get("init_adapter") or guards.get("init_adapter") or None

    log_data = parse_train_log(run_path / "train.out", warnings)
    checkpoints = scan_checkpoints(
        run_path, origin_run_dir, log_data["train_points"], log_data["eval_points"], warnings
    )

    latest_step = log_data["max_tqdm_step"]
    if latest_step is None and log_data["train_points"]:
        latest_step = log_data["train_points"][-1]["step"]
    if latest_step is None and checkpoints:
        latest_step = checkpoints[-1]["step"]

    return {
        "id": dir_name,
        "label": default_label(preset, dir_name),
        "color": color,
        "status": status_meta.get("status", "unknown"),
        "latest_step": latest_step,
        "target_steps": eff.get("stop_budget_steps") or log_data["tqdm_total"],
        "origin_run_dir": origin_run_dir,
        "is_local_snapshot": bool(origin_run_dir and str(run_path.resolve()) != origin_run_dir.rstrip("/")),
        "init_adapter": init_adapter,
        "base_revision": guards.get("base_revision"),
        "dataset": eff.get("dataset_repo_id") or (run_meta.get("dataset_inputs") or {}).get("repo_id"),
        "split_text": log_data["split_text"] or (f"eval_split={eff['eval_split']}" if "eval_split" in eff else None),
        "peak_lr": eff.get("peak_lr"),
        "floor_lr": eff.get("floor_lr"),
        "warmup_steps": eff.get("warmup_steps"),
        "seed": eff.get("seed"),
        "global_batch": eff.get("global_batch"),
        "train_points": log_data["train_points"],
        "eval_points": log_data["eval_points"],
        "checkpoints": checkpoints,
        "warnings": warnings,
    }


def build_csv(runs: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        ["run_label", "run_id", "record_type", "step", "train_loss", "train_loss_step", "eval_loss", "lr", "checkpoint_path"]
    )
    for run in runs:
        for p in run["train_points"]:
            writer.writerow([run["label"], run["id"], "train", p["step"], p["loss"], p["step"], "", p["lr"], ""])
        for p in run["eval_points"]:
            writer.writerow([run["label"], run["id"], "eval", p["step"], "", "", p["eval_loss"], "", ""])
        for ck in run["checkpoints"]:
            writer.writerow(
                [
                    run["label"],
                    run["id"],
                    "checkpoint",
                    ck["step"],
                    "" if ck["train_loss"] is None else ck["train_loss"],
                    "" if ck["train_loss_step"] is None else ck["train_loss_step"],
                    "" if ck["eval_loss"] is None else ck["eval_loss"],
                    "" if ck["lr"] is None else ck["lr"],
                    ck["deploy_path"],
                ]
            )
    return buf.getvalue()


def fmt_loss(v: float | None) -> str:
    return "missing" if v is None else f"{v:.4f}"


def fmt_lr(v: float | None) -> str:
    return "missing" if v is None else f"{v:.2e}"


def fmt_step_metric(v: float | None, step: int | None, exact: bool, formatter) -> str:
    if v is None:
        return "missing"
    base = formatter(v)
    return base if exact or step is None else f"{base} (step {step:,})"


def render_toggles(runs: list[dict[str, Any]]) -> str:
    parts = []
    for idx, run in enumerate(runs):
        parts.append(
            f'<label class="toggle"><input type="checkbox" checked data-run-idx="{idx}">'
            f'<span class="swatch" style="background:{run["color"]}"></span>'
            f'{html.escape(run["label"])}</label>'
        )
    return "".join(parts)


def render_ckpt_rows(runs: list[dict[str, Any]]) -> str:
    rows = []
    for idx, run in enumerate(runs):
        sw = f'<span class="swatch" style="background:{run["color"]};margin-right:6px"></span>'
        lbl = html.escape(run["label"])
        for ck in run["checkpoints"]:
            alias_txt = f" ({', '.join(ck['aliases'])})" if ck["aliases"] else ""
            step_txt = f"{ck['step']:,}{alias_txt}"
            val_txt = fmt_loss(ck["eval_loss"])
            tr_txt = fmt_step_metric(ck["train_loss"], ck["train_loss_step"], ck["train_loss_exact"], fmt_loss)
            lr_txt = fmt_step_metric(ck["lr"], ck["lr_step"], ck["lr_exact"], fmt_lr)
            path_esc = html.escape(ck["deploy_path"])
            rows.append(
                f'<tr tabindex="0" data-run-idx="{idx}" data-step="{ck["step"]}">'
                f"<td>{sw}{lbl}</td>"
                f'<td class="num">{html.escape(step_txt)}</td>'
                f'<td class="num">{html.escape(val_txt)}</td>'
                f'<td class="num">{html.escape(tr_txt)}</td>'
                f'<td class="num">{html.escape(lr_txt)}</td>'
                f'<td class="mono">{path_esc}</td></tr>'
            )
    if not rows:
        return '<tr><td colspan="6" class="muted">No saved checkpoints found under native/checkpoints/.</td></tr>'
    return "\n".join(rows)


def render_meta_rows(runs: list[dict[str, Any]]) -> tuple[str, str]:
    rows = []
    notes = []
    for run in runs:
        sw = f'<span class="swatch" style="background:{run["color"]};margin-right:6px"></span>'
        prog = "missing" if run["latest_step"] is None else f"{run['latest_step']:,}"
        if run["target_steps"]:
            prog += f" / {run['target_steps']:,}"
        init_txt = (
            f"Warm-start adapter: {run['init_adapter']}"
            if run["init_adapter"]
            else f"Fresh base ({run['base_revision'] or 'unspecified'})"
        )
        sched_txt = (
            f"{fmt_lr(run['peak_lr'])} -> {fmt_lr(run['floor_lr'])}, warmup {run['warmup_steps']:,}"
            if run["peak_lr"] is not None and run["warmup_steps"] is not None
            else "missing"
        )
        data_txt = (
            f"{run['dataset'] or 'missing'}; {run['split_text'] or 'split missing'}; "
            f"seed {run['seed'] if run['seed'] is not None else '?'}; "
            f"batch {run['global_batch'] if run['global_batch'] is not None else '?'}"
        )
        init_cls = ' class="mono"' if run["init_adapter"] else ""
        rows.append(
            f"<tr><td>{sw}{html.escape(run['label'])}</td>"
            f"<td>{html.escape(run['status'])} ({html.escape(prog)})</td>"
            f"<td{init_cls}>{html.escape(init_txt)}</td>"
            f"<td>{html.escape(sched_txt)}</td>"
            f"<td>{html.escape(data_txt)}</td></tr>"
        )
        if run["is_local_snapshot"] and run["origin_run_dir"]:
            notes.append(
                f'<div class="muted">{html.escape(run["label"])}: local snapshot input; '
                f'deployment paths derived from run.json run_dir ({html.escape(run["origin_run_dir"])}).</div>'
            )
        for w in run["warnings"]:
            notes.append(f'<div class="warn">{html.escape(run["label"])}: {html.escape(w)}</div>')
    return "\n".join(rows), "\n".join(notes)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Checkpoint Selection Report</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #f6f7f9; color: #0f172a; font: 14px/1.45 system-ui, -apple-system, sans-serif; }
  .page { max-width: 1320px; margin: 0 auto; padding: 14px 16px 26px; display: grid; gap: 10px; }
  .page > * { min-width: 0; }
  .top { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 8px; border-bottom: 1px solid #d8dee6; padding-bottom: 8px; }
  .top h1 { margin: 0; font-size: 18px; }
  .muted { color: #475569; font-size: 12.5px; }
  .panel { background: #fff; border: 1px solid #d8dee6; border-radius: 8px; padding: 10px 12px; }
  .notice { border-left: 3px solid #64748b; font-size: 12.5px; color: #334155; }
  .sel-panel { border-color: #2563eb; background: #fcfdff; display: grid; gap: 7px; }
  .row { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px 14px; }
  .path-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  #selected-path { flex: 1 1 360px; min-width: 220px; font: 12.5px ui-monospace, SFMono-Regular, Menlo, monospace; padding: 6px 8px; border: 1px solid #94a3b8; border-radius: 6px; }
  .btn { font: 600 12.5px system-ui, sans-serif; padding: 6px 12px; border: 1px solid #94a3b8; border-radius: 6px; background: #fff; color: #0f172a; cursor: pointer; }
  .btn-primary { background: #1d4ed8; border-color: #1d4ed8; color: #fff; }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .toggles { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
  .toggle { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; font-size: 13px; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  #main-chart { width: 100%; height: 620px; }
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; }
  th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
  th { background: #f1f5f9; white-space: nowrap; }
  .num { text-align: right; white-space: nowrap; }
  #ckpt-body tr { cursor: pointer; }
  #ckpt-body tr:hover { background: #f8fafc; }
  #ckpt-body tr.selected { background: #eff6ff; outline: 1px solid #2563eb; outline-offset: -1px; }
  #ckpt-body tr.hidden-run { display: none; }
  .mono { font: 11.5px ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all; }
  .warn, .copy-fb { color: #92400e; font-size: 12px; margin-top: 4px; }
  .copy-ok { color: #166534; }
</style>
<script type="text/javascript">__PLOTLY_JS__</script>
</head>
<body>
<div class="page">
  <header class="top">
    <h1>Checkpoint Selection Report</h1>
    <div class="muted">Static snapshot generated __GENERATED_AT__ &middot; Read-only offline HTML</div>
  </header>

  <div class="panel notice">
    <strong>Static snapshot:</strong> Browser interactions do not modify training or checkpoints.
    <strong>Validation loss is a shortlist diagnostic, not robot rollout success.</strong>
    No automatic best-checkpoint or winner selection is performed.
    Do not assume identical recipes, splits, or schedules across runs, or that warm-start alone caused differences.
  </div>

  <section class="panel sel-panel" aria-label="Selected saved checkpoint">
    <div class="row">
      <div id="sel-title"><strong>Selected checkpoint:</strong> none</div>
      <div id="sel-metrics" class="muted"></div>
    </div>
    <div class="path-row">
      <label class="muted" for="selected-path">Deployment path (pretrained_model):</label>
      <input id="selected-path" type="text" readonly value="">
      <button type="button" class="btn btn-primary" id="copy-btn" disabled>Copy path</button>
    </div>
    <div id="copy-status" class="muted" role="status" aria-live="polite"></div>
  </section>

  <section class="panel row" aria-label="Chart controls">
    <div class="toggles" id="run-toggles">__TOGGLES__</div>
    <div class="toggles">
      <label class="muted" for="smooth-slider">Train smoothing:
        <input id="smooth-slider" type="range" min="0" max="0.9" step="0.05" value="0.6">
        <strong id="smooth-label">EMA 0.60</strong>
      </label>
      <button type="button" class="btn" id="csv-btn">Download metrics CSV</button>
    </div>
  </section>

  <section class="panel" aria-label="Training curves">
    <div id="main-chart"></div>
  </section>

  <section class="panel" aria-label="Saved checkpoints">
    <div class="row" style="margin-bottom:6px">
      <strong>Saved checkpoints</strong>
      <span class="muted">Click any checkpoint diamond on the chart or row below to select it.</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Run</th><th class="num">Exact step</th><th class="num">Val loss</th>
            <th class="num">Train loss</th><th class="num">LR</th>
            <th>Original deployment path (pretrained_model)</th>
          </tr>
        </thead>
        <tbody id="ckpt-body">__CKPT_ROWS__</tbody>
      </table>
    </div>
  </section>

  <section class="panel" aria-label="Run metadata summary">
    <div style="margin-bottom:6px"><strong>Run metadata</strong></div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Run</th><th>Status / progress</th><th>Initialization</th><th>LR schedule</th><th>Dataset / split / seed</th></tr>
        </thead>
        <tbody>__META_ROWS__</tbody>
      </table>
    </div>
    __META_NOTES__
  </section>
</div>

<script id="report-data" type="application/json">__PAYLOAD_JSON__</script>
<script type="text/javascript">
(function () {
  "use strict";
  var REPORT = JSON.parse(document.getElementById("report-data").textContent);
  var RUNS = REPORT.runs;
  var state = { visible: RUNS.map(function () { return true; }), alpha: 0.6 };

  var chartEl = document.getElementById("main-chart");
  var pathInput = document.getElementById("selected-path");
  var copyBtn = document.getElementById("copy-btn");
  var copyStatus = document.getElementById("copy-status");
  var ckptBody = document.getElementById("ckpt-body");
  var togglesEl = document.getElementById("run-toggles");

  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function fmtInt(v) { return (v === null || v === undefined) ? "missing" : Number(v).toLocaleString("en-US"); }
  function fmtLoss(v) { return (v === null || v === undefined) ? "missing" : Number(v).toFixed(4); }
  function fmtLr(v) { return (v === null || v === undefined) ? "missing" : Number(v).toExponential(2); }
  function fmtCkptTrain(ck) {
    if (ck.train_loss === null) return "missing";
    return ck.train_loss_exact ? fmtLoss(ck.train_loss) : (fmtLoss(ck.train_loss) + " (step " + fmtInt(ck.train_loss_step) + ")");
  }
  function fmtCkptLr(ck) {
    if (ck.lr === null) return "missing";
    return ck.lr_exact ? fmtLr(ck.lr) : (fmtLr(ck.lr) + " (step " + fmtInt(ck.lr_step) + ")");
  }
  function ema(values, alpha) {
    if (!values.length || alpha <= 0) return values.slice();
    var out = [], acc = null;
    for (var i = 0; i < values.length; i++) {
      acc = (acc === null) ? values[i] : (alpha * acc + (1 - alpha) * values[i]);
      out.push(acc);
    }
    return out;
  }
  function findCkpt(rIdx, step) {
    var list = RUNS[rIdx] ? RUNS[rIdx].checkpoints : [];
    for (var i = 0; i < list.length; i++) if (list[i].step === step) return list[i];
    return null;
  }
  function guideShapes(step) {
    var line = { color: "#0f172a", width: 1.5, dash: "dash" };
    return [
      { type: "line", xref: "x", yref: "paper", x0: step, x1: step, y0: 0.66, y1: 1.00, line: line },
      { type: "line", xref: "x2", yref: "paper", x0: step, x1: step, y0: 0.33, y1: 0.59, line: line },
      { type: "line", xref: "x3", yref: "paper", x0: step, x1: step, y0: 0.00, y1: 0.25, line: line }
    ];
  }

  var traces = [];
  var meta = [];

  function pushCkptMarkers(run, rIdx, xaxis, yaxis, field, filled) {
    var xs = [], ys = [], cds = [], txts = [], lbl = esc(run.label);
    run.checkpoints.forEach(function (ck) {
      if (ck[field] === null) return;
      xs.push(ck.step);
      ys.push(ck[field]);
      cds.push({ kind: "ckpt", runIdx: rIdx, step: ck.step });
      txts.push(
        "<b>" + lbl + "</b> (saved checkpoint)<br>Step: " + fmtInt(ck.step)
        + "<br>Val loss: " + fmtLoss(ck.eval_loss)
        + "<br>Train loss: " + fmtCkptTrain(ck)
        + "<br>LR: " + fmtCkptLr(ck)
      );
    });
    if (!xs.length) return;
    traces.push({
      type: "scatter", mode: "markers", x: xs, y: ys, customdata: cds, text: txts,
      xaxis: xaxis, yaxis: yaxis, legendgroup: run.id, showlegend: false,
      marker: filled
        ? { symbol: "diamond", size: 11, color: run.color, line: { color: "#fff", width: 2 } }
        : { symbol: "diamond", size: 8, color: "#fff", line: { color: run.color, width: 2 } },
      hovertemplate: "%{text}<extra></extra>"
    });
    meta.push({ runIdx: rIdx, kind: "ckpt" });
  }

  RUNS.forEach(function (run, rIdx) {
    var lbl = esc(run.label), tx = [], ty = [], lx = [], ly = [];
    run.train_points.forEach(function (p) {
      if (p.loss !== null) { tx.push(p.step); ty.push(p.loss); }
      if (p.lr !== null) { lx.push(p.step); ly.push(p.lr); }
    });
    var ex = run.eval_points.map(function (p) { return p.step; });
    var ey = run.eval_points.map(function (p) { return p.eval_loss; });
    var hasLegend = false;

    if (tx.length) {
      traces.push({
        type: "scatter", mode: "lines", x: tx, y: ty, xaxis: "x", yaxis: "y",
        name: run.label + " (raw)", legendgroup: run.id, showlegend: false,
        line: { color: run.color, width: 1.2 }, opacity: 0.22,
        hovertemplate: "<b>" + lbl + "</b><br>Step %{x:,}<br>Raw train loss: %{y:.4f}<extra></extra>"
      });
      meta.push({ runIdx: rIdx, kind: "train_raw" });
      traces.push({
        type: "scatter", mode: "lines", x: tx, y: ema(ty, state.alpha), customdata: ty,
        xaxis: "x", yaxis: "y", name: run.label, legendgroup: run.id, showlegend: true,
        line: { color: run.color, width: 2.2 },
        hovertemplate: "<b>" + lbl + "</b><br>Step %{x:,}<br>Smoothed: %{y:.4f}<br>Raw: %{customdata:.4f}<extra></extra>"
      });
      meta.push({ runIdx: rIdx, kind: "train_smooth", rawY: ty });
      hasLegend = true;
    }
    pushCkptMarkers(run, rIdx, "x", "y", "train_loss", false);

    if (ex.length) {
      traces.push({
        type: "scatter", mode: "lines+markers", x: ex, y: ey, xaxis: "x2", yaxis: "y2",
        name: run.label, legendgroup: run.id, showlegend: !hasLegend,
        line: { color: run.color, width: 2 }, marker: { size: 6, color: run.color },
        hovertemplate: "<b>" + lbl + "</b><br>Step %{x:,}<br>Val loss: %{y:.4f}<extra></extra>"
      });
      meta.push({ runIdx: rIdx, kind: "eval" });
      hasLegend = true;
    }
    pushCkptMarkers(run, rIdx, "x2", "y2", "eval_loss", true);

    if (lx.length) {
      traces.push({
        type: "scatter", mode: "lines", x: lx, y: ly, xaxis: "x3", yaxis: "y3",
        name: run.label, legendgroup: run.id, showlegend: !hasLegend,
        line: { color: run.color, width: 2 },
        hovertemplate: "<b>" + lbl + "</b><br>Step %{x:,}<br>LR: %{y:.2e}<extra></extra>"
      });
      meta.push({ runIdx: rIdx, kind: "lr" });
    }
    pushCkptMarkers(run, rIdx, "x3", "y3", "lr", false);
  });

  var spike = { showspikes: true, spikemode: "across", spikethickness: 1, spikecolor: "#94a3b8", spikedash: "dot", gridcolor: "#e2e8f0", zeroline: false };
  Plotly.newPlot(chartEl, traces, {
    margin: { l: 70, r: 20, t: 34, b: 48 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff", hovermode: "closest",
    legend: { orientation: "h", x: 0, y: 1.07 },
    xaxis: Object.assign({ anchor: "y", matches: "x3", showticklabels: false }, spike),
    yaxis: { domain: [0.66, 1.00], title: { text: "Training loss" }, gridcolor: "#e2e8f0", zeroline: false, rangemode: "tozero" },
    xaxis2: Object.assign({ anchor: "y2", matches: "x3", showticklabels: false }, spike),
    yaxis2: { domain: [0.33, 0.59], title: { text: "Validation loss" }, gridcolor: "#e2e8f0", zeroline: false, rangemode: "tozero" },
    xaxis3: Object.assign({ anchor: "y3", title: { text: "New optimizer updates" }, separatethousands: true }, spike),
    yaxis3: { domain: [0.00, 0.25], title: { text: "Learning rate" }, gridcolor: "#e2e8f0", zeroline: false, exponentformat: "e", rangemode: "tozero" },
    shapes: []
  }, {
    responsive: true,
    displaylogo: false,
    toImageButtonOptions: { format: "png", filename: "training_curves", scale: 2 },
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
  });

  function setRunVisible(rIdx, vis) {
    state.visible[rIdx] = vis;
    var indices = [], flags = [];
    for (var i = 0; i < meta.length; i++) {
      if (meta[i].runIdx !== rIdx) continue;
      indices.push(i);
      flags.push(meta[i].kind === "train_smooth" ? (vis && state.alpha > 0) : vis);
    }
    if (indices.length) Plotly.restyle(chartEl, { visible: flags }, indices);
    ckptBody.querySelectorAll('tr[data-run-idx="' + rIdx + '"]').forEach(function (tr) {
      tr.classList.toggle("hidden-run", !vis);
    });
  }

  togglesEl.addEventListener("change", function (e) {
    if (e.target && e.target.dataset.runIdx !== undefined) {
      setRunVisible(parseInt(e.target.dataset.runIdx, 10), e.target.checked);
    }
  });

  chartEl.on("plotly_legendclick", function (evt) {
    var m = evt && meta[evt.curveNumber];
    if (!m) return false;
    var next = !state.visible[m.runIdx];
    var cb = togglesEl.querySelector('input[data-run-idx="' + m.runIdx + '"]');
    if (cb) cb.checked = next;
    setRunVisible(m.runIdx, next);
    return false;
  });

  chartEl.on("plotly_click", function (evt) {
    if (!evt || !evt.points || !evt.points.length) return;
    for (var i = 0; i < evt.points.length; i++) {
      var cd = evt.points[i].customdata;
      if (cd && cd.kind === "ckpt") { selectCheckpoint(cd.runIdx, cd.step); return; }
    }
    var p = evt.points[0], m = meta[p.curveNumber];
    if (m && typeof p.x === "number" && findCkpt(m.runIdx, p.x)) selectCheckpoint(m.runIdx, p.x);
  });

  var slider = document.getElementById("smooth-slider");
  var smoothLabel = document.getElementById("smooth-label");
  slider.addEventListener("input", function () {
    state.alpha = parseFloat(slider.value) || 0;
    smoothLabel.textContent = state.alpha > 0 ? ("EMA " + state.alpha.toFixed(2)) : "Off";
    var rawIdx = [], smIdx = [], smY = [], smVis = [];
    for (var i = 0; i < meta.length; i++) {
      if (meta[i].kind === "train_raw") rawIdx.push(i);
      if (meta[i].kind === "train_smooth") {
        smIdx.push(i);
        smY.push(ema(meta[i].rawY, state.alpha));
        smVis.push(state.visible[meta[i].runIdx] && state.alpha > 0);
      }
    }
    if (rawIdx.length) Plotly.restyle(chartEl, { opacity: state.alpha > 0 ? 0.22 : 0.85 }, rawIdx);
    if (smIdx.length) Plotly.restyle(chartEl, { y: smY, visible: smVis }, smIdx);
  });

  function selectCheckpoint(rIdx, step) {
    var run = RUNS[rIdx], ck = findCkpt(rIdx, step);
    if (!run || !ck) return;
    var titleEl = document.getElementById("sel-title");
    titleEl.textContent = "";
    var sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = run.color;
    sw.style.marginRight = "6px";
    titleEl.appendChild(sw);
    var b = document.createElement("strong");
    b.textContent = run.label + " — step " + fmtInt(ck.step);
    titleEl.appendChild(b);

    document.getElementById("sel-metrics").textContent =
      "Val loss: " + fmtLoss(ck.eval_loss)
      + " | Train loss: " + fmtCkptTrain(ck)
      + " | LR: " + fmtCkptLr(ck)
      + " | Source: " + ck.path_source;

    pathInput.value = ck.deploy_path;
    copyBtn.disabled = false;
    copyStatus.textContent = "";
    copyStatus.className = "muted";

    ckptBody.querySelectorAll("tr[data-run-idx]").forEach(function (tr) {
      var match = parseInt(tr.dataset.runIdx, 10) === rIdx && parseInt(tr.dataset.step, 10) === step;
      tr.classList.toggle("selected", match);
      tr.setAttribute("aria-selected", match ? "true" : "false");
    });
    Plotly.relayout(chartEl, { shapes: guideShapes(step) });
  }

  ckptBody.addEventListener("click", function (e) {
    var tr = e.target.closest("tr[data-run-idx]");
    if (tr) selectCheckpoint(parseInt(tr.dataset.runIdx, 10), parseInt(tr.dataset.step, 10));
  });
  ckptBody.addEventListener("keydown", function (e) {
    if (e.key !== "Enter" && e.key !== " ") return;
    var tr = e.target.closest("tr[data-run-idx]");
    if (tr) { e.preventDefault(); selectCheckpoint(parseInt(tr.dataset.runIdx, 10), parseInt(tr.dataset.step, 10)); }
  });

  pathInput.addEventListener("click", function () { pathInput.select(); });
  copyBtn.addEventListener("click", function () {
    var text = pathInput.value;
    if (!text) return;
    function fallback(msg) {
      pathInput.focus();
      pathInput.select();
      copyStatus.className = "muted copy-fb";
      copyStatus.textContent = msg + " Path selected — press Ctrl+C / Cmd+C to copy.";
    }
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function" && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function () {
        copyStatus.className = "muted copy-ok";
        copyStatus.textContent = "Copied path to clipboard.";
      }, function () {
        fallback("Clipboard write was blocked.");
      });
    } else {
      fallback("Clipboard API unavailable on this origin.");
    }
  });

  document.getElementById("csv-btn").addEventListener("click", function () {
    var blob = new Blob([REPORT.csv], { type: "text/csv;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "training_metrics.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  });

  for (var i = 0; i < RUNS.length; i++) {
    if (RUNS[i].checkpoints.length) {
      selectCheckpoint(i, RUNS[i].checkpoints[RUNS[i].checkpoints.length - 1].step);
      break;
    }
  }
})();
</script>
</body>
</html>
"""


def render_html(runs: list[dict[str, Any]], generated_at: str) -> str:
    payload = {"generated_at": generated_at, "runs": runs, "csv": build_csv(runs)}
    payload_json = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    meta_rows, meta_notes = render_meta_rows(runs)
    plotly_js = get_plotlyjs().replace("</script>", "<\\/script>")
    return (
        HTML_TEMPLATE.replace("__GENERATED_AT__", html.escape(generated_at))
        .replace("__PLOTLY_JS__", plotly_js)
        .replace("__TOGGLES__", render_toggles(runs))
        .replace("__CKPT_ROWS__", render_ckpt_rows(runs))
        .replace("__META_ROWS__", meta_rows)
        .replace("__META_NOTES__", meta_notes)
        .replace("__PAYLOAD_JSON__", payload_json)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build one offline HTML report for inspecting runs and selecting saved checkpoints."
    )
    parser.add_argument("runs", nargs="+", type=Path, help="One or more run directories.")
    parser.add_argument("-o", "--output", type=Path, default=Path("report.html"), help="Output HTML path.")
    args = parser.parse_args(argv)

    missing = [str(p) for p in args.runs if not p.is_dir()]
    if missing:
        print(f"ERROR: run directory not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    runs = [load_run(p, PALETTE[i % len(PALETTE)]) for i, p in enumerate(args.runs)]
    counts: dict[str, int] = {}
    for run in runs:
        counts[run["label"]] = counts.get(run["label"], 0) + 1
    seen: dict[str, int] = {}
    for run in runs:
        lbl = run["label"]
        if counts[lbl] > 1:
            seen[lbl] = seen.get(lbl, 0) + 1
            run["label"] = f"{lbl} #{seen[lbl]}"

    if sum(len(r["train_points"]) + len(r["eval_points"]) + len(r["checkpoints"]) for r in runs) == 0:
        print("ERROR: no metrics or saved checkpoints found; refusing to emit empty report.", file=sys.stderr)
        return 1

    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(runs, generated_at), encoding="utf-8")

    for run in runs:
        print(
            f"{run['label']}: {len(run['train_points'])} train, "
            f"{len(run['eval_points'])} eval, {len(run['checkpoints'])} checkpoints ({run['status']})"
        )
        for w in run["warnings"]:
            print(f"  warning: {w}", file=sys.stderr)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
