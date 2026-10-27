"""Statistics (Wilson and bootstrap intervals) and tolerant reading and repair of the study CSVs."""
from __future__ import annotations

import csv
from pathlib import Path
import shutil

import numpy as np



# -------------------- evaluation/stats.py --------------------
# Small statistics helpers for reporting trial results.
def wilson_interval(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a success rate. Returns (low, high) as fractions."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def bootstrap_ci(x, stat=np.median, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap confidence interval for a statistic (default: the median)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


# -------------------- evaluation/stats.py --------------------
# Reading the study CSVs when they are not tidy.
_LAYOUT_START = ("fixed", "randmk")

def _split_glued(row: list, n: int, levels: set) -> list:
    """Split a long row where a later row starts (a level name followed by a layout name). Returns the pieces."""
    for j in range(1, len(row) - 1):
        if row[j] in levels and row[j + 1].startswith(_LAYOUT_START):
            return [row[:j]] + _split_glued(row[j:], n, levels)
    return [row]

def read_rows(path, fields: list[str]) -> tuple[list[dict], int]:
    """(rows as dicts, number of rows dropped). Values are strings, as csv.DictReader gives them."""
    path = Path(path)
    with open(path, newline="") as f:
        raw = list(csv.reader(f))
    if not raw:
        return [], 0
    header, body = raw[0], raw[1:]
    n_head = len(header)
    prefix_ok = header == fields[:n_head]                      # the header is an older, shorter version of `fields`
    names = fields if prefix_ok else header
    levels = {r[0] for r in body if len(r) >= 2 and r[1].startswith(_LAYOUT_START)}
    out, dropped = [], 0
    for row in body:
        if not row:
            continue
        pieces = [row] if len(row) <= len(names) else _split_glued(row, len(names), levels)
        for k, piece in enumerate(pieces):
            complete = len(piece) >= n_head and len(piece) <= len(names)
            last_piece_of_glued = k == len(pieces) - 1
            if complete and (len(pieces) == 1 or last_piece_of_glued or len(piece) == len(names)):
                out.append({n: v for n, v in zip(names, piece)})
            else:
                dropped += 1                                  # truncated by an interrupted write
    return out, dropped

def repair(path, fields: list[str]) -> tuple[int, int]:
    """Rewrite `path` with the current header and the good rows; keep the original as <path>.bak. Returns (kept, dropped)."""
    path = Path(path)
    rows, dropped = read_rows(path, fields)
    shutil.copy(path, str(path) + ".bak")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    return len(rows), dropped

_END_STATE = ("miss_m", "yaw_err_deg", "min_clearance_m", "pos_rmse_m", "final_pos_err_m", "near_time_s")

def apply_cap(rows: list[dict], cap: float | None) -> list[dict]:
    """The same trials as if every run had been capped at `cap` seconds. A run is deterministic, so a run that took longer than
    `cap` would have timed out at `cap`: it becomes a timeout (not docked, duration = cap) whatever happened afterwards, and the
    columns that summarise the whole run (final miss, RMSE, closest approach ...) are blanked because they include time after the cap.
    Runs that ended before `cap` are untouched. This can only shorten a run: it cannot turn a 200 s timeout into a success."""
    if not cap:
        return rows
    out = []
    for r in rows:
        try:
            dur = float(r.get("duration_s", "nan") or "nan")
        except ValueError:
            dur = float("nan")
        if dur > cap + 1e-6:
            r = dict(r, outcome="timeout", docked="0", duration_s=f"{cap:g}", **{k: "" for k in _END_STATE})
        out.append(r)
    return out
