"""Paired comparison of search configurations.

Every config meets the same start pose, marker layout, dock height and people on the same seed, so the right
question about two configs is asked seed by seed: on the SAME trial, did config A do better than config B?

A pair is one (level, layout, seed) that both configs finished. Trials of one seed are not independent (the same
start and room at several dock heights), so confidence intervals resample whole seeds (clusters of pairs), not
single pairs. Nothing here is a significance verdict: it gives the size of the difference and how well the data
pin it down.

    docking          wins (A docked, B did not), losses, both docked, neither; success difference with a
                     cluster-bootstrap 95% CI; exact McNemar p on the discordant pairs
    binary events    contact, false dock, timeout/gave up: difference in rate, same CI
    continuous       time to dock, path length (pairs where BOTH docked: a failed run has no docking time),
                     position RMSE, final position error, closest approach, time near obstacles (all pairs):
                     median paired difference A - B with a cluster-bootstrap CI

`compare(rows, a, b)` returns a list of dicts, one per metric; `paired_tables(rows)` and `scope_rows(rows)` build the scopes
(all rooms pooled, each room, each room and dock-height band); `format_report` prints them.
"""
from __future__ import annotations

import math

import numpy as np

COMPARISONS = [("C-info", "B"), ("C-sweep", "B"), ("C-info", "C-sweep")]
# Dock-marker height bands [m]. With the head locked level (camera 0.19 m up, 0.40 m plate, 0.30 m black square) the whole square is
# in view from the 0.5 m standoff only up to about 0.33 m; at 0.35 m the top is cut off and a level head fails; from 0.40 m up a
# level head cannot see the marker at all. So three bands, not two: the 0.35 m case belongs to neither side.
BANDS = [("level", "level camera sees it (<= 0.30 m)", lambda z: z <= 0.30 + 1e-9),
         ("borderline", "borderline (0.35 m)", lambda z: 0.30 + 1e-9 < z < 0.40),
         ("tilt", "tilt needed (>= 0.40 m)", lambda z: z >= 0.40)]

# (key, label, kind, unit, higher_is_better, needs_both_docked)
METRICS = [
    ("docked", "docking success", "binary", "pp", True, False),
    ("contact", "contact (wall, pillar, person)", "binary", "pp", False, False),
    ("false_dock", "false dock", "binary", "pp", False, False),
    ("failed", "timeout / gave up", "binary", "pp", False, False),
    ("duration_s", "time to dock", "cont", "s", False, True),
    ("path_length_m", "path length", "cont", "m", False, True),
    ("pos_rmse_m", "position RMSE", "cont", "m", False, False),
    ("final_pos_err_m", "final position error", "cont", "m", False, False),
    ("min_clearance_m", "closest approach to obstacles", "cont", "m", True, False),
    ("near_time_s", "time within 0.10 m of an obstacle", "cont", "s", False, False),
]
KEYS = {m[0]: m for m in METRICS}


def _f(r, k):
    try:
        return float(r.get(k, "nan") or "nan")
    except (ValueError, TypeError):
        return float("nan")


def _value(r, key):
    if key == "docked":
        return float(int(float(r.get("docked", 0) or 0)))
    if key == "contact":
        return float(r.get("outcome") == "contact")
    if key == "false_dock":
        return float(r.get("outcome") == "false dock")
    if key == "failed":
        return float(r.get("outcome") in ("timeout", "gave up"))
    return _f(r, key)


def dock_band(r) -> str:
    z = _f(r, "dock_height_m")
    if z != z:
        return "?"
    return next(k for k, _, f in BANDS if f(z))


def pairs(rows, a: str, b: str):
    """[(cluster, row_a, row_b)] for every (level, layout, seed) both configs finished; cluster = (level, seed)."""
    by = {}
    for r in rows:
        by.setdefault((r["level"], r["layout"], int(r["seed"])), {})[r["config"]] = r
    out = []
    for (lv, lay, seed), d in by.items():
        if a in d and b in d:
            out.append(((lv, seed), d[a], d[b]))
    return out


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact p for `wins` against `losses` among the discordant pairs (1.0 with none)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def _cluster_boot(diffs, clusters, stat, n_boot=2000, seed=0, min_clusters=4):
    """95% percentile CI of stat(diffs) resampling whole clusters. nan when there are too few clusters."""
    diffs = np.asarray(diffs, float)
    if len(diffs) == 0:
        return float("nan"), float("nan")
    keys = list(dict.fromkeys(clusters))
    if len(keys) < min_clusters:
        return float("nan"), float("nan")
    idx_of = {k: i for i, k in enumerate(keys)}
    groups = [[] for _ in keys]
    for d, c in zip(diffs, clusters):
        groups[idx_of[c]].append(d)
    groups = [np.array(g) for g in groups]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots[i] = stat(np.concatenate([groups[j] for j in pick]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compare(rows, a: str, b: str) -> list:
    """One dict per metric: n pairs, the mean/median difference A - B and its 95% cluster-bootstrap CI (+ docking counts)."""
    prs = pairs(rows, a, b)
    out = []
    for key, label, kind, unit, higher, both in METRICS:
        use = [(c, ra, rb) for c, ra, rb in prs if not both or (_value(ra, "docked") and _value(rb, "docked"))]
        va = np.array([_value(ra, key) for _, ra, _ in use])
        vb = np.array([_value(rb, key) for _, _, rb in use])
        ok = np.isfinite(va) & np.isfinite(vb)
        cl = [c for (c, _, _), k in zip(use, ok) if k]
        d = (va - vb)[ok]
        row = dict(metric=key, label=label, kind=kind, unit=unit, higher_is_better=higher, n=int(ok.sum()),
                   n_seeds=len(set(cl)), a=a, b=b)
        if not len(d):
            row.update(diff=float("nan"), lo=float("nan"), hi=float("nan"))
        elif kind == "binary":
            scale = 100.0
            row.update(diff=float(d.mean() * scale))
            lo, hi = _cluster_boot(d, cl, np.mean)
            row.update(lo=lo * scale, hi=hi * scale, rate_a=float(va[ok].mean() * scale), rate_b=float(vb[ok].mean() * scale))
            if key == "docked":
                w, l = int(((va[ok] == 1) & (vb[ok] == 0)).sum()), int(((va[ok] == 0) & (vb[ok] == 1)).sum())
                row.update(wins=w, losses=l, both=int(((va[ok] == 1) & (vb[ok] == 1)).sum()),
                           neither=int(((va[ok] == 0) & (vb[ok] == 0)).sum()), p_mcnemar=mcnemar_exact(w, l))
        else:
            row.update(diff=float(np.median(d)))
            lo, hi = _cluster_boot(d, cl, np.median)
            row.update(lo=lo, hi=hi, median_a=float(np.median(va[ok])), median_b=float(np.median(vb[ok])))
        out.append(row)
    return out


def verdict(row) -> str:
    """'better' / 'worse' / 'unclear' for A against B, from whether the CI excludes zero (nan CI: 'few pairs')."""
    if row["n"] == 0 or row["lo"] != row["lo"]:
        return "few pairs"
    if row["lo"] > 0:
        return "better" if row["higher_is_better"] else "worse"
    if row["hi"] < 0:
        return "worse" if row["higher_is_better"] else "better"
    return "unclear"


# ------------------------------------------------------------------------------------------------ scopes
def scope_rows(rows, by: str = "room") -> dict:
    """{scope label: rows}. by='room': all rooms pooled, then each room. by='band': the same split by dock-marker height band (BANDS). by='headline': all rooms per band, then each room in the level band."""
    levels = list(dict.fromkeys(r["level"] for r in rows))
    if by == "headline":
        # all rooms in each height band, then each room within the band where a level head can see the marker
        out = {}
        for key, label, _ in BANDS:
            s2 = [r for r in rows if dock_band(r) == key]
            if s2:
                out[f"all rooms / {label}"] = s2
        lv = [r for r in rows if dock_band(r) == "level"]
        for room in levels:
            s2 = [r for r in lv if r["level"] == room]
            if s2:
                out[f"{room} / level band"] = s2
        return out
    out = {"all rooms": rows}
    for lv in levels:
        out[lv] = [r for r in rows if r["level"] == lv]
    if by == "band":
        out = {}
        for scope in ["all rooms"] + levels:
            sel = rows if scope == "all rooms" else [r for r in rows if r["level"] == scope]
            for key, label, _ in BANDS:
                s2 = [r for r in sel if dock_band(r) == key]
                if s2:
                    out[f"{scope} / {label}"] = s2
    return out


def paired_tables(rows, by: str = "room", comparisons=COMPARISONS) -> list:
    """Flat list of dicts: scope, a, b and the compare() fields, for every scope and comparison with at least one pair."""
    out = []
    for scope, rs in scope_rows(rows, by).items():
        for a, b in comparisons:
            if not pairs(rs, a, b):
                continue
            for m in compare(rs, a, b):
                out.append(dict(scope=scope, **m))
    return out


def _ci(m, fmt):
    if m["n"] == 0:
        return "-"
    if m["lo"] != m["lo"]:
        return f"{fmt(m['diff'])}  [few seeds]"
    return f"{fmt(m['diff'])} [{fmt(m['lo'])}, {fmt(m['hi'])}]"


def format_report(rows, by: str = "room", comparisons=COMPARISONS) -> str:
    lines = []
    sp = lambda v: f"{v:+.0f}"
    sv = lambda v: f"{v:+.1f}"
    sm = lambda v: f"{v:+.3f}"
    for scope, rs in scope_rows(rows, by).items():
        block = []
        for a, b in comparisons:
            if not pairs(rs, a, b):
                continue
            ms = {m["metric"]: m for m in compare(rs, a, b)}
            d = ms["docked"]
            cnt = f"{d.get('wins', 0)}W {d.get('losses', 0)}L {d.get('both', 0)}both {d.get('neither', 0)}neither"
            block.append(f"  {a:8s} vs {b:8s} {d['n']:4d} {d['n_seeds']:4d}  {cnt:34s} "
                         f"{_ci(d, sp):26s} p={d.get('p_mcnemar', float('nan')):.3f}  "
                         f"{_ci(ms['duration_s'], sv):28s} {_ci(ms['pos_rmse_m'], sm):28s} {_ci(ms['min_clearance_m'], sm):26s}")
        if block:
            lines.append(f"\n=== {scope} ===")
            lines.append(f"  {'A':8s} vs {'B':8s} {'pairs':>4s} {'seeds':>4s}  {'docking (A vs B, same trial)':34s} "
                         f"{'success diff pp [95% CI]':26s} {'McNemar':8s}  {'time to dock diff s [CI]':28s} "
                         f"{'pos RMSE diff m [CI]':28s} {'closest approach diff m [CI]':26s}")
            lines += block
    lines.append("\nDifferences are A - B. Success: percentage points, positive = A docks more often. Time: both-docked pairs only, negative = A faster.")
    lines.append("RMSE: negative = A estimates better. Closest approach: positive = A keeps more distance. CIs resample whole seeds; 'few seeds' = under 4.")
    return "\n".join(lines)


def taxonomy(rows) -> list:
    """Failure taxonomy: per (level, config) the share of trials in each outcome. [dict(level, config, n, docked, false dock, ...)]"""
    cats = ["docked", "false dock", "contact", "timeout", "gave up"]
    out = []
    for lv in dict.fromkeys(r["level"] for r in rows):
        for c in dict.fromkeys(r["config"] for r in rows):
            rs = [r for r in rows if r["level"] == lv and r["config"] == c]
            if rs:
                d = dict(level=lv, config=c, n=len(rs))
                for k in cats:
                    d[k] = sum(r["outcome"] == k for r in rs) / len(rs)
                out.append(d)
    return out
