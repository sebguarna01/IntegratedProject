"""Paired comparison of search configurations (statistics only; scripts/analyse.py draws the tables and figures).

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

`compare(rows, a, b)` returns a list of dicts, one per metric; scripts/analyse.py builds the scopes and prints them.
"""
from __future__ import annotations

import math

import numpy as np

COMPARISONS = [("C-info", "B"), ("C-sweep", "B"), ("C-info", "C-sweep")]
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
