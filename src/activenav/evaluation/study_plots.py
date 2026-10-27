"""Charts for the search study (`scripts/search_study.py --report --plots`).

Input: the rows of the study CSVs (list of dicts, newest row per level/layout/config/seed). Output: PNG files in two folders.
  summary/  the high-level view: 01 scorecard of everything pooled, 02 the same four headline metrics room by room,
            03 docking success per room by dock-marker height band (level camera sees it / borderline / tilt needed), 04 paired difference against B per room (forest plot),
            05 how every trial ended (failure taxonomy).
  detail/   the in-depth view: success by level and by dock height, docked over time, time to dock, estimation, safety and effort
            (one panel per level), effects_of_dock_height (every metric against the dock height) and rooms/<level>_*.png.
`make_all(rows, out_dir, view="summary" | "detail" | "both")` picks which.
Columns that older CSVs lack are skipped, not faked. Static images: the printed report is the table view of the same data.

Colours are the validated categorical slots 1-3 (blue, orange, aqua), always in the same order, and each config also has its own
marker, so identity never rests on colour alone (the aqua is under 3:1 on the light surface, hence the labels).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from activenav.evaluation.stats import wilson_interval

CONFIGS = ["B", "C-sweep", "C-info"]
COLOUR = {"B": "#2a78d6", "C-sweep": "#eb6834", "C-info": "#1baf7a"}
MARKER = {"B": "o", "C-sweep": "s", "C-info": "^"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"B": "B  fixed head", "C-sweep": "C-sweep  pan sweep", "C-info": "C-info  chosen gaze"}


# ------------------------------------------------------------------------------------------------ helpers
def _num(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, "nan") or "nan"))
        except ValueError:
            out.append(float("nan"))
    return np.array(out, float)


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2.0, "font.family": "DejaVu Sans",
    })


def _groups(rows):
    """[(label, rows)] per (level, layout); the layout is only spelled out when a level has several."""
    keys = list(dict.fromkeys((r["level"], r["layout"]) for r in rows))
    per_level = {}
    for lv, lay in keys:
        per_level.setdefault(lv, []).append(lay)
    out = []
    for lv, lay in keys:
        label = lv if len(per_level[lv]) == 1 else f"{lv}\n{lay.replace('fixed', '').strip('+') or 'default'}"
        out.append((label, [r for r in rows if r["level"] == lv and r["layout"] == lay]))
    return out


def _legend(fig, y=0.995):
    handles = [plt.Line2D([], [], color=COLOUR[c], marker=MARKER[c], lw=2, ms=7, label=NAMES[c]) for c in CONFIGS]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.99, y), fontsize=9)


def _cfg_rows(rows, c):
    return [r for r in rows if r["config"] == c]


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _bar_offsets(n=3, width=0.26):
    return [(i - (n - 1) / 2) * width for i in range(n)], width


def _have(rows, key):
    return bool(np.isfinite(_num(rows, key)).any())


# ------------------------------------------------------------------------------------------------ figures
def fig_success_by_level(rows, path):
    groups = _groups(rows)
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(groups) + 2), 4.4))
    offs, w = _bar_offsets()
    for j, c in enumerate(CONFIGS):
        for i, (_, g) in enumerate(groups):
            rs = _cfg_rows(g, c)
            if not rs:
                continue
            k = sum(r["docked"] == "1" for r in rs)
            lo, hi = wilson_interval(k, len(rs))
            p = k / len(rs)
            ax.bar(i + offs[j], 100 * p, width=w - 0.04, color=COLOUR[c], zorder=3)
            ax.plot([i + offs[j]] * 2, [100 * lo, 100 * hi], color=INK, lw=1.2, zorder=4)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylim(0, 105)
    ax.set_ylabel("docked [%]   (whisker: 95% Wilson interval)")
    ax.set_title("Docking success by level", pad=26)
    _legend(fig, 0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def fig_success_vs_height(rows, path):
    hs = sorted({h for h in _num(rows, "dock_height_m") if np.isfinite(h)})
    if len(hs) < 2:
        return None
    levels = list(dict.fromkeys(r["level"] for r in rows))
    panels = [("all levels pooled", rows)] + [(lv, [r for r in rows if r["level"] == lv]) for lv in levels if len(levels) > 1]
    ncol = min(3, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(max(4.6, 0.8 * len(hs) + 2.5) * ncol, 3.7 * nrow + 0.5), squeeze=False, sharey=True)
    offs = {"B": -0.14, "C-sweep": 0.0, "C-info": 0.14}                  # side by side: B and C-sweep often coincide exactly
    for ax, (title, rs) in zip(axs.ravel(), panels):
        for c in CONFIGS:
            for i, h in enumerate(hs):
                sub = [r for r in _cfg_rows(rs, c) if abs(float(r.get("dock_height_m") or "nan") - h) < 1e-6]
                if not sub:
                    continue
                k = sum(r["docked"] == "1" for r in sub)
                lo, hi = wilson_interval(k, len(sub))
                x = i + offs[c]
                ax.plot([x, x], [100 * lo, 100 * hi], color=COLOUR[c], lw=1.6, zorder=2)
                ax.plot([x], [100 * k / len(sub)], color=COLOUR[c], marker=MARKER[c], ms=7, mec=SURFACE, mew=1.3, ls="", zorder=3)
        ax.set_title(title)
        ax.set_xticks(range(len(hs)), [f"{h:g}" for h in hs])
        ax.set_ylim(-4, 106)
        ax.set_xlabel("dock marker centre height [m]")
    for ax in axs[:, 0]:
        ax.set_ylabel("docked [%]   (whisker: 95% Wilson interval)")
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Docking success against dock-marker height", x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.0)
    _legend(fig, 0.955)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)


def fig_docked_over_time(rows, path):
    groups = _groups(rows)
    ncol = min(3, len(groups))
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow + 0.5), squeeze=False, sharey=True)
    tmax = max(float(r["duration_s"]) for r in rows)
    for ax, (label, g) in zip(axs.ravel(), groups):
        for c in CONFIGS:
            rs = _cfg_rows(g, c)
            if not rs:
                continue
            td = np.sort([float(r["duration_s"]) for r in rs if r["docked"] == "1"])
            t = np.concatenate([[0.0], td, [tmax]])
            frac = np.concatenate([[0.0], np.arange(1, len(td) + 1), [len(td)]]) / len(rs)
            ax.step(t, 100 * frac, where="post", color=COLOUR[c], lw=2.0, zorder=3)
        ax.set_title(label.replace("\n", "  "))
        ax.set_xlim(0, tmax)
        ax.set_ylim(0, 102)
        ax.set_xlabel("time [s]")
    for ax in axs[:, 0]:
        ax.set_ylabel("docked by then [%]")
    for ax in axs.ravel()[len(groups):]:
        ax.axis("off")
    fig.suptitle("Docked over time (a failed run never reaches the curve: it is not 'slow')", x=0.01, ha="left",
                 fontweight="bold", fontsize=11, y=1.0)
    _legend(fig, 0.955)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)


def _box(ax, x, vals, colour, w=0.2):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return
    lo, q1, med, q3, hi = np.percentile(vals, [10, 25, 50, 75, 90])
    ax.plot([x, x], [lo, hi], color=colour, lw=1.0, zorder=2)
    ax.add_patch(plt.Rectangle((x - w / 2, q1), w, max(q3 - q1, 1e-9), color=colour, alpha=0.35, lw=0, zorder=3))
    ax.plot([x - w / 2, x + w / 2], [med, med], color=colour, lw=2.4, zorder=4)


def fig_time_to_dock(rows, path):
    groups = _groups(rows)
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(groups) + 2), 4.4))
    offs, w = _bar_offsets()
    for j, c in enumerate(CONFIGS):
        for i, (_, g) in enumerate(groups):
            td = [float(r["duration_s"]) for r in _cfg_rows(g, c) if r["docked"] == "1"]
            _box(ax, i + offs[j], td, COLOUR[c], w=w - 0.06)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylabel("time to dock [s]   (successful runs; box: quartiles, bar: median, whisker: 10th-90th percentile)")
    ax.set_ylim(bottom=0)
    ax.set_title("Time to dock, successful runs only", pad=26)
    _legend(fig, 0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def _panel_metric(ax, groups, key, ylabel, title, ref=None, pct=False):
    groups = [g for g in groups if np.isfinite(_num(g[1], key)).any()]        # levels without this metric (older CSVs) are left out
    offs, w = _bar_offsets()
    for j, c in enumerate(CONFIGS):
        for i, (_, g) in enumerate(groups):
            v = _num(_cfg_rows(g, c), key)
            _box(ax, i + offs[j], 100 * v if pct else v, COLOUR[c], w=w - 0.06)
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(range(len(groups)), [g[0].replace("\n", " ") for g in groups], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def fig_estimation(rows, path):
    keys = [("nees_consistent", "time inside the 95% band [%]", "Filter consistency (higher is better)", None, True),
            ("final_pos_err_m", "final position error [m]", "Position error at the end", None, False),
            ("landmark_err_m", "mean landmark error [m]", "Landmark map error", None, False),
            ("sigma_xy_mean_m", "mean sigma xy [m]", "Filter's own uncertainty", None, False)]
    keys = [k for k in keys if _have(rows, k[0])]
    if not keys:
        return None
    groups = _groups(rows)
    ncol = 2
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(max(8.5, 1.5 * len(groups) * 2), 3.9 * nrow + 0.5), squeeze=False)
    for ax, (k, yl, ti, ref, pct) in zip(axs.ravel(), keys):
        _panel_metric(ax, groups, k, yl, ti, ref, pct)
        ax.set_ylim(bottom=0)
    for ax in axs.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle("State estimation (all runs of a config; box: quartiles, bar: median, whisker: 10th-90th percentile)", x=0.01, ha="left", fontweight="bold",
                 fontsize=11, y=1.0)
    _legend(fig, 0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, path)


def fig_safety_effort(rows, path):
    keys = [("near_time_s", "seconds within 0.10 m of an obstacle", "Time close to obstacles"),
            ("blocked_s", "seconds held back by the reflex", "Safety reflex holding the robot"),
            ("head_pan_travel_rad", "total pan motion [rad]", "Head effort: pan"),
            ("head_tilt_travel_rad", "total tilt motion [rad]", "Head effort: tilt"),
            ("n_recover", "stuck recoveries per run", "Deadlock recoveries"),
            ("path_length_m", "path length [m]", "Distance driven")]
    keys = [k for k in keys if _have(rows, k[0])]
    if not keys:
        return None
    groups = _groups(rows)
    ncol = 2
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(max(8.5, 1.5 * len(groups) * 2), 3.7 * nrow + 0.5), squeeze=False)
    for ax, (k, yl, ti) in zip(axs.ravel(), keys):
        _panel_metric(ax, groups, k, yl, ti)
        ax.set_ylim(bottom=0)
    for ax in axs.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle("Safety and effort (all runs of a config; box: quartiles, bar: median, whisker: 10th-90th percentile)", x=0.01, ha="left", fontweight="bold",
                 fontsize=11, y=1.0)
    _legend(fig, 0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def _levels(rows):
    return list(dict.fromkeys(r["level"] for r in rows))


def _scorecard_axes(axs, rows, titles=True, ticks=True):
    """The four umbrella metrics on four axes: success, speed, estimation, safety."""
    xs = np.arange(len(CONFIGS))
    ax = axs[0]                                                         # 1. docking success
    for i, c in enumerate(CONFIGS):
        rs = _cfg_rows(rows, c)
        if not rs:
            continue
        k = sum(r["docked"] == "1" for r in rs)
        lo, hi = wilson_interval(k, len(rs))
        ax.bar(i, 100 * k / len(rs), width=0.6, color=COLOUR[c], zorder=3)
        ax.plot([i, i], [100 * lo, 100 * hi], color=INK, lw=1.2, zorder=4)
        ax.text(i, 100 * hi + 2, f"{k}/{len(rs)}", ha="center", fontsize=8, color=INK2)
    ax.set_ylim(0, 114)
    ax.set_ylabel("docked [%]")
    ax = axs[1]                                                         # 2. time to dock, successful runs
    for i, c in enumerate(CONFIGS):
        _box(ax, i, [float(r["duration_s"]) for r in _cfg_rows(rows, c) if r["docked"] == "1"], COLOUR[c], w=0.5)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("time to dock [s]")
    ax = axs[2]                                                         # 3. estimation
    if _have(rows, "nees_consistent"):
        for i, c in enumerate(CONFIGS):
            _box(ax, i, 100 * _num(_cfg_rows(rows, c), "nees_consistent"), COLOUR[c], w=0.5)
        ax.set_ylim(0, 102)
        ax.set_ylabel("time inside 95% band [%]")
        t3 = "3  Estimation (consistency)"
    else:
        for i, c in enumerate(CONFIGS):
            _box(ax, i, _num(_cfg_rows(rows, c), "landmark_err_m"), COLOUR[c], w=0.5)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("landmark error [m]")
        t3 = "3  Estimation (landmark error)"
    ax = axs[3]                                                         # 4. safety
    for i, c in enumerate(CONFIGS):
        _box(ax, i, _num(_cfg_rows(rows, c), "min_clearance_m"), COLOUR[c], w=0.5)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_ylabel("closest approach [m]")
    if titles:
        for ax, t in zip(axs, ["1  Success", "2  Speed (successful runs)", t3, "4  Safety (below 0 = contact)"]):
            ax.set_title(t)
    for ax in axs:
        ax.set_xticks(xs, CONFIGS if ticks else [""] * len(CONFIGS))


def fig_scorecard(rows, path, title=None):
    """The four umbrella metrics, pooled over everything in `rows`."""
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    _scorecard_axes(axs, rows)
    fig.suptitle(title or f"Scorecard: the four headline metrics ({len(rows)} trials, all rooms and heights pooled)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def fig_summary_by_room(rows, path):
    """One row per room (plus 'all rooms'), the four headline metrics across: the one-page room-level comparison."""
    levels = _levels(rows)
    sets = ([("all rooms", rows)] if len(levels) > 1 else []) + [(lv, [r for r in rows if r["level"] == lv]) for lv in levels]
    fig, axs = plt.subplots(len(sets), 4, figsize=(15, 2.35 * len(sets) + 0.9), squeeze=False, sharey="col")
    for i, (name, rs) in enumerate(sets):
        _scorecard_axes(axs[i], rs, titles=(i == 0), ticks=(i == len(sets) - 1))
        axs[i][0].annotate(f"{name}\n(n={len(rs)})", xy=(0, 0.5), xycoords="axes fraction", xytext=(-52, 0), textcoords="offset points",
                           ha="right", va="center", fontsize=9, fontweight="bold")
        for ax in axs[i][1:]:
            ax.set_ylabel("")
    fig.suptitle("Headline metrics room by room (every dock height pooled; whiskers 10th-90th percentile)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0.06, 0, 1, 0.97))
    return _save(fig, path)


def fig_height_bands(rows, path):
    """Docking success per room by dock-marker height band: level camera sees it / borderline / tilt needed (paired.BANDS)."""
    from activenav.evaluation import paired
    h = _num(rows, "dock_height_m")
    if not np.isfinite(h).any():
        return None
    bands = [(label, f) for _, label, f in paired.BANDS if any(np.isfinite(v) and f(v) for v in h)]
    if len(bands) < 2:
        return None
    levels = _levels(rows)
    fig, axs = plt.subplots(len(bands), 1, figsize=(max(9, 1.6 * len(levels)), 2.9 * len(bands) + 0.9), sharey=True, squeeze=False)
    axs = axs[:, 0]
    offs, w = _bar_offsets()
    for ax, (title, sel) in zip(axs, bands):
        for j, c in enumerate(CONFIGS):
            for i, lv in enumerate(levels):
                rs = [r for r in rows if r["level"] == lv and r["config"] == c and np.isfinite(float(r.get("dock_height_m") or "nan"))
                      and sel(float(r["dock_height_m"]))]
                if not rs:
                    continue
                k = sum(r["docked"] == "1" for r in rs)
                lo, hi = wilson_interval(k, len(rs))
                ax.bar(i + offs[j], 100 * k / len(rs), width=w - 0.04, color=COLOUR[c], zorder=3)
                ax.plot([i + offs[j]] * 2, [100 * lo, 100 * hi], color=INK, lw=1.1, zorder=4)
                ax.text(i + offs[j], 100 * hi + 2, f"{k}/{len(rs)}", ha="center", fontsize=6.5, color=INK2, rotation=90)
        ax.set_xticks(range(len(levels)), levels, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 118)
        ax.set_title(title)
    for ax in axs:
        ax.set_ylabel("docked [%]")
    axs[-1].set_xlabel("whisker: 95% Wilson interval over trials (heights of one seed are not independent: read as a guide)", fontsize=8)
    for ax in axs[:-1]:
        ax.set_xticklabels([])
    fig.suptitle("Docking success per room, by how high the dock marker hangs", x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.0)
    _legend(fig, 0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def fig_height_effects(rows, path, title=None):
    """Every metric against the dock-marker height (rooms in `rows` pooled): median per height, whisker = quartiles (success: Wilson)."""
    hs = sorted({h for h in _num(rows, "dock_height_m") if np.isfinite(h)})
    if len(hs) < 2:
        return None
    # key, label, which runs, kind
    spec = [("docked", "docked [%]", "all", "rate"),
            ("duration_s", "time to dock [s]", "docked", "q"),
            ("head_pan_travel_rad", "head pan travel [rad]", "all", "q"),
            ("head_tilt_travel_rad", "head tilt travel [rad]", "all", "q"),
            ("path_length_m", "path length [m]", "all", "q"),
            ("nees_consistent", "filter consistent [% of time]", "all", "q100"),
            ("final_pos_err_m", "final position error [m]", "all", "q"),
            ("min_clearance_m", "closest approach [m]", "all", "q"),
            ("near_time_s", "time within 0.10 m of an obstacle [s]", "all", "q")]
    spec = [x for x in spec if x[0] in ("docked", "duration_s") or (_have(rows, x[0]) and np.nanmax(np.abs(_num(rows, x[0]))) > 0)]
    ncol = 3
    nrow = int(np.ceil(len(spec) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(max(4.6, 0.62 * len(hs) + 3) * ncol, 3.1 * nrow + 0.7), squeeze=False)
    offs = {"B": -0.15, "C-sweep": 0.0, "C-info": 0.15}
    for ax, (key, label, which, kind) in zip(axs.ravel(), spec):
        for c in CONFIGS:
            xs, ys = [], []
            for i, h in enumerate(hs):
                sub = [r for r in _cfg_rows(rows, c) if abs(float(r.get("dock_height_m") or "nan") - h) < 1e-6]
                if which == "docked":
                    sub = [r for r in sub if r["docked"] == "1"]
                if not sub:
                    continue
                x = i + offs[c]
                if kind == "rate":
                    k = sum(r["docked"] == "1" for r in sub)
                    lo, hi = wilson_interval(k, len(sub))
                    m, lo, hi = 100 * k / len(sub), 100 * lo, 100 * hi
                else:
                    v = _num(sub, key) * (100 if kind == "q100" else 1)
                    v = v[np.isfinite(v)]
                    if not len(v):
                        continue
                    m = float(np.median(v))
                    lo, hi = (np.percentile(v, [25, 75]) if len(v) >= 3 else (m, m))
                ax.plot([x, x], [lo, hi], color=COLOUR[c], lw=1.4, zorder=2)
                xs.append(x)
                ys.append(m)
            ax.plot(xs, ys, color=COLOUR[c], lw=1.0, alpha=0.5, zorder=2)
            ax.plot(xs, ys, color=COLOUR[c], marker=MARKER[c], ms=5.5, mec=SURFACE, mew=1.0, ls="", zorder=3)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(len(hs)), [f"{h:g}" for h in hs], fontsize=8)
        if key == "min_clearance_m":
            ax.axhline(0, color=INK, lw=1.0)
        if kind == "rate":
            ax.set_ylim(-4, 106)
        elif key not in ("min_clearance_m",):
            ax.set_ylim(bottom=0)
        ax.set_xlabel("dock marker centre height [m]", fontsize=8)
    for ax in axs.ravel()[len(spec):]:
        ax.axis("off")
    fig.suptitle(title or "Effect of the dock-marker height (median; whisker: quartiles, success: 95% Wilson; time to dock from successful runs only)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.0)
    _legend(fig, 0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


# ------------------------------------------------------------------------------------------------ paired comparison
# (metric key, panel title, x label, scale, sign): sign makes "right of zero" always mean "the active-head config did better"
_FOREST = [
    ("docked", "Docking success", "pp more often", 1.0, +1),
    ("duration_s", "Time to dock (both docked)", "s faster", 1.0, -1),
    ("pos_rmse_m", "Position RMSE", "mm lower", 1000.0, -1),
    ("min_clearance_m", "Closest approach", "cm more room", 100.0, +1),
]


def fig_paired_effects(rows, path, by="room"):
    """Forest plot of the paired difference against B, one row per room (plus all rooms): C-info and C-sweep, same seed and layout.
    Right of zero = better than B in every panel. Whiskers: 95% CI resampling whole seeds."""
    from activenav.evaluation import paired
    scopes = paired.scope_rows(rows, by)
    prs = {sc: {c: paired.compare(rs, c, "B") for c in ("C-info", "C-sweep") if paired.pairs(rs, c, "B")} for sc, rs in scopes.items()}
    prs = {sc: d for sc, d in prs.items() if d}
    if not prs:
        return None
    names = list(prs)
    fig, axs = plt.subplots(1, len(_FOREST), figsize=(15, 0.62 * len(names) + 2.0), sharey=True, squeeze=False)
    axs = axs[0]
    y = np.arange(len(names))[::-1]
    off = {"C-info": 0.17, "C-sweep": -0.17}
    for ax, (key, title, xl, scale, sign) in zip(axs, _FOREST):
        for c in ("C-info", "C-sweep"):
            for yi, sc in zip(y, names):
                m = next((m for m in prs[sc].get(c, []) if m["metric"] == key), None)
                if m is None or m["n"] == 0:
                    continue
                d = sign * m["diff"] * scale
                lo, hi = sorted((sign * m["lo"] * scale, sign * m["hi"] * scale))
                ax.plot([d], [yi + off[c]], MARKER[c], color=COLOUR[c], ms=7, zorder=4, mec=SURFACE, mew=0.6)
                if lo == lo:
                    ax.plot([lo, hi], [yi + off[c]] * 2, color=COLOUR[c], lw=1.6, zorder=3)
        ax.axvline(0, color=INK, lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(f"{xl}  \u2192  better than B")
        ax.set_yticks(y, names)
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=COLOUR[c], marker=MARKER[c], lw=2, ms=7, label=f"{c} vs B") for c in ("C-info", "C-sweep")]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0), fontsize=9)
    fig.suptitle("Paired difference against B: same start, layout, dock height and people on every seed (whiskers: 95% CI over seeds)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


_OUTCOME = [("docked", "docked", "#0e7c66"), ("false dock", "false dock", "#d9a300"), ("contact", "contact", "#c23b3b"),
            ("timeout", "timeout", "#8d8c88"), ("gave up", "gave up", "#c9c8c3")]


def fig_outcomes(rows, path):
    """How every trial ended, per room and config: the failure taxonomy as stacked bars (share of trials)."""
    from activenav.evaluation import paired
    tax = paired.taxonomy(rows)
    if not tax:
        return None
    levels = _levels(rows)
    fig, ax = plt.subplots(figsize=(11, 0.34 * len(tax) + 0.5 * len(levels) + 1.4))
    ypos, labels, ylev = [], [], []
    y = 0.0
    for lv in levels:
        for c in CONFIGS:
            t = next((t for t in tax if t["level"] == lv and t["config"] == c), None)
            if t is None:
                continue
            left = 0.0
            for k, _, col in _OUTCOME:
                w = 100 * t[k]
                if w > 0:
                    ax.barh(y, w, left=left, color=col, height=0.78, zorder=3)
                    if w >= 7:
                        ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=8, color="white" if k in ("docked", "contact", "timeout") else INK)
                left += w
            ypos.append(y)
            labels.append(f"{c}  (n={t['n']})")
            y -= 1.0
        ylev.append((lv, y + 0.5 * len([1 for c in CONFIGS if any(t["level"] == lv and t["config"] == c for t in tax)]) + 0.5))
        y -= 0.6
    ax.set_yticks(ypos, labels, fontsize=8.5)
    for lv, yy in ylev:
        ax.annotate(lv, xy=(0, yy), xycoords=("axes fraction", "data"), xytext=(-110, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=9, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of trials [%]")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in _OUTCOME], labels=[n for _, n, _ in _OUTCOME],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=9)
    fig.tight_layout(rect=(0.1, 0, 1, 1))
    return _save(fig, path)


def make_all(rows, out_dir, view: str = "both") -> list[Path]:
    """Write the figures the data supports; returns the paths. view: 'summary' (high level), 'detail' (in depth) or 'both'."""
    _style()
    out = Path(out_dir)
    made = []
    if view in ("summary", "both"):
        s = out / "summary"
        made += [fig_scorecard(rows, s / "01_scorecard_all_rooms.png"),
                 fig_summary_by_room(rows, s / "02_headline_by_room.png"),
                 fig_height_bands(rows, s / "03_success_by_dock_height_band.png"),
                 fig_paired_effects(rows, s / "04_paired_difference_vs_B.png", by="headline"),
                 fig_outcomes(rows, s / "05_how_trials_ended.png")]
    if view in ("detail", "both"):
        d = out / "detail"
        made += [fig_success_by_level(rows, d / "success_by_level.png"),
                 fig_success_vs_height(rows, d / "success_vs_dock_height.png"),
                 fig_height_effects(rows, d / "effects_of_dock_height.png"),
                 fig_docked_over_time(rows, d / "docked_over_time.png"),
                 fig_time_to_dock(rows, d / "time_to_dock.png"),
                 fig_estimation(rows, d / "estimation.png"),
                 fig_safety_effort(rows, d / "safety_effort.png"),
                 fig_paired_effects(rows, d / "paired_difference_by_height_band.png", by="band")]
        levels = _levels(rows)
        if len(levels) > 1:
            for lv in levels:
                rs = [r for r in rows if r["level"] == lv]
                made += [fig_scorecard(rs, d / "rooms" / f"{lv}_scorecard.png", f"{lv}: the four headline metrics ({len(rs)} trials, heights pooled)"),
                         fig_height_effects(rs, d / "rooms" / f"{lv}_dock_height_effects.png", f"{lv}: effect of the dock-marker height")]
    return [m for m in made if m is not None]
