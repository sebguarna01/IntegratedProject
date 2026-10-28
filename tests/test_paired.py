"""Paired comparison of configs: counts, exact test, cluster bootstrap, tolerant CSV reading."""
import math

import numpy as np
import pytest

from activenav.evaluation import paired


def _row(level, config, seed, docked, t=60.0, layout="fixed+dock0.3", outcome=None, z=0.30, rmse=0.02, clr=0.2):
    return dict(level=level, layout=layout, config=config, seed=str(seed), docked="1" if docked else "0",
                outcome=outcome or ("docked" if docked else "timeout"), duration_s=str(t), dock_height_m=str(z),
                pos_rmse_m=str(rmse), final_pos_err_m="0.03", min_clearance_m=str(clr), path_length_m="10", near_time_s="0")


def _rows(n=20, a_extra_wins=14, a_extra_losses=0, level="r1"):
    """B docks on the first 5 seeds only; A docks on those plus `a_extra_wins` more and loses `a_extra_losses` of B's."""
    rows = []
    for s in range(n):
        b = s < 5
        a = b or (s - 5) < a_extra_wins
        if b and s < a_extra_losses:
            a = False
        rows += [_row(level, "B", s, b), _row(level, "C-info", s, a, t=50.0, rmse=0.015)]
    return rows


def test_mcnemar_exact_matches_the_binomial():
    assert paired.mcnemar_exact(0, 0) == 1.0
    assert paired.mcnemar_exact(5, 5) == 1.0
    assert paired.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert paired.mcnemar_exact(8, 2) == pytest.approx(2 * sum(math.comb(10, i) for i in range(3)) / 2 ** 10)


def test_pairs_need_both_configs_on_the_same_level_layout_and_seed():
    rows = [_row("r", "B", 1, 1), _row("r", "C-info", 1, 1), _row("r", "B", 2, 1), _row("r", "C-info", 2, 1, layout="other"),
            _row("r", "C-info", 3, 1)]
    pr = paired.pairs(rows, "C-info", "B")
    assert len(pr) == 1 and pr[0][0] == ("r", 1)


def test_win_loss_counts_and_the_success_difference():
    m = {x["metric"]: x for x in paired.compare(_rows(20, 14, 0), "C-info", "B")}["docked"]
    assert (m["wins"], m["losses"], m["both"], m["neither"]) == (14, 0, 5, 1)
    assert m["diff"] == pytest.approx(100 * 14 / 20)
    assert m["lo"] > 0 and m["p_mcnemar"] < 0.001 and paired.verdict(m) == "better"
    assert m["rate_a"] == pytest.approx(95.0) and m["rate_b"] == pytest.approx(25.0)


def test_a_null_comparison_is_unclear_not_better():
    rows = []
    rng = np.random.default_rng(1)
    for s in range(40):
        d = rng.random() < 0.5
        rows += [_row("r", "B", s, d), _row("r", "C-info", s, d if rng.random() < 0.8 else not d)]
    m = {x["metric"]: x for x in paired.compare(rows, "C-info", "B")}["docked"]
    assert m["lo"] <= 0 <= m["hi"] and paired.verdict(m) == "unclear"


def test_time_uses_only_pairs_where_both_docked_and_a_median_difference():
    ms = {x["metric"]: x for x in paired.compare(_rows(20, 14, 0), "C-info", "B")}
    t = ms["duration_s"]
    assert t["n"] == 5 and t["diff"] == pytest.approx(-10.0)          # A takes 50 s, B 60 s, on the 5 seeds both docked
    assert ms["pos_rmse_m"]["n"] == 20 and ms["pos_rmse_m"]["diff"] == pytest.approx(-0.005)


def test_too_few_seeds_gives_no_interval():
    rows = [_row("r", "B", s, 1) for s in range(3)] + [_row("r", "C-info", s, 1) for s in range(3)]
    m = {x["metric"]: x for x in paired.compare(rows, "C-info", "B")}["docked"]
    assert m["n"] == 3 and m["lo"] != m["lo"] and paired.verdict(m) == "few pairs"


def test_clusters_are_seeds_not_single_trials():
    """Ten dock heights of one seed are one cluster: 3 seeds are too few for an interval however many trials they hold."""
    rows = []
    for s in range(3):
        for z in (0.2, 0.25, 0.3, 0.35, 0.45, 0.6, 0.9, 0.5, 0.55, 0.4):
            rows += [_row("r", "B", s, 0, layout=f"h{z}", z=z), _row("r", "C-info", s, 1, layout=f"h{z}", z=z)]
    m = {x["metric"]: x for x in paired.compare(rows, "C-info", "B")}["docked"]
    assert m["n"] == 30 and m["n_seeds"] == 3 and m["lo"] != m["lo"]


# ------------------------------------------------------------------------------------------------ tolerant CSV reading
def _write(path, lines):
    path.write_text("\n".join(lines) + "\n")


def test_reader_keeps_extra_columns_of_a_stale_header_and_drops_a_truncated_row(tmp_path):
    from activenav.evaluation.stats import read_rows, repair
    fields = ["level", "layout", "config", "seed", "docked", "extra"]
    f = tmp_path / "r.csv"
    # header from an old version (no 'extra'); one row written by a newer version; one truncated row glued to the next complete one
    _write(f, ["level,layout,config,seed,docked",
               "roomA,fixed+dock0.2,B,0,1",
               "roomA,fixed+dock0.2,C-info,0,1,7",
               "roomA,fixed+dock0.3,B,1,0,roomA,fixed+dock0.3,C-info,1,1,9"])
    rows, dropped = read_rows(f, fields)
    assert [(r["config"], r["seed"], r.get("extra")) for r in rows] == [("B", "0", None), ("C-info", "0", "7"), ("C-info", "1", "9")]
    assert dropped == 1
    kept, dropped = repair(f, fields)
    assert (kept, dropped) == (3, 1) and (tmp_path / "r.csv.bak").exists()
    assert f.read_text().splitlines()[0] == ",".join(fields)
    rows2, d2 = read_rows(f, fields)
    assert len(rows2) == 3 and d2 == 0


def test_apply_cap_turns_long_runs_into_timeouts_and_leaves_short_ones():
    from activenav.evaluation.stats import apply_cap
    short = _row("r", "B", 1, 1, t=120.0)
    long_ok = _row("r", "B", 2, 1, t=250.0)
    long_to = _row("r", "B", 3, 0, t=900.0)
    out = apply_cap([short, long_ok, long_to], 200)
    assert out[0] == short                                           # untouched
    assert out[1]["outcome"] == "timeout" and out[1]["docked"] == "0" and out[1]["duration_s"] == "200" and out[1]["pos_rmse_m"] == ""
    assert out[2]["outcome"] == "timeout" and out[2]["duration_s"] == "200"
    assert apply_cap([long_ok], None) == [long_ok]
    assert long_ok["outcome"] == "docked"                            # the input rows are not modified
