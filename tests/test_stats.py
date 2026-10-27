import numpy as np

from activenav.evaluation.stats import bootstrap_ci, wilson_interval


def test_wilson_zero_successes():
    lo, hi = wilson_interval(0, 30)
    assert lo == 0.0
    assert abs(hi - 0.1135) < 0.002


def test_wilson_half():
    lo, hi = wilson_interval(50, 100)
    assert abs(lo - 0.4038) < 0.002 and abs(hi - 0.5962) < 0.002


def test_bootstrap_ci_contains_median_and_is_reproducible():
    x = np.random.default_rng(0).normal(5.0, 1.0, 200)
    a = bootstrap_ci(x, seed=1)
    assert a == bootstrap_ci(x, seed=1)
    assert a[0] < np.median(x) < a[1]
