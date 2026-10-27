"""The controller/estimator must never touch simulator ground truth.

Only src/activenav/sim (defines it) and src/activenav/evaluation (uses it) may mention it.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "activenav"
ALLOWED = {"sim", "evaluation"}


def test_no_ground_truth_outside_evaluation():
    offenders = []
    for f in SRC.rglob("*.py"):
        if f.relative_to(SRC).parts[0] in ALLOWED:
            continue
        if "ground_truth" in f.read_text():
            offenders.append(str(f.relative_to(SRC)))
    assert not offenders, f"ground_truth referenced outside evaluation/: {offenders}"
