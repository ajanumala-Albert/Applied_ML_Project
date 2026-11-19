"""Class imbalance handling for weighted models."""

import pandas as pd

def _report_distribution(y: pd.Series) -> None:
    counts = pd.Series(y).value_counts().sort_index()
    total = len(y)
    for cls, cnt in counts.items():
        label = "Early readmission (<30d)" if cls == 1 else "No early readmission"
        print(f"  Class {cls} ({label}): {cnt:,}  ({cnt/total:.1%})")


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Return scale_pos_weight for XGBoost (ratio of negatives to positives)."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"\nXGBoost scale_pos_weight = {spw:.2f}  (negatives / positives)")
    return float(spw)
