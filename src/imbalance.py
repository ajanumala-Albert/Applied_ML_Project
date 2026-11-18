"""Class imbalance handling: SMOTE resampling and reporting."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_smote(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to the training set only. Returns resampled arrays."""
    print("\n--- Class imbalance handling ---")
    print("Before SMOTE:")
    _report_distribution(y_train)

    smote = SMOTE(random_state=random_seed, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    _report_distribution(pd.Series(y_res))
    print(
        "\nNote: SMOTE is applied to the training set only. "
        "Validation and test sets remain untouched to avoid data leakage."
    )
    return X_res, y_res


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
