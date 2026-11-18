"""Calibration diagnostics: reliability curve and Brier score."""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def run_calibration_diagnostics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: str,
    model_name: str = "Final Model",
) -> dict:
    """Plot reliability curve and report Brier score."""
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_prob)

    # Reliability (calibration) curve
    fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "o-", color="steelblue",
            label=f"{model_name}\n(Brier = {brier:.4f})")
    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)
    ax.set_title("Reliability Curve (Calibration)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(plots_dir, "calibration_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Interpret calibration direction
    overconf_bins = (mean_pred > fraction_pos).sum()
    underconf_bins = (mean_pred < fraction_pos).sum()

    print("\n========== Calibration Diagnostics ==========")
    print(f"Brier score : {brier:.4f}  (lower is better; perfect = 0)")
    print(f"Saved reliability curve → {out_path}")

    if overconf_bins > underconf_bins:
        direction = "OVER-confident"
        note = (
            "The model assigns higher probabilities than justified — "
            "predicted ~30% doesn't mean 30% actually readmit. "
            "Consider Platt scaling or isotonic regression before using "
            "probabilities directly in downstream decisions (e.g. care routing)."
        )
    else:
        direction = "UNDER-confident"
        note = (
            "The model is conservative — predicted probabilities are lower "
            "than the true positive rate. The model's scores rank patients "
            "well (AUC) but should not be used as raw probability estimates "
            "without recalibration."
        )

    print(f"Calibration direction: {direction}")
    print(f"Interpretation: {note}")

    return {"brier_score": brier, "direction": direction}
