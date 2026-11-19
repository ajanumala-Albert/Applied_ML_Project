"""Baseline model training, evaluation, and comparison."""

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

from src.pipeline import (
    build_logreg_pipeline,
    build_rf_pipeline,
    build_xgb_pipeline,
)
from src.imbalance import compute_scale_pos_weight


def evaluate(model, X, y, label: str = "") -> dict:
    """Return a dict of evaluation metrics for a fitted model."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y, y_prob),
        "f1": f1_score(y, y_pred, zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "brier": brier_score_loss(y, y_prob),
    }
    if label:
        print(f"\n--- {label} ---")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v:.4f}")
    return metrics


def train_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> dict:
    """Train three baseline models and report validation metrics."""
    spw = compute_scale_pos_weight(y_train)
    results = {}

    print("\n========== Baseline model training ==========")

    # --- Logistic Regression ---
    print("\nTraining Logistic Regression (baseline)...")
    lr = build_logreg_pipeline(X_train, C=1.0, solver="saga",
                                class_weight="balanced", random_state=random_state)
    lr.fit(X_train, y_train)
    results["LogisticRegression"] = {
        "model": lr,
        "val_metrics": evaluate(lr, X_val, y_val, "LogisticRegression — Validation"),
    }

    # --- Random Forest ---
    print("\nTraining Random Forest...")
    rf = build_rf_pipeline(X_train, n_estimators=100, max_depth=10,
                            class_weight="balanced", random_state=random_state)
    rf.fit(X_train, y_train)
    results["RandomForest"] = {
        "model": rf,
        "val_metrics": evaluate(rf, X_val, y_val, "RandomForest — Validation"),
    }

    # --- XGBoost ---
    print("\nTraining XGBoost...")
    xgb = build_xgb_pipeline(
        X_train, learning_rate=0.1, max_depth=6, n_estimators=100,
        scale_pos_weight=spw, random_state=random_state
    )
    xgb.fit(X_train, y_train)
    results["XGBoost"] = {
        "model": xgb,
        "val_metrics": evaluate(xgb, X_val, y_val, "XGBoost — Validation"),
    }

    _print_comparison_table(results)
    return results


def _print_comparison_table(results: dict) -> None:
    print("\n========== Baseline comparison (validation set) ==========")
    header = f"{'Model':<22} {'ROC-AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Brier':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        m = r["val_metrics"]
        print(
            f"{name:<22} {m['roc_auc']:>8.4f} {m['f1']:>8.4f} "
            f"{m['precision']:>10.4f} {m['recall']:>8.4f} {m['brier']:>8.4f}"
        )

    print(
        "\nNote: Because early readmission represents approximately 11% "
        "of records, accuracy alone could be misleading. In this project, "
        "recall is treated as our primary metric because our objective is "
        "to identify as many patients at risk of early readmission as "
        "possible for preventive support. Precision and F1 are also "
        "reported to quantify the false-alert trade-off, while ROC-AUC "
        "evaluates ranking ability and Brier score evaluates probability "
        "quality."
    )
