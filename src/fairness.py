"""Fairness & bias audit on the sensitive attribute 'race'."""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score


def _group_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray,
                   groups: pd.Series) -> pd.DataFrame:
    """Return per-group TPR, FPR, positive prediction rate, ROC-AUC."""
    rows = []
    for group in sorted(groups.unique()):
        mask = groups == group
        yt = y_true[mask]
        yp = y_pred[mask]
        ypr = y_prob[mask]
        if yt.sum() == 0 or (yt == 0).sum() == 0:
            continue  # skip groups with only one class
        tp = ((yt == 1) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ppr = yp.mean()  # positive prediction rate
        try:
            auc = roc_auc_score(yt, ypr)
        except Exception:
            auc = float("nan")
        rows.append({"group": group, "n": int(mask.sum()),
                     "TPR": tpr, "FPR": fpr, "PPR": ppr, "ROC_AUC": auc})
    return pd.DataFrame(rows).set_index("group")


def _demographic_parity_diff(ppr: pd.Series) -> float:
    return float(ppr.max() - ppr.min())


def _equalized_odds_tpr_gap(tpr: pd.Series) -> float:
    return float(tpr.max() - tpr.min())


def _compute_reweigh_weights(y_train: pd.Series, groups_train: pd.Series) -> np.ndarray:
    """
    Reweighing mitigation: assign sample weights inversely proportional to
    P(group) * P(label|group), so each (group, label) cell gets equal weight.
    """
    n = len(y_train)
    weights = np.ones(n)
    for group in groups_train.unique():
        for label in [0, 1]:
            mask = (groups_train == group) & (y_train == label)
            count = mask.sum()
            if count == 0:
                continue
            p_group = (groups_train == group).sum() / n
            p_label = (y_train == label).sum() / n
            p_joint = count / n
            # weight = P(group) * P(label) / P(group, label)  [ideal = 1 if independent]
            w = (p_group * p_label) / p_joint
            weights[mask.values] = w
    # Normalise so sum equals n
    weights = weights * (n / weights.sum())
    return weights


def run_fairness_audit(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_col: str,
    plots_dir: str,
    seed: int = 42,
) -> None:
    """Full fairness audit: metrics before/after reweighing mitigation."""
    from src.pipeline import build_xgb_pipeline
    from src.imbalance import compute_scale_pos_weight

    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    print("\n========== Fairness & Bias Audit ==========")
    print(f"Sensitive attribute: '{sensitive_col}'")

    # Extract race from raw test/train (before preprocessing)
    if sensitive_col not in X_test.columns:
        print(f"Warning: '{sensitive_col}' not found in X_test. Skipping fairness audit.")
        return

    groups_test = X_test[sensitive_col].astype(object).fillna("Unknown")
    groups_train = X_train[sensitive_col].astype(object).fillna("Unknown")

    print(f"\nGroup distribution in test set:\n{groups_test.value_counts().to_string()}")

    # --- BEFORE mitigation ---
    y_pred_before = model.predict(X_test)
    y_prob_before = model.predict_proba(X_test)[:, 1]
    metrics_before = _group_metrics(y_test, y_pred_before, y_prob_before, groups_test)

    dp_before = _demographic_parity_diff(metrics_before["PPR"])
    eo_before = _equalized_odds_tpr_gap(metrics_before["TPR"])

    print("\n--- Before mitigation ---")
    print(metrics_before[["n", "TPR", "FPR", "PPR", "ROC_AUC"]].round(4).to_string())
    print(f"\nDemographic Parity Difference (PPR gap): {dp_before:.4f}")
    print(f"Equalized Odds — TPR gap:                 {eo_before:.4f}")

    # --- Reweighing mitigation ---
    print("\n--- Applying Reweighing mitigation ---")
    sample_weights = _compute_reweigh_weights(y_train, groups_train)
    spw = compute_scale_pos_weight(y_train)

    # Refit final model (XGBoost) with sample weights
    # We need to pass sample_weight to the classifier's fit method.
    # This requires fitting the pipeline in two stages.
    from src.feature_engineering import DiabetesFeatureEngineer
    from xgboost import XGBClassifier

    engineer = model.named_steps["engineer"]
    preprocessor = model.named_steps["preprocessor"]

    X_train_eng = engineer.transform(X_train)
    X_train_proc = preprocessor.transform(X_train_eng)

    xgb_params = model.named_steps["classifier"].get_params()
    fair_clf = XGBClassifier(**xgb_params)
    fair_clf.fit(X_train_proc, y_train, sample_weight=sample_weights)

    # Predict on test using the same pre-fitted preprocessor
    X_test_eng = engineer.transform(X_test)
    X_test_proc = preprocessor.transform(X_test_eng)

    y_pred_after = fair_clf.predict(X_test_proc)
    y_prob_after = fair_clf.predict_proba(X_test_proc)[:, 1]
    metrics_after = _group_metrics(y_test, y_pred_after, y_prob_after, groups_test)

    dp_after = _demographic_parity_diff(metrics_after["PPR"])
    eo_after = _equalized_odds_tpr_gap(metrics_after["TPR"])

    print("\n--- After Reweighing mitigation ---")
    print(metrics_after[["n", "TPR", "FPR", "PPR", "ROC_AUC"]].round(4).to_string())
    print(f"\nDemographic Parity Difference (PPR gap): {dp_after:.4f}")
    print(f"Equalized Odds — TPR gap:                 {eo_after:.4f}")

    # --- Summary comparison ---
    overall_auc_before = roc_auc_score(y_test, y_prob_before)
    overall_auc_after = roc_auc_score(y_test, y_prob_after)

    print("\n--- Before vs After summary ---")
    print(f"{'Metric':<35} {'Before':>8} {'After':>8}")
    print("-" * 55)
    print(f"{'Demographic Parity Diff (PPR gap)':<35} {dp_before:>8.4f} {dp_after:>8.4f}")
    print(f"{'Equalized Odds (TPR gap)':<35} {eo_before:>8.4f} {eo_after:>8.4f}")
    print(f"{'Overall ROC-AUC':<35} {overall_auc_before:>8.4f} {overall_auc_after:>8.4f}")
    print(
        "\nNote: Reweighing trades a small amount of overall AUC for reduced "
        "disparity between racial groups. The appropriate fairness criterion "
        "(demographic parity vs equalized odds) depends on the downstream use case."
    )

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (df, title) in zip(axes, [
        (metrics_before, "Before Reweighing"),
        (metrics_after, "After Reweighing"),
    ]):
        df_plot = df[df.index != "Unknown"] if "Unknown" in df.index else df
        df_plot["TPR"].plot(kind="bar", ax=ax, color="steelblue", alpha=0.8)
        ax.set_title(f"TPR by Race Group\n{title}", fontsize=11)
        ax.set_xlabel("Race")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Fairness Audit — Equalized Odds (TPR by Race)", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, "fairness_tpr_by_race.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved fairness plot → {out_path}")
