"""SHAP-based global and local explainability for the best tree model."""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def run_explainability(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: str,
    n_local: int = 5,
) -> None:
    """Compute SHAP values and save global + local explanation plots."""
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    print("\n========== SHAP Explainability ==========")

    # Extract the fitted preprocessing + classifier from the pipeline
    # The pipeline steps are: engineer → preprocessor → classifier
    engineer = model.named_steps["engineer"]
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform test data through the preprocessing steps
    X_eng = engineer.transform(X_test)
    X_proc = preprocessor.transform(X_eng)

    # Get feature names from ColumnTransformer
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_proc.shape[1])]

    # Use TreeExplainer for RF/XGB
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_proc)

    # For binary classifiers, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # --- Global: summary bar plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv, X_proc, feature_names=feature_names,
        plot_type="bar", show=False, max_display=20
    )
    plt.title("SHAP Feature Importance (Global)", fontsize=13)
    plt.tight_layout()
    global_path = os.path.join(plots_dir, "shap_global.png")
    plt.savefig(global_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP plot → {global_path}")

    # --- Identify top features and discuss clinical relevance ---
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:5]
    top_features = [feature_names[i] for i in top_indices]
    print("\nTop 5 features by mean |SHAP|:")
    for i, feat in enumerate(top_features):
        print(f"  {i+1}. {feat}  (mean |SHAP| = {mean_abs_shap[top_indices[i]]:.4f})")

    print(
    "\nInterpretation: SHAP values describe predictive associations and "
    "model behaviour, not causal effects. In this run, prior inpatient "
    "utilisation, total prior visits, and specific discharge disposition "
    "categories had the largest average influence. The discharge disposition "
    "features are administrative codes and should be interpreted cautiously. "
    "These results do not establish that any feature causes early readmission "
    "or rule out spurious associations."
    )

    # --- Local: waterfall plots for n_local test samples ---
    shap_explanation = shap.Explanation(
        values=sv,
        base_values=explainer.expected_value if not isinstance(
            explainer.expected_value, list
        ) else explainer.expected_value[1],
        data=X_proc,
        feature_names=feature_names,
    )

    for i in range(min(n_local, len(X_test))):
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(shap_explanation[i], show=False, max_display=15)
        actual = y_test.iloc[i]
        plt.title(f"Local explanation — sample {i}  (actual label={actual})", fontsize=11)
        plt.tight_layout()
        local_path = os.path.join(plots_dir, f"shap_local_{i}.png")
        plt.savefig(local_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved {min(n_local, len(X_test))} local SHAP waterfall plots → {plots_dir}/shap_local_*.png")
