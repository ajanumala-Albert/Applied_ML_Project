"""
Early Readmission Prediction — Diabetes 130 Hospitals
======================================================
Dataset:  OpenML ID 4541 (Option 3)
Target:   early_readmission (1 = readmitted within 30 days)
Sensitive attribute: race

Run with:
    python main.py
"""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

# Ensure project root is on the path when running from subdirectory
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_data, split_data
from src.feature_engineering import report_engineered_features
from src.imbalance import  compute_scale_pos_weight
from src.models import train_baselines, evaluate
from src.tuning import run_all_tuning
from src.pipeline import (
    build_logreg_pipeline,
    build_rf_pipeline,
    build_xgb_pipeline,
    infer_column_types,
)
from src.explainability import run_explainability
from src.calibration import run_calibration_diagnostics
from src.fairness import run_fairness_audit


_ROOT = Path(__file__).parent


def load_config(path: Path = _ROOT / "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print(" Early Readmission Prediction — Diabetes 130 Hospitals")
    print("=" * 60)

    config = load_config()
    seed = config["dataset"]["random_seed"]
    artifacts_dir = str(_ROOT / config["paths"]["artifacts"])
    plots_dir = str(_ROOT / config["paths"]["plots"])
    config["paths"]["artifacts"] = artifacts_dir
    config["paths"]["plots"] = plots_dir
    config["paths"]["run_log"] = str(_ROOT / config["paths"]["run_log"])
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    X, y = load_data(config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)

    # ------------------------------------------------------------------ #
    # 2. Feature engineering preview                                       #
    # ------------------------------------------------------------------ #
    report_engineered_features(X_train)

    # ------------------------------------------------------------------ #
    # 3. Baseline models (trained on raw train; SMOTE applied inside CV)  #
    # ------------------------------------------------------------------ #
    baseline_results = train_baselines(X_train, y_train, X_val, y_val, seed=seed)

    # ------------------------------------------------------------------ #
    # 4. Hyperparameter tuning — subsample to 20 % for speed/RAM         #
    #    (best params are then applied to the full train+val set)         #
    # ------------------------------------------------------------------ #
    tune_frac = config.get("tuning", {}).get("subsample_frac", 0.2)
    X_tune = X_train.sample(frac=tune_frac, random_state=seed)
    y_tune = y_train.loc[X_tune.index]
    print(f"\nTuning on {len(X_tune):,} rows ({int(tune_frac*100)}% subsample of training set)")
    best_params = run_all_tuning(X_tune, y_tune, config)

    # ------------------------------------------------------------------ #
    # 5. Train final model on train+val with best XGBoost params          #
    #    (XGBoost typically wins on tabular data; check baseline results)  #
    # ------------------------------------------------------------------ #
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    spw = compute_scale_pos_weight(y_trainval)
    xgb_best = best_params["XGBoost"]

    print("\n========== Training final model on train+val ==========")
    final_model = build_xgb_pipeline(
        X_trainval,
        learning_rate=xgb_best.get("learning_rate", 0.1),
        max_depth=xgb_best.get("max_depth", 6),
        n_estimators=xgb_best.get("n_estimators", 100),
        subsample=xgb_best.get("subsample", 0.8),
        scale_pos_weight=spw,
        random_state=seed,
    )
    final_model.fit(X_trainval, y_trainval)
    evaluate(final_model, X_test, y_test, "Final XGBoost model — Test set")

    # Save final model
    model_path = os.path.join(artifacts_dir, "final_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"\nFinal model saved → {model_path}")

    # ------------------------------------------------------------------ #
    # 6. SHAP explainability                                               #
    # ------------------------------------------------------------------ #
    run_explainability(final_model, X_test, y_test, plots_dir)

    # ------------------------------------------------------------------ #
    # 7. Calibration diagnostics                                           #
    # ------------------------------------------------------------------ #
    run_calibration_diagnostics(final_model, X_test, y_test, plots_dir,
                                 model_name="XGBoost (tuned)")

    # ------------------------------------------------------------------ #
    # 8. Fairness & bias audit                                             #
    # ------------------------------------------------------------------ #
    sensitive_col = config["dataset"]["sensitive_attr"]
    run_fairness_audit(
        final_model,
        X_trainval, y_trainval,
        X_test, y_test,
        sensitive_col=sensitive_col,
        plots_dir=plots_dir,
        seed=seed,
    )

    print("\n" + "=" * 60)
    print(" Done. Outputs:")
    print(f"   Plots     : {plots_dir}")
    print(f"   Artifacts : {artifacts_dir}")
    print(f"   Run log   : {config['paths']['run_log']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
