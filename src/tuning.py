"""Hyperparameter tuning with Optuna + StratifiedKFold cross-validation."""

import csv
import json
import os
from pathlib import Path

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.pipeline import (
    build_logreg_pipeline,
    build_rf_pipeline,
    build_xgb_pipeline,
)
from src.imbalance import compute_scale_pos_weight

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _write_log_header(log_path: str) -> None:
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "trial", "params", "roc_auc_cv", "artifact"])


def _append_run_log(
    log_path: str,
    model_name: str,
    trial_number: int,
    params: dict,
    roc_auc_cv: float,
    artifact_path: str,
) -> None:
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            trial_number,
            json.dumps(params),
            f"{roc_auc_cv:.6f}",
            artifact_path,
        ])


def tune_logreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    cv_folds: int,
    run_log_path: str,
    artifacts_dir: str,
    seed: int = 42,
) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 10.0, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        pipe = build_logreg_pipeline(X_train, C=C, solver=solver, random_state=seed)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    for trial in study.trials:
        _append_run_log(run_log_path, "LogisticRegression",
                        trial.number, trial.params, trial.value or 0.0, "")

    best = study.best_params
    print(f"  [LogReg] Best params: {best}  |  CV ROC-AUC: {study.best_value:.4f}")
    return best


def tune_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    cv_folds: int,
    run_log_path: str,
    artifacts_dir: str,
    seed: int = 42,
) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=25)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        pipe = build_rf_pipeline(
            X_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    for trial in study.trials:
        _append_run_log(run_log_path, "RandomForest",
                        trial.number, trial.params, trial.value or 0.0, "")

    best = study.best_params
    print(f"  [RF] Best params: {best}  |  CV ROC-AUC: {study.best_value:.4f}")
    return best


def tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    cv_folds: int,
    run_log_path: str,
    artifacts_dir: str,
    seed: int = 42,
) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    spw = compute_scale_pos_weight(y_train)

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=25)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        pipe = build_xgb_pipeline(
            X_train,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            scale_pos_weight=spw,
            random_state=seed,
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    for trial in study.trials:
        _append_run_log(run_log_path, "XGBoost",
                        trial.number, trial.params, trial.value or 0.0, "")

    best = study.best_params
    print(f"  [XGB] Best params: {best}  |  CV ROC-AUC: {study.best_value:.4f}")
    return best


def run_all_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> dict:
    """Run Optuna tuning for all three models and return best params dict."""
    n_trials = config["tuning"]["n_trials"]
    cv_folds = config["tuning"]["cv_folds"]
    seed = config["dataset"]["random_seed"]
    run_log = config["paths"]["run_log"]
    artifacts_dir = config["paths"]["artifacts"]
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    _write_log_header(run_log)

    print("\n========== Hyperparameter tuning with Optuna ==========")
    print(f"n_trials={n_trials}, cv_folds={cv_folds}, metric=ROC-AUC")

    print("\n[1/3] Tuning Logistic Regression...")
    lr_params = tune_logreg(X_train, y_train, n_trials, cv_folds,
                             run_log, artifacts_dir, seed)

    print("\n[2/3] Tuning Random Forest...")
    rf_params = tune_rf(X_train, y_train, n_trials, cv_folds,
                        run_log, artifacts_dir, seed)

    print("\n[3/3] Tuning XGBoost...")
    xgb_params = tune_xgb(X_train, y_train, n_trials, cv_folds,
                           run_log, artifacts_dir, seed)

    return {
        "LogisticRegression": lr_params,
        "RandomForest": rf_params,
        "XGBoost": xgb_params,
    }
