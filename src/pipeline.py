"""Build full sklearn Pipelines for each model type."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.feature_engineering import DiabetesFeatureEngineer


def _make_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def infer_column_types(X: pd.DataFrame) -> tuple[list, list]:
    """Infer numeric and categorical column names after feature engineering."""
    engineer = DiabetesFeatureEngineer()
    X_eng = engineer.transform(X.head(5))
    numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_eng.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return numeric_cols, categorical_cols


def build_logreg_pipeline(
    X: pd.DataFrame,
    C: float = 1.0,
    solver: str = "saga",
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Pipeline:
    numeric_cols, categorical_cols = infer_column_types(X)
    preprocessor = _make_preprocessor(numeric_cols, categorical_cols)
    return Pipeline([
        ("engineer", DiabetesFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=C,
            solver=solver,
            class_weight=class_weight,
            max_iter=2000,
            random_state=random_state,
        )),
    ])


def build_rf_pipeline(
    X: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_leaf: int = 1,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Pipeline:
    numeric_cols, categorical_cols = infer_column_types(X)
    preprocessor = _make_preprocessor(numeric_cols, categorical_cols)
    return Pipeline([
        ("engineer", DiabetesFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=1,
        )),
    ])


def build_xgb_pipeline(
    X: pd.DataFrame,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    n_estimators: int = 100,
    subsample: float = 0.8,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    numeric_cols, categorical_cols = infer_column_types(X)
    preprocessor = _make_preprocessor(numeric_cols, categorical_cols)
    return Pipeline([
        ("engineer", DiabetesFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=1,
            verbosity=0,
        )),
    ])
