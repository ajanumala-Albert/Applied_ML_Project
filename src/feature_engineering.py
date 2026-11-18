"""Custom feature engineering and sklearn transformer definitions."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain-specific engineered features before encoding."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, np.ndarray):
            raise ValueError("Expected a pandas DataFrame.")

        # Feature 1: medication intensity relative to stay length
        if "num_medications" in X.columns and "time_in_hospital" in X.columns:
            num_med = pd.to_numeric(X["num_medications"], errors="coerce").fillna(0)
            time = pd.to_numeric(X["time_in_hospital"], errors="coerce").fillna(1)
            X["num_medications_per_day"] = num_med / (time + 1)

        # Feature 2: total prior visit burden
        visit_cols = ["number_outpatient", "number_emergency", "number_inpatient"]
        present = [c for c in visit_cols if c in X.columns]
        if present:
            visits = sum(
                pd.to_numeric(X[c], errors="coerce").fillna(0) for c in present
            )
            X["total_visits"] = visits

        # Feature 3: diagnostic complexity — has all 3 diagnosis codes filled
        diag_cols = ["diag_1", "diag_2", "diag_3"]
        present_diag = [c for c in diag_cols if c in X.columns]
        if len(present_diag) == 3:
            filled = (
                X["diag_1"].notna()
                & X["diag_2"].notna()
                & X["diag_3"].notna()
            )
            X["high_diag_complexity"] = filled.astype(int)

        return X

    def get_feature_names_out(self, input_features=None):
        return None  # handled downstream


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes, encodes, and scales."""
    # Identify column types after feature engineering (on a small sample)
    sample = X.head(10).copy()
    engineer = DiabetesFeatureEngineer()
    sample_eng = engineer.transform(sample)

    numeric_cols = sample_eng.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = sample_eng.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def report_engineered_features(X: pd.DataFrame) -> None:
    """Print a preview of the 3 engineered features."""
    engineer = DiabetesFeatureEngineer()
    X_eng = engineer.transform(X.head(5))
    new_cols = ["num_medications_per_day", "total_visits", "high_diag_complexity"]
    present = [c for c in new_cols if c in X_eng.columns]
    print("\n--- Engineered features (first 5 rows) ---")
    print(X_eng[present].to_string())
