"""Data loading, cleaning, binarization, and train/val/test splitting."""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


MISSING_MARKER = "?"
HIGH_MISSING_COLS = ["weight", "payer_code", "medical_specialty"]


def load_data(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the Diabetes 130 Hospitals dataset and return cleaned X, y."""
    openml_id = config["dataset"]["openml_id"]
    print(f"Fetching dataset from OpenML (id={openml_id})...")
    dataset = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
    X: pd.DataFrame = dataset.data.copy()
    y_raw: pd.Series = dataset.target.copy()

    # Binarize target: early readmission (<30 days) = 1, otherwise = 0
    y = (y_raw == "<30").astype(int)
    y.name = "early_readmission"

    # Replace '?' sentinel with NaN
    X = X.replace(MISSING_MARKER, np.nan)

    # Drop high-missingness columns
    drop_cols = config["dataset"]["drop_cols"]
    existing_drop = [c for c in drop_cols if c in X.columns]
    X = X.drop(columns=existing_drop)

    print("\n--- Dataset overview ---")
    print(f"Shape: {X.shape}")
    print(f"\nClass distribution:\n{y.value_counts()}")
    print(f"\nPositive rate: {y.mean():.3f}")

    _report_missing(X)
    _report_dtypes(X)

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """Stratified 70/15/15 train/val/test split."""
    seed = config["dataset"]["random_seed"]
    test_size = config["dataset"]["test_size"]
    val_size = config["dataset"]["val_size"]

    # First cut off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Then cut val from the remaining
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, stratify=y_temp, random_state=seed
    )

    print(f"\n--- Split sizes ---")
    print(f"Train : {len(X_train):>7,} rows  ({y_train.mean():.3f} positive rate)")
    print(f"Val   : {len(X_val):>7,} rows  ({y_val.mean():.3f} positive rate)")
    print(f"Test  : {len(X_test):>7,} rows  ({y_test.mean():.3f} positive rate)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def _report_missing(X: pd.DataFrame) -> None:
    missing = X.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("\nNo missing values.")
    else:
        print("\n--- Missing value rates (top columns) ---")
        print(missing.head(15).to_string())


def _report_dtypes(X: pd.DataFrame) -> None:
    print("\n--- Column dtypes ---")
    dtype_counts = X.dtypes.value_counts()
    print(dtype_counts.to_string())
