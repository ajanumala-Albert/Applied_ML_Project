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

    # Replace the dataset's '?' sentinel with NaN.
    # The literal "None" values in max_glu_serum and A1Cresult
    # are retained as explicit categories because "not measured"
    # may itself contain useful information.
    X = X.replace(MISSING_MARKER, np.nan)

    # Drop the columns selected in config.yaml.
    # This is a project-specific decision based on the EDA missingness profile.
    drop_cols = config["dataset"]["drop_cols"]
    existing_drop = [
        c for c in drop_cols
        if c in X.columns
    ]

    print("\n--- Data cleaning treatment ---")

    identifier_cols = {
        "encounter_id",
        "patient_nbr",
    }

    for col in existing_drop:
        if col in identifier_cols:
            print(f"Dropping {col}: identifier column")
        else:
            missing_rate = X[col].isna().mean() * 100
            print(
                f"Dropping {col}: "
                f"{missing_rate:.2f}% missing values"
            )

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
    """Split encounters by patient to prevent patient overlap between partitions and data leakage."""

    seed = config["dataset"]["random_seed"]
    test_size = config["dataset"]["test_size"]
    val_size = config["dataset"]["val_size"]
    group_col = config["dataset"]["group_col"]

    if group_col not in X.columns:
        raise ValueError(f"Missing grouping column: {group_col}")

    # Create one stratification label per patient.
    # This is used only to balance the patient groups across splits.
    # A patient is positive if any of their encounters is positive.
    patient_labels = (
        pd.DataFrame({
            group_col: X[group_col],
            "target": y,
        })
        .groupby(group_col)["target"]
        .max()
    )

    train_patients, temporary_patients = train_test_split(
        patient_labels.index,
        test_size=test_size + val_size,
        stratify=patient_labels,
        random_state=seed,
    )

    val_patients, test_patients = train_test_split(
        temporary_patients,
        test_size=test_size / (test_size + val_size),
        stratify=patient_labels.loc[temporary_patients],
        random_state=seed,
    )

    def select_patients(patient_ids):
        mask = X[group_col].isin(patient_ids)

        return (
            X.loc[mask].drop(columns=[group_col]),
            y.loc[mask],
        )

    X_train, y_train = select_patients(train_patients)
    X_val, y_val = select_patients(val_patients)
    X_test, y_test = select_patients(test_patients)

    assert set(train_patients).isdisjoint(val_patients)
    assert set(train_patients).isdisjoint(test_patients)
    assert set(val_patients).isdisjoint(test_patients)

    print("\n--- Patient-level split sizes ---")
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
