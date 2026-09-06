"""
CostGuard - Data Preparation Pipeline
Handles loading raw credit card transaction data, validation, stratified train/test
splitting, feature scaling (leakage-free), and class imbalance handling.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE  # noqa: F401 — available for experimentation

# ==========================================
# Configuration & Constants
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
TARGET_COLUMN = "Class"
FEATURES_TO_SCALE = ["Time", "Amount"]
DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42


def load_raw_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load raw transaction dataset from CSV.
    Raises FileNotFoundError if file does not exist.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please ensure 'creditcard.csv' is placed in the data/ directory."
        )
    df = pd.read_csv(data_path)
    return df


def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data schema, report null values, and ensure valid target distribution.
    Returns cleaned DataFrame.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    # Check for missing values
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        # In creditcard dataset, nulls should be zero; if any, drop them cleanly
        df = df.dropna().reset_index(drop=True)

    # Ensure target is binary {0, 1}
    unique_targets = set(df[TARGET_COLUMN].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Unexpected target values: {unique_targets}. Expected subset of {{0, 1}}.")

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified split into train and test sets to preserve the minority class ratio.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features_to_scale: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Scale specified features (Time, Amount) using RobustScaler (resilient to outliers).
    Scaler is strictly fitted on X_train only to prevent data leakage into X_test.
    """
    if features_to_scale is None:
        features_to_scale = FEATURES_TO_SCALE

    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    cols_present = [col for col in features_to_scale if col in X_train.columns]
    if cols_present:
        # Fit scaler exclusively on training data
        scaler.fit(X_train[cols_present])
        X_train_scaled[cols_present] = scaler.transform(X_train[cols_present])
        X_test_scaled[cols_present] = scaler.transform(X_test[cols_present])

    return X_train_scaled, X_test_scaled, scaler




def compute_imbalance_ratio(y: pd.Series) -> float:
    """Compute ratio of majority class (0) to minority class (1)."""
    counts = y.value_counts()
    neg_count = counts.get(0, 0)
    pos_count = counts.get(1, 0)
    if pos_count == 0:
        return float("inf")
    return float(neg_count / pos_count)


def prepare_data(
    data_path: Path = DEFAULT_DATA_PATH,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, RobustScaler]:
    """
    End-to-end data preparation entrypoint:
    Loads -> Validates -> Stratified Splits -> Scales features (leakage-free).
    """
    df = load_raw_data(data_path)
    df = validate_and_clean_data(df)
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=test_size, random_state=random_state
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    print("=" * 60)
    print("CostGuard - Data Preparation Pipeline Execution")
    print("=" * 60)

    # 1. Load and clean
    raw_df = load_raw_data()
    print(f"Loaded raw dataset from: {DEFAULT_DATA_PATH}")
    print(f"Total Transactions:      {len(raw_df):,}")
    print(f"Total Columns:           {raw_df.shape[1]}")

    cleaned_df = validate_and_clean_data(raw_df)
    missing_values = cleaned_df.isnull().sum().sum()
    print(f"Missing Values:          {missing_values}")

    # Class balance in full dataset
    full_fraud_count = int(cleaned_df[TARGET_COLUMN].sum())
    full_legit_count = len(cleaned_df) - full_fraud_count
    fraud_pct = (full_fraud_count / len(cleaned_df)) * 100
    print(f"Class Balance:           Legitimate={full_legit_count:,} | Fraud={full_fraud_count:,} ({fraud_pct:.3f}%)")

    # 2. Stratified train/test split & scaling
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    scale_pos_weight = compute_imbalance_ratio(y_train)

    print("\n--- Train / Test Split Summary ---")
    print(f"Train Set Shape:         {X_train.shape} (Fraud: {y_train.sum():,}, Legit: {(y_train == 0).sum():,})")
    print(f"Test Set Shape:          {X_test.shape}  (Fraud: {y_test.sum():,}, Legit: {(y_test == 0).sum():,})")
    print(f"Imbalance Ratio (N/P):   {scale_pos_weight:.1f}:1")
    print(f"Features Scaled:         {FEATURES_TO_SCALE} via RobustScaler (fitted on train only)")
    print("Data preparation complete and verified successfully.")
    print("=" * 60)
