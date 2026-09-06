"""
CostGuard - Model Training & Calibration Pipeline
Trains baseline Logistic Regression and tuned calibrated XGBoost models
for fraud detection, reporting precision, recall, F1, and PR-AUC.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

import sys

# ==========================================
# Configuration & Paths
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_RANDOM_STATE = 42

from src.data_prep import prepare_data, compute_imbalance_ratio


def evaluate_model_performance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate classification performance on test set at specified decision threshold.
    Returns dictionary of standard metrics.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
        # Min-max scale decision function to [0, 1] if predict_proba is absent
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)

    y_pred = (y_prob >= threshold).astype(int)

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def train_baseline_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LogisticRegression:
    """
    Train baseline Logistic Regression with balanced class weighting.
    """
    print("Training Baseline Logistic Regression (class_weight='balanced')...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    lr.fit(X_train, y_train)
    return lr


def save_artifact(obj: Any, filename: str) -> Path:
    """Save serialized model artifact to models/ directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / filename
    joblib.dump(obj, filepath)
    return filepath


def load_artifact(filename: str) -> Any:
    """Load serialized model artifact from models/ directory."""
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model artifact not found: {filepath}")
    return joblib.load(filepath)


if __name__ == "__main__":
    print("=" * 65)
    print("CostGuard - Model Training Pipeline: Baseline Logistic Regression")
    print("=" * 65)

    # 1. Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    save_artifact(scaler, "scaler.joblib")

    # 2. Train baseline
    baseline_lr = train_baseline_logistic_regression(X_train, y_train)
    save_artifact(baseline_lr, "baseline_lr.joblib")

    # 3. Evaluate baseline
    lr_results = evaluate_model_performance(baseline_lr, X_test, y_test, threshold=0.5)

    print("\n--- Baseline Logistic Regression Performance (Threshold = 0.50) ---")
    print(f"Precision:         {lr_results['precision']:.4f}")
    print(f"Recall:            {lr_results['recall']:.4f}")
    print(f"F1-Score:          {lr_results['f1']:.4f}")
    print(f"PR-AUC (Avg Prec): {lr_results['pr_auc']:.4f}")
    print(f"ROC-AUC:           {lr_results['roc_auc']:.4f}")
    print(f"Confusion Matrix:  TP={lr_results['true_positives']}, FP={lr_results['false_positives']}, "
          f"TN={lr_results['true_negatives']}, FN={lr_results['false_negatives']}")
    print("=" * 65)
