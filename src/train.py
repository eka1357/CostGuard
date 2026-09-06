"""
CostGuard - Model Training & Calibration Pipeline
Trains baseline Logistic Regression and calibrated XGBoost models
for cost-aware fraud detection, evaluating precision, recall, F1, PR-AUC, and ROC-AUC.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier

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
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)

    y_pred = (y_prob >= threshold).astype(int)

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
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


def train_xgboost_with_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[XGBClassifier, CalibratedClassifierCV]:
    """
    Train XGBoost model and calibrate output probabilities using Platt scaling
    (sigmoid calibration) via 3-fold cross-validation.
    
    Returns:
        (raw_xgb_model, calibrated_model)
        - raw_xgb_model is fitted on all X_train and compatible with SHAP TreeExplainer.
        - calibrated_model outputs calibrated posterior probabilities for cost optimization.
    """
    scale_pos = compute_imbalance_ratio(y_train)
    print(f"Training primary XGBoost classifier (scale_pos_weight={scale_pos:.1f})...")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    print("Fitting calibrated ensemble via CalibratedClassifierCV(cv=3, method='sigmoid')...")
    calibrated_xgb = CalibratedClassifierCV(
        estimator=XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        method="sigmoid",
        cv=3,
    )
    calibrated_xgb.fit(X_train, y_train)

    return xgb, calibrated_xgb


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


def train_and_evaluate_all() -> Dict[str, Any]:
    """
    Train both baseline Logistic Regression and Calibrated XGBoost,
    evaluate on test set, save artifacts, and return comparison summary.
    """
    # 1. Load data
    print("Loading and preparing dataset...")
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    save_artifact(scaler, "scaler.joblib")

    # 2. Train and evaluate Baseline Logistic Regression
    lr_model = train_baseline_logistic_regression(X_train, y_train)
    save_artifact(lr_model, "baseline_lr.joblib")
    lr_results = evaluate_model_performance(lr_model, X_test, y_test, threshold=0.5)

    # 3. Train and calibrate XGBoost
    raw_xgb, calibrated_xgb = train_xgboost_with_calibration(X_train, y_train)
    save_artifact(raw_xgb, "xgb_model.joblib")
    save_artifact(calibrated_xgb, "calibrated_xgb.joblib")

    xgb_raw_results = evaluate_model_performance(raw_xgb, X_test, y_test, threshold=0.5)
    xgb_cal_results = evaluate_model_performance(calibrated_xgb, X_test, y_test, threshold=0.5)

    # Save test set and predictions for fast dashboard loading
    # Include raw (unscaled) Amount for per-transaction cost optimization
    from src.data_prep import load_raw_data, DEFAULT_DATA_PATH
    raw_df = load_raw_data(DEFAULT_DATA_PATH)
    raw_amounts = raw_df.loc[X_test.index, "Amount"].to_numpy()

    test_artifacts = {
        "X_test": X_test,
        "y_test": y_test,
        "lr_probs": lr_results["y_prob"],
        "xgb_cal_probs": xgb_cal_results["y_prob"],
        "raw_amounts": raw_amounts,
    }
    save_artifact(test_artifacts, "test_eval_cache.joblib")

    return {
        "baseline_lr": lr_results,
        "raw_xgb": xgb_raw_results,
        "calibrated_xgb": xgb_cal_results,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("CostGuard - Complete Model Training & Evaluation")
    print("=" * 70)

    results = train_and_evaluate_all()

    lr = results["baseline_lr"]
    xgb_raw = results["raw_xgb"]
    xgb_cal = results["calibrated_xgb"]

    print("\n" + "=" * 70)
    print(f"{'Metric':<20} | {'Baseline LR':<15} | {'XGBoost (Raw)':<15} | {'XGBoost (Calibrated)':<15}")
    print("-" * 70)
    print(f"{'Precision':<20} | {lr['precision']:<15.4f} | {xgb_raw['precision']:<15.4f} | {xgb_cal['precision']:<15.4f}")
    print(f"{'Recall':<20} | {lr['recall']:<15.4f} | {xgb_raw['recall']:<15.4f} | {xgb_cal['recall']:<15.4f}")
    print(f"{'F1-Score':<20} | {lr['f1']:<15.4f} | {xgb_raw['f1']:<15.4f} | {xgb_cal['f1']:<15.4f}")
    print(f"{'PR-AUC':<20} | {lr['pr_auc']:<15.4f} | {xgb_raw['pr_auc']:<15.4f} | {xgb_cal['pr_auc']:<15.4f}")
    print(f"{'ROC-AUC':<20} | {lr['roc_auc']:<15.4f} | {xgb_raw['roc_auc']:<15.4f} | {xgb_cal['roc_auc']:<15.4f}")
    print(f"{'Brier Score':<20} | {lr['brier_score']:<15.4f} | {xgb_raw['brier_score']:<15.4f} | {xgb_cal['brier_score']:<15.4f}")
    print(f"{'True Positives':<20} | {lr['true_positives']:<15} | {xgb_raw['true_positives']:<15} | {xgb_cal['true_positives']:<15}")
    print(f"{'False Positives':<20} | {lr['false_positives']:<15} | {xgb_raw['false_positives']:<15} | {xgb_cal['false_positives']:<15}")
    print(f"{'False Negatives':<20} | {lr['false_negatives']:<15} | {xgb_raw['false_negatives']:<15} | {xgb_cal['false_negatives']:<15}")
    print("=" * 70)
