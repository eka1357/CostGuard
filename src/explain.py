"""
CostGuard - SHAP Explainability Engine
Generates local Shapley feature attributions (SHAP) for high-risk transactions
flagged at the cost-optimal decision threshold.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys
import joblib
import numpy as np
import pandas as pd
import shap

# ==========================================
# Configuration & Constants
# ==========================================
DEFAULT_TOP_N: int = 10
DEFAULT_TOP_K_DRIVERS: int = 3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
from src.train import load_artifact, save_artifact
from src.cost_optimizer import optimize_thresholds, DEFAULT_TRANSACTION_VALUE, DEFAULT_INVESTIGATION_COST


def load_model_and_test_data() -> Tuple[Any, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load primary XGBoost model and cached test predictions.
    Returns:
        (xgb_model, X_test, y_test, y_prob, raw_amounts)
    """
    xgb_model = load_artifact("xgb_model.joblib")
    test_cache = load_artifact("test_eval_cache.joblib")
    X_test = test_cache["X_test"]
    y_test = test_cache["y_test"].to_numpy()
    y_prob = test_cache["xgb_cal_probs"]
    raw_amounts = test_cache["raw_amounts"]
    return xgb_model, X_test, y_test, y_prob, raw_amounts


def get_top_flagged_transactions(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    top_n: int = DEFAULT_TOP_N,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Filter transactions flagged at or above the decision threshold,
    and rank them by fraud probability in descending order.

    Returns:
        (X_flagged_top, y_flagged_top, y_prob_top, summary_df)
    """
    flagged_mask = y_prob >= threshold
    if not np.any(flagged_mask):
        raise ValueError(f"No transactions were flagged at threshold {threshold:.4f}")

    flagged_indices = np.where(flagged_mask)[0]
    # Sort flagged instances by risk probability descending
    sorted_order = np.argsort(y_prob[flagged_indices])[::-1]
    top_indices = flagged_indices[sorted_order[:top_n]]

    X_top = X_test.iloc[top_indices].copy()
    y_top = y_test[top_indices]
    prob_top = y_prob[top_indices]

    summary_df = pd.DataFrame({
        "rank": np.arange(1, len(top_indices) + 1),
        "test_index": top_indices,
        "original_index": X_top.index.to_numpy(),
        "fraud_probability": prob_top,
        "ground_truth": y_top,
        "status": np.where(y_top == 1, "True Fraud (TP)", "False Alarm (FP)"),
    })

    return X_top, y_top, prob_top, summary_df


def compute_shap_explanations(
    model: Any,
    X_sample: pd.DataFrame,
) -> shap.Explanation:
    """
    Compute SHAP TreeExplainer values for the given sample feature matrix.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return shap_values


def extract_case_drivers(
    shap_values: shap.Explanation,
    feature_names: List[str],
    case_idx: int,
    top_k: int = DEFAULT_TOP_K_DRIVERS,
) -> Dict[str, Any]:
    """
    Extract the top positive drivers (increasing fraud risk) and
    top negative drivers (mitigating risk) for a single instance.
    """
    vals = shap_values.values[case_idx]
    data = shap_values.data[case_idx]

    # Positive contributions (push toward fraud)
    pos_indices = np.where(vals > 0)[0]
    pos_sorted = pos_indices[np.argsort(vals[pos_indices])[::-1]]
    top_pos = [
        {
            "feature": feature_names[i],
            "value": float(data[i]),
            "shap_attribution": float(vals[i]),
        }
        for i in pos_sorted[:top_k]
    ]

    # Negative contributions (push toward legitimate)
    neg_indices = np.where(vals < 0)[0]
    neg_sorted = neg_indices[np.argsort(vals[neg_indices])]
    top_neg = [
        {
            "feature": feature_names[i],
            "value": float(data[i]),
            "shap_attribution": float(vals[i]),
        }
        for i in neg_sorted[:top_k]
    ]

    return {
        "case_idx": case_idx,
        "base_value": float(
            shap_values.base_values[case_idx]
            if np.ndim(shap_values.base_values) > 0
            else shap_values.base_values
        ),
        "top_positive_drivers": top_pos,
        "top_negative_drivers": top_neg,
    }


def explain_top_flagged(
    top_n: int = DEFAULT_TOP_N,
    transaction_value: float = DEFAULT_TRANSACTION_VALUE,
    investigation_cost: float = DEFAULT_INVESTIGATION_COST,
) -> Dict[str, Any]:
    """
    End-to-end SHAP explanation routine:
    1. Loads model and test predictions.
    2. Identifies cost-optimal threshold.
    3. Extracts top-N highest risk flagged cases.
    4. Computes TreeExplainer attributions.
    5. Saves explanation cache for dashboard use.
    """
    xgb_model, X_test, y_test, y_prob, raw_amounts = load_model_and_test_data()

    # Determine cost-optimal threshold using per-transaction amounts
    opt_summary = optimize_thresholds(
        y_true=y_test,
        y_prob=y_prob,
        amounts=raw_amounts,
        investigation_cost=investigation_cost,
    )
    cost_opt_threshold = opt_summary["cost_optimal"]["threshold"]

    # Filter top flagged cases
    X_top, y_top, prob_top, summary_df = get_top_flagged_transactions(
        X_test=X_test,
        y_test=y_test,
        y_prob=y_prob,
        threshold=cost_opt_threshold,
        top_n=top_n,
    )

    # Compute SHAP values
    shap_values = compute_shap_explanations(xgb_model, X_top)
    feature_names = list(X_top.columns)

    # Extract top drivers for each transaction
    drivers_list = []
    for i in range(len(X_top)):
        drivers = extract_case_drivers(shap_values, feature_names, case_idx=i)
        drivers_list.append(drivers)

    summary_df["top_drivers"] = [
        ", ".join(
            f"{d['feature']} ({d['shap_attribution']:+.2f})"
            for d in item["top_positive_drivers"]
        )
        for item in drivers_list
    ]

    # Save artifacts for dashboard
    cache_payload = {
        "cost_optimal_threshold": cost_opt_threshold,
        "summary_df": summary_df,
        "X_top": X_top,
        "y_top": y_top,
        "prob_top": prob_top,
        "shap_values": shap_values,
        "drivers_list": drivers_list,
        "feature_names": feature_names,
    }
    save_artifact(cache_payload, "shap_explanations.joblib")

    return cache_payload


def print_shap_report(payload: Dict[str, Any]) -> None:
    """Print readable SHAP feature attribution report for top flagged cases."""
    df = payload["summary_df"]
    thresh = payload["cost_optimal_threshold"]
    drivers = payload["drivers_list"]

    print("=" * 86)
    print(f"CostGuard - SHAP Attributions for Top {len(df)} Flagged Alerts")
    print(f"Decision Point: Cost-Optimal Threshold = {thresh:.4f}")
    print("=" * 86)
    print(f"{'Rank':<5} | {'Tx Index':<10} | {'Risk Prob':<10} | {'Verdict':<20} | {'Top Contributing Fraud Drivers'}")
    print("-" * 86)
    for idx, row in df.iterrows():
        print(
            f"{row['rank']:<5} | {row['original_index']:<10} | {row['fraud_probability']:<10.2%} | "
            f"{row['status']:<20} | {row['top_drivers']}"
        )
    print("=" * 86)

    # Print detailed breakdown for #1 ranked case
    case_0 = drivers[0]
    row_0 = df.iloc[0]
    print(f"\nDetailed Breakdown for Highest-Risk Alert (Rank #1, Tx #{row_0['original_index']}):")
    print(f"  Assessed Risk:   {row_0['fraud_probability']:.2%}")
    print(f"  Base Log-Odds:   {case_0['base_value']:.3f}")
    print("  Key Risk Escalators (Positive SHAP):")
    for d in case_0["top_positive_drivers"]:
        print(f"    - {d['feature']:<8} = {d['value']:>9.4f}  (SHAP contribution: {d['shap_attribution']:>+6.3f})")
    print("  Key Protective Factors (Negative SHAP):")
    for d in case_0["top_negative_drivers"]:
        print(f"    - {d['feature']:<8} = {d['value']:>9.4f}  (SHAP contribution: {d['shap_attribution']:>+6.3f})")
    print("=" * 86)


if __name__ == "__main__":
    payload = explain_top_flagged(top_n=DEFAULT_TOP_N)
    print_shap_report(payload)
