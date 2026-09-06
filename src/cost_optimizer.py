"""
CostGuard - Cost Matrix & Threshold Optimization Engine
Optimizes the classification threshold to maximize net financial dollars saved
rather than optimizing for statistical heuristics like F1-Score or Accuracy.

Key design decision: savings are computed per-transaction using the actual
Amount column from the dataset, not a flat assumed value.  The investigation
cost (labor/ops per alert) remains a scalar because it does not vary by
transaction.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# Named Cost Matrix Assumptions (Constants)
# ==========================================
# Stated assumptions for GIBC V2 Track 02 submission:
# 1. Transaction value: actual per-transaction Amount from the dataset.
#    A flat fallback (DEFAULT_TRANSACTION_VALUE) is available when Amount
#    is not supplied, for backward compatibility and unit tests.
# 2. Cost to review/investigate each flagged alert (labor/ops): $8.00
# 3. Investigation cost applies to all flagged alerts (both TP and FP)
DEFAULT_TRANSACTION_VALUE: float = 100.0   # flat fallback only
DEFAULT_INVESTIGATION_COST: float = 8.0
DEFAULT_INVESTIGATE_ALL_ALERTS: bool = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
from src.train import load_artifact


def calculate_savings_and_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    amounts: Optional[np.ndarray] = None,
    transaction_value: float = DEFAULT_TRANSACTION_VALUE,
    investigation_cost: float = DEFAULT_INVESTIGATION_COST,
    investigate_all_alerts: bool = DEFAULT_INVESTIGATE_ALL_ALERTS,
) -> Dict[str, Any]:
    """
    Calculate confusion matrix, statistical metrics, and net financial dollars saved
    at a specific decision threshold.

    When `amounts` is provided, fraud loss per transaction is the actual Amount.
    When `amounts` is None, the flat `transaction_value` is used for every fraud.

    Financial Formulation (per-transaction mode):
    - Baseline Loss = sum(Amount_i for all fraud transactions)
    - Fraud Loss Prevented = sum(Amount_i for caught frauds, i.e. True Positives)
    - Investigation Expenses = num_alerts * investigation_cost
    - Net Dollars Saved = Fraud Loss Prevented - Investigation Expenses
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    fraud_mask = (y_true == 1)

    if amounts is not None:
        amounts = np.asarray(amounts, dtype=float)
        baseline_loss = float(np.sum(amounts[fraud_mask]))
        caught_mask = fraud_mask & (y_pred == 1)
        fraud_loss_prevented = float(np.sum(amounts[caught_mask]))
    else:
        total_frauds = int(tp + fn)
        baseline_loss = total_frauds * transaction_value
        fraud_loss_prevented = tp * transaction_value

    total_alerts = int(tp + fp)
    if investigate_all_alerts:
        investigation_expenses = total_alerts * investigation_cost
    else:
        investigation_expenses = fp * investigation_cost

    dollars_saved = fraud_loss_prevented - investigation_expenses
    model_total_loss = baseline_loss - dollars_saved

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_alerts": total_alerts,
        "baseline_loss": float(baseline_loss),
        "fraud_loss_prevented": float(fraud_loss_prevented),
        "model_total_loss": float(model_total_loss),
        "investigation_expenses": float(investigation_expenses),
        "dollars_saved": float(dollars_saved),
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    transaction_value: float = DEFAULT_TRANSACTION_VALUE,
    investigation_cost: float = DEFAULT_INVESTIGATION_COST,
    num_steps: int = 500,
    investigate_all_alerts: bool = DEFAULT_INVESTIGATE_ALL_ALERTS,
) -> pd.DataFrame:
    """
    Evaluate savings and performance across a dense spectrum of candidate thresholds.

    Per-transaction mode (amounts provided):
        Uses O(N * T) vectorized computation — for each threshold, sums the Amount
        of caught frauds and subtracts investigation costs.

    Flat mode (amounts=None):
        Uses O(N log N) searchsorted for instantaneous TP/FP counting.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    total_pos = int(np.sum(pos_mask))
    total_neg = int(np.sum(neg_mask))

    thresholds = np.linspace(0.001, 0.999, num_steps)

    # Pre-sort for fast binary search of TP/FP counts
    sorted_pos_probs = np.sort(y_prob[pos_mask])
    sorted_neg_probs = np.sort(y_prob[neg_mask])

    tp_arr = len(sorted_pos_probs) - np.searchsorted(sorted_pos_probs, thresholds, side="left")
    fp_arr = len(sorted_neg_probs) - np.searchsorted(sorted_neg_probs, thresholds, side="left")
    fn_arr = total_pos - tp_arr
    tn_arr = total_neg - fp_arr
    total_alerts = tp_arr + fp_arr

    with np.errstate(divide="ignore", invalid="ignore"):
        precision_arr = np.where(total_alerts > 0, tp_arr / total_alerts, 0.0)
        recall_arr = np.where(total_pos > 0, tp_arr / total_pos, 0.0)
        denom = precision_arr + recall_arr
        f1_arr = np.where(denom > 0, 2.0 * precision_arr * recall_arr / denom, 0.0)

    if investigate_all_alerts:
        investigation_expenses = total_alerts * investigation_cost
    else:
        investigation_expenses = fp_arr * investigation_cost

    if amounts is not None:
        # Per-transaction savings: sum Amount of caught frauds at each threshold
        amounts = np.asarray(amounts, dtype=float)
        fraud_amounts = amounts[pos_mask]
        fraud_probs = y_prob[pos_mask]

        # Sort fraud transactions by probability for efficient prefix-sum computation
        sort_idx = np.argsort(fraud_probs)
        fraud_probs_sorted = fraud_probs[sort_idx]
        fraud_amounts_sorted = fraud_amounts[sort_idx]
        cumsum_amounts = np.cumsum(fraud_amounts_sorted)
        total_fraud_amount = cumsum_amounts[-1] if len(cumsum_amounts) > 0 else 0.0

        # For each threshold, caught frauds are those with prob >= threshold
        # Using searchsorted: index i means i elements have prob < threshold
        cut_indices = np.searchsorted(fraud_probs_sorted, thresholds, side="left")
        # Amount of caught frauds = total_fraud_amount - cumsum up to cut_index
        fraud_loss_prevented = np.where(
            cut_indices > 0,
            total_fraud_amount - cumsum_amounts[cut_indices - 1],
            total_fraud_amount,
        )

        baseline_loss = total_fraud_amount
    else:
        # Flat mode: each fraud is worth transaction_value
        fraud_loss_prevented = tp_arr * transaction_value
        baseline_loss = total_pos * transaction_value

    dollars_saved = fraud_loss_prevented - investigation_expenses
    model_total_loss = baseline_loss - dollars_saved

    df_sweep = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision_arr,
        "recall": recall_arr,
        "f1": f1_arr,
        "true_positives": tp_arr,
        "false_positives": fp_arr,
        "true_negatives": tn_arr,
        "false_negatives": fn_arr,
        "total_alerts": total_alerts,
        "baseline_loss": baseline_loss,
        "fraud_loss_prevented": fraud_loss_prevented,
        "model_total_loss": model_total_loss,
        "investigation_expenses": investigation_expenses,
        "dollars_saved": dollars_saved,
    })

    return df_sweep


def optimize_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    transaction_value: float = DEFAULT_TRANSACTION_VALUE,
    investigation_cost: float = DEFAULT_INVESTIGATION_COST,
    num_steps: int = 1000,
    investigate_all_alerts: bool = DEFAULT_INVESTIGATE_ALL_ALERTS,
) -> Dict[str, Any]:
    """
    Find both the Cost-Optimal threshold (maximizing dollars saved)
    and the F1-Optimal threshold (maximizing F1-score), comparing the financial impact.
    """
    df_sweep = sweep_thresholds(
        y_true=y_true,
        y_prob=y_prob,
        amounts=amounts,
        transaction_value=transaction_value,
        investigation_cost=investigation_cost,
        num_steps=num_steps,
        investigate_all_alerts=investigate_all_alerts,
    )

    # Optimal row for Expected Dollars Saved
    best_cost_idx = df_sweep["dollars_saved"].idxmax()
    cost_optimal = df_sweep.loc[best_cost_idx].to_dict()

    # Optimal row for F1-score
    best_f1_idx = df_sweep["f1"].idxmax()
    f1_optimal = df_sweep.loc[best_f1_idx].to_dict()

    # Performance at standard default 0.50 threshold
    default_metrics = calculate_savings_and_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.50,
        amounts=amounts,
        transaction_value=transaction_value,
        investigation_cost=investigation_cost,
        investigate_all_alerts=investigate_all_alerts,
    )

    # Financial delta calculations
    savings_cost_opt = cost_optimal["dollars_saved"]
    savings_f1_opt = f1_optimal["dollars_saved"]
    savings_default = default_metrics["dollars_saved"]

    dollar_gain_over_f1 = savings_cost_opt - savings_f1_opt
    pct_gain_over_f1 = (
        (dollar_gain_over_f1 / abs(savings_f1_opt)) * 100.0 if savings_f1_opt != 0 else 0.0
    )

    dollar_gain_over_default = savings_cost_opt - savings_default
    pct_gain_over_default = (
        (dollar_gain_over_default / abs(savings_default)) * 100.0
        if savings_default != 0
        else 0.0
    )

    # Theoretical Bayes threshold: p* = C_invest / V (meaningful in flat mode)
    theoretical_threshold = float(investigation_cost / transaction_value)

    return {
        "transaction_value": transaction_value,
        "investigation_cost": investigation_cost,
        "theoretical_bayes_threshold": theoretical_threshold,
        "cost_optimal": cost_optimal,
        "f1_optimal": f1_optimal,
        "default_05": default_metrics,
        "dollar_gain_over_f1": float(dollar_gain_over_f1),
        "pct_gain_over_f1": float(pct_gain_over_f1),
        "dollar_gain_over_default": float(dollar_gain_over_default),
        "pct_gain_over_default": float(pct_gain_over_default),
        "sweep_curve": df_sweep,
    }


def print_comparison_report(results: Dict[str, Any]) -> None:
    """Print clean comparison report showing financial impact of cost optimization."""
    co = results["cost_optimal"]
    f1 = results["f1_optimal"]
    df = results["default_05"]

    print("=" * 80)
    print("CostGuard - Cost-Optimal vs F1-Optimal Threshold Analysis")
    print("=" * 80)
    print(f"Alert Investigation Cost = ${results['investigation_cost']:.2f}")
    print(f"Transaction Values:        Per-transaction Amount from dataset")
    print("-" * 80)
    print(f"{'Decision Criterion':<26} | {'Threshold':<10} | {'Recall':<8} | {'FP Alerts':<10} | {'Dollars Saved':<14}")
    print("-" * 80)
    print(
        f"{'Default (Standard 0.50)':<26} | {df['threshold']:<10.3f} | "
        f"{df['recall']:<8.2%} | {df['false_positives']:<10} | ${df['dollars_saved']:<14,.2f}"
    )
    print(
        f"{'F1-Score Optimal':<26} | {f1['threshold']:<10.3f} | "
        f"{f1['recall']:<8.2%} | {f1['false_positives']:<10} | ${f1['dollars_saved']:<14,.2f}"
    )
    print(
        f"{'Cost-Optimal (CostGuard)':<26} | {co['threshold']:<10.3f} | "
        f"{co['recall']:<8.2%} | {co['false_positives']:<10} | ${co['dollars_saved']:<14,.2f}"
    )
    print("=" * 80)
    print("HEADLINE FINANCIAL IMPACT:")
    print(f"-> Cost-Optimal saves ${results['dollar_gain_over_f1']:+,.2f} more than F1-Optimal ({results['pct_gain_over_f1']:+.2f}%)")
    print(f"-> Cost-Optimal saves ${results['dollar_gain_over_default']:+,.2f} more than Default 0.50 ({results['pct_gain_over_default']:+.2f}%)")
    print(
        f"-> At Cost-Optimal threshold ({co['threshold']:.3f}), the system stops {int(co['true_positives'])} frauds "
        f"while incurring only {int(co['false_positives'])} false alarms."
    )
    print(f"-> Baseline fraud loss (no model): ${co['baseline_loss']:,.2f}")
    print("=" * 80)


if __name__ == "__main__":
    test_cache = load_artifact("test_eval_cache.joblib")
    y_test = test_cache["y_test"].to_numpy()
    y_prob = test_cache["xgb_cal_probs"]
    X_test = test_cache["X_test"]

    # Extract per-transaction Amount (already scaled by RobustScaler — use raw values)
    # The Amount column in X_test has been scaled; we need the original Amount.
    # Reload raw amounts from the dataset for accurate dollar calculations.
    from src.data_prep import load_raw_data, DEFAULT_DATA_PATH, TARGET_COLUMN

    raw_df = load_raw_data(DEFAULT_DATA_PATH)
    # X_test.index preserves original DataFrame indices from the stratified split
    test_amounts = raw_df.loc[X_test.index, "Amount"].to_numpy()

    results = optimize_thresholds(
        y_true=y_test,
        y_prob=y_prob,
        amounts=test_amounts,
        investigation_cost=DEFAULT_INVESTIGATION_COST,
    )

    print_comparison_report(results)
