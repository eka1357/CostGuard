"""
CostGuard - End-to-End Orchestrator Pipeline
Executes the entire CostGuard lifecycle in sequence:
Data Preparation -> Model Training & Calibration -> Cost Matrix Optimization -> SHAP Explainability.
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import prepare_data, compute_imbalance_ratio, DEFAULT_DATA_PATH
from src.train import train_and_evaluate_all
from src.cost_optimizer import (
    optimize_thresholds,
    print_comparison_report,
    DEFAULT_TRANSACTION_VALUE,
    DEFAULT_INVESTIGATION_COST,
)
from src.explain import explain_top_flagged, print_shap_report


def run_full_pipeline(
    transaction_value: float = DEFAULT_TRANSACTION_VALUE,
    investigation_cost: float = DEFAULT_INVESTIGATION_COST,
) -> None:
    """Run all pipeline stages sequentially and report execution benchmarks."""
    start_total = time.time()

    print("=" * 80)
    print("COSTGUARD: END-TO-END PIPELINE EXECUTION")
    print("Optimizing Fraud Detection for Expected Dollars Saved")
    print("=" * 80)

    # Stage 1: Data Preparation
    print("\n>>> STAGE 1: Data Preparation & Stratified Split...")
    t0 = time.time()
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    ratio = compute_imbalance_ratio(y_train)
    print(f"    Train size: {len(X_train):,}, Test size: {len(X_test):,}")
    print(f"    Imbalance ratio: {ratio:.1f}:1 | Elapsed: {time.time() - t0:.2f}s")

    # Stage 2: Training & Calibration
    print("\n>>> STAGE 2: Model Training & Probability Calibration...")
    t0 = time.time()
    train_results = train_and_evaluate_all()
    xgb_cal = train_results["calibrated_xgb"]
    print(f"    Calibrated XGBoost PR-AUC: {xgb_cal['pr_auc']:.4f}, F1: {xgb_cal['f1']:.4f}")
    print(f"    Elapsed: {time.time() - t0:.2f}s")

    # Stage 3: Cost Optimization
    print("\n>>> STAGE 3: Cost Matrix & Threshold Optimization...")
    t0 = time.time()
    y_test_arr = y_test.to_numpy()
    y_prob = xgb_cal["y_prob"]
    cost_summary = optimize_thresholds(
        y_true=y_test_arr,
        y_prob=y_prob,
        transaction_value=transaction_value,
        investigation_cost=investigation_cost,
    )
    print_comparison_report(cost_summary)
    print(f"    Elapsed: {time.time() - t0:.2f}s")

    # Stage 4: Local SHAP Explainability
    print("\n>>> STAGE 4: Local SHAP Feature Attribution for High-Risk Alerts...")
    t0 = time.time()
    shap_summary = explain_top_flagged(
        top_n=10,
        transaction_value=transaction_value,
        investigation_cost=investigation_cost,
    )
    print_shap_report(shap_summary)
    print(f"    Elapsed: {time.time() - t0:.2f}s")

    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"PIPELINE RUN COMPLETE IN {total_time:.2f}s")
    print("Interactive dashboard ready: streamlit run app/dashboard.py")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
