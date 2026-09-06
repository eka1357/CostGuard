"""
CostGuard - Pipeline Test Suite
Sanity checks and integration tests for data preparation, model evaluation,
cost matrix optimization, and SHAP explainability.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# Set up project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import (
    validate_and_clean_data,
    split_data,
    scale_features,
    compute_imbalance_ratio,
    TARGET_COLUMN,
)
from src.train import (
    evaluate_model_performance,
    load_artifact,
)
from src.cost_optimizer import (
    calculate_savings_and_metrics,
    sweep_thresholds,
    optimize_thresholds,
    DEFAULT_TRANSACTION_VALUE,
    DEFAULT_INVESTIGATION_COST,
)
from src.explain import (
    get_top_flagged_transactions,
    extract_case_drivers,
)


# ==========================================
# Fixtures
# ==========================================
@pytest.fixture
def synthetic_data():
    """Generates synthetic credit card transaction DataFrame for fast unit testing."""
    np.random.seed(42)
    n_samples = 1000
    n_fraud = 20

    data = {f"V{i}": np.random.randn(n_samples) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 172800, n_samples)
    data["Amount"] = np.random.exponential(scale=50, size=n_samples)

    labels = np.zeros(n_samples, dtype=int)
    fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
    labels[fraud_indices] = 1
    data[TARGET_COLUMN] = labels

    return pd.DataFrame(data)


@pytest.fixture
def trained_artifacts():
    """Load pre-trained artifacts from models/ directory if available."""
    models_dir = PROJECT_ROOT / "models"
    required = ["baseline_lr.joblib", "xgb_model.joblib", "calibrated_xgb.joblib", "scaler.joblib"]
    if not all((models_dir / f).exists() for f in required):
        pytest.skip("Model artifacts not yet generated in models/ directory.")
    return {
        "scaler": load_artifact("scaler.joblib"),
        "lr": load_artifact("baseline_lr.joblib"),
        "xgb": load_artifact("xgb_model.joblib"),
        "calibrated_xgb": load_artifact("calibrated_xgb.joblib"),
    }


# ==========================================
# 1. Data Preparation Tests
# ==========================================
class TestDataPreparation:

    def test_validate_and_clean_data(self, synthetic_data):
        df_clean = validate_and_clean_data(synthetic_data)
        assert len(df_clean) == len(synthetic_data)
        assert df_clean.isnull().sum().sum() == 0

        # Verify missing column raises ValueError
        df_invalid = synthetic_data.drop(columns=[TARGET_COLUMN])
        with pytest.raises(ValueError):
            validate_and_clean_data(df_invalid)

    def test_split_data_stratification(self, synthetic_data):
        X_train, X_test, y_train, y_test = split_data(synthetic_data, test_size=0.2, random_state=42)
        assert len(X_train) == 800
        assert len(X_test) == 200

        train_fraud_rate = y_train.mean()
        test_fraud_rate = y_test.mean()
        # Stratification maintains approximately equal class distribution
        assert abs(train_fraud_rate - test_fraud_rate) < 0.01

    def test_scale_features_no_leakage(self, synthetic_data):
        X_train, X_test, y_train, y_test = split_data(synthetic_data, test_size=0.2, random_state=42)
        X_train_s, X_test_s, scaler = scale_features(X_train, X_test, features_to_scale=["Time", "Amount"])

        # Scaler must have been fitted on 800 samples (train only)
        assert scaler.n_features_in_ == 2
        assert not X_train_s["Amount"].isnull().any()
        assert not X_test_s["Amount"].isnull().any()

    def test_compute_imbalance_ratio(self):
        y = pd.Series([0] * 990 + [1] * 10)
        ratio = compute_imbalance_ratio(y)
        assert ratio == 99.0


# ==========================================
# 2. Model Evaluation Tests
# ==========================================
class TestModelEvaluation:

    def test_evaluate_model_performance(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        X_dummy = pd.DataFrame({"feat": y_prob})

        class MockModel:
            def predict_proba(self, X):
                p1 = X["feat"].to_numpy()
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])

        results = evaluate_model_performance(MockModel(), X_dummy, y_true, threshold=0.5)

        assert results["true_positives"] == 4
        assert results["false_positives"] == 0
        assert results["true_negatives"] == 4
        assert results["false_negatives"] == 0
        assert results["precision"] == 1.0
        assert results["recall"] == 1.0
        assert results["f1"] == 1.0
        assert 0.0 <= results["brier_score"] <= 1.0

    def test_loaded_model_predictions(self, trained_artifacts):
        xgb = trained_artifacts["xgb"]
        calibrated = trained_artifacts["calibrated_xgb"]

        cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        X_sample = pd.DataFrame(np.random.randn(5, 30), columns=cols)

        raw_probs = xgb.predict_proba(X_sample)[:, 1]
        cal_probs = calibrated.predict_proba(X_sample)[:, 1]

        assert len(raw_probs) == 5
        assert len(cal_probs) == 5
        assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))


# ==========================================
# 3. Cost Optimizer Tests
# ==========================================
class TestCostOptimizer:

    def test_savings_formula_with_known_values(self):
        # Scenario: 5 actual frauds, 5 legits
        # Model catches 4 frauds (TP=4), misses 1 (FN=1)
        # Model flags 1 legit as fraud (FP=1), correctly clears 4 (TN=4)
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.8, 0.2, 0.2, 0.1, 0.1])
        amounts = np.array([200, 50, 300, 100, 150, 80, 40, 60, 30, 20])

        # --- Per-transaction Amount mode ---
        res = calculate_savings_and_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.5,
            amounts=amounts,
            investigation_cost=8.0,
            investigate_all_alerts=True,
        )

        assert res["true_positives"] == 4
        assert res["false_positives"] == 1
        assert res["false_negatives"] == 1
        assert res["total_alerts"] == 5

        # Baseline loss = sum of all fraud amounts = 200+50+300+100+150 = $800
        # Caught frauds (prob >= 0.5): indices 0,1,2,3 -> amounts 200,50,300,100 = $650
        # Investigation expenses = 5 alerts * $8 = $40
        # Dollars saved = $650 - $40 = $610
        assert res["baseline_loss"] == 800.0
        assert res["fraud_loss_prevented"] == 650.0
        assert res["investigation_expenses"] == 40.0
        assert res["dollars_saved"] == 610.0

        # --- Flat mode (backward compat) ---
        res_flat = calculate_savings_and_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.5,
            transaction_value=100.0,
            investigation_cost=8.0,
            investigate_all_alerts=True,
        )
        # Formula: 4 * 100 - (4 + 1) * 8 = 400 - 40 = $360
        assert res_flat["dollars_saved"] == 360.0

    def test_sweep_thresholds_shape_and_bounds(self):
        y_true = np.random.choice([0, 1], size=200, p=[0.9, 0.1])
        y_prob = np.random.uniform(0, 1, size=200)

        df_sweep = sweep_thresholds(y_true, y_prob, num_steps=50)
        assert len(df_sweep) == 50
        assert "dollars_saved" in df_sweep.columns
        assert "threshold" in df_sweep.columns
        assert df_sweep["threshold"].iloc[0] < df_sweep["threshold"].iloc[-1]

    def test_cost_optimal_guarantee(self):
        # By definition, cost-optimal threshold must achieve >= dollars saved than F1-optimal
        y_true = np.random.choice([0, 1], size=500, p=[0.95, 0.05])
        y_prob = np.random.beta(0.5, 5.0, size=500)

        opt = optimize_thresholds(y_true, y_prob, num_steps=100)
        cost_savings = opt["cost_optimal"]["dollars_saved"]
        f1_savings = opt["f1_optimal"]["dollars_saved"]

        assert cost_savings >= f1_savings


# ==========================================
# 4. Explainability Tests
# ==========================================
class TestExplainability:

    def test_get_top_flagged_ranking(self):
        n = 50
        X = pd.DataFrame(np.random.randn(n, 4), columns=["A", "B", "C", "D"])
        y = np.random.choice([0, 1], size=n)
        probs = np.linspace(0.01, 0.99, n)

        X_top, y_top, prob_top, summary_df = get_top_flagged_transactions(
            X_test=X,
            y_test=y,
            y_prob=probs,
            threshold=0.50,
            top_n=5,
        )

        assert len(summary_df) == 5
        # Must be sorted in descending order of fraud probability
        assert list(summary_df["fraud_probability"]) == sorted(summary_df["fraud_probability"], reverse=True)
        assert np.all(summary_df["fraud_probability"] >= 0.50)

    def test_extract_case_drivers(self):
        class MockExplanation:
            values = np.array([[2.5, -1.2, 0.8, -0.4]])
            data = np.array([[10.0, 20.0, 30.0, 40.0]])
            base_values = -3.0

        drivers = extract_case_drivers(
            MockExplanation(),
            feature_names=["f1", "f2", "f3", "f4"],
            case_idx=0,
            top_k=2,
        )

        assert len(drivers["top_positive_drivers"]) == 2
        assert drivers["top_positive_drivers"][0]["feature"] == "f1"
        assert drivers["top_positive_drivers"][0]["shap_attribution"] == 2.5
        assert drivers["top_negative_drivers"][0]["feature"] == "f2"
        assert drivers["top_negative_drivers"][0]["shap_attribution"] == -1.2
