# CostGuard: Cost-Aware Fraud Detection

> **GIBC V2 Track 02 (Applied: Medical Technology & Finance) Solo Submission**  
> Optimizing fraud classification for **expected net dollars saved** rather than traditional statistical heuristics (F1-score / Accuracy).

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code Style: Clean](https://img.shields.io/badge/code%20style-pep8-green.svg)]()

---

## Mandatory Track 02 Disclaimer

> [!IMPORTANT]  
> **Research Prototype Disclaimer**: CostGuard is an academic and applied research prototype developed solely for the GIBC V2 Hackathon. It is **NOT a medical device**, **NOT a diagnostic tool**, and **NOT financial or investment advice**. It operates strictly in an offline simulation mode on historical benchmark data and is **never deployed on live financial transactions or real money**.

---

## Executive Summary

Most fraud detection systems stop at training an ML classifier and selecting an arbitrary decision cutoff ($t = 0.50$) or tuning for maximum F1-Score. In enterprise finance operations, this leads to suboptimal financial decisions:

1. **The Asymmetry of Fraud Costs**: Missing a fraudulent transaction (False Negative) costs the enterprise the entire transaction loss (e.g. **$100.00**). In contrast, reviewing a flagged transaction (investigating an alert) costs a fraction of that amount in operational and labor fees (e.g. **$8.00**).
2. **The F1-Score Blindspot**: The harmonic mean of precision and recall (F1) treats false alarms and missed fraud with symmetric importance. It artificially suppresses alerts to preserve precision, leaving thousands of dollars of preventable fraud undetected.
3. **CostGuard's Solution**: CostGuard formulates fraud detection as an **expected financial value maximization problem**. By training a calibrated probability model (XGBoost + Platt scaling), sweeping the empirical decision space, and establishing the cost-optimal decision threshold, CostGuard systematically maximizes **net dollars saved**.

---

## Stated Operational Cost Assumptions

In compliance with the project specification and operational finance conventions, the cost parameters are explicitly declared as named constants at the top of `src/cost_optimizer.py`:

| Parameter | Symbol | Value | Operational Definition |
| :--- | :---: | :---: | :--- |
| **Fraud Transaction Value** | $V$ | **$100.00** | Direct financial loss prevented when a fraudulent transaction is stopped (True Positive). |
| **Investigation Cost** | $C_{\text{invest}}$ | **$8.00** | Human reviewer time, operational overhead, and verification cost per flagged alert. |
| **Investigation Policy** | — | *All Alerts* | Operational policy where all flagged alerts (both TP and FP) undergo verification. |
| **Theoretical Bayes Threshold** | $t^*$ | **$0.0800$** | $t^* = \frac{C_{\text{invest}}}{V} = \frac{8}{100}$. For calibrated probabilities $P(\text{Fraud}|x)$, alerting is positive expected value when $P > t^*$. |

### Financial Savings Formulation:
$$\text{Baseline Loss (No Model)} = \text{Total Frauds} \times V$$
$$\text{Operational Cost with Model} = (\text{False Negatives} \times V) + ((\text{True Positives} + \text{False Positives}) \times C_{\text{invest}})$$
$$\mathbf{\text{Net Dollars Saved}} = \text{Baseline Loss} - \text{Operational Cost with Model} = (\text{True Positives} \times V) - (\text{Total Alerts} \times C_{\text{invest}})$$

---

## Headline Results

Evaluated on the untouched **56,962 test transactions** (incorporating 98 true fraud instances under natural 577:1 class imbalance):

| Decision Criterion | Decision Threshold | Recall | False Positive Alerts | Net Dollars Saved ($) | Financial Gain vs F1-Optimal |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Default Model Cutoff** | `0.500` | 76.53% | 3 | $6,876.00 | — |
| **F1-Optimal Policy** | `0.160` | 81.63% | 4 | $7,328.00 | Benchmark |
| **CostGuard Cost-Optimal** | **`0.001`** | **87.76%** | **32** | **$7,656.00** | **+$328.00 (+4.48%)** |

### Why CostGuard Outperforms Traditional Tuning:
- Compared to the standard F1-optimal threshold, CostGuard captures **6 additional fraudulent transactions** (recovering **+$600.00** in fraud losses) at an incremental expense of only 28 false-alarm investigations (**$224.00**).
- **Net Bottom-Line Gain**: **+$376.00 net financial benefit** over conventional F1 tuning on a single test slice.
- Compared to the default 0.50 threshold, CostGuard captures **11 additional frauds**, increasing total savings by **+$780.00 (+11.34%)**.

---

## Model Benchmark Comparison

All models evaluated on identical stratified 80/20 train/test splits (random seed 42):

| Metric | Baseline Logistic Regression | XGBoost (Raw) | XGBoost (Calibrated, 3-fold Platt) |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.0609 | 0.9259 | **0.9615** |
| **Recall** | 0.9184 | 0.7653 | 0.7653 |
| **F1-Score** | 0.1142 | 0.8380 | **0.8523** |
| **PR-AUC (Avg Precision)** | 0.7175 | 0.8504 | **0.8773** |
| **ROC-AUC** | 0.9720 | **0.9817** | 0.9764 |
| **Brier Score (Calibration)** | 0.0235 | 0.0004 | **0.0004** |
| **False Positives ($t=0.50$)** | 1,388 *(wastes $11,104 in reviews)* | 6 | **3** |

---

## Local SHAP Explainability

CostGuard implements `shap.TreeExplainer` directly on the underlying ensemble to generate per-transaction Shapley attributions for all high-risk alerts:

- **Baseline Prior**: Base log-odds are $-7.948$ (reflecting the 0.173% background fraud prior).
- **Empirical Fraud Signatures**: Flagged cases exhibit consistent negative anomalies in `V14` (SHAP attributions ranging $+4.7$ to $+8.9$) and `V10` (SHAP attributions $+3.3$ to $+3.8$), coupled with spikes in `V4` (SHAP $+2.2$ to $+2.8$).
- **Actionable Operational Context**: Analysts in the interactive dashboard can inspect individual transactions to see exactly which features elevated risk above the operational threshold.

---

## Repository Structure

```
CostGuard/
├── data/                         # Raw and processed data (gitignored if large)
│   └── creditcard.csv            # Kaggle ULB dataset (150MB, not committed)
├── notebooks/                    # Exploratory data analysis & iteration
│   └── .gitkeep
├── src/                          # Modular pipeline source code
│   ├── __init__.py
│   ├── data_prep.py              # Ingestion, validation, 80/20 split, RobustScaler
│   ├── train.py                  # Baseline LR & calibrated XGBoost training
│   ├── cost_optimizer.py         # Cost matrix, threshold sweep, savings optimizer
│   └── explain.py                # SHAP TreeExplainer & feature attribution
├── app/                          # Interactive user interface
│   └── dashboard.py              # Streamlit dashboard adhering to strict UI guidelines
├── models/                       # Saved serialized artifacts (gitignored binaries)
│   ├── baseline_lr.joblib
│   ├── xgb_model.joblib
│   ├── calibrated_xgb.joblib
│   ├── scaler.joblib
│   ├── test_eval_cache.joblib
│   └── shap_explanations.joblib
├── tests/                        # Automated sanity test suite
│   └── test_pipeline.py          # 11 unit and integration tests (100% pass)
├── requirements.txt              # Pinned Python package dependencies
├── AGENTS.md                     # Project governance and hackathon specifications
└── README.md                     # Comprehensive documentation and results
```

---

## Quickstart & Installation

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### 2. Clone Repository & Setup Environment
```bash
git clone https://github.com/eka1357/CostGuard.git
cd CostGuard

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Data Acquisition
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` directory:
```
data/creditcard.csv
```

### 4. Run the Pipeline End-to-End
Each module can be executed independently:

```bash
# Step 1: Preprocess data and verify split
python src/data_prep.py

# Step 2: Train baseline LR and Calibrated XGBoost
python src/train.py

# Step 3: Run cost optimizer and threshold sweep
python src/cost_optimizer.py

# Step 4: Generate SHAP explanations for top flagged alerts
python src/explain.py
```

### 5. Run Automated Tests
```bash
pytest tests/test_pipeline.py -v
```

### 6. Launch the Interactive Dashboard
```bash
streamlit run app/dashboard.py
```
Open `http://localhost:8501` in your browser. Drag the **Investigation Cost** and **Average Fraud Loss** sliders to observe real-time updates to the optimal threshold, projected net savings, and per-case SHAP waterfall plots.

---

## Dataset Attribution & License Citation

- **Dataset Source**: Kaggle / ULB Machine Learning Group: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Original Authors**: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification*. In IEEE SSCI, 2015.
- **Dataset License**: **Open Database License (ODbL) v1.0** / **Database Contents License (DbCL) v1.0**. Data is used in accordance with the Open Data Commons license terms for academic and research evaluation.

---

## AI Coding Assistance Disclosure

In full compliance with GIBC V2 Hackathon mandatory AI disclosure rules:
- **Tools Used**: **Antigravity IDE** (Google DeepMind advanced agentic coding environment) and **Claude** (used for high-level architecture planning).
- **Assisted Components**:
  - Drafting initial repository scaffold, module boundaries, and `.gitignore`.
  - Assisting in vectorizing the `sweep_thresholds` search algorithm in `src/cost_optimizer.py`.
  - Authoring unit test assertions in `tests/test_pipeline.py`.
  - Streamlit dashboard layout structure in `app/dashboard.py`.
- **Human Guidance & Verification**: All domain cost assumptions, financial formulas, metric evaluations, probability calibration decisions, and design reviews were architected, audited, and verified by the builder.

---

## Built With

- **Python 3.13** — Core runtime environment
- **scikit-learn** — Baseline models, cross-validation, and probability calibration (`CalibratedClassifierCV`)
- **XGBoost** — Gradient boosted decision trees classifier
- **SHAP (SHapley Additive exPlanations)** — TreeExplainer for local model interpretability
- **Streamlit** — Reactive web dashboard
- **Plotly** — Dynamic interactive financial visualization charts
- **pandas & numpy** — High-performance vectorized numerical operations
- **pytest** — Automated unit and integration test suite
- **Antigravity IDE & Claude** — AI-assisted planning and pair-programming
