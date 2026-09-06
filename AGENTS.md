# AGENTS.md — CostGuard: Cost-Aware Fraud Detection

## Project Summary
Building a solo submission for GIBC V2, Track 02 (Applied: Medical Technology & Finance).
Deadline: Sep 21, 2026, 11:45pm Taipei time (UTC+8). Deliverable is a working prototype +
Devpost submission with 6 required components (see "Submission Requirements" below).

**Core idea:** A fraud detection system that optimizes for *expected dollars saved*, not
accuracy/F1. Most fraud detection projects stop at a classifier + metrics. The
differentiator here is:
1. A cost matrix (assumed $ value per transaction, assumed $ cost per false-positive
   investigation) used to pick the classification threshold that maximizes expected
   savings — not the F1-optimal threshold.
2. Per-transaction SHAP explanations for flagged cases.
3. An interactive dashboard where a judge can drag a cost-tradeoff slider and watch the
   optimal threshold and projected savings update live.

Judging criteria this must satisfy (Track 02): Innovation & Impact, Technical Feasibility,
Rigor & Validation, Presentation. Every design decision should be justifiable against one
of these four.

## Tech Stack
- Python 3.11+
- pandas, numpy — data handling
- scikit-learn — baseline logistic regression, preprocessing, metrics
- xgboost or lightgbm — main model
- imbalanced-learn — class imbalance handling (SMOTE / class weights)
- shap — explainability
- streamlit — interactive dashboard (preferred over Gradio for chart flexibility)
- matplotlib / plotly — static and interactive charts
- pytest — basic sanity tests on the pipeline (not extensive, just enough to show rigor)

## Dataset
Kaggle "Credit Card Fraud Detection" (ULB) — https://www.kaggle.com/mlg-ulb/creditcardfraud
Fallback: IEEE-CIS Fraud Detection dataset if more feature richness is wanted.
Requires a free Kaggle account + API token (kaggle.json) to download via CLI, or manual
download from the browser.

## Repo Structure (target)
```
costguard/
  data/                  # raw + processed data (gitignored if large)
  notebooks/             # EDA, exploratory model iteration
  src/
    data_prep.py         # loading, cleaning, imbalance handling
    train.py             # model training + calibration
    cost_optimizer.py     # cost matrix + threshold optimization logic
    explain.py           # SHAP wrapper
  app/
    dashboard.py         # Streamlit app
  models/                # saved model artifacts
  tests/
    test_pipeline.py
  README.md              # setup instructions, results, assumptions (REQUIRED for submission)
  requirements.txt
  AGENTS.md              # this file
```

## Coding Conventions
- Keep functions small and testable; the pipeline (data_prep -> train -> cost_optimizer ->
  explain) should be runnable end-to-end from one script or notebook.
- Every assumption (transaction value, investigation cost, etc.) must be a named constant
  at the top of `cost_optimizer.py`, not a magic number buried in code — these get called
  out explicitly in the README as stated assumptions.
- Prefer clarity over cleverness; a judge or another developer should be able to read
  `train.py` top to bottom and understand the pipeline without cross-referencing other files.
- No hardcoded file paths outside a config section at the top of each script.

## Definition of Done (per phase)
1. **Data + baseline**: pipeline runs end-to-end on raw data -> cleaned data -> logistic
   regression baseline with precision/recall/F1 printed.
2. **Real model + cost framework**: XGBoost trained and calibrated; cost matrix defined;
   script outputs the cost-optimal threshold vs. the F1-optimal threshold, with dollars
   saved at each.
3. **Explainability**: SHAP values computed for at least the top N flagged transactions;
   a plot or table showing feature contributions per flagged case.
4. **Dashboard**: Streamlit app loads the trained model + cost framework, has a working
   slider for the false-positive-to-fraud-loss cost ratio, and live-updates a chart of
   threshold vs. expected savings, plus a table of currently-flagged transactions with
   SHAP explanations.
5. **Docs**: README with setup steps, how to run, stated assumptions, and headline
   results (e.g. "X% more savings than F1-optimized threshold").

## Mandatory Compliance (official rules — non-negotiable)
- **AI disclosure**: This project is built with AI coding assistance (Antigravity agent +
  Claude for planning). This MUST be listed in "Built With" and the README must state
  which parts were AI-assisted. Undisclosed use is treated as misrepresentation by the
  organizers and can result in disqualification.
- **Track 02 required disclaimer**: README must explicitly state this is a research
  prototype — NOT a medical device, NOT a diagnostic tool, and NOT financial advice.
- **Dataset license**: cite both the source AND the license of the dataset in the README
  (not just "from Kaggle"). Confirm the exact license before submission.
- **No live deployment**: never frame this as operating on real money or real patients —
  simulation / historical data only, stated explicitly.
- **Repo must be public and unrestricted** with no undocumented paid API keys or unusual
  dependencies — document any special setup so a judge with no context can run it.
- **English only** for all documentation and video.
- **Original work window**: all code must be written July 11–Sep 21, 2026. Using an
  existing public dataset is fine; the code/pipeline itself must be new.
- Repo state exactly at the Sep 21, 11:45pm Taipei deadline is what gets judged — commits
  after that may be disregarded. Do not plan on post-deadline fixes.

## Submission Requirements (Devpost — all 6 required)
1. Project Description — what it does, problem solved, how it works under the hood.
2. Source Code — public GitHub repo link with README (setup, prerequisites, usage).
3. Demo Video — 2-5 min, YouTube/Vimeo/Youku, unlisted OK, English audio or subtitles,
   must show the dashboard in action and explain the cost-savings methodology.
4. Built With — full list of languages/frameworks/libraries/datasets used.
5. Team Information — solo, so just the builder's real full name on a Devpost account.
6. Screenshots — 3+ images: dashboard view, SHAP explanation plot, savings/threshold curve.

## Out of Scope (don't build these — solo, 2-week timeline)
- No real-time streaming / Kafka / production deployment.
- No user authentication or multi-user dashboard.
- No mobile app or separate frontend framework — Streamlit is sufficient.
- No cloud deployment required for judging; local run via README instructions is fine,
  though deploying to Streamlit Community Cloud is a nice-to-have if time allows.
