# Early Readmission Prediction — Diabetes 130 Hospitals

**Course:** Applied ML: Foundations, Engineering & Evaluation (MSA-DATI07-05)  
**Dataset:** Diabetes 130 Hospitals — OpenML ID 4541 (Option 3)  
**Problem:** Binary classification — predict whether a diabetic patient will be readmitted to hospital within 30 days (`early_readmission = 1`) or not (`= 0`).

## Group members

| Member | Contributions |
|--------|---------------|
| Adarsh | EDA, feature engineering, pipeline |
| Saathvika | Modelling, hyperparameter tuning, run log |
| Sean | Explainability, calibration, fairness audit |

> AI coding assistant (Claude Code) was used to scaffold the initial implementation.

## Project structure

```
AML_Project/
├── config.yaml              # Central configuration (paths, seeds, tuning params)
├── requirements.txt         # Python dependencies
├── README.md
├── main.py                  # Orchestrates the full pipeline end to end
├── run_log.csv              # Appended by tuning.py — one row per Optuna trial
├── src/
│   ├── data_loader.py       # Load, clean, binarize, split
│   ├── feature_engineering.py  # Custom features + sklearn transformers
│   ├── imbalance.py         # Class imbalance utilities (scale_pos_weight)
│   ├── models.py            # Baseline training & evaluation
│   ├── tuning.py            # Optuna hyperparameter search
│   ├── pipeline.py          # Full sklearn Pipeline builder
│   ├── explainability.py    # SHAP global + local explanations
│   ├── calibration.py       # Reliability curve + Brier score
│   └── fairness.py          # Fairness metrics + reweighing mitigation
├── artifacts/               # Saved model files (created at runtime)
├── plots/                   # All figures (created at runtime)
└── notebooks/
    └── exploration.ipynb    # Supplementary EDA visualizations
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

Outputs are written to `plots/` and `artifacts/`. The Optuna run log is appended to `run_log.csv`.

## Results summary

| Model | ROC-AUC | F1 | Recall | Brier |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.629 | 0.239 | 0.504 | 0.226 |
| Random Forest (baseline) | 0.645 | 0.255 | 0.536 | 0.237 |
| XGBoost (baseline) | 0.660 | 0.265 | 0.533 | 0.211 |
| **XGBoost (Optuna-tuned, test set)** | **0.679** | **0.278** | **0.622** | **0.224** |

Fairness audit (race): Demographic Parity Difference 0.115 → 0.112 after reweighing; Equalized Odds TPR gap unchanged at 0.196.

## Technical requirements checklist

- [x] EDA & problem framing — dataset overview, missing values, class distribution
- [x] Feature engineering — 3 engineered features with domain justification
- [x] Class imbalance handling — `scale_pos_weight` (XGBoost), `class_weight='balanced'` (LR/RF)
- [x] Modelling progression — LogReg → RF → XGBoost with shared metrics
- [x] sklearn Pipeline — `engineer → preprocessor → classifier` per model
- [x] Hyperparameter tuning — Optuna TPE, 10 trials, 3-fold stratified CV
- [x] Run log — `run_log.csv` (30 rows: params + ROC-AUC per trial)
- [x] Explainability — SHAP global bar + 5 local waterfall plots
- [x] Calibration — reliability curve, Brier score, over/under-confidence diagnosis
- [x] Fairness & bias audit — per-race TPR/FPR/PPR, reweighing mitigation, before/after

## Dataset notes

- Source: UCI / OpenML — 130 US hospitals, 1999–2008
- ~100,000 patient encounter records
- Target: `readmitted` binarized → `<30` days = 1, otherwise = 0 (~11% positive class)
- Sensitive attribute for fairness audit: `race`
- Missing values encoded as `?` in the raw data — replaced with NaN during loading
- Dropped high-missingness columns: `weight`, `payer_code`, `medical_specialty`
