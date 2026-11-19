# Early Readmission Prediction: Diabetes 130 Hospitals

**Course:** Applied ML: Foundations, Engineering & Evaluation (MSA-DATI07-05)  
**Dataset:** Diabetes 130 Hospitals, OpenML ID 4541  
**Task:** Predict whether a patient will be readmitted within 30 days.

The original `readmitted` target is converted to:

- `1`: readmitted within 30 days (`<30`)
- `0`: readmitted after 30 days or not readmitted (`>30` or `NO`)

## Group members

| Member | Contributions |
|---|---|
| Adarsh | EDA, feature engineering, pipeline |
| Saathvika | Modelling, hyperparameter tuning, run log, code review |
| Sean | Explainability, calibration, fairness audit |

> The implementation was based primarily on the course materials and homework notebook code. Claude Code and ChatGPT 5.6 Luna were used as supplementary review tools to check syntax, identify typographical errors, and review minor code edits made to reflect decisions taken by the project team. The project team made the methodological decisions, ran the code, reviewed the outputs, and validated the final implementation.

## Project structure

```text
Applied_ML_Project/
├── config.yaml
├── requirements.txt
├── README.md
├── main.py
├── run_log.csv
├── artifacts/
│   └── final_model.pkl
├── plots/
│   ├── calibration_curve.png
│   ├── fairness_tpr_by_race.png
│   ├── shap_global.png
│   └── shap_local_*.png
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── imbalance.py
│   ├── models.py
│   ├── tuning.py
│   ├── pipeline.py
│   ├── explainability.py
│   ├── calibration.py
│   └── fairness.py
└── supplementary/
    ├── eda_standalone.ipynb
    └── eda_standalone.html
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

The workflow downloads the dataset from OpenML, performs patient-level splitting, trains baseline models, tunes the models with Optuna, evaluates the final XGBoost model, and generates explainability, calibration, and fairness outputs.

Results are written to `plots/`, `artifacts/`, and `run_log.csv`.

## Data preparation

- The raw `?` marker is converted to `NaN`.
- `weight`, `payer_code`, and `medical_specialty` are dropped because of high missingness.
- `encounter_id` is dropped as an identifier.
- `patient_nbr` is temporarily retained for patient-level splitting and removed before modelling.
- Literal `None` values in `max_glu_serum` and `A1Cresult` are retained as explicit categories.
- The data is split approximately into 70% training, 15% validation, and 15% test encounters, with all encounters from a patient assigned to the same partition.

## Feature engineering and modelling

The workflow creates:

- `num_medications_per_day`
- `total_visits`
- `high_diag_complexity`

`admission_type_id`, `discharge_disposition_id`, and `admission_source_id` are treated as categorical codes and one-hot encoded rather than treated as continuous numeric variables.

Class imbalance is handled using:

- `class_weight="balanced"` for Logistic Regression and Random Forest
- `scale_pos_weight` for XGBoost

The compared models are Logistic Regression, Random Forest, and XGBoost. XGBoost achieved the strongest baseline validation performance and was selected as the final model. Its tuned parameters were then used to retrain the model on the combined training and validation data.

## Results summary

Baseline results are reported on the validation set. The final tuned XGBoost result is reported on the held-out test set.

| Model | ROC-AUC | F1 | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6673 | 0.2726 | 0.1808 | 0.5539 | 0.2183 |
| Random Forest | 0.6681 | 0.2728 | 0.1773 | 0.5908 | 0.2341 |
| XGBoost | 0.6852 | 0.2857 | 0.1895 | 0.5798 | 0.2114 |
| **XGBoost, tuned test result** | **0.6603** | **0.2602** | **0.1680** | **0.5769** | **0.2252** |

Because early readmission represents approximately 11% of records, accuracy alone could be misleading. In this project, recall is treated as the primary metric because the objective is to identify as many patients at risk of early readmission as possible for preventive support. Precision and F1 are also reported to quantify the false-alert trade-off.

### Hyperparameter tuning

Optuna tuning used 10 trials and 3-fold stratified cross-validation on a 20% subsample of the training data. The selected XGBoost configuration was:

- Learning rate: `0.0293`
- Maximum depth: `3`
- Number of estimators: `200`
- Subsample fraction: `0.5853`
- Cross-validation ROC-AUC: `0.6576`

The selected parameters were then used to retrain XGBoost on the combined training and validation data before final evaluation on the held-out test set.

## Explainability

SHAP is used to generate global feature importance and local waterfall explanations.

In the final run, the most influential features included prior inpatient utilisation, total prior visits, and specific discharge disposition categories. SHAP values describe predictive associations and model behaviour. They should not be interpreted as evidence of causal relationships.

## Calibration

Calibration is assessed using a reliability curve and Brier score.

The final model was overconfident, meaning its predicted probabilities were generally higher than the observed event frequencies. The model may still be useful for ranking patients by relative risk, but its raw probabilities should be interpreted cautiously before being used in probability-based decisions.

No post-hoc calibration method was applied in the current workflow. Platt scaling or isotonic regression could be evaluated in future work.

## Fairness audit

Fairness is assessed across race groups using:

- True Positive Rate, TPR
- False Positive Rate, FPR
- Predicted Positive Rate, PPR
- Group-level ROC-AUC
- PPR gap
- TPR gap, representing the equal opportunity component

The reported TPR gap represents the equal opportunity component of equalized odds. FPR is also reported by group, but the figure visualises TPR only.

Before and after reweighting:

| Metric | Before | After |
|---|---:|---:|
| PPR gap | 0.1277 | 0.1431 |
| TPR gap | 0.2612 | 0.2086 |
| Overall ROC-AUC | 0.6603 | 0.6609 |

Reweighting reduced the TPR gap but increased the PPR gap. Therefore, the mitigation improved one fairness criterion while worsening another.

## Technical checklist

- [x] Exploratory data analysis and problem framing
- [x] Missingness and identifier analysis
- [x] Patient-level train, validation, and test splitting
- [x] Feature engineering
- [x] Categorical encoding and numeric preprocessing
- [x] Class imbalance handling
- [x] Logistic Regression, Random Forest, and XGBoost comparison
- [x] Scikit-learn pipelines
- [x] Optuna hyperparameter tuning
- [x] SHAP global and local explanations
- [x] Calibration diagnostics
- [x] Fairness audit and reweighting comparison
- [x] Reproducible run log and saved final model