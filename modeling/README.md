# Stage 4 — Predictive Modeling: Table of Contents & Final Checklist

End-to-end modeling of medical insurance `charges`, from clean data to a deployable,
explainable model and a business action plan. All artifacts live in this `modeling/` folder.

## How to reproduce
Run with the project venv, in order:
```
.venv/Scripts/python.exe modeling/chunk1_load.py    # ... through chunk35_dashboard.py
```
Each script imports `model_utils.py` (splits, metrics, results registry, model factory, plotting).
All runs are seeded `random_state=42`.

## Table of contents (39 chunks)

| # | Chunk | Script | Key output |
|---|---|---|---|
| 1 | Load & verify | `chunk1_load.py` | clean 1,337×7 confirmed |
| 2 | Features: age/BMI | `chunk2_features.py` | `insurance_features.csv` |
| 3 | Features: interactions | `chunk3_interactions.py` | `smoker_bmi`, etc. |
| 4 | Features: aggregates | `chunk4_aggregates.py` | group means (leakage flagged) |
| 5 | Encoding | `chunk5_encode.py` | `insurance_model_matrix.csv` |
| 6 | Train/test split | `chunk6_split.py` | leak-safe splits (stratified) |
| 7 | Scaling | `chunk7_scale.py` | scaled splits + `scaler.joblib` |
| 8 | Linear Regression | `chunk8_linear.py` | baseline; coef chart |
| 9 | Ridge | `chunk9_ridge.py` | alpha=1 |
| 10 | Lasso | `chunk10_lasso.py` | drops region features |
| 11 | Elastic-Net | `chunk11_elasticnet.py` | |
| 12 | Polynomial (deg 2) | `chunk12_polynomial.py` | rediscovers smoker×bmi |
| 13 | Decision Tree | `chunk13_decision_tree.py` | importances |
| 14 | Random Forest | `chunk14_random_forest.py` | breaks linear ceiling |
| 15 | Gradient Boosting | `chunk15_gradient_boosting.py` | **production model** |
| 16 | XGBoost | `chunk16_xgboost.py` | |
| 17 | SVR (+ target-scaled) | `chunk17_svr.py` | best single model |
| 18 | KNN | `chunk18_knn.py` | |
| 19 | Comparison table | `chunk19_comparison.py` | `model_comparison.csv`, chart |
| 20 | Voting ensemble | `chunk20_voting.py` | ties best |
| 21 | Stacking (Linear meta) | `chunk21_stacking_linear.py` | |
| 22 | Stacking (XGB meta) | `chunk22_stacking_xgb.py` | |
| 23 | MedCost-AdaBoost | `chunk23_adaboost.py` | underperforms (reported honestly) |
| 24 | Bagging | `chunk24_bagging.py` | |
| 25 | Cross-validation | `chunk25_cross_validation.py` | honest ~0.85 CV; models tied |
| 26 | Residual analysis | `chunk26_residuals.py` | diagnostics figure |
| 27 | SHAP global | `chunk27_shap_global.py` | beeswarm + importance |
| 28 | SHAP dependence | `chunk28_shap_dependence.py` | BMI-30 cliff |
| 29 | Individual explanations | `chunk29_individual.py` | 5 waterfalls |
| 30 | Business insights | `chunk30_insights.py` | `business_insights.md` |
| 31 | Recommendations | (doc) | `recommendations.md` |
| 32 | Financial impact | `chunk32_financial.py` | `financial_impact.md`, chart |
| 33 | Literature comparison | (doc) | `literature_comparison.md` |
| 34 | Limitations & future work | (doc) | `limitations_future_work.md` |
| 35 | Results dashboard | `chunk35_dashboard.py` | 6-panel dashboard |
| 36 | Executive summary | (doc) | `executive_summary.md` |
| 37 | Technical summary | (doc) | `technical_summary.md` |
| 38 | The complete story | (doc) | `the_complete_story.md` |
| 39 | This checklist & TOC | (doc) | `README.md` |

## Deliverables index
- **Narrative:** `results/the_complete_story.md` (Acts 1–6), `results/executive_summary.md`, `results/technical_summary.md`
- **Business:** `results/business_insights.md`, `results/recommendations.md`, `results/financial_impact.md`
- **Analysis:** `results/literature_comparison.md`, `results/limitations_future_work.md`
- **Data:** `results/model_comparison.csv`, `results/model_results.json`, `results/shap_vs_builtin_importance.csv`
- **Figures:** 13 PNG (+PDF) in `figures/` — coefficient, importances, comparison, residuals, SHAP (beeswarm/bar/dependence), waterfalls, financial, dashboard
- **Models:** 8 fitted estimators in `results/models/`; splits + scaler in `splits/`

## Headline results
- **Production model:** Gradient Boosting — **CV R² ≈ 0.85** (honest), test R² 0.92, RMSE ~$3,400, MAE ~$2,100.
- **Top driver:** `smoker_bmi` (mean |SHAP| $6,183); smoking + obesity can add >$25k to a prediction.
- **Not drivers:** region, sex.
- **Action:** smoker×BMI repricing + targeted wellness ≈ **$15M/yr per 10,000 members** (conservative).

## Final checklist
- [x] All 39 chunks completed
- [x] Clean data verified; leakage controlled (train-only target encoding)
- [x] 16 models built, tuned (GridSearchCV), and honestly cross-validated
- [x] Best model selected on principle (accuracy tie → simplest, explainable)
- [x] Residual diagnostics + SHAP global/dependence/individual explanations
- [x] Business insights, recommendations, and ROI-ranked financial impact
- [x] Literature benchmark + limitations reconciled honestly
- [x] Executive, technical, and hiring-manager narratives written
- [x] All code runs error-free with the project venv (seeded, reproducible)
- [x] All outputs saved inside the `Medical Insurance Cost Dataset` folder

**Status: COMPLETE.** Recommended next step (not started, per scope): productionize with a Tweedie/GLM
comparison, conformal prediction intervals, and acquisition of claims/diagnosis data.
