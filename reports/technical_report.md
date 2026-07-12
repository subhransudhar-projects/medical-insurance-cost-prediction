# Technical Report — Detailed Methodology
*Medical Insurance Cost Prediction · actual verified results*

## 1. Data preparation
- **Source:** 1,338 raw rows → **1,337** after removing one exact duplicate; 0 missing values; dtypes
  validated (int: age, children; float: bmi, charges; categorical: sex, smoker, region).
- **Outliers:** high-`charges` cases (10.4% by IQR) were **retained** — they are legitimate high-cost
  smokers, not data errors (removing them would bias the model against real tail risk).
- **Feature engineering (20 features):**
  - Non-linear: `age_squared`, `bmi_squared`; ordinal `age_group` (5 bands), `bmi_category` (4 clinical bands).
  - Interactions: `smoker_bmi`, `smoker_age`, `age_bmi`, `smoker_children` (smoker terms via a 0/1 switch).
  - Group aggregates: `region_mean_charges`, `smoker_mean_charges`, `region_mean_bmi`.
- **Encoding:** one-hot `region`; binary `sex` (female=1), `smoker` (yes=1).
- **Scaling:** `StandardScaler` on 14 continuous/engineered features; 6 binary indicators passed through.
- **Leakage control:** the two target-encoded aggregates are **refit on the training fold only** and mapped
  onto test rows, so no test target information enters training. (Verified immaterial to results: dropping
  these features changes CV R² by 0.0002.)

## 2. Protocol
- **Split:** 80/20, `random_state=42`, **stratified on `smoker`** (train 20.49% / test 20.52%).
- **Tuning:** `GridSearchCV`, 5-fold `KFold(shuffle=True, random_state=42)`, scoring R². Scale-sensitive
  models use `Pipeline(StandardScaler, model)` so scaling is refit per fold.
- **Logging:** all runs recorded to a JSON results registry for reproducible comparison.

## 3. Model development & results

**Baseline (linear) models** — plateau ~0.90 test / ~0.825 CV:
| Model | Test R² | CV R² |
|---|---|---|
| Linear Regression | 0.896 | 0.825 |
| Ridge (α=1) | 0.898 | 0.825 |
| Lasso (α=10) | 0.898 | 0.825 |
| Elastic-Net (α=0.001, l1=0.1) | 0.897 | 0.825 |
| Polynomial (deg 2) | 0.892 | — |

**Tree-based models** — break the ceiling to ~0.92 test / ~0.84 CV:
| Model | Test R² | CV R² |
|---|---|---|
| Decision Tree (depth 5) | 0.894 | 0.823 |
| Random Forest | 0.918 | 0.840 |
| Gradient Boosting | 0.920 | 0.840 |
| XGBoost | 0.919 | 0.844 |

**Other models:**
| Model | Test R² | Note |
|---|---|---|
| SVR (inputs scaled only) | 0.363 | underfits — target on $-scale |
| SVR (target-scaled) | 0.924 | best single model after fix |
| KNN (k=10, distance) | 0.895 | memorizes (train R²=0.998) |

**Ensembles:**
| Model | Test R² |
|---|---|
| Voting (top 3) | 0.924 |
| Stacking (Linear meta) | 0.922 |
| Stacking (XGBoost meta) | 0.917 |
| Bagging | 0.898 |
| MedCost-AdaBoost | 0.709 |

## 4. Model evaluation
- **Cross-validation (5-fold, training set):** top tier clusters at **R² ≈ 0.84–0.85 ± 0.04**. The
  CV-mean spread across the top 5 models (0.008) is **5× smaller** than the fold-to-fold std (0.040) →
  **the top models are statistically indistinguishable.**
- **Test-split optimism:** the single held-out split scores ~0.92 vs CV ~0.85 — a measured **+0.076**
  optimism. We report the CV figure as the honest one.
- **Feature importance:** consistent across models and metrics — `smoker_bmi` dominant, then age terms;
  region/sex negligible. SHAP and built-in importances agree on ranking.

## 5. Model interpretation (SHAP, TreeExplainer on Gradient Boosting)
- **Base value:** $13,356. **Top drivers (mean |SHAP|):** `smoker_bmi` $6,183, `age` $1,429, `age_bmi`
  $935, `age_squared` $918, `smoker_age` $832.
- Smoking-related features = **62%** of total SHAP weight; region+sex = **3.6%**.
- **Dependence:** a step-change at BMI 30 for smokers; monotonic age effect (−$1,598 at ≤30 → +$1,991 at >50).
- **Local:** 5 representative cases explained via waterfalls (e.g., 54-yo obese smoker → $42,667 predicted
  vs $42,000 actual).

## 6. Validation
- **Test performance (production Gradient Boosting):** R² 0.920, RMSE $3,404, MAE $2,108.
- **Business validation:** every modeling finding matches the independent EDA/visualization conclusions
  (smoking dominates; smoker×BMI interaction; region/sex not drivers).
- **Stability:** consistent across folds (std 0.039–0.042 for the top tier).

## 7. Residual diagnostics (production model, test set)
- **Unbiased on average** (mean residual −$716) and **homoscedastic** (corr(|resid|, fitted) = −0.03, p=0.58).
- **But non-normal:** Shapiro-Wilk W=0.52, p≈2e-26 — a heavy right tail of ~15–20 under-predicted cases,
  overwhelmingly **high-cost non-smokers** whose drivers are absent from the feature set.
- **Implication:** point predictions are reliable for the bulk; Gaussian prediction intervals would be
  wrong (use quantile/conformal); an unmodeled high-cost segment points to a data gap, not a model flaw.

## 8. Limitations
- Small (1,337 rows), cross-sectional; only 7 raw features (no claims history / diagnoses / behavior).
- ~15% of cost variation is unexplained, concentrated in the high-cost tail.
- `smoker` likely self-reported; findings specific to this population/cost regime.
- Feature redundancy makes linear coefficients uninterpretable (we relied on SHAP instead).

## 9. Future work
- **Tweedie / Gamma GLM** for the skewed, non-negative insurance target.
- **Quantile regression / conformal prediction** for honest intervals.
- **Deep tabular models (e.g. TabNet)** only if the feature set grows substantially.
- **Acquire claims/diagnosis data** — the highest-value step to break the ~0.85 ceiling.
- Productionize with drift monitoring and fairness audits (sex/region excluded as rating factors).
