# Technical Summary — Medical Insurance Cost Modeling
*For a data-science audience · ~2 pages*

## 1. Data & feature engineering
- **Source:** `insurance_cleaned.csv`, 1,337 rows × 7 columns (age, sex, bmi, children, smoker, region,
  charges), validated in Stage 1 (0 missing, 0 duplicates, types correct).
- **Engineered features (20 total):**
  - Non-linear terms: `age_squared`, `bmi_squared`; ordinal `age_group` (5 bands), `bmi_category` (4).
  - Interactions: `smoker_age`, `smoker_bmi`, `age_bmi`, `smoker_children` (smoker terms via a 0/1 switch).
  - Group aggregates: `region_mean_charges`, `smoker_mean_charges`, `region_mean_bmi`.
  - Encodings: one-hot `region`, binary `sex` (female=1), binary `smoker` (yes=1).
- **Leakage control:** the two target-encoded aggregates were **refit on the training fold only** and mapped
  onto test rows (Chunk 6). This lowers headline scores but removes target leakage.

## 2. Protocol
- **Split:** 80/20, `random_state=42`, **stratified on `smoker`** (train 20.49% / test 20.52% smokers).
- **Scaling:** `StandardScaler` on the 14 continuous/engineered features; 6 indicators passed through.
  Scale-sensitive models used `Pipeline(StandardScaler, model)` so scaling is **refit inside each CV fold**.
- **Tuning:** `GridSearchCV`, 5-fold `KFold(shuffle, rs=42)`, scoring R². Metrics logged to a JSON registry.

## 3. Models & results (16 total)

| Model | Test R² | 5-fold CV R² | Test RMSE | Notes |
|---|---|---|---|---|
| Voting (top 3) | 0.924 | 0.848 | $3,316 | best by test R² (ties SVR) |
| SVR (target-scaled) | 0.924 | 0.847 | $3,317 | best single model |
| Stacking (Linear meta) | 0.922 | — | $3,352 | ties, not beats, base |
| Gradient Boosting | 0.920 | 0.840 | $3,404 | **selected for production** |
| XGBoost | 0.919 | 0.844 | $3,422 | |
| Random Forest | 0.918 | 0.840 | $3,440 | |
| Linear / Ridge / Lasso / ENet | ~0.897 | ~0.825 | ~$3,840 | strong, interpretable baselines |
| KNN | 0.895 | 0.812 | $3,887 | memorizes (train R²=0.998) |
| MedCost-AdaBoost | 0.709 | 0.687 | $6,475 | chases the outlier tail |
| SVR (inputs only) | 0.363 | 0.229 | $9,587 | underfits without target scaling |

## 4. Key technical findings
- **The top ~6 models are a statistical tie:** spread of CV means = 0.008 vs CV std = 0.040. Ranking within
  the tier is split-noise.
- **Honest performance is ~0.85 CV R², not 0.92.** The single test split is **+0.076 optimistic** — measured
  directly, and the reason we quote CV.
- **Ensembles did not beat the best single model** — base models are correlated (all lean on `smoker_bmi`),
  so averaging cancels little. We chose **Gradient Boosting** for production (tied accuracy, simpler,
  SHAP-native).
- **SVR is a cautionary tale:** worst model as-specified (0.36), best after `TransformedTargetRegressor`
  target scaling (0.924) — a diagnosis, not a recipe.

## 5. Diagnostics & explainability
- **Residuals:** unbiased on average (mean −$716), **no heteroscedasticity** (corr |resid|,fitted = −0.03),
  but strongly non-normal (Shapiro W=0.52, p≈2e-26) with a heavy right tail — high-cost non-smokers the
  features can't explain.
- **SHAP (TreeExplainer):** base value $13,356; top drivers `smoker_bmi` ($6,183 mean |SHAP|), `age`
  ($1,429), `age_bmi`, `age_squared`, `smoker_age`. Dependence plots show a **step-change at BMI 30 for
  smokers**. SHAP and built-in importances agree on ranking.

## 6. Reproducibility
Modular scripts `chunk1..chunk35`, a shared `model_utils.py` (splits, metrics, registry, model factory),
persisted splits + scaler + fitted models (`results/models/*.joblib`), a results registry
(`results/model_results.json`), and a comparison CSV. All runs seeded `random_state=42`.

## 7. Recommended next steps
Tweedie/Gamma GLM and quantile/conformal intervals for the skewed target; acquire claims/diagnosis data to
break the ~0.85 ceiling; wrap in a monitored, fairness-audited pipeline (sex/region excluded).
