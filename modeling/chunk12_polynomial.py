"""
Stage 4 · Chunk 12 — Baseline: Polynomial Regression (degree 2).

Built from the RAW encoded features (age, bmi, children, sex, smoker, region)
so degree-2 expansion generates ALL pairwise interactions + squares
automatically — including smoker x bmi — without relying on Stage-4 manual
feature engineering. This is the bridge from linear to non-linear modeling.

Pipeline: PolynomialFeatures(2) -> StandardScaler -> LinearRegression.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

from model_utils import load_splits, evaluate_and_log, load_registry

X_train, X_test, y_train, y_test = load_splits(scaled=False)

# Raw encoded features only (drop engineered squares/interactions/aggregates/ordinals;
# drop region_southeast as the reference level).
raw_cols = ["age", "bmi", "children", "sex", "smoker",
            "region_northeast", "region_northwest", "region_southwest"]
Xtr, Xte = X_train[raw_cols], X_test[raw_cols]

pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression()),
])
pipe.fit(Xtr, y_train)

n_poly = pipe.named_steps["poly"].n_output_features_
print("=" * 66)
print("CHUNK 12 — POLYNOMIAL REGRESSION (degree 2)")
print("=" * 66)
print(f"Raw inputs        : {len(raw_cols)}  ->  polynomial features: {n_poly}")

rec = evaluate_and_log(
    "Polynomial Regression (deg 2)", pipe, Xtr, y_train, Xte, y_test,
    best_params={"degree": 2, "raw_features": len(raw_cols), "poly_features": int(n_poly)},
    note="PolynomialFeatures(2)+StandardScaler+LinearRegression on raw encoded features",
    scaled="pipeline",
)

# --- Which auto-generated interactions matter most? ---
names = pipe.named_steps["poly"].get_feature_names_out(raw_cols)
coef = pd.Series(pipe.named_steps["lr"].coef_, index=names)
print("\n--- Top 8 polynomial terms by |standardized coefficient| ---")
print(coef.reindex(coef.abs().sort_values(ascending=False).index).head(8).round(1).to_string())
print("\n(Note whether the auto-discovered 'smoker bmi' term surfaces near the top —")
print(" that would independently reproduce our key EDA interaction.)")

reg = load_registry()
comp = reg[reg["model"].str.contains("Linear|Ridge|Lasso|Elastic|Polynomial")][
    ["model", "test_r2", "test_rmse", "test_mae"]].round(4)
print("\n--- Full linear-family comparison (test set) ---")
print(comp.to_string(index=False))
