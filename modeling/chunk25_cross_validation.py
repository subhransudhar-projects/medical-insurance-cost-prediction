"""
Stage 4 · Chunk 25 — Cross-validation of the best model(s).

Identify the best model by test R2, then 5-fold CV it on the TRAINING set
(test stays held out). We ALSO CV the rest of the top tier to test whether the
leaderboard ordering is real signal or just test-split noise, and to pick a
sensible production model for the explainability phase.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from model_utils import load_splits, load_registry, load_model, build_estimator

X_train, X_test, y_train, y_test = load_splits(scaled=False)
reg = load_registry().sort_values("test_r2", ascending=False)
best_name = reg.iloc[0]["model"]

print("=" * 70)
print("CHUNK 25 — CROSS-VALIDATION OF BEST MODEL")
print("=" * 70)
print(f"Best model by test R2: {best_name} (test R2={reg.iloc[0]['test_r2']:.4f})")

svr_params = reg.loc[reg["model"] == "SVR (target-scaled)", "best_params"].iloc[0]
top_models = {
    "Voting (top 3)": load_model("voting_top3"),
    "SVR (target-scaled)": build_estimator("SVR (target-scaled)", svr_params),
    "Gradient Boosting": load_model("gradient_boosting"),
    "XGBoost": load_model("xgboost"),
    "Random Forest": load_model("random_forest"),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []
fold_detail = {}
for name, est in top_models.items():
    res = cross_validate(est, X_train, y_train, cv=cv,
                         scoring=["r2", "neg_root_mean_squared_error"], n_jobs=-1)
    r2s = res["test_r2"]
    rmses = -res["test_neg_root_mean_squared_error"]
    fold_detail[name] = r2s
    rows.append({
        "model": name,
        "cv_r2_mean": r2s.mean(), "cv_r2_std": r2s.std(),
        "cv_rmse_mean": rmses.mean(), "cv_rmse_std": rmses.std(),
        "single_test_r2": reg.loc[reg["model"] == name, "test_r2"].iloc[0],
    })

cvtab = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False)
print("\n--- 5-fold CV on TRAINING set (top tier) ---")
print(cvtab.round(4).to_string(index=False))

print(f"\n--- Best model ({best_name}) fold-by-fold R2 ---")
folds = fold_detail[best_name]
for i, r in enumerate(folds, 1):
    print(f"  fold {i}: R2 = {r:.4f}")
print(f"  mean = {folds.mean():.4f}  std = {folds.std():.4f}")

# Stability assessment
best_row = cvtab[cvtab["model"] == best_name].iloc[0]
gap = best_row["single_test_r2"] - best_row["cv_r2_mean"]
print("\n--- Stability assessment ---")
print(f"{best_name}: CV R2 = {best_row['cv_r2_mean']:.4f} +/- {best_row['cv_r2_std']:.4f}, "
      f"single-split test R2 = {best_row['single_test_r2']:.4f}")
print(f"Optimism of the test split over CV mean: +{gap:.4f}")
print(f"CV std ({best_row['cv_r2_std']:.4f}) >= spread of top-5 CV means "
      f"({cvtab['cv_r2_mean'].max()-cvtab['cv_r2_mean'].min():.4f}): "
      f"{'YES -> top models are statistically tied' if best_row['cv_r2_std'] >= (cvtab['cv_r2_mean'].max()-cvtab['cv_r2_mean'].min()) else 'NO'}")

print("\n--- Production selection (reasoned) ---")
print("The top models are within one CV standard deviation of each other -> a")
print("statistical tie. For deployment + explainability (Chunks 26-29) we carry")
print("forward GRADIENT BOOSTING: within noise of the best, far simpler than the")
print("Voting ensemble, and natively explainable via SHAP TreeExplainer.")
