"""
Stage 4 · Chunk 20 — Ensemble: Voting Regressor (top 3 models).

Averages the predictions of the three best individual models. If the models
make partly-independent errors, the average cancels some noise and can beat
each one alone.
"""
import pandas as pd
from sklearn.ensemble import VotingRegressor
from model_utils import (load_splits, evaluate_and_log, load_registry,
                         build_estimator, save_model)

X_train, X_test, y_train, y_test = load_splits(scaled=False)
reg = load_registry().sort_values("test_r2", ascending=False)

top3 = reg.head(3)[["model", "best_params", "test_r2"]].to_dict("records")
print("=" * 66); print("CHUNK 20 — VOTING REGRESSOR (top 3)"); print("=" * 66)
print("Base models (top 3 by test R2):")
for r in top3:
    print(f"  {r['model']:24} test R2={r['test_r2']:.4f}  params={r['best_params']}")

estimators = [(r["model"], build_estimator(r["model"], r["best_params"])) for r in top3]
voting = VotingRegressor(estimators=estimators, n_jobs=-1)
voting.fit(X_train, y_train)

rec = evaluate_and_log("Voting (top 3)", voting, X_train, y_train, X_test, y_test,
                       best_params={"members": [r["model"] for r in top3]},
                       note="VotingRegressor of top-3 individual models", scaled="mixed")
save_model(voting, "voting_top3")

print("\n--- Voting vs its members (test set) ---")
reg2 = load_registry()
names = [r["model"] for r in top3] + ["Voting (top 3)"]
print(reg2[reg2["model"].isin(names)][["model", "test_r2", "test_rmse", "test_mae"]]
      .sort_values("test_r2", ascending=False).round(4).to_string(index=False))

best_member = top3[0]["test_r2"]
verdict = "OUTPERFORMS" if rec["test_r2"] > best_member else "does NOT beat"
print(f"\nVerdict: the ensemble {verdict} the best individual model "
      f"({rec['test_r2']:.4f} vs {best_member:.4f}).")
