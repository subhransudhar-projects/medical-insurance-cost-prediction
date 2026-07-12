"""Stage 4 · Chunk 19 — Comprehensive model comparison table + charts."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import load_registry, set_style, save_fig, RESULTS, VERMILLION, BLUE

set_style()
reg = load_registry().sort_values("test_r2", ascending=False).reset_index(drop=True)

cols = ["model", "train_r2", "test_r2", "train_rmse", "test_rmse", "train_mae", "test_mae", "overfit_gap_r2"]
table = reg[cols].copy()

print("=" * 100)
print("CHUNK 19 — MODEL COMPARISON (ranked by test R2)")
print("=" * 100)
with pd.option_context("display.float_format", lambda v: f"{v:,.4f}" if abs(v) < 100 else f"{v:,.0f}"):
    print(table.to_string(index=False))

print("\n--- Best hyperparameters ---")
for _, r in reg.iterrows():
    print(f"  {r['model']:32} {r['best_params']}")

top3 = reg.head(3)["model"].tolist()
print(f"\nTOP 3 by test R2: {top3}")

# Save full comparison (incl. params) to CSV
out = reg[cols + ["best_params", "note"]].copy()
out["best_params"] = out["best_params"].apply(json.dumps)
out.to_csv(RESULTS / "model_comparison.csv", index=False)
print(f"[saved] results/model_comparison.csv")

# --- Bar charts: test R2 and test RMSE ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
order = reg.sort_values("test_r2")  # ascending for horizontal bars
colors_r2 = [VERMILLION if m in top3 else "#9aa7bd" for m in order["model"]]
axes[0].barh(order["model"], order["test_r2"], color=colors_r2)
for i, v in enumerate(order["test_r2"]):
    axes[0].text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9, color="#1a2b45")
axes[0].set_title("Test R2 by Model  (top 3 highlighted)", loc="left")
axes[0].set_xlabel("test R2 (higher is better)")
axes[0].set_xlim(0, 1.02)

order2 = reg.sort_values("test_rmse", ascending=False)
colors_rmse = [VERMILLION if m in top3 else "#9aa7bd" for m in order2["model"]]
axes[1].barh(order2["model"], order2["test_rmse"], color=colors_rmse)
for i, v in enumerate(order2["test_rmse"]):
    axes[1].text(v + 80, i, f"${v:,.0f}", va="center", fontsize=9, color="#1a2b45")
axes[1].set_title("Test RMSE by Model  (top 3 highlighted, lower is better)", loc="left")
axes[1].set_xlabel("test RMSE (USD)")
axes[1].set_xlim(0, order2["test_rmse"].max() * 1.18)

fig.suptitle("Stage 4 — Model Comparison (12 models)", fontsize=17, fontweight="bold", y=1.0)
fig.tight_layout()
save_fig(fig, "chunk19_model_comparison")
