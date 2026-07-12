"""
Stage 4 · Chunk 7 — Scale numeric features (StandardScaler, fit on TRAIN only).

Scaled  : continuous + count + engineered numerics (age, bmi, children,
          age_squared, bmi_squared, the 4 interactions, the 3 group aggregates,
          age_group_ord, bmi_category_ord).
Left as-is : 0/1 indicators (sex, smoker, region_* one-hots) — scaling binaries
          adds nothing and hurts interpretability.

Why scale: distance/gradient models (Linear family, SVR, KNN) are sensitive to
feature magnitude; tree models (DT/RF/GB/XGB/AdaBoost/Bagging) are NOT and will
use the UNSCALED splits from Chunk 6.

Note on rigor: for the GridSearchCV model chunks we will wrap StandardScaler in
a Pipeline so scaling is refit inside each CV fold (fully leak-free). The scaled
CSVs saved here satisfy this chunk's verification and provide a ready-made scaled
set; the fitted scaler is persisted for reproducibility.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
SPLITS = BASE / "splits"

X_train = pd.read_csv(SPLITS / "X_train.csv")
X_test = pd.read_csv(SPLITS / "X_test.csv")

INDICATORS = ["sex", "smoker", "region_northeast", "region_northwest",
              "region_southeast", "region_southwest"]
scale_cols = [c for c in X_train.columns if c not in INDICATORS]

scaler = StandardScaler().fit(X_train[scale_cols])

X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[scale_cols] = scaler.transform(X_train[scale_cols])
X_test_s[scale_cols] = scaler.transform(X_test[scale_cols])

print("=" * 70)
print("CHUNK 7 — SCALE NUMERIC FEATURES")
print("=" * 70)
print(f"\nScaled ({len(scale_cols)}) : {scale_cols}")
print(f"Passthrough ({len(INDICATORS)}) : {INDICATORS}")

# --- Verify: train scaled means ~0, stds ~1 (population std, ddof=0) ---
tr_mean = X_train_s[scale_cols].mean().abs().max()
tr_std = X_train_s[scale_cols].std(ddof=0)
print("\n--- Verification on TRAIN (scaled columns) ---")
print(f"max |mean| across scaled cols : {tr_mean:.2e}  (~0 expected)")
print(f"std range across scaled cols  : [{tr_std.min():.4f}, {tr_std.max():.4f}]  (~1 expected)")

print("\nPer-column check (first 6 scaled cols):")
chk = pd.DataFrame({
    "train_mean": X_train_s[scale_cols].mean(),
    "train_std":  X_train_s[scale_cols].std(ddof=0),
    "test_mean":  X_test_s[scale_cols].mean(),
    "test_std":   X_test_s[scale_cols].std(ddof=0),
}).round(4)
print(chk.head(6).to_string())
print("\n(Test means are near — but not exactly — 0/1: correct, since the scaler")
print(" was fit on TRAIN only and applied to unseen TEST rows.)")

# --- Persist ---
X_train_s.to_csv(SPLITS / "X_train_scaled.csv", index=False)
X_test_s.to_csv(SPLITS / "X_test_scaled.csv", index=False)
joblib.dump({"scaler": scaler, "scale_cols": scale_cols, "indicators": INDICATORS},
            SPLITS / "scaler.joblib")
print(f"\n[saved] splits/X_train_scaled.csv, X_test_scaled.csv, scaler.joblib")
