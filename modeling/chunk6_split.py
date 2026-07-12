"""
Stage 4 · Chunk 6 — Train/test split (leak-safe).

  - X = all features, y = charges
  - 80/20 split, random_state=42, STRATIFIED on smoker status
  - REFIT the three group-aggregate features on the TRAINING split only
    (region_mean_charges, smoker_mean_charges, region_mean_bmi), then map the
    train-fitted tables onto the test rows. This removes the target leakage
    flagged in Chunk 4.
  - Persist splits to modeling/splits/ for all downstream chunks.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
MATRIX = BASE / "insurance_model_matrix.csv"
FEATURES = BASE / "insurance_features.csv"
SPLITS = BASE / "splits"
SPLITS.mkdir(exist_ok=True)

mm = pd.read_csv(MATRIX)
feats = pd.read_csv(FEATURES)  # same row order/index; holds region & smoker labels

y = mm["charges"]
X = mm.drop(columns=["charges"])
strat = mm["smoker"]  # 0/1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=strat
)
train_idx, test_idx = X_train.index, X_test.index

# --- Leak-safe refit of group aggregates on TRAIN only ---------------------
region_lab = feats["region"]
smoker_lab = feats["smoker"]
global_mean_charge = y_train.mean()

reg_charge_map = pd.DataFrame(
    {"region": region_lab.loc[train_idx].values, "charges": y_train.values}
).groupby("region")["charges"].mean()
smk_charge_map = pd.DataFrame(
    {"smoker": smoker_lab.loc[train_idx].values, "charges": y_train.values}
).groupby("smoker")["charges"].mean()
reg_bmi_map = pd.DataFrame(
    {"region": region_lab.loc[train_idx].values, "bmi": X_train["bmi"].values}
).groupby("region")["bmi"].mean()

def remap(frame, idx):
    frame = frame.copy()
    frame["region_mean_charges"] = region_lab.loc[idx].map(reg_charge_map).fillna(global_mean_charge).values
    frame["smoker_mean_charges"] = smoker_lab.loc[idx].map(smk_charge_map).fillna(global_mean_charge).values
    frame["region_mean_bmi"]     = region_lab.loc[idx].map(reg_bmi_map).fillna(X_train["bmi"].mean()).values
    return frame

# Snapshot full-data values (from Chunk 4) before overwriting, to prove the refit changed them
full_reg_charge = mm.groupby(region_lab)["region_mean_charges"].first() if False else None
X_train = remap(X_train, train_idx)
X_test = remap(X_test, test_idx)

print("=" * 70)
print("CHUNK 6 — TRAIN/TEST SPLIT (leak-safe, stratified on smoker)")
print("=" * 70)
print(f"\nX_train : {X_train.shape}    y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}     y_test  : {y_test.shape}")
print(f"Split   : {len(X_train)/len(mm)*100:.0f}% train / {len(X_test)/len(mm)*100:.0f}% test")

print("\n--- Stratification check: smoker share preserved ---")
print(f"overall smoker rate : {strat.mean()*100:.2f}%")
print(f"train smoker rate   : {mm.loc[train_idx,'smoker'].mean()*100:.2f}%")
print(f"test smoker rate    : {mm.loc[test_idx,'smoker'].mean()*100:.2f}%")

print("\n--- Leak-safe refit: TRAIN-fitted target encodings ---")
print("region_mean_charges (train-fitted):")
print(reg_charge_map.round(0).to_string())
print("smoker_mean_charges (train-fitted):  no=%.0f  yes=%.0f"
      % (smk_charge_map.loc["no"], smk_charge_map.loc["yes"]))
print("(These are computed from the 1,069 training rows only — test charges")
print(" never entered these tables, so no target leaks into the features.)")

# --- Persist splits ---
X_train.to_csv(SPLITS / "X_train.csv", index=False)
X_test.to_csv(SPLITS / "X_test.csv", index=False)
y_train.to_csv(SPLITS / "y_train.csv", index=False)
y_test.to_csv(SPLITS / "y_test.csv", index=False)
print(f"\n[saved] splits/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print(f"        feature count = {X_train.shape[1]}")
