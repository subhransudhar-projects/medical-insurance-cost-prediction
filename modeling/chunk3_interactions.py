"""
Stage 4 · Chunk 3 — Feature engineering: interaction terms.

Story: our EDA showed the effect of bmi on charges is far larger for smokers
than non-smokers. Interaction terms encode that reality so models can exploit
it directly rather than having to rediscover it from raw columns.

Reads/writes insurance_features.csv (accumulating pipeline; idempotent).
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
FEATURES = BASE / "insurance_features.csv"

df = pd.read_csv(FEATURES)

# smoker is categorical yes/no; interactions need its binary form (yes=1, no=0).
smoker_bin = (df["smoker"] == "yes").astype(int)

df["smoker_age"]      = smoker_bin * df["age"]        # smoking effect intensifying with age
df["smoker_bmi"]      = smoker_bin * df["bmi"]        # smoking combined with high BMI (key EDA finding)
df["age_bmi"]         = df["age"] * df["bmi"]         # joint age/BMI risk
df["smoker_children"] = smoker_bin * df["children"]   # smoking x dependents

new_cols = ["smoker_age", "smoker_bmi", "age_bmi", "smoker_children"]

print("=" * 68)
print("CHUNK 3 — INTERACTION TERMS")
print("=" * 68)
print(f"\nShape now : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"New cols  : {new_cols}")

print("\n--- Summary of new interaction features ---")
print(df[new_cols].describe().round(2).to_string())

print("\n--- How they separate smokers vs non-smokers (mean) ---")
comp = df.assign(smoker=df["smoker"]).groupby("smoker")[new_cols].mean().round(1)
print(comp.to_string())
print("\nNote: smoker_* terms are 0 for every non-smoker by construction, so they")
print("act as 'switches' that let the model apply an extra age/BMI slope only to smokers.")

print("\n--- First 5 rows (key columns) ---")
show = ["age", "bmi", "children", "smoker", "smoker_age", "smoker_bmi", "age_bmi", "smoker_children"]
print(df[show].head(5).to_string(index=False))

df.to_csv(FEATURES, index=False)
print(f"\n[saved] {FEATURES.name}  ({df.shape[1]} columns)")
