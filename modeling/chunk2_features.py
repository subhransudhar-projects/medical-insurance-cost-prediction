"""
Stage 4 · Chunk 2 — Feature engineering: age & BMI transformations.

Story: insurance costs do not rise linearly with age or BMI. We add category
bands and squared terms so linear models can 'see' the true, curved patterns
and so tree models get ready-made segment features.

Outputs an enriched dataset (insurance_features.csv) used by later chunks.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATASET_DIR = BASE.parent
CLEANED = DATASET_DIR / "insurance_cleaned.csv"
OUT = BASE / "insurance_features.csv"

df = pd.read_csv(CLEANED)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

# --- age_group: 18-25, 26-35, 36-45, 46-55, 56-64 -------------------------
AGE_GROUP_ORDER = ["18-25", "26-35", "36-45", "46-55", "56-64"]
df["age_group"] = pd.cut(
    df["age"], bins=[17, 25, 35, 45, 55, 64],
    labels=AGE_GROUP_ORDER, right=True,
).astype(pd.CategoricalDtype(AGE_GROUP_ORDER, ordered=True))

# --- bmi_category: Underweight/Normal/Overweight/Obese --------------------
BMI_CATEGORY_ORDER = ["Underweight", "Normal", "Overweight", "Obese"]
df["bmi_category"] = pd.cut(
    df["bmi"], bins=[0, 18.5, 25, 30, np.inf],
    labels=BMI_CATEGORY_ORDER, right=False,
).astype(pd.CategoricalDtype(BMI_CATEGORY_ORDER, ordered=True))

# --- Non-linear numeric terms ---------------------------------------------
df["age_squared"] = df["age"] ** 2
df["bmi_squared"] = df["bmi"] ** 2

print("=" * 64)
print("CHUNK 2 — FEATURE ENGINEERING (age & BMI)")
print("=" * 64)
print(f"\nEnriched shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"New columns   : age_group, bmi_category, age_squared, bmi_squared")

print("\n--- age_group value counts ---")
print(df["age_group"].value_counts().reindex(AGE_GROUP_ORDER).to_string())

print("\n--- bmi_category value counts ---")
print(df["bmi_category"].value_counts().reindex(BMI_CATEGORY_ORDER).to_string())

print("\n--- age_squared / bmi_squared sanity (ranges) ---")
print(f"age        : min {df['age'].min()}  max {df['age'].max()}")
print(f"age_squared: min {df['age_squared'].min()}  max {df['age_squared'].max()}")
print(f"bmi        : min {df['bmi'].min():.2f}  max {df['bmi'].max():.2f}")
print(f"bmi_squared: min {df['bmi_squared'].min():.2f}  max {df['bmi_squared'].max():.2f}")

print("\nFirst 3 rows (new columns):")
print(df[["age", "age_group", "age_squared", "bmi", "bmi_category", "bmi_squared"]]
      .head(3).to_string(index=False))

df.to_csv(OUT, index=False)
print(f"\n[saved] {OUT.name}  ({df.shape[1]} columns)")
