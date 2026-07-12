"""
Stage 4 · Chunk 5 — Encode categorical variables into a numeric model matrix.

Encoding scheme (as specified):
  - region : one-hot  -> region_northeast/northwest/southeast/southwest
  - sex    : binary   -> male=0, female=1
  - smoker : binary   -> no=0,  yes=1
Additionally (required so the matrix is fully numeric for modeling):
  - age_group    : ordinal 0..4 (ordered band)
  - bmi_category : ordinal 0..3 (ordered clinical category)

Writes insurance_model_matrix.csv (fully numeric). Keeps insurance_features.csv
(human-readable) untouched.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
FEATURES = BASE / "insurance_features.csv"
OUT = BASE / "insurance_model_matrix.csv"

df = pd.read_csv(FEATURES)

AGE_ORDER = ["18-25", "26-35", "36-45", "46-55", "56-64"]
BMI_ORDER = ["Underweight", "Normal", "Overweight", "Obese"]

m = df.copy()

# --- Binary encodings ---
m["sex"] = (m["sex"] == "female").astype(int)      # male=0, female=1
m["smoker"] = (m["smoker"] == "yes").astype(int)   # no=0, yes=1

# --- Ordinal encodings for engineered ordered categoricals ---
m["age_group_ord"] = m["age_group"].map({k: i for i, k in enumerate(AGE_ORDER)}).astype(int)
m["bmi_category_ord"] = m["bmi_category"].map({k: i for i, k in enumerate(BMI_ORDER)}).astype(int)
m = m.drop(columns=["age_group", "bmi_category"])

# --- One-hot region (keep all 4 levels; tree models are fine, linear models
#     use regularization — noted for Chunk 8) ---
region_dummies = pd.get_dummies(df["region"], prefix="region").astype(int)
m = m.drop(columns=["region"]).join(region_dummies)

# Reorder: put charges (target) last
target = m.pop("charges")
m["charges"] = target

print("=" * 70)
print("CHUNK 5 — ENCODE CATEGORICAL VARIABLES")
print("=" * 70)
print(f"\nModel matrix shape : {m.shape[0]} rows x {m.shape[1]} columns")
print(f"\nAll columns ({m.shape[1]}):")
for c in m.columns:
    print(f"  - {c:24} {str(m[c].dtype)}")

print("\n--- Encoding sanity checks ---")
print(f"sex    : {dict(df['sex'].value_counts())}  ->  female=1 count {int(m['sex'].sum())}")
print(f"smoker : {dict(df['smoker'].value_counts())}  ->  yes=1 count {int(m['smoker'].sum())}")
print("region one-hot column sums (should match region counts):")
for c in [c for c in m.columns if c.startswith("region_")]:
    print(f"  {c:22} {int(m[c].sum())}")

all_numeric = all(pd.api.types.is_numeric_dtype(m[c]) for c in m.columns)
print(f"\nAll columns numeric: {all_numeric}")

print("\n--- First 5 rows (transposed for readability) ---")
print(m.head(5).T.to_string())

m.to_csv(OUT, index=False)
print(f"\n[saved] {OUT.name}  ({m.shape[1]} columns, fully numeric)")
