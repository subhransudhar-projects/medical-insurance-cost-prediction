"""
Stage 4 · Chunk 1 — Load the cleaned dataset and verify its clean state.

Story: we begin the modeling journey from a clean, validated foundation so we
can trust that every downstream model is built on solid ground.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent            # .../Medical Insurance Cost Dataset/modeling
DATASET_DIR = BASE.parent
CLEANED = DATASET_DIR / "insurance_cleaned.csv"

EXPECTED_COLS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]

df = pd.read_csv(CLEANED)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

print("=" * 64)
print("CHUNK 1 — LOAD & VERIFY CLEAN STATE")
print("=" * 64)
print(f"\nSource file        : {CLEANED.name}")
print(f"Shape              : {df.shape[0]} rows x {df.shape[1]} columns")

# --- Column-name check ---
cols_ok = list(df.columns) == EXPECTED_COLS
print(f"\nColumns present    : {list(df.columns)}")
print(f"Columns as expected: {cols_ok}  (age, sex, bmi, children, smoker, region, charges)")

# --- Missing-value check ---
missing = df.isnull().sum()
print(f"\nMissing values     : {int(missing.sum())} total")
if missing.sum():
    print(missing[missing > 0])

# --- Duplicates ---
print(f"Exact duplicates   : {int(df.duplicated().sum())}")

# --- Data types ---
print("\nData types:")
print(df.dtypes.to_string())

# --- Head ---
print("\nFirst 3 rows:")
print(df.head(3).to_string(index=False))

# --- Verdict ---
clean = cols_ok and missing.sum() == 0 and df.duplicated().sum() == 0
print("\n" + "-" * 64)
print(f"VERDICT: {'CLEAN — ready to model' if clean else 'ISSUES FOUND — review above'}")
print("-" * 64)
