"""
DATA STORY — Chunk 1: Load data & initial inspection.

Loads the raw Medical Insurance Cost dataset and produces a first-look profile:
head, tail, dtypes, shape, and summary statistics (numeric + categorical).
Exact columns: age, sex, bmi, children, smoker, region, charges.
"""
from pathlib import Path
import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

FOLDER = Path(__file__).resolve().parent.parent            # Medical Insurance Cost Dataset
# The brief says "Excel file", but the dataset ships as insurance.csv (no .xlsx exists).
SRC = FOLDER / "insurance.csv"

df = pd.read_csv(SRC)

def banner(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)

banner("SOURCE")
print(f"Loaded: {SRC.name}  (note: dataset is CSV, not Excel — no .xlsx present)")

banner("SHAPE  (rows, columns)")
print(f"{df.shape[0]} rows  x  {df.shape[1]} columns")

banner("FIRST 5 ROWS")
print(df.head(5).to_string())

banner("LAST 5 ROWS")
print(df.tail(5).to_string())

banner("COLUMN NAMES & DATA TYPES")
info = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null": df.notna().sum(),
    "nulls": df.isna().sum(),
    "unique": df.nunique(),
})
print(info.to_string())

banner("SUMMARY STATISTICS — NUMERIC")
print(df.describe().T.to_string())

banner("SUMMARY STATISTICS — CATEGORICAL")
print(df.describe(include="object").T.to_string())

banner("QUICK DATA-QUALITY FLAGS")
print(f"Exact duplicate rows : {df.duplicated().sum()}")
print(f"Any missing values   : {df.isna().any().any()}  (total {int(df.isna().sum().sum())})")
print(f"charges — min/max     : ${df['charges'].min():,.0f}  /  ${df['charges'].max():,.0f}")
print(f"charges — mean vs med : ${df['charges'].mean():,.0f}  vs  ${df['charges'].median():,.0f}  "
      f"(mean > median => right-skewed)")
