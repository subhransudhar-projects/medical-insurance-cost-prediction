"""
data_preprocessing.py
=====================
Loading, validation, and cleaning for the Medical Insurance Cost dataset.

Schema (exact column names): age, sex, bmi, children, smoker, region, charges
"""
from __future__ import annotations
import pandas as pd

EXPECTED_COLS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
CATEGORICAL_COLS = ["sex", "smoker", "region"]
VALID_CATEGORIES = {
    "sex": {"male", "female"},
    "smoker": {"yes", "no"},
    "region": {"northeast", "northwest", "southeast", "southwest"},
}


def load_data(path: str) -> pd.DataFrame:
    """Load the insurance dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV (raw or cleaned).

    Returns
    -------
    pandas.DataFrame
        The loaded data with any stray unnamed index columns dropped.
    """
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")


def validate(df: pd.DataFrame) -> dict:
    """Validate schema, missing values, duplicates, and categorical levels.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to validate.

    Returns
    -------
    dict
        Report with keys: ``schema_ok``, ``n_missing``, ``n_duplicates``,
        ``bad_categories``, and ``is_clean`` (bool).
    """
    schema_ok = list(df.columns[:7]) == EXPECTED_COLS or set(EXPECTED_COLS).issubset(df.columns)
    bad = {}
    for col, valid in VALID_CATEGORIES.items():
        if col in df.columns:
            offenders = set(df[col].astype(str).str.strip().str.lower()) - valid
            if offenders:
                bad[col] = sorted(offenders)
    report = {
        "schema_ok": bool(schema_ok),
        "n_missing": int(df.isnull().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "bad_categories": bad,
    }
    report["is_clean"] = (
        report["schema_ok"] and report["n_missing"] == 0
        and report["n_duplicates"] == 0 and not bad
    )
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: normalized categoricals, exact duplicates removed.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data.

    Returns
    -------
    pandas.DataFrame
        Cleaned data with a fresh index. Charges/outliers are intentionally
        retained (high-cost smokers are legitimate, not errors).
    """
    out = df.copy()
    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.lower()
    return out.drop_duplicates().reset_index(drop=True)
