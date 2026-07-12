"""
feature_engineering.py
======================
Feature creation and encoding for the insurance cost model.

Includes LEAK-SAFE target encoding: fit the group aggregates on the training
split only, then apply the fitted tables to any split.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

AGE_GROUP_ORDER = ["18-25", "26-35", "36-45", "46-55", "56-64"]
BMI_CATEGORY_ORDER = ["Underweight", "Normal", "Overweight", "Obese"]


def add_age_bmi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add non-linear age/BMI terms and ordinal category bands.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing ``age`` and ``bmi``.

    Returns
    -------
    pandas.DataFrame
        Copy with ``age_group``, ``bmi_category``, ``age_squared``, ``bmi_squared``.
    """
    out = df.copy()
    out["age_group"] = pd.cut(out["age"], [17, 25, 35, 45, 55, 64], labels=AGE_GROUP_ORDER)
    out["bmi_category"] = pd.cut(out["bmi"], [0, 18.5, 25, 30, np.inf],
                                 labels=BMI_CATEGORY_ORDER, right=False)
    out["age_squared"] = out["age"] ** 2
    out["bmi_squared"] = out["bmi"] ** 2
    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction terms (smoker acts as a 0/1 switch).

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing ``smoker`` (yes/no or 0/1), ``age``, ``bmi``, ``children``.

    Returns
    -------
    pandas.DataFrame
        Copy with ``smoker_age``, ``smoker_bmi``, ``age_bmi``, ``smoker_children``.
    """
    out = df.copy()
    s = out["smoker"]
    if not pd.api.types.is_numeric_dtype(s):
        s = s.map({"yes": 1, "no": 0})
    out["smoker_age"] = s * out["age"]
    out["smoker_bmi"] = s * out["bmi"]
    out["age_bmi"] = out["age"] * out["bmi"]
    out["smoker_children"] = s * out["children"]
    return out


def fit_target_encodings(train_df: pd.DataFrame) -> dict:
    """Fit group-aggregate (target) encodings on the TRAINING split only.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training rows containing ``region``, ``smoker``, ``bmi``, ``charges``.

    Returns
    -------
    dict
        Fitted lookup tables and global fallbacks, to pass to
        :func:`apply_target_encodings`.
    """
    return {
        "region_mean_charges": train_df.groupby("region")["charges"].mean(),
        "smoker_mean_charges": train_df.groupby("smoker")["charges"].mean(),
        "region_mean_bmi": train_df.groupby("region")["bmi"].mean(),
        "global_charge": train_df["charges"].mean(),
        "global_bmi": train_df["bmi"].mean(),
    }


def apply_target_encodings(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """Apply train-fitted target encodings to any split (no leakage).

    Parameters
    ----------
    df : pandas.DataFrame
        Data with ``region`` and ``smoker`` labels.
    maps : dict
        Output of :func:`fit_target_encodings`.

    Returns
    -------
    pandas.DataFrame
        Copy with ``region_mean_charges``, ``smoker_mean_charges``, ``region_mean_bmi``.
    """
    out = df.copy()
    out["region_mean_charges"] = out["region"].map(maps["region_mean_charges"]).fillna(maps["global_charge"])
    out["smoker_mean_charges"] = out["smoker"].map(maps["smoker_mean_charges"]).fillna(maps["global_charge"])
    out["region_mean_bmi"] = out["region"].map(maps["region_mean_bmi"]).fillna(maps["global_bmi"])
    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode region; binary-encode sex (female=1) and smoker (yes=1);
    ordinal-encode the engineered bands. Returns a fully numeric matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with raw + engineered columns.

    Returns
    -------
    pandas.DataFrame
        Fully numeric feature matrix (drops the raw string band columns).
    """
    out = df.copy()
    if not pd.api.types.is_numeric_dtype(out["sex"]):
        out["sex"] = (out["sex"] == "female").astype(int)
    if not pd.api.types.is_numeric_dtype(out["smoker"]):
        out["smoker"] = (out["smoker"] == "yes").astype(int)
    if "age_group" in out.columns:
        out["age_group_ord"] = out["age_group"].map({k: i for i, k in enumerate(AGE_GROUP_ORDER)}).astype(int)
        out["bmi_category_ord"] = out["bmi_category"].map({k: i for i, k in enumerate(BMI_CATEGORY_ORDER)}).astype(int)
        out = out.drop(columns=["age_group", "bmi_category"])
    dummies = pd.get_dummies(df["region"], prefix="region").astype(int)
    return out.drop(columns=["region"]).join(dummies)
