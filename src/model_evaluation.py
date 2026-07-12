"""
model_evaluation.py
===================
Metrics, cross-validation, and diagnostics for insurance cost regression.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_validate, KFold

RANDOM_STATE = 42


def regression_metrics(y_true, y_pred) -> dict:
    """Compute R², RMSE, and MAE.

    Parameters
    ----------
    y_true, y_pred : array-like
        Actual and predicted charges.

    Returns
    -------
    dict
        ``{"r2", "rmse", "mae"}``.
    """
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """Score a fitted model on train and test and return a flat record.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_train, y_train, X_test, y_test : array-like

    Returns
    -------
    dict
        Train/test R², RMSE, MAE plus the train-test R² overfit gap.
    """
    tr = regression_metrics(y_train, model.predict(X_train))
    te = regression_metrics(y_test, model.predict(X_test))
    return {
        "train_r2": tr["r2"], "test_r2": te["r2"],
        "train_rmse": tr["rmse"], "test_rmse": te["rmse"],
        "train_mae": tr["mae"], "test_mae": te["mae"],
        "overfit_gap_r2": tr["r2"] - te["r2"],
    }


def cross_validate_model(model, X, y, cv_splits=5) -> dict:
    """5-fold CV of R² and RMSE with stability (std) reported.

    Parameters
    ----------
    model : sklearn estimator (unfitted; cloned per fold)
    X, y : array-like
    cv_splits : int, default 5

    Returns
    -------
    dict
        ``{"cv_r2_mean", "cv_r2_std", "cv_rmse_mean", "cv_rmse_std", "fold_r2"}``.
    """
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(model, X, y, cv=cv,
                         scoring=["r2", "neg_root_mean_squared_error"], n_jobs=-1)
    r2 = res["test_r2"]
    rmse = -res["test_neg_root_mean_squared_error"]
    return {
        "cv_r2_mean": float(r2.mean()), "cv_r2_std": float(r2.std()),
        "cv_rmse_mean": float(rmse.mean()), "cv_rmse_std": float(rmse.std()),
        "fold_r2": r2.tolist(),
    }


def residual_diagnostics(y_true, y_pred) -> dict:
    """Bias, heteroscedasticity, and normality checks on residuals.

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    dict
        Mean/median/std residual, Shapiro-Wilk (W, p), and the
        corr(|residual|, fitted) heteroscedasticity check with its p-value.
    """
    from scipy import stats
    resid = np.asarray(y_true) - np.asarray(y_pred)
    sw_w, sw_p = stats.shapiro(resid)
    het_r, het_p = stats.pearsonr(np.abs(resid), np.asarray(y_pred))
    return {
        "mean_residual": float(resid.mean()),
        "median_residual": float(np.median(resid)),
        "std_residual": float(resid.std()),
        "shapiro_w": float(sw_w), "shapiro_p": float(sw_p),
        "normal": bool(sw_p >= 0.05),
        "hetero_corr": float(het_r), "hetero_p": float(het_p),
    }


def feature_importance(model, feature_names) -> pd.Series:
    """Return sorted feature importances for a tree-based model.

    Parameters
    ----------
    model : fitted estimator exposing ``feature_importances_``
    feature_names : list of str

    Returns
    -------
    pandas.Series
        Importances indexed by feature name, descending.
    """
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
