"""
Shared utilities for the Stage 4 modeling chunks.

Provides: split loading, consistent metric computation (R2, RMSE, MAE on train
and test), a JSON results registry (so the Chunk 19 comparison table is a simple
read), and a common publication plotting style.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

BASE = Path(__file__).resolve().parent
SPLITS = BASE / "splits"
FIGDIR = BASE / "figures"
RESULTS = BASE / "results"
FIGDIR.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
REGISTRY = RESULTS / "model_results.json"

# Colorblind-safe accents (consistent with the Stage-3 visualization suite)
VERMILLION = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"
NAVY = "#1a2b45"


def set_style():
    sns.set_theme(style="whitegrid", palette="colorblind", font="DejaVu Sans")
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.titlesize": 15, "axes.titleweight": "semibold",
        "axes.labelsize": 12.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def load_splits(scaled=False):
    """Return X_train, X_test, y_train, y_test. scaled=True loads standardized X."""
    sfx = "_scaled" if scaled else ""
    X_train = pd.read_csv(SPLITS / f"X_train{sfx}.csv")
    X_test = pd.read_csv(SPLITS / f"X_test{sfx}.csv")
    y_train = pd.read_csv(SPLITS / "y_train.csv")["charges"]
    y_test = pd.read_csv(SPLITS / "y_test.csv")["charges"]
    return X_train, X_test, y_train, y_test


def metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_and_log(name, model, X_train, y_train, X_test, y_test,
                     best_params=None, note="", scaled=None, verbose=True):
    """Score a fitted model on train+test, print a summary, and log to the registry."""
    pred_tr = model.predict(X_train)
    pred_te = model.predict(X_test)
    m_tr, m_te = metrics(y_train, pred_tr), metrics(y_test, pred_te)
    rec = {
        "model": name,
        "train_r2": m_tr["r2"], "test_r2": m_te["r2"],
        "train_rmse": m_tr["rmse"], "test_rmse": m_te["rmse"],
        "train_mae": m_tr["mae"], "test_mae": m_te["mae"],
        "overfit_gap_r2": m_tr["r2"] - m_te["r2"],
        "best_params": best_params or {},
        "scaled_input": scaled,
        "note": note,
    }
    _append_registry(rec)
    if verbose:
        print(f"\n{'='*66}\n{name}\n{'='*66}")
        if best_params:
            print(f"Best params : {best_params}")
        print(f"{'metric':<8}{'TRAIN':>14}{'TEST':>14}")
        print(f"{'R2':<8}{m_tr['r2']:>14.4f}{m_te['r2']:>14.4f}")
        print(f"{'RMSE':<8}{m_tr['rmse']:>14,.0f}{m_te['rmse']:>14,.0f}")
        print(f"{'MAE':<8}{m_tr['mae']:>14,.0f}{m_te['mae']:>14,.0f}")
        print(f"Overfit gap (train-test R2): {rec['overfit_gap_r2']:.4f}")
    return rec


def _append_registry(rec):
    data = []
    if REGISTRY.exists():
        data = json.loads(REGISTRY.read_text())
    data = [d for d in data if d["model"] != rec["model"]]  # replace same-name
    data.append(rec)
    REGISTRY.write_text(json.dumps(data, indent=2))


def load_registry():
    if REGISTRY.exists():
        return pd.DataFrame(json.loads(REGISTRY.read_text()))
    return pd.DataFrame()


def save_fig(fig, name):
    fig.savefig(FIGDIR / f"{name}.png")
    fig.savefig(FIGDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"[saved] figures/{name}.png")


MODELS = RESULTS / "models"
MODELS.mkdir(exist_ok=True)


def save_model(model, key):
    """Persist a fitted estimator under a short key (e.g. 'random_forest')."""
    import joblib
    joblib.dump(model, MODELS / f"{key}.joblib")
    print(f"[saved] results/models/{key}.joblib")


def load_model(key):
    import joblib
    return joblib.load(MODELS / f"{key}.joblib")


def build_estimator(name, params):
    """Reconstruct a fresh (unfitted) estimator from a registry model name + tuned params.
    Scale-sensitive models are wrapped in a StandardScaler pipeline. Polynomial
    Regression is excluded (it uses a column subset) and unsupported here."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.compose import TransformedTargetRegressor
    p = dict(params or {})
    if name == "Linear Regression":
        return Pipeline([("s", StandardScaler()), ("m", LinearRegression())])
    if name == "Ridge Regression":
        return Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=p.get("alpha", 1.0), random_state=42))])
    if name == "Lasso Regression":
        return Pipeline([("s", StandardScaler()), ("m", Lasso(alpha=p.get("alpha", 1.0), max_iter=100000, random_state=42))])
    if name == "Elastic-Net":
        return Pipeline([("s", StandardScaler()), ("m", ElasticNet(alpha=p.get("alpha", 0.001), l1_ratio=p.get("l1_ratio", 0.1), max_iter=100000, random_state=42))])
    if name == "Decision Tree":
        return DecisionTreeRegressor(random_state=42, **p)
    if name == "Random Forest":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **p)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=42, **p)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=42, n_jobs=1, objective="reg:squarederror", verbosity=0, **p)
    if name == "KNN":
        return Pipeline([("s", StandardScaler()), ("m", KNeighborsRegressor(**p))])
    if name == "SVR":
        return Pipeline([("s", StandardScaler()), ("m", SVR(kernel="rbf", **p))])
    if name == "SVR (target-scaled)":
        base = Pipeline([("s", StandardScaler()), ("m", SVR(kernel="rbf", **p))])
        return TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
    raise ValueError(f"build_estimator: unsupported model '{name}'")


def plot_importances(importances, feature_names, title, fname, color=GREEN, top=10):
    """Horizontal bar chart of top-N feature importances. Returns the full Series."""
    imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top_s = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_s.index, top_s.values, color=color)
    for y_i, v in enumerate(top_s.values):
        ax.text(v + top_s.values.max() * 0.01, y_i, f"{v:.3f}", va="center",
                ha="left", fontsize=9.5, color=NAVY, fontweight="medium")
    ax.set_title(title, loc="left")
    ax.set_xlabel("importance (variance reduction share)")
    ax.margins(x=0.12)
    save_fig(fig, fname)
    return imp
