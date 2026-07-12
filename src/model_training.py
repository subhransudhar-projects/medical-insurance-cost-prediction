"""
model_training.py
=================
Model factory and tuning pipelines for insurance cost regression.

Scale-sensitive models are wrapped in a StandardScaler pipeline so that, under
cross-validation, scaling is refit inside each fold (leak-free).
"""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              VotingRegressor, StackingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold

RANDOM_STATE = 42


def _pipe(model):
    """Wrap a scale-sensitive estimator in a StandardScaler pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def get_model_grid() -> dict:
    """Return the tuning grid for every candidate model.

    Returns
    -------
    dict
        ``{name: (estimator, param_grid)}`` covering linear, tree, SVR, and KNN
        families. Ensembles are built separately via :func:`build_ensembles`.
    """
    rs = RANDOM_STATE
    return {
        "Linear Regression": (_pipe(LinearRegression()), {}),
        "Ridge Regression": (_pipe(Ridge(random_state=rs)),
                             {"model__alpha": [0.01, 0.1, 1, 10, 100]}),
        "Lasso Regression": (_pipe(Lasso(max_iter=100000, random_state=rs)),
                            {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}),
        "Elastic-Net": (_pipe(ElasticNet(max_iter=100000, random_state=rs)),
                       {"model__alpha": [0.001, 0.01, 0.1, 1, 10],
                        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}),
        "Decision Tree": (DecisionTreeRegressor(random_state=rs),
                         {"max_depth": [3, 5, 7, 10, 15],
                          "min_samples_split": [2, 5, 10],
                          "min_samples_leaf": [1, 2, 4]}),
        "Random Forest": (RandomForestRegressor(random_state=rs, n_jobs=-1),
                         {"n_estimators": [50, 100, 200],
                          "max_depth": [5, 10, 15, None],
                          "min_samples_split": [2, 5, 10]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=rs),
                             {"n_estimators": [50, 100, 200],
                              "learning_rate": [0.01, 0.05, 0.1],
                              "max_depth": [3, 5, 7]}),
        "SVR (target-scaled)": (
            TransformedTargetRegressor(regressor=_pipe(SVR(kernel="rbf")),
                                       transformer=StandardScaler()),
            {"regressor__model__C": [0.1, 1, 10, 100],
             "regressor__model__gamma": [0.01, 0.1, 1],
             "regressor__model__epsilon": [0.01, 0.1, 0.5]}),
        "KNN": (_pipe(KNeighborsRegressor()),
               {"model__n_neighbors": [3, 5, 7, 10, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2]}),
    }


def tune_model(estimator, param_grid, X_train, y_train, cv_splits=5):
    """Grid-search a single model with 5-fold CV and return the best fitted estimator.

    Parameters
    ----------
    estimator : sklearn estimator or Pipeline
        The model (or pipeline) to tune.
    param_grid : dict
        Hyperparameter grid (empty dict -> just fit).
    X_train, y_train : array-like
        Training features and target (``charges``).
    cv_splits : int, default 5
        Number of CV folds.

    Returns
    -------
    tuple
        ``(best_estimator, best_params, best_cv_r2)``.
    """
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    if not param_grid:
        est = estimator.fit(X_train, y_train)
        from sklearn.model_selection import cross_val_score
        return est, {}, float(cross_val_score(estimator, X_train, y_train, cv=cv, scoring="r2").mean())
    grid = GridSearchCV(estimator, param_grid, scoring="r2", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


def build_ensembles(base_estimators: list, meta=None):
    """Build a Voting and a Stacking ensemble from fitted/paramed base estimators.

    Parameters
    ----------
    base_estimators : list of (str, estimator)
        Named base models (e.g. the top 3-4 individual models).
    meta : sklearn estimator, optional
        Final estimator for stacking (default: LinearRegression).

    Returns
    -------
    dict
        ``{"voting": VotingRegressor, "stacking": StackingRegressor}`` (unfitted).
    """
    meta = meta or LinearRegression()
    return {
        "voting": VotingRegressor(estimators=base_estimators, n_jobs=-1),
        "stacking": StackingRegressor(estimators=base_estimators, final_estimator=meta,
                                      cv=5, n_jobs=-1),
    }
