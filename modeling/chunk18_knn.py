"""Stage 4 · Chunk 18 — Non-linear: K-Nearest Neighbors Regressor."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold
from model_utils import load_splits, evaluate_and_log, load_registry

X_train, X_test, y_train, y_test = load_splits(scaled=False)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor())])
grid = GridSearchCV(
    pipe,
    {"knn__n_neighbors": [3, 5, 7, 10, 15],
     "knn__weights": ["uniform", "distance"],
     "knn__p": [1, 2]},
    scoring="r2", cv=cv, n_jobs=-1,
)
grid.fit(X_train, y_train)

print("=" * 66); print("CHUNK 18 — K-NEAREST NEIGHBORS"); print("=" * 66)
print(f"Best params         : {grid.best_params_}")
print(f"Best CV R2 (5-fold) : {grid.best_score_:.4f}")
rec = evaluate_and_log("KNN", grid.best_estimator_, X_train, y_train, X_test, y_test,
                       best_params={k.replace('knn__',''): v for k, v in grid.best_params_.items()},
                       note="Pipeline(StandardScaler+KNN)", scaled="pipeline")

reg = load_registry()
print("\n--- All models so far ranked by test R2 ---")
print(reg[["model", "test_r2", "test_rmse", "test_mae"]].sort_values("test_r2", ascending=False)
      .round(4).to_string(index=False))
