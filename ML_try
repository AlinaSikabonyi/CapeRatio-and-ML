
from __future__ import annotations
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class Config:
    cape_path: Path
    returns_path: Path
    out_model: Path = Path("best_model.joblib")
    random_state: int = 42
    n_splits: int = 5


def load_and_merge(cape_path: Path, returns_path: Path) -> pd.DataFrame:
    """Load CSVs and merge on Date. Expect columns: Date, cape_ratio and Date, tya_real_return."""
    cape = pd.read_csv(cape_path)
    returns = pd.read_csv(returns_path)

    # parse dates
    cape["Date"] = pd.to_datetime(cape["Date"])  # fail loud if missing
    returns["Date"] = pd.to_datetime(returns["Date"])  # fail loud if missing

    # rename to consistent columns
    cape = cape.rename(columns={cape.columns[1]: "cape_ratio"})
    returns = returns.rename(columns={returns.columns[1]: "tya_real_return"})

    df = pd.merge_asof(cape.sort_values("Date"), returns.sort_values("Date"), on="Date")

    # drop rows with missing returns or cape
    df = df.dropna(subset=["cape_ratio", "tya_real_return"]).reset_index(drop=True)
    return df


def create_features(df: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    """Create features from cape_ratio and date info.

    Features created:
    - log_cape
    - lagged cape values (1..max_lag)
    - rolling mean/std (3, 6, 12 months)
    - year and month as categorical (one-hot later by pipeline if desired)
    """
    df = df.copy()
    df["log_cape"] = np.log(df["cape_ratio"].replace(0, np.nan))

    # create lags
    for lag in range(1, max_lag + 1):
        df[f"cape_lag_{lag}"] = df["cape_ratio"].shift(lag)

    # rolling statistics on cape (window in months / rows)
    for w in (3, 6, 12):
        df[f"cape_roll_mean_{w}"] = df["cape_ratio"].rolling(window=w, min_periods=1).mean()
        df[f"cape_roll_std_{w}"] = df["cape_ratio"].rolling(window=w, min_periods=1).std().fillna(0)

    # percent change
    df["cape_pct_change"] = df["cape_ratio"].pct_change().fillna(0)

    # time features
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    # target
    df["target"] = df["tya_real_return"]

    # drop rows with NaNs introduced by lags
    df = df.dropna().reset_index(drop=True)
    return df


def train_test_split_time_series(df: pd.DataFrame, test_size: int = 24):
    """Simple holdout split respecting time order. test_size is number of rows in the test set (e.g., months)."""
    if test_size <= 0 or test_size >= len(df) - 2:
        raise ValueError("test_size must be positive and smaller than the dataset length")
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test


def get_models(random_state: int = 42):
    """Return a dictionary of candidate models. Try to use xgboost/lightgbm if installed."""
    models = {
        "ridge": Ridge(random_state=random_state),
        "rf": RandomForestRegressor(random_state=random_state, n_jobs=-1),
    }
    try:
        import xgboost as xgb  # type: ignore

        models["xgb"] = xgb.XGBRegressor(random_state=random_state, n_jobs=-1)
    except Exception:
        pass
    try:
        import lightgbm as lgb  # type: ignore

        models["lgbm"] = lgb.LGBMRegressor(random_state=random_state, n_jobs=-1)
    except Exception:
        pass
    return models


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def fit_with_time_series_cv(X: pd.DataFrame, y: pd.Series, random_state: int, n_splits: int = 5):
    """Fit multiple models using TimeSeriesSplit + GridSearchCV. Returns best estimator and cv results."""
    models = get_models(random_state=random_state)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {}

    for name, model in models.items():
        print(f"Training model: {name}")

        pipe = Pipeline(
            [("scaler", StandardScaler()), ("model", model)],
        )

        # small, sensible grids to keep runtime modest for demo purposes
        if name == "ridge":
            param_grid = {"model__alpha": [0.1, 1.0, 10.0]}
        elif name == "rf":
            param_grid = {"model__n_estimators": [100, 300], "model__max_depth": [3, 6, None]}
        elif name == "xgb":
            param_grid = {"model__n_estimators": [100, 300], "model__max_depth": [3, 6]}
        elif name == "lgbm":
            param_grid = {"model__n_estimators": [100, 300], "model__max_depth": [3, 6]}
        else:
            param_grid = {}

        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X, y)

        results[name] = {
            "best_estimator": search.best_estimator_,
            "best_score": -search.best_score_,
            "cv_results": search.cv_results_,
        }
    return results


def plot_predictions(df_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_test["Date"], y_true, label="Actual", linewidth=2)
    ax.plot(df_test["Date"], y_pred, label="Predicted", linestyle="--")
    ax.set_title("Actual vs Predicted: 10-year annualized real returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Real Return (%)")
    ax.legend()
    fig.tight_layout()
    out_file = out_dir / "prediction_plot.png"
    fig.savefig(out_file, dpi=150)
    print(f"Saved prediction plot to {out_file}")


def feature_importance(estimator, feature_names: list, out_dir: Path):
    """Plot or print feature importances. Works for tree models; for linear, use coef_."""
    importances = None
    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        model = estimator.named_steps["model"]
    else:
        model = estimator

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)

    else:
        print("No feature importance available for this model type. Using permutation importance could be added.")

    if importances is not None:
        idx = np.argsort(importances)[::-1]
        topk = min(12, len(feature_names))
        fig, ax = plt.subplots(figsize=(8, topk * 0.4 + 1))
        ax.barh([feature_names[i] for i in idx[:topk]][::-1], importances[idx][:topk][::-1])
        ax.set_title("Feature importances (top {})".format(topk))
        fig.tight_layout()
        out_file = out_dir / "feature_importances.png"
        fig.savefig(out_file, dpi=150)
        print(f"Saved feature importances to {out_file}")


def main(cfg: Config):
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    df = load_and_merge(cfg.cape_path, cfg.returns_path)
    df_features = create_features(df, max_lag=12)

    # holdout latest 24 rows for final test if data is monthly use 24 ~ 2 years; change as you see fit
    train_df, test_df = train_test_split_time_series(df_features, test_size=24)

    feature_cols = [c for c in train_df.columns if c not in ("Date", "tya_real_return", "target")]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    results = fit_with_time_series_cv(X_train, y_train, random_state=cfg.random_state, n_splits=cfg.n_splits)

    # select best model by cv score (lowest RMSE)
    best_name, best_info = min(results.items(), key=lambda kv: kv[1]["best_score"])  # smallest RMSE
    best_estimator = best_info["best_estimator"]
    print(f"Best model: {best_name} with CV RMSE: {best_info['best_score']:.4f}")

    # evaluate on test set
    y_pred = best_estimator.predict(X_test)
    metrics = evaluate(y_test.values, y_pred)
    print("Test performance:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # save model
    joblib.dump(best_estimator, cfg.out_model)
    print(f"Saved best model to {cfg.out_model}")

    # plots
    plot_predictions(test_df, y_test.values, y_pred, out_dir)
    feature_importance(best_estimator, feature_cols, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cape", required=True, help="Path to cape_ratio.csv")
    parser.add_argument("--returns", required=True, help="Path to tya_real_return.csv")
    parser.add_argument("--out", default="best_model.joblib", help="Output file for the trained model")
    args = parser.parse_args()

    cfg = Config(cape_path=Path(args.cape), returns_path=Path(args.returns), out_model=Path(args.out))
    main(cfg)
