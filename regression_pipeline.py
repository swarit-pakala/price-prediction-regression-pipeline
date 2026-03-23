from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = "test.csv"                # <-- change this
TARGET_COL = "YOUR_TARGET_COLUMN"     # <-- change this
OUTPUT_DIR = "regression_project_outputs"

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

HIGH_MISSING_THRESHOLD = 0.75
HIGH_ZERO_THRESHOLD = 0.75

# Tuned search size
MAX_SEARCH_ITERS = 20


# =============================================================================
# HELPERS
# =============================================================================
def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned = []
    seen = {}

    for col in df.columns:
        new_col = str(col).strip()
        new_col = new_col.replace("\n", " ").replace("\t", " ")
        new_col = "_".join(new_col.split())

        if new_col in seen:
            seen[new_col] += 1
            new_col = f"{new_col}_{seen[new_col]}"
        else:
            seen[new_col] = 0

        cleaned.append(new_col)

    df.columns = cleaned
    return df


def load_table(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path_obj)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path_obj)
    if suffix == ".parquet":
        return pd.read_parquet(path_obj)

    raise ValueError(f"Unsupported file type: {suffix}")


def make_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    pct_errors = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(pct_errors) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "MAPE_percent": safe_mape(y_true, y_pred),
    }


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def make_one_hot_encoder() -> OneHotEncoder:
    # compatible with multiple sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def total_combinations(grid: dict) -> int:
    total = 1
    for values in grid.values():
        total *= len(values)
    return total


# =============================================================================
# CUSTOM TRANSFORMERS
# =============================================================================
class DropHighMissingColumns(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.columns_to_drop_: list[str] = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        missing_ratio = X.isna().mean()
        self.columns_to_drop_ = missing_ratio[missing_ratio > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.drop(columns=self.columns_to_drop_, errors="ignore")


class DropHighZeroColumns(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.columns_to_drop_: list[str] = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            self.columns_to_drop_ = []
            return self

        zero_ratio = (X[numeric_cols] == 0).mean()
        self.columns_to_drop_ = zero_ratio[zero_ratio > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.drop(columns=self.columns_to_drop_, errors="ignore")


class DropConstantColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop_: list[str] = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        nunique = X.nunique(dropna=False)
        self.columns_to_drop_ = nunique[nunique <= 1].index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.drop(columns=self.columns_to_drop_, errors="ignore")


# =============================================================================
# EDA + REPORTING
# =============================================================================
def save_basic_eda_reports(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    eda_dir = out_dir / "eda"
    eda_dir.mkdir(exist_ok=True)

    info_df = pd.DataFrame({
        "column": X.columns,
        "dtype": X.dtypes.astype(str).values,
        "missing_count": X.isna().sum().values,
        "missing_pct": (X.isna().mean() * 100).values,
        "nunique": X.nunique(dropna=False).values,
    }).sort_values(["missing_pct", "nunique"], ascending=[False, False])

    info_df.to_csv(eda_dir / "feature_overview.csv", index=False)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        desc = X[numeric_cols].describe().T
        desc.to_csv(eda_dir / "numeric_summary.csv")

        corr_df = X[numeric_cols].copy()
        corr_df["__target__"] = y.values
        correlations = corr_df.corr(numeric_only=True)["__target__"].drop("__target__").sort_values(
            key=lambda s: s.abs(), ascending=False
        )
        correlations.to_csv(eda_dir / "numeric_correlations_with_target.csv", header=["correlation"])

        # correlation bar plot (top 20 by absolute value)
        top_corr = correlations.reindex(correlations.abs().sort_values(ascending=False).index).head(20)
        plt.figure(figsize=(10, 6))
        plt.barh(top_corr.index[::-1], top_corr.values[::-1])
        plt.title("Top Numeric Correlations with Target")
        plt.xlabel("Correlation")
        plt.tight_layout()
        plt.savefig(eda_dir / "top_numeric_correlations.png", dpi=200)
        plt.close()

    # target histogram
    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=30)
    plt.title("Target Distribution")
    plt.xlabel("Target")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(eda_dir / "target_distribution.png", dpi=200)
    plt.close()

    # missingness plot
    missing_pct = (X.isna().mean() * 100).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(missing_pct.index[::-1], missing_pct.values[::-1])
    plt.title("Top 20 Columns by Missing Percentage")
    plt.xlabel("Missing %")
    plt.tight_layout()
    plt.savefig(eda_dir / "top_missing_columns.png", dpi=200)
    plt.close()


# =============================================================================
# PIPELINE BUILDERS
# =============================================================================
def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, selector(dtype_include=np.number)),
            ("cat", categorical_pipeline, selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )
    return preprocessor


def build_full_pipeline(model) -> Pipeline:
    feature_cleaner = Pipeline(steps=[
        ("drop_high_missing", DropHighMissingColumns(threshold=HIGH_MISSING_THRESHOLD)),
        ("drop_high_zero", DropHighZeroColumns(threshold=HIGH_ZERO_THRESHOLD)),
        ("drop_constant", DropConstantColumns()),
    ])

    pipeline = Pipeline(steps=[
        ("feature_cleaner", feature_cleaner),
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])
    return pipeline


def build_candidate_models() -> dict:
    return {
        "baseline_dummy": DummyRegressor(strategy="median"),
        "elastic_net": ElasticNet(max_iter=20000),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
    }


def build_param_grids() -> dict:
    return {
        "elastic_net": {
            "model__alpha": [0.0005, 0.001, 0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.05, 0.2, 0.5, 0.8, 0.95, 1.0],
        },
        "random_forest": {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [None, 8, 15, 25, 40],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": [1.0, "sqrt", 0.5],
        },
        "extra_trees": {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [None, 8, 15, 25, 40],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": [1.0, "sqrt", 0.5],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_iter": [200, 400, 700],
            "model__max_leaf_nodes": [15, 31, 63, 127],
            "model__min_samples_leaf": [10, 20, 30, 50],
            "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
        },
    }


# =============================================================================
# MODEL SELECTION
# =============================================================================
def compare_models_cv(X_train: pd.DataFrame, y_train: pd.Series, out_dir: Path) -> pd.DataFrame:
    models = build_candidate_models()
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, model in models.items():
        pipeline = build_full_pipeline(model)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )

        rows.append({
            "model": name,
            "cv_rmse_mean": -scores["test_rmse"].mean(),
            "cv_rmse_std": scores["test_rmse"].std(),
            "cv_mae_mean": -scores["test_mae"].mean(),
            "cv_r2_mean": scores["test_r2"].mean(),
        })

    leaderboard = pd.DataFrame(rows).sort_values("cv_rmse_mean", ascending=True)
    leaderboard.to_csv(out_dir / "model_comparison_cv.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(leaderboard["model"], leaderboard["cv_rmse_mean"])
    plt.title("Cross-Validated RMSE by Model")
    plt.ylabel("RMSE (lower is better)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_cv.png", dpi=200)
    plt.close()

    return leaderboard


def tune_top_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    leaderboard: pd.DataFrame,
    out_dir: Path,
) -> tuple[str, object, pd.DataFrame]:
    models = build_candidate_models()
    grids = build_param_grids()
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    candidate_names = [
        m for m in leaderboard["model"].tolist()
        if m != "baseline_dummy" and m in grids
    ][:3]

    tuning_rows = []
    best_name = None
    best_estimator = None
    best_score = math.inf

    for name in candidate_names:
        pipeline = build_full_pipeline(models[name])
        param_grid = grids[name]
        n_iter = min(MAX_SEARCH_ITERS, total_combinations(param_grid))

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        search.fit(X_train, y_train)

        tuned_rmse = -search.best_score_
        tuning_rows.append({
            "model": name,
            "best_cv_rmse": tuned_rmse,
            "best_params": json.dumps(search.best_params_),
        })

        if tuned_rmse < best_score:
            best_score = tuned_rmse
            best_name = name
            best_estimator = search.best_estimator_

    tuning_df = pd.DataFrame(tuning_rows).sort_values("best_cv_rmse", ascending=True)
    tuning_df.to_csv(out_dir / "tuned_model_results.csv", index=False)

    return best_name, best_estimator, tuning_df


# =============================================================================
# DIAGNOSTICS
# =============================================================================
def save_predictions_and_metrics(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    out_dir: Path,
) -> dict:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = regression_metrics(y_train, train_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    metrics = {
        "train": train_metrics,
        "test": test_metrics,
    }
    save_json(metrics, out_dir / "metrics.json")

    preds_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": test_pred,
        "residual": y_test.values - test_pred,
        "abs_error": np.abs(y_test.values - test_pred),
    })
    preds_df.to_csv(out_dir / "test_predictions.csv", index=False)

    return metrics


def plot_regression_diagnostics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
) -> None:
    pred = model.predict(X_test)
    residuals = y_test.values - pred

    # Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, pred, alpha=0.7)
    min_val = min(np.min(y_test), np.min(pred))
    max_val = max(np.max(y_test), np.max(pred))
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_predicted.png", dpi=200)
    plt.close()

    # Residuals vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(pred, residuals, alpha=0.7)
    plt.axhline(0, linewidth=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_vs_predicted.png", dpi=200)
    plt.close()

    # Residual histogram
    plt.figure(figsize=(7, 6))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_distribution.png", dpi=200)
    plt.close()


def save_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
) -> None:
    result = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        scoring="r2",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    importance_df.to_csv(out_dir / "permutation_feature_importance.csv", index=False)

    top = importance_df.head(20).sort_values("importance_mean", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Decrease in R² when Permuted")
    plt.title("Top 20 Permutation Feature Importances")
    plt.tight_layout()
    plt.savefig(out_dir / "top_feature_importance.png", dpi=200)
    plt.close()


def save_feature_cleaning_summary(best_model, out_dir: Path) -> None:
    cleaner = best_model.named_steps["feature_cleaner"]

    summary = {
        "dropped_high_missing_columns": cleaner.named_steps["drop_high_missing"].columns_to_drop_,
        "dropped_high_zero_columns": cleaner.named_steps["drop_high_zero"].columns_to_drop_,
        "dropped_constant_columns": cleaner.named_steps["drop_constant"].columns_to_drop_,
    }
    save_json(summary, out_dir / "feature_cleaning_summary.json")


# =============================================================================
# MAIN
# =============================================================================
def main():
    out_dir = make_output_dir(OUTPUT_DIR)

    # -------------------------------------------------------------------------
    # 1) Load data
    # -------------------------------------------------------------------------
    df = load_table(DATA_PATH)
    df = sanitize_column_names(df)
    df = df.drop_duplicates().reset_index(drop=True)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"TARGET_COL='{TARGET_COL}' not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    # -------------------------------------------------------------------------
    # 2) Separate features/target
    # -------------------------------------------------------------------------
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    X = df.drop(columns=[TARGET_COL]).copy()

    valid_rows = y.notna()
    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    if len(X) < 50:
        print("Warning: dataset is small. Results may be unstable.")

    # -------------------------------------------------------------------------
    # 3) Train/test split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # -------------------------------------------------------------------------
    # 4) Save EDA on training data only
    # -------------------------------------------------------------------------
    save_basic_eda_reports(X_train, y_train, out_dir)

    # -------------------------------------------------------------------------
    # 5) Baseline model comparison
    # -------------------------------------------------------------------------
    leaderboard = compare_models_cv(X_train, y_train, out_dir)
    print("\nCross-validated model comparison:")
    print(leaderboard)

    # -------------------------------------------------------------------------
    # 6) Tune top models
    # -------------------------------------------------------------------------
    best_name, best_model, tuned_df = tune_top_models(X_train, y_train, leaderboard, out_dir)

    print("\nTuned model results:")
    print(tuned_df)

    if best_model is None:
        raise RuntimeError("No tuned model was selected.")

    print(f"\nBest tuned model: {best_name}")

    # -------------------------------------------------------------------------
    # 7) Final evaluation
    # -------------------------------------------------------------------------
    metrics = save_predictions_and_metrics(best_model, X_train, X_test, y_train, y_test, out_dir)
    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))

    # -------------------------------------------------------------------------
    # 8) Diagnostics + feature importance
    # -------------------------------------------------------------------------
    plot_regression_diagnostics(best_model, X_test, y_test, out_dir)
    save_permutation_importance(best_model, X_test, y_test, out_dir)
    save_feature_cleaning_summary(best_model, out_dir)

    # -------------------------------------------------------------------------
    # 9) Save fitted model
    # -------------------------------------------------------------------------
    joblib.dump(best_model, out_dir / "best_regression_pipeline.joblib")

    # -------------------------------------------------------------------------
    # 10) Save project summary
    # -------------------------------------------------------------------------
    summary = {
        "data_path": DATA_PATH,
        "target_column": TARGET_COL,
        "n_rows_total_after_target_cleaning": int(len(X)),
        "n_features_raw": int(X.shape[1]),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "best_model_name": best_name,
        "test_rmse": metrics["test"]["RMSE"],
        "test_mae": metrics["test"]["MAE"],
        "test_r2": metrics["test"]["R2"],
    }
    save_json(summary, out_dir / "project_summary.json")

    print(f"\nDone. All outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
