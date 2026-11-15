"""Reusable metrics, preprocessing pipelines, hurdle models, and calibration helpers for 90-day CLV."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, TweedieRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    median_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_SKEWED_FEATURES: list[str] = [
    "frequency",
    "monetary",
    "aov",
    "avg_items_per_invoice",
    "avg_unit_price",
    "n_unique_products",
    "spend_last_30d",
    "spend_last_60d",
    "spend_last_90d",
    "invoices_last_30d",
    "invoices_last_60d",
    "invoices_last_90d",
    "return_invoice_count",
    "return_value_abs",
    "return_value_ratio",
]

NUMERIC_LINEAR_RAW_FEATURES: list[str] = [
    "recency_days",
    "n_active_months",
    "n_active_weeks",
    "tenure_days_in_window",
    "is_tenure_at_edge",
    "mean_interpurchase_days",
    "is_single_invoice",
    "spend_last_90d_ratio",
    "return_rate",
    "is_uk",
]

ALL_NUMERIC_FEATURES: list[str] = NUMERIC_SKEWED_FEATURES + NUMERIC_LINEAR_RAW_FEATURES
CATEGORICAL_FEATURES: list[str] = ["country"]


def make_stratification_bins(
    y: pd.Series | np.ndarray,
    n_positive_bins: int = 9,
) -> np.ndarray:
    """Create discrete strata keeping exact zeros separate and binning positives evenly.

    With ~42% zero targets, zero rows form bin 0 while positive values are binned into
    ``n_positive_bins`` quantile strata so extreme wholesale whales distribute evenly across folds.
    """
    y_arr = np.asarray(y, dtype=float)
    bins = np.zeros(len(y_arr), dtype=int)
    pos_mask = y_arr > 0
    if pos_mask.any():
        pos_series = pd.Series(y_arr[pos_mask])
        pos_bins = pd.qcut(
            pos_series.rank(method="first"),
            q=min(n_positive_bins, int(pos_mask.sum())),
            labels=False,
        )
        bins[pos_mask] = pos_bins.to_numpy(dtype=int) + 1
    return bins


def normalized_gini(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """Compute Lorenz-based normalized Gini coefficient with order-invariant tie averaging.

    Uses average rank weighting: ``Gini(y, s) = sum(y_i * (2 * rank_avg(s_i) - n - 1)) / (n * sum(y))``
    normalized by ``Gini(y, y)``. Guarantees 1.0 for perfect ordering, -1.0 for inverse,
    and exact 0.0 for constant predictions regardless of DataFrame row order.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have equal length.")
    n = len(yt)
    if n < 2 or float(yt.sum()) <= 0.0:
        return 0.0

    def _raw_gini(values: np.ndarray, scores: np.ndarray) -> float:
        ranks = rankdata(scores, method="average")
        total = float(values.sum())
        return float(np.sum(values * (2.0 * ranks - n - 1.0)) / (n * total))

    perfect_gini = _raw_gini(yt, yt)
    if np.isclose(perfect_gini, 0.0):
        return 0.0
    return float(_raw_gini(yt, yp) / perfect_gini)


def capture_and_lift_at_k(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    top_frac: float = 0.10,
) -> tuple[float, float]:
    """Compute share of actual total spend captured by top ``top_frac`` predicted customers, plus lift.

    Handles ties cleanly via group-level pro-rata allocation if multiple rows share the exact
    boundary predicted value, so constant predictions yield exact capture == ``top_frac`` and
    exact lift == ``1.0`` independent of row order.
    """
    if not (0.0 < top_frac <= 1.0):
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have equal length.")

    total_actual = float(yt.sum())
    n = len(yt)
    if n == 0 or total_actual <= 0.0:
        return 0.0, 0.0

    quota = float(top_frac * n)
    df_tmp = pd.DataFrame({"y": yt, "pred": yp})
    grouped = (
        df_tmp.groupby("pred", sort=True)
        .agg(cnt=("y", "count"), y_sum=("y", "sum"))
        .sort_index(ascending=False)
    )

    captured = 0.0
    remaining = quota
    for cnt, y_sum in zip(grouped["cnt"], grouped["y_sum"], strict=False):
        if remaining <= 0.0:
            break
        take_frac = min(1.0, remaining / float(cnt))
        captured += float(y_sum) * take_frac
        remaining -= float(cnt)

    capture = float(captured / total_actual)
    lift = float(capture / top_frac)
    return capture, lift


def rmse_excluding_top_k(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    top_k: int = 1,
) -> float:
    """Compute RMSE after dropping the ``top_k`` largest true target observations."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if top_k <= 0 or top_k >= len(yt):
        raise ValueError(f"top_k must satisfy 0 < top_k < len(y_true) ({len(yt)}), got {top_k}")
    keep_idx = np.argsort(yt)[:-top_k]
    return float(root_mean_squared_error(yt[keep_idx], yp[keep_idx]))


def evaluate_clv_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute full Phase 4b suite of ranking, business capture, calibration, and error metrics."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have equal length.")

    # Constant predictions yield NaN Spearman in scipy; map undefined monotonic correlation to 0.0
    if np.std(yp) < 1e-12 or np.std(yt) < 1e-12:
        sp_rho = 0.0
    else:
        res = spearmanr(yt, yp)
        sp_rho = float(res.statistic) if not np.isnan(res.statistic) else 0.0

    cap10, lift10 = capture_and_lift_at_k(yt, yp, top_frac=0.10)
    cap20, lift20 = capture_and_lift_at_k(yt, yp, top_frac=0.20)

    sum_true = float(yt.sum())
    sum_pred = float(yp.sum())
    total_rev_err_pct = ((sum_pred - sum_true) / sum_true * 100.0) if sum_true > 0.0 else 0.0

    return {
        "spearman": sp_rho,
        "normalized_gini": normalized_gini(yt, yp),
        "top_10_capture": cap10,
        "lift_at_10": lift10,
        "top_20_capture": cap20,
        "lift_at_20": lift20,
        "total_rev_error_pct": total_rev_err_pct,
        "mae": float(mean_absolute_error(yt, yp)),
        "medae": float(median_absolute_error(yt, yp)),
        "rmse": float(root_mean_squared_error(yt, yp)),
        "rmse_excl_top1": rmse_excluding_top_k(yt, yp, top_k=1) if len(yt) > 1 else np.nan,
        "rmse_excl_top5": rmse_excluding_top_k(yt, yp, top_k=5) if len(yt) > 5 else np.nan,
    }


def calibration_by_predicted_decile(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute mean predicted vs. mean actual spend across predicted quantile bins (1=lowest, N=top)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have equal length.")

    s_pred = pd.Series(yp)
    if s_pred.nunique() == 1:
        decile = pd.Series(np.ones(len(yt), dtype=int), name="pred_decile")
    else:
        pct_rank = s_pred.rank(method="average", pct=True)
        # Discretize percentile rank safely even when tied low/zero predictions share a bin
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        decile = pd.cut(
            pct_rank, bins=edges, labels=range(1, n_bins + 1), include_lowest=True
        ).astype(int)

    df_cal = pd.DataFrame({"pred_decile": decile, "y_true": yt, "y_pred": yp})
    tot_true = float(yt.sum())
    tot_pred = float(yp.sum())

    out = (
        df_cal.groupby("pred_decile", observed=True)
        .agg(
            n_customers=("y_true", "count"),
            mean_pred=("y_pred", "mean"),
            mean_actual=("y_true", "mean"),
            median_actual=("y_true", "median"),
            sum_pred=("y_pred", "sum"),
            sum_actual=("y_true", "sum"),
        )
        .assign(
            actual_share_pct=lambda d: (
                (d["sum_actual"] / tot_true * 100.0) if tot_true > 0 else 0.0
            ),
            pred_share_pct=lambda d: (d["sum_pred"] / tot_pred * 100.0) if tot_pred > 0 else 0.0,
            pred_to_actual_ratio=lambda d: np.where(
                d["mean_actual"] > 0, d["mean_pred"] / d["mean_actual"], np.nan
            ),
        )
        .sort_index()
    )
    return out


class RuleBaselineRegressor(BaseEstimator, RegressorMixin):
    """Heuristic CLV baselines compatible with sklearn CV cloning and evaluation."""

    VALID_RULES = ("train_mean", "train_median", "run_rate", "freq_aov_run_rate")

    def __init__(
        self,
        rule: str = "run_rate",
        lookback_days: float = 270.0,
        horizon_days: float = 90.0,
    ) -> None:
        self.rule = rule
        self.lookback_days = lookback_days
        self.horizon_days = horizon_days

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "RuleBaselineRegressor":
        if self.rule not in self.VALID_RULES:
            raise ValueError(
                f"Unsupported baseline rule {self.rule!r}; expected one of {self.VALID_RULES}."
            )
        y_arr = np.asarray(y, dtype=float)
        self.train_mean_ = float(np.mean(y_arr))
        self.train_median_ = float(np.median(y_arr))
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("RuleBaselineRegressor must be fitted before predict().")
        scale = float(self.horizon_days) / float(self.lookback_days)
        n = len(X)
        if self.rule == "train_mean":
            preds = np.full(n, self.train_mean_, dtype=float)
        elif self.rule == "train_median":
            preds = np.full(n, self.train_median_, dtype=float)
        elif self.rule == "run_rate":
            if "monetary" not in X.columns:
                raise ValueError("Missing 'monetary' column for run_rate baseline.")
            preds = X["monetary"].to_numpy(dtype=float) * scale
        elif self.rule == "freq_aov_run_rate":
            for col in ("aov", "frequency"):
                if col not in X.columns:
                    raise ValueError(f"Missing {col!r} column for freq_aov_run_rate baseline.")
            preds = X["aov"].to_numpy(dtype=float) * X["frequency"].to_numpy(dtype=float) * scale
        else:
            raise ValueError(f"Unsupported rule: {self.rule!r}")
        return np.clip(preds, 0.0, None)


def cross_validate_clv_estimator(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    strata: np.ndarray,
    fit_params_fn: Any | None = None,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    """Evaluate a CLV estimator across CV folds without cross-fold leakage.

    Clones the estimator independently per fold, collects validation metrics across all splits,
    and returns ``(summary_mean_std_dict, per_fold_df, mean_oof_predictions)``.
    """
    n_samples = len(X)
    oof_sum = np.zeros(n_samples, dtype=float)
    oof_cnt = np.zeros(n_samples, dtype=int)
    fold_rows = []

    y_arr = y.to_numpy(dtype=float)
    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X, strata)):
        est = clone(estimator)
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        if fit_params_fn is not None:
            extra_kwargs = fit_params_fn(tr_idx)
            est.fit(X_tr, y_tr, **extra_kwargs)
        else:
            est.fit(X_tr, y_tr)

        preds = np.clip(np.asarray(est.predict(X_val), dtype=float), 0.0, None)
        oof_sum[val_idx] += preds
        oof_cnt[val_idx] += 1

        fold_metrics = evaluate_clv_predictions(y_val, preds)
        fold_metrics["fold"] = fold_idx
        fold_rows.append(fold_metrics)

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in fold_df.columns if c != "fold"]
    summary: dict[str, float] = {}
    for col in metric_cols:
        summary[f"{col}_mean"] = float(fold_df[col].mean())
        summary[f"{col}_std"] = float(fold_df[col].std(ddof=1))

    mean_oof = np.where(oof_cnt > 0, oof_sum / np.maximum(oof_cnt, 1), np.nan)
    return summary, fold_df, mean_oof


class CountryGrouper(BaseEstimator, TransformerMixin):
    """Group rare categorical countries outside UK + Top-N training countries into 'Other'."""

    def __init__(
        self, top_n: int = 5, uk_label: str = "United Kingdom", other_label: str = "Other"
    ) -> None:
        self.top_n = top_n
        self.uk_label = uk_label
        self.other_label = other_label

    def _to_1d_strings(self, X: Any) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            ser = X.iloc[:, 0]
        else:
            arr = np.asarray(X)
            ser = pd.Series(arr.ravel() if arr.ndim > 1 else arr)
        return ser.fillna(self.other_label).astype(str)

    def fit(self, X: Any, y: Any = None) -> "CountryGrouper":
        ser = self._to_1d_strings(X)
        non_uk = ser[ser != self.uk_label]
        top_non_uk = list(non_uk.value_counts().head(self.top_n).index)
        allowed = []
        if (ser == self.uk_label).any():
            allowed.append(self.uk_label)
        for c in top_non_uk:
            if c not in allowed:
                allowed.append(c)
        self.kept_countries_ = allowed
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "kept_countries_"):
            raise RuntimeError("CountryGrouper must be fitted before transform().")
        ser = self._to_1d_strings(X)
        mapped = ser.where(ser.isin(self.kept_countries_), other=self.other_label)
        return mapped.to_numpy(dtype=object).reshape(-1, 1)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        if input_features is not None and len(input_features) > 0:
            return np.asarray([str(input_features[0])], dtype=object)
        return np.asarray(["country"], dtype=object)


def _log1p_nonnegative(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    return np.log1p(np.clip(arr, 0.0, None))


def build_linear_clv_preprocessor(top_n_countries: int = 5) -> ColumnTransformer:
    """Fold-safe preprocessor for linear/GLM CLV models (impute, log1p skewed, scale, group country)."""
    skewed_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(_log1p_nonnegative, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    raw_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("country_grouper", CountryGrouper(top_n=top_n_countries)),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("skewed_num", skewed_pipe, NUMERIC_SKEWED_FEATURES),
            ("raw_num", raw_pipe, NUMERIC_LINEAR_RAW_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_tree_clv_preprocessor(
    top_n_countries: int = 5,
    impute_missing: bool = False,
) -> ColumnTransformer:
    """Preprocessor for tree CLV models; keeps structural NaNs intact unless ``impute_missing=True``."""
    num_transformer: Any = SimpleImputer(strategy="median") if impute_missing else "passthrough"
    cat_pipe = Pipeline(
        steps=[
            ("country_grouper", CountryGrouper(top_n=top_n_countries)),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, ALL_NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


class Log1pSmearingRegressor(BaseEstimator, RegressorMixin):
    """Fit a regressor on ``log1p(y)`` and apply Duan smearing upon retransformation."""

    def __init__(
        self,
        regressor: Any = None,
        use_smearing: bool = True,
        max_log_pred: float = 20.0,
    ) -> None:
        self.regressor = regressor
        self.use_smearing = use_smearing
        self.max_log_pred = max_log_pred

    def fit(self, X: Any, y: pd.Series | np.ndarray) -> "Log1pSmearingRegressor":
        y_arr = np.asarray(y, dtype=float)
        if np.any(y_arr < 0):
            raise ValueError("Target values must be >= 0 for log1p regression.")
        y_log = np.log1p(y_arr)
        base = Ridge(alpha=1.0) if self.regressor is None else self.regressor
        self.regressor_ = clone(base)
        self.regressor_.fit(X, y_log)

        if self.use_smearing:
            pred_log_train = np.clip(self.regressor_.predict(X), -20.0, self.max_log_pred)
            resid = np.clip(y_log - pred_log_train, -30.0, 30.0)
            self.smearing_factor_ = float(np.mean(np.exp(resid)))
        else:
            self.smearing_factor_ = 1.0
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "regressor_"):
            raise RuntimeError("Log1pSmearingRegressor must be fitted before predict().")
        pred_log = np.clip(self.regressor_.predict(X), -20.0, self.max_log_pred)
        preds = self.smearing_factor_ * np.exp(pred_log) - 1.0
        return np.clip(preds, 0.0, None)


class HurdleCLVRegressor(BaseEstimator, RegressorMixin):
    """Two-stage Hurdle CLV estimator: P(target > 0) * E[target | target > 0]."""

    def __init__(self, classifier: Any = None, regressor: Any = None) -> None:
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X: Any, y: pd.Series | np.ndarray) -> "HurdleCLVRegressor":
        y_arr = np.asarray(y, dtype=float)
        y_active = (y_arr > 0).astype(int)

        clf_base = LogisticRegression(max_iter=1000) if self.classifier is None else self.classifier
        reg_base = (
            Log1pSmearingRegressor(Ridge(alpha=1.0), use_smearing=True)
            if self.regressor is None
            else self.regressor
        )

        self.classifier_ = clone(clf_base)
        self.classifier_.fit(X, y_active)

        pos_mask = y_arr > 0
        self.has_positives_ = bool(pos_mask.any())
        self.regressor_ = clone(reg_base)
        if self.has_positives_:
            X_pos = X.iloc[pos_mask] if hasattr(X, "iloc") else X[pos_mask]
            self.regressor_.fit(X_pos, y_arr[pos_mask])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "classifier_"):
            raise RuntimeError("HurdleCLVRegressor must be fitted before predict_proba().")
        return np.asarray(self.classifier_.predict_proba(X)[:, 1], dtype=float)

    def predict_conditional(self, X: Any) -> np.ndarray:
        if not hasattr(self, "regressor_"):
            raise RuntimeError("HurdleCLVRegressor must be fitted before predict_conditional().")
        if not self.has_positives_:
            return np.zeros(len(X), dtype=float)
        return np.clip(np.asarray(self.regressor_.predict(X), dtype=float), 0.0, None)

    def predict(self, X: Any) -> np.ndarray:
        return np.clip(self.predict_proba(X) * self.predict_conditional(X), 0.0, None)


def evaluate_classifier_metrics(
    y_active_true: pd.Series | np.ndarray,
    prob_pred: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute Stage-1 binary probability quality metrics (PR-AUC, ROC-AUC, Brier, ECE)."""
    yt = np.asarray(y_active_true, dtype=int)
    pp = np.clip(np.asarray(prob_pred, dtype=float), 1e-7, 1.0 - 1e-7)
    if len(yt) != len(pp):
        raise ValueError("y_active_true and prob_pred must have equal length.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(pp, edges[1:-1], right=True)
    ece = 0.0
    n = len(yt)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            ece += (float(mask.sum()) / n) * abs(float(yt[mask].mean()) - float(pp[mask].mean()))

    return {
        "pr_auc": float(average_precision_score(yt, pp)),
        "roc_auc": float(roc_auc_score(yt, pp)),
        "brier_score": float(brier_score_loss(yt, pp)),
        "ece_10bin": float(ece),
        "mean_pred_prob": float(pp.mean()),
        "actual_active_rate": float(yt.mean()),
    }


def build_candidate_models(random_state: int = 42, top_n_countries: int = 5) -> dict[str, Any]:
    """Build all pure-sklearn candidate CLV model pipelines for Step 4 & Step 5 comparison."""
    lin_pre = build_linear_clv_preprocessor(top_n_countries=top_n_countries)
    tree_pre_hgb = build_tree_clv_preprocessor(
        top_n_countries=top_n_countries, impute_missing=False
    )
    tree_pre_rf = build_tree_clv_preprocessor(top_n_countries=top_n_countries, impute_missing=True)

    # Family A: Regularized Linear Models on log1p(target) + Duan Smearing
    ridge_pipe = Pipeline(
        [
            ("prep", clone(lin_pre)),
            ("model", Log1pSmearingRegressor(Ridge(alpha=10.0), use_smearing=True)),
        ]
    )
    elastic_pipe = Pipeline(
        [
            ("prep", clone(lin_pre)),
            (
                "model",
                Log1pSmearingRegressor(
                    ElasticNet(alpha=0.05, l1_ratio=0.3, random_state=random_state, max_iter=5000),
                    use_smearing=True,
                ),
            ),
        ]
    )

    # Family B: Tweedie GLM (power=1.5 compound Poisson-Gamma) on raw target
    tweedie_pipe = Pipeline(
        [
            ("prep", clone(lin_pre)),
            ("model", TweedieRegressor(power=1.5, alpha=1.0, link="log", max_iter=1000)),
        ]
    )

    # Family C: HistGradientBoostingRegressor (log1p+smearing vs raw-target poisson)
    hgb_log_pipe = Pipeline(
        [
            ("prep", clone(tree_pre_hgb)),
            (
                "model",
                Log1pSmearingRegressor(
                    HistGradientBoostingRegressor(
                        loss="squared_error",
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=30,
                        random_state=random_state,
                    ),
                    use_smearing=True,
                ),
            ),
        ]
    )
    hgb_poisson_pipe = Pipeline(
        [
            ("prep", clone(tree_pre_hgb)),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="poisson",
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Family D: Two-Stage Hurdle Models (P(active) * E[spend | active])
    hurdle_hgb = HurdleCLVRegressor(
        classifier=Pipeline(
            [
                ("prep", clone(tree_pre_hgb)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=30,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        regressor=Pipeline(
            [
                ("prep", clone(tree_pre_hgb)),
                (
                    "reg",
                    Log1pSmearingRegressor(
                        HistGradientBoostingRegressor(
                            loss="squared_error",
                            max_iter=200,
                            learning_rate=0.05,
                            max_leaf_nodes=31,
                            min_samples_leaf=25,
                            random_state=random_state,
                        ),
                        use_smearing=True,
                    ),
                ),
            ]
        ),
    )
    hurdle_logit = HurdleCLVRegressor(
        classifier=Pipeline(
            [
                ("prep", clone(lin_pre)),
                ("clf", LogisticRegression(C=0.5, max_iter=1000)),
            ]
        ),
        regressor=Pipeline(
            [
                ("prep", clone(lin_pre)),
                ("reg", Log1pSmearingRegressor(Ridge(alpha=10.0), use_smearing=True)),
            ]
        ),
    )

    # Family E: ExtraTreesRegressor robustness check (bagged randomized trees on imputed raw target)
    extratrees_pipe = Pipeline(
        [
            ("prep", clone(tree_pre_rf)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=10,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "A1: Ridge (log1p + Duan)": ridge_pipe,
        "A2: ElasticNet (log1p + Duan)": elastic_pipe,
        "B: Tweedie GLM (power=1.5, log link)": tweedie_pipe,
        "C1: HistGBR (log1p + Duan)": hgb_log_pipe,
        "C2: HistGBR (Poisson raw target)": hgb_poisson_pipe,
        "D1: Hurdle (HistGB Clf x HistGB Reg)": hurdle_hgb,
        "D2: Hurdle (Logit Clf x Ridge Reg)": hurdle_logit,
        "E: ExtraTrees (raw target check)": extratrees_pipe,
    }


def ranking_capture_score(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """Composite CLV tuning objective: 0.5 * Normalized Gini + 0.5 * Top-10% Capture."""
    cap10, _ = capture_and_lift_at_k(y_true, y_pred, top_frac=0.10)
    return 0.5 * normalized_gini(y_true, y_pred) + 0.5 * cap10


CLV_FEATURE_GROUPS: dict[str, list[str]] = {
    "Frequency & Cadence": [
        "frequency",
        "n_active_months",
        "n_active_weeks",
        "mean_interpurchase_days",
        "is_single_invoice",
    ],
    "Historical Monetary & Assortment": [
        "monetary",
        "n_unique_products",
    ],
    "Basket Size & Pricing": [
        "aov",
        "avg_items_per_invoice",
        "avg_unit_price",
    ],
    "Recency & Recent Momentum": [
        "recency_days",
        "spend_last_30d",
        "spend_last_60d",
        "spend_last_90d",
        "invoices_last_30d",
        "invoices_last_60d",
        "invoices_last_90d",
        "spend_last_90d_ratio",
    ],
    "Returns & Refunds": [
        "return_invoice_count",
        "return_value_abs",
        "return_value_ratio",
        "return_rate",
    ],
    "Tenure & Geography": [
        "tenure_days_in_window",
        "is_tenure_at_edge",
        "is_uk",
        "country",
    ],
}


def cv_permutation_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    cv: Any,
    strata: np.ndarray,
    feature_groups: Mapping[str, Sequence[str]] | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute validation-fold permutation importance for single features or collinear groups.

    When ``feature_groups`` maps group names to multiple columns, columns within a group
    are permuted jointly using the same row permutation on each validation fold so within-group
    joint distribution is preserved while breaking joint predictive signal with ``y``.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    if len(X) != len(y):
        raise ValueError("X and y must have equal length.")

    groups: dict[str, list[str]] = (
        {str(col): [str(col)] for col in X.columns}
        if feature_groups is None
        else {str(name): list(cols) for name, cols in feature_groups.items()}
    )
    missing_cols = sorted({c for cols in groups.values() for c in cols} - set(X.columns))
    if missing_cols:
        raise ValueError(f"Columns missing from X: {missing_cols}")

    y_arr = np.asarray(y, dtype=float)
    rng = np.random.default_rng(random_state)
    fold_records: list[dict[str, Any]] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, strata)):
        est_fold = clone(estimator)
        est_fold.fit(X.iloc[tr_idx], y_arr[tr_idx])

        X_val = X.iloc[va_idx]
        y_val = y_arr[va_idx]
        base_metrics = evaluate_clv_predictions(y_val, est_fold.predict(X_val))

        X_perm = X_val.copy()
        n_val = len(X_val)

        for group_name, cols in groups.items():
            gini_drops = []
            cap10_drops = []
            spear_drops = []
            mae_increases = []

            for _ in range(n_repeats):
                perm = rng.permutation(n_val)
                for col in cols:
                    X_perm[col] = X_val[col].iloc[perm].to_numpy()
                perm_metrics = evaluate_clv_predictions(y_val, est_fold.predict(X_perm))
                gini_drops.append(base_metrics["normalized_gini"] - perm_metrics["normalized_gini"])
                cap10_drops.append(base_metrics["top_10_capture"] - perm_metrics["top_10_capture"])
                spear_drops.append(base_metrics["spearman"] - perm_metrics["spearman"])
                mae_increases.append(perm_metrics["mae"] - base_metrics["mae"])

            # Restore unshuffled columns cleanly before evaluating the next group
            for col in cols:
                X_perm[col] = X_val[col].to_numpy()

            fold_records.append(
                {
                    "fold": fold_idx,
                    "feature": group_name,
                    "n_features": len(cols),
                    "gini_drop": float(np.mean(gini_drops)),
                    "cap10_drop": float(np.mean(cap10_drops)),
                    "spearman_drop": float(np.mean(spear_drops)),
                    "mae_increase": float(np.mean(mae_increases)),
                }
            )

    fold_df = pd.DataFrame(fold_records)
    summary_df = (
        fold_df.groupby(["feature", "n_features"], sort=False)
        .agg(
            gini_drop_mean=("gini_drop", "mean"),
            gini_drop_std=("gini_drop", "std"),
            cap10_drop_mean=("cap10_drop", "mean"),
            cap10_drop_std=("cap10_drop", "std"),
            spearman_drop_mean=("spearman_drop", "mean"),
            spearman_drop_std=("spearman_drop", "std"),
            mae_increase_mean=("mae_increase", "mean"),
            mae_increase_std=("mae_increase", "std"),
        )
        .reset_index()
        .fillna(0.0)
        .sort_values("gini_drop_mean", ascending=False)
        .reset_index(drop=True)
    )
    return summary_df


def compute_quantile_pdp(
    fitted_estimator: Any,
    X: pd.DataFrame,
    feature: str,
    quantiles: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99),
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate 1D partial dependence across observed feature quantiles for skewed features."""
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found in X.")
    non_null = pd.to_numeric(X[feature], errors="coerce").dropna().to_numpy(dtype=float)
    if len(non_null) == 0:
        raise ValueError(f"Feature '{feature}' has no non-null numeric values.")

    if sample_size is not None and sample_size < len(X):
        X_base = X.sample(n=sample_size, random_state=random_state).copy()
    else:
        X_base = X.copy()

    raw_grid = np.quantile(non_null, quantiles)
    # Round minimally only to dedupe integer-stepped count quantiles without losing scale
    unique_vals, first_idx = np.unique(raw_grid, return_index=True)
    order = np.argsort(first_idx)
    grid_vals = unique_vals[order]
    q_arr = np.asarray(quantiles, dtype=float)[first_idx[order]]

    rows = []
    X_tmp = X_base.copy()
    orig_dtype = X_base[feature].dtype
    for q_val, grid_v in zip(q_arr, grid_vals, strict=False):
        # Preserve integer compatibility cleanly if original feature was int-typed
        assign_v = (
            int(round(grid_v)) if pd.api.types.is_integer_dtype(orig_dtype) else float(grid_v)
        )
        X_tmp[feature] = assign_v
        preds = np.asarray(fitted_estimator.predict(X_tmp), dtype=float)
        rows.append(
            {
                "feature": feature,
                "quantile": float(q_val),
                "grid_value": float(assign_v),
                "mean_pred": float(np.mean(preds)),
                "median_pred": float(np.median(preds)),
                "p10_pred": float(np.quantile(preds, 0.10)),
                "p90_pred": float(np.quantile(preds, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_clv_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    ref_pred: pd.Series | np.ndarray | None = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute customer-level bootstrap CIs for all CLV metrics and optional paired delta vs reference."""
    if n_bootstrap < 10:
        raise ValueError(f"n_bootstrap must be >= 10, got {n_bootstrap}")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have equal length.")
    yref = None
    if ref_pred is not None:
        yref = np.asarray(ref_pred, dtype=float)
        if len(yref) != len(yt):
            raise ValueError("y_true and ref_pred must have equal length.")

    point = evaluate_clv_predictions(yt, yp)
    point_ref = evaluate_clv_predictions(yt, yref) if yref is not None else None
    metric_names = list(point.keys())

    rng = np.random.default_rng(random_state)
    n = len(yt)
    samples: dict[str, list[float]] = {m: [] for m in metric_names}
    delta_samples: dict[str, list[float]] = {m: [] for m in metric_names}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt_b = yt[idx]
        yp_b = yp[idx]
        m_b = evaluate_clv_predictions(yt_b, yp_b)
        for m in metric_names:
            samples[m].append(m_b[m])
        if yref is not None:
            m_ref_b = evaluate_clv_predictions(yt_b, yref[idx])
            for m in metric_names:
                delta_samples[m].append(m_b[m] - m_ref_b[m])

    alpha_low = (1.0 - ci) / 2.0
    alpha_high = 1.0 - alpha_low
    rows = []
    for m in metric_names:
        arr = np.asarray(samples[m], dtype=float)
        row: dict[str, Any] = {
            "metric": m,
            "point_estimate": float(point[m]),
            "boot_mean": float(np.nanmean(arr)),
            "boot_std": float(np.nanstd(arr, ddof=1)),
            "ci_low": float(np.nanquantile(arr, alpha_low)),
            "ci_high": float(np.nanquantile(arr, alpha_high)),
        }
        if point_ref is not None:
            d_arr = np.asarray(delta_samples[m], dtype=float)
            row["delta_vs_ref"] = float(point[m] - point_ref[m])
            row["delta_ci_low"] = float(np.nanquantile(d_arr, alpha_low))
            row["delta_ci_high"] = float(np.nanquantile(d_arr, alpha_high))
        rows.append(row)

    return pd.DataFrame(rows).set_index("metric")


def evaluate_whale_sensitivity_slices(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    exclude_top_ks: Sequence[int] = (0, 1, 5),
) -> pd.DataFrame:
    """Evaluate all CLV metrics on the full sample and after excluding the top-k true whales."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have equal length.")

    n = len(yt)
    order = np.argsort(yt)
    rows = []
    for k in exclude_top_ks:
        if k < 0 or k >= n:
            raise ValueError(f"exclude_top_k must satisfy 0 <= k < n ({n}), got {k}")
        keep = np.arange(n) if k == 0 else order[: n - k]
        metrics = evaluate_clv_predictions(yt[keep], yp[keep])
        metrics["slice"] = f"All (n={n})" if k == 0 else f"Excl Top-{k} (n={n - k})"
        metrics["n_customers"] = int(len(keep))
        metrics["actual_sum_gbp"] = float(yt[keep].sum())
        rows.append(metrics)

    cols = [
        "slice",
        "n_customers",
        "actual_sum_gbp",
        "spearman",
        "normalized_gini",
        "top_10_capture",
        "lift_at_10",
        "top_20_capture",
        "lift_at_20",
        "total_rev_error_pct",
        "mae",
        "medae",
        "rmse",
    ]
    return pd.DataFrame(rows)[cols]


def compute_slice_error_table(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str = "target",
    y_pred_col: str = "pred_frozen_tweedie",
    baseline_col: str | None = "pred_run_rate",
    sort_by: str | None = "actual_sum_gbp",
    ascending: bool = False,
) -> pd.DataFrame:
    """Compute diagnostic CLV error, bias, calibration, and ranking metrics per cohort slice."""
    required = [group_col, y_true_col, y_pred_col] + (
        [baseline_col] if baseline_col is not None else []
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from DataFrame: {missing}")

    total_actual_all = float(df[y_true_col].sum())
    rows: list[dict[str, Any]] = []

    for group_val, sub in df.groupby(group_col, observed=True, sort=True):
        yt = sub[y_true_col].to_numpy(dtype=float)
        yp = sub[y_pred_col].to_numpy(dtype=float)
        n = len(sub)
        m = evaluate_clv_predictions(yt, yp)
        sum_true = float(yt.sum())
        sum_pred = float(yp.sum())
        row: dict[str, Any] = {
            "slice": str(group_val),
            "n_customers": int(n),
            "pct_customers": float(n / len(df) * 100.0) if len(df) > 0 else 0.0,
            "actual_sum_gbp": sum_true,
            "pred_sum_gbp": sum_pred,
            "actual_share_pct": float(sum_true / total_actual_all * 100.0)
            if total_actual_all > 0.0
            else 0.0,
            "actual_mean_gbp": float(np.mean(yt)),
            "pred_mean_gbp": float(np.mean(yp)),
            "zero_target_pct": float(np.mean(yt == 0.0) * 100.0),
            "total_rev_error_pct": float((sum_pred - sum_true) / sum_true * 100.0)
            if sum_true > 0.0
            else np.nan,
            "mean_bias_pred_minus_true": float(np.mean(yp - yt)),
            "mae": float(m["mae"]),
            "medae": float(m["medae"]),
            "rmse": float(m["rmse"]),
            "spearman": float(m["spearman"]),
            "normalized_gini": float(m["normalized_gini"]),
        }
        if baseline_col is not None:
            yb = sub[baseline_col].to_numpy(dtype=float)
            bm = evaluate_clv_predictions(yt, yb)
            row["run_rate_rev_err_pct"] = (
                float((yb.sum() - sum_true) / sum_true * 100.0) if sum_true > 0.0 else np.nan
            )
            row["run_rate_mae"] = float(bm["mae"])
            row["run_rate_rmse"] = float(bm["rmse"])
        rows.append(row)

    out = pd.DataFrame(rows)
    if sort_by is not None:
        if sort_by not in out.columns:
            raise ValueError(f"sort_by column '{sort_by}' not found in computed slice metrics.")
        out = out.sort_values(sort_by, ascending=ascending)
    return out.set_index("slice")


def compute_targeting_simulation(
    y_true: pd.Series | np.ndarray,
    pred_map: Mapping[str, pd.Series | np.ndarray],
    top_fracs: Sequence[float] = (0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Translate ranking lift at business budget tiers into captured GBP revenue vs. random and oracle."""
    yt = np.asarray(y_true, dtype=float)
    n = len(yt)
    total_actual = float(yt.sum())
    rows: list[dict[str, Any]] = []

    for frac in top_fracs:
        oracle_cap, _ = capture_and_lift_at_k(yt, yt, top_frac=float(frac))
        random_gbp = total_actual * float(frac)
        n_targeted = float(frac * n)
        for model_name, raw_pred in pred_map.items():
            yp = np.asarray(raw_pred, dtype=float)
            cap, lift = capture_and_lift_at_k(yt, yp, top_frac=float(frac))
            captured_gbp = cap * total_actual
            # Empirical buyer hit rate on top k descending ties-averaged cleanly via stable sort
            k_int = max(1, int(round(n_targeted)))
            if np.std(yp) < 1e-12:
                hit_rate_pct = float((yt > 0).mean() * 100.0)
            else:
                top_idx = np.argsort(-yp, kind="mergesort")[:k_int]
                hit_rate_pct = float((yt[top_idx] > 0).mean() * 100.0)
            rows.append(
                {
                    "top_pct": int(round(frac * 100)),
                    "model": str(model_name),
                    "n_targeted_approx": k_int,
                    "actual_captured_gbp": float(captured_gbp),
                    "capture_share_pct": float(cap * 100.0),
                    "lift_vs_random": float(lift),
                    "incremental_vs_random_gbp": float(captured_gbp - random_gbp),
                    "pct_of_oracle_ceiling": float(cap / oracle_cap * 100.0)
                    if oracle_cap > 0.0
                    else 0.0,
                    "avg_actual_spend_targeted_gbp": float(captured_gbp / n_targeted)
                    if n_targeted > 0
                    else 0.0,
                    "active_buyer_hit_rate_pct": hit_rate_pct,
                }
            )
    return pd.DataFrame(rows)


def compute_coverage_ratio_breakdown(
    sales: pd.DataFrame,
    returns: pd.DataFrame,
    eligible_ids: Sequence[Any],
    cutoff: pd.Timestamp | str,
    horizon_days: int = 90,
) -> pd.DataFrame:
    """Partition all target-window registered customers cleanly into eligible vs. out-of-scope cohorts."""
    cutoff_ts = pd.Timestamp(cutoff)
    target_end = cutoff_ts + pd.Timedelta(days=horizon_days)

    s_tg = sales[(sales["InvoiceDate"] >= cutoff_ts) & (sales["InvoiceDate"] < target_end)]
    r_tg = returns[(returns["InvoiceDate"] >= cutoff_ts) & (returns["InvoiceDate"] < target_end)]

    elig_set = set(eligible_ids)
    all_tg_ids = set(s_tg["Customer ID"]) | set(r_tg["Customer ID"])
    sale_tg_ids = set(s_tg["Customer ID"])
    oos_ids = all_tg_ids - elig_set

    prior_ids = set(sales.loc[sales["InvoiceDate"] < cutoff_ts, "Customer ID"]) | set(
        returns.loc[returns["InvoiceDate"] < cutoff_ts, "Customer ID"]
    )
    ret_only_ids = oos_ids - sale_tg_ids
    dormant_ids = (oos_ids & sale_tg_ids) & prior_ids
    cold_start_ids = (oos_ids & sale_tg_ids) - prior_ids

    gross_by_cust = (
        s_tg.groupby("Customer ID")["TotalPrice"].sum() if len(s_tg) > 0 else pd.Series(dtype=float)
    )
    ret_by_cust = (
        r_tg.groupby("Customer ID")["TotalPrice"].sum() if len(r_tg) > 0 else pd.Series(dtype=float)
    )
    tot_gross = float(gross_by_cust.sum())
    tot_net = float(gross_by_cust.sum() + ret_by_cust.sum())

    cohorts = [
        ("Eligible (Model-Scored, Lookback Active)", elig_set & all_tg_ids, len(elig_set)),
        ("Out-of-Scope: Cold-Start (New Buyers)", cold_start_ids, len(cold_start_ids)),
        ("Out-of-Scope: Dormant-Reactivated", dormant_ids, len(dormant_ids)),
        ("Out-of-Scope: Returns-Only", ret_only_ids, len(ret_only_ids)),
    ]
    rows: list[dict[str, Any]] = []
    for label, active_ids, scored_or_cohort_n in cohorts:
        id_list = list(active_ids)
        g_sum = float(gross_by_cust.reindex(id_list, fill_value=0.0).sum()) if id_list else 0.0
        r_sum = float(ret_by_cust.reindex(id_list, fill_value=0.0).sum()) if id_list else 0.0
        n_sum = g_sum + r_sum
        rows.append(
            {
                "cohort": label,
                "cohort_pool_customers": int(scored_or_cohort_n),
                "target_active_customers": int(len(id_list)),
                "gross_spend_gbp": g_sum,
                "returns_gbp": r_sum,
                "net_spend_gbp": n_sum,
                "gross_share_pct": float(g_sum / tot_gross * 100.0) if tot_gross > 0 else 0.0,
                "net_share_pct": float(n_sum / tot_net * 100.0) if tot_net > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("cohort")
