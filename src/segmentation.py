"""Reusable unsupervised RFM segmentation and validation utilities."""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RFM_COLUMNS: list[str] = ["recency_days", "frequency", "monetary"]
DEFAULT_RANDOM_STATE: int = 42


def transform_rfm_features(
    X: pd.DataFrame | np.ndarray,
    recency_transform: str = "sqrt",
    feature_cols: Sequence[str] = tuple(RFM_COLUMNS),
) -> np.ndarray:
    """Apply positive skewness compression to RFM columns before standardization.

    Applies ``sqrt`` (default) or ``log1p`` to ``recency_days`` and ``log1p`` to all
    heavily right-skewed spend/count columns (``frequency``, ``monetary``, and optionally
    ``aov``). Supports both DataFrames (extracting ``feature_cols`` safely) and 2D arrays.
    Defined at module scope so ``joblib`` can serialize the full sklearn Pipeline.
    """
    cols = list(feature_cols)
    if isinstance(X, pd.DataFrame):
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required clustering columns: {missing}")
        arr = X[cols].to_numpy(dtype=float, copy=True)
    else:
        arr = np.asarray(X, dtype=float).copy()
        if arr.ndim != 2 or arr.shape[1] != len(cols):
            raise ValueError(
                f"Expected 2D array with {len(cols)} columns {cols}, got shape {arr.shape}"
            )

    if np.any(arr < 0):
        raise ValueError("RFM clustering features must be non-negative prior to sqrt/log1p.")

    for idx, col_name in enumerate(cols):
        if col_name == "recency_days":
            if recency_transform == "sqrt":
                arr[:, idx] = np.sqrt(arr[:, idx])
            elif recency_transform == "log1p":
                arr[:, idx] = np.log1p(arr[:, idx])
            else:
                raise ValueError(f"Unsupported recency_transform: {recency_transform!r}")
        else:
            arr[:, idx] = np.log1p(arr[:, idx])

    return arr


def build_rfm_preprocessor(
    feature_cols: Sequence[str] = tuple(RFM_COLUMNS),
    recency_transform: str = "sqrt",
) -> Pipeline:
    """Create the stateless transform + StandardScaler preprocessing Pipeline."""
    cols = tuple(feature_cols)
    return Pipeline(
        steps=[
            (
                "transform",
                FunctionTransformer(
                    func=transform_rfm_features,
                    kw_args={"recency_transform": recency_transform, "feature_cols": cols},
                    validate=False,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )


def build_rfm_pipeline(
    n_clusters: int = 4,
    feature_cols: Sequence[str] = tuple(RFM_COLUMNS),
    recency_transform: str = "sqrt",
    n_init: int = 20,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Create a single end-to-end sklearn Pipeline: transform -> StandardScaler -> KMeans."""
    if n_init < 20:
        raise ValueError(f"KMeans n_init must be >= 20 for reproducibility, got {n_init}")
    pre = build_rfm_preprocessor(feature_cols=feature_cols, recency_transform=recency_transform)
    return Pipeline(
        steps=[
            *pre.steps,
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    n_init=n_init,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _rank_quintile_scores(series: pd.Series, ascending_is_better: bool = True) -> pd.Series:
    """Assign deterministic 1-5 scores using average percentile ranks to handle ties cleanly.

    Using ``rank(method="average", pct=True)`` avoids ``pd.qcut`` duplicate-edge failures
    while guaranteeing that tied observations (such as all single-invoice customers with
    ``frequency == 1``) always receive the exact same RFM score.
    """
    ordered = series if ascending_is_better else -series
    pct_rank = ordered.rank(method="average", pct=True)
    scores = pd.cut(
        pct_rank,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    )
    return scores.astype("int64")


def map_rfm_segment(r_score: int, f_score: int, m_score: int) -> str:
    """Map integer (1-5) R, F, M quintile scores to mutually exclusive rule-based segments."""
    if r_score >= 4:
        if f_score >= 4 and m_score >= 4:
            return "Champions"
        if f_score >= 4:
            return "Loyal Customers"
        if f_score >= 2:
            return "Potential Loyalists"
        return "Recent / New Customers"

    if r_score == 3:
        if f_score >= 4:
            return "Loyal Customers"
        if f_score >= 2:
            return "Need Attention"
        return "About to Sleep"

    # r_score <= 2
    if f_score >= 4 or m_score >= 4:
        return "At Risk (High Value)"
    return "Hibernating / Lost"


def score_rfm_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rank-based 1-5 R/F/M scores, combined RFM_score, and rule-based segment names."""
    missing = [col for col in RFM_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required RFM columns: {missing}")

    out = pd.DataFrame(index=df.index)
    out.index.name = df.index.name

    out["R_score"] = _rank_quintile_scores(df["recency_days"], ascending_is_better=False)
    out["F_score"] = _rank_quintile_scores(df["frequency"], ascending_is_better=True)
    out["M_score"] = _rank_quintile_scores(df["monetary"], ascending_is_better=True)
    out["RFM_score_sum"] = out["R_score"] + out["F_score"] + out["M_score"]
    out["RFM_code"] = (
        out["R_score"].astype(str) + out["F_score"].astype(str) + out["M_score"].astype(str)
    )
    out["rule_segment"] = [
        map_rfm_segment(int(r), int(f), int(m))
        for r, f, m in zip(out["R_score"], out["F_score"], out["M_score"], strict=False)
    ]
    return out


def evaluate_kmeans_k_range(
    X_scaled: pd.DataFrame | np.ndarray,
    k_values: Iterable[int] = range(2, 11),
    n_init: int = 20,
    n_resamples: int = 20,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Sweep candidate K values and return internal validity + bootstrap stability metrics.

    For each K, fits a reference KMeans (n_init >= 20), computes Inertia, Silhouette,
    Davies-Bouldin, Calinski-Harabasz, and evaluates cluster stability as the mean
    Adjusted Rand Index across ``n_resamples`` bootstrap fits predicted back onto the full
    dataset. Also flags tiny clusters representing under 3% of observations.
    """
    if n_init < 20:
        raise ValueError(f"KMeans n_init must be >= 20 for reproducibility, got {n_init}")

    X_arr = np.asarray(X_scaled, dtype=float)
    n_samples = len(X_arr)
    rng = np.random.default_rng(random_state)

    rows = []
    labels_by_k: dict[int, np.ndarray] = {}

    for k in k_values:
        k_int = int(k)
        if k_int < 2 or k_int >= n_samples:
            raise ValueError(f"K must satisfy 2 <= K < n_samples ({n_samples}), got {k_int}")

        ref_km = KMeans(n_clusters=k_int, n_init=n_init, random_state=random_state)
        ref_labels = ref_km.fit_predict(X_arr)
        labels_by_k[k_int] = ref_labels

        counts = pd.Series(ref_labels).value_counts().sort_values(ascending=False)
        min_sz = int(counts.min())
        min_pct = float(min_sz / n_samples * 100.0)
        max_pct = float(counts.max() / n_samples * 100.0)
        sizes_str = " / ".join(f"{int(c)} ({c / n_samples * 100:.1f}%)" for c in counts.values)

        ari_scores = []
        for b in range(n_resamples):
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            boot_km = KMeans(
                n_clusters=k_int,
                n_init=n_init,
                random_state=random_state + 100 + b,
            )
            boot_km.fit(X_arr[boot_idx])
            boot_pred_full = boot_km.predict(X_arr)
            ari_scores.append(adjusted_rand_score(ref_labels, boot_pred_full))

        rows.append(
            {
                "K": k_int,
                "inertia": float(ref_km.inertia_),
                "silhouette": float(silhouette_score(X_arr, ref_labels)),
                "davies_bouldin": float(davies_bouldin_score(X_arr, ref_labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X_arr, ref_labels)),
                "stability_ari_mean": float(np.mean(ari_scores)),
                "stability_ari_std": float(np.std(ari_scores)),
                "min_cluster_size": min_sz,
                "min_cluster_pct": round(min_pct, 2),
                "max_cluster_pct": round(max_pct, 2),
                "flag_lt_3pct": bool(min_pct < 3.0),
                "cluster_sizes": sizes_str,
            }
        )

    return pd.DataFrame(rows), labels_by_k


def evaluate_robustness_checks(
    X_rfm_scaled: pd.DataFrame | np.ndarray,
    X_aov_scaled: pd.DataFrame | np.ndarray,
    kmeans_labels: Sequence[int],
    n_clusters: int,
    n_init: int = 20,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare K-Means assignments against GMM, Ward Hierarchical, and RFM+AOV sensitivity.

    Returns ``(summary_df, assignments_df)`` with Adjusted Rand Index (ARI),
    Normalized Mutual Information (NMI), Silhouette, and minimum cluster sizes.
    """
    if n_init < 20:
        raise ValueError(f"KMeans n_init must be >= 20 for reproducibility, got {n_init}")

    X_rfm = np.asarray(X_rfm_scaled, dtype=float)
    X_aov = np.asarray(X_aov_scaled, dtype=float)
    km_arr = np.asarray(kmeans_labels, dtype=int)
    n_samples = len(X_rfm)

    if len(X_aov) != n_samples or len(km_arr) != n_samples:
        raise ValueError(
            "Input feature arrays and kmeans_labels must share the exact same row count."
        )

    gmm_labels = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=10,
        random_state=random_state,
    ).fit_predict(X_rfm)

    ward_labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    ).fit_predict(X_rfm)

    km_aov_labels = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    ).fit_predict(X_aov)

    assignments = pd.DataFrame(
        {
            "kmeans_rfm": km_arr,
            "gmm_rfm": gmm_labels,
            "ward_rfm": ward_labels,
            "kmeans_rfm_aov": km_aov_labels,
        }
    )

    specs = [
        ("KMeans (3D RFM Baseline)", km_arr, X_rfm),
        ("Gaussian Mixture (Full Cov, 3D RFM)", gmm_labels, X_rfm),
        ("Ward Hierarchical (3D RFM)", ward_labels, X_rfm),
        ("KMeans Sensitivity (RFM + log1p(aov))", km_aov_labels, X_aov),
    ]

    summary_rows = []
    for method_name, lbls, feat_space in specs:
        counts = pd.Series(lbls).value_counts().sort_values(ascending=False)
        min_pct = float(counts.min() / n_samples * 100.0)
        summary_rows.append(
            {
                "method": method_name,
                "n_clusters": int(n_clusters),
                "ari_vs_kmeans": round(float(adjusted_rand_score(km_arr, lbls)), 4),
                "nmi_vs_kmeans": round(float(normalized_mutual_info_score(km_arr, lbls)), 4),
                "silhouette": round(float(silhouette_score(feat_space, lbls)), 4),
                "min_cluster_pct": round(min_pct, 2),
                "cluster_sizes": " / ".join(
                    f"{int(c)} ({c / n_samples * 100:.1f}%)" for c in counts.values
                ),
            }
        )

    return pd.DataFrame(summary_rows), assignments


def profile_clusters(
    df: pd.DataFrame,
    labels: Sequence[int],
    extra_cols: Sequence[str] = ("aov", "return_rate", "is_uk"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-cluster size, revenue share, R/F/M statistics, and relative-to-global ratios.

    Returns ``(profile_df, relative_means_df)`` sorted descending by ``mean_M``.
    """
    required = list(RFM_COLUMNS) + [c for c in extra_cols if c not in RFM_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required profiling columns: {missing}")

    lbl_arr = np.asarray(labels)
    if len(lbl_arr) != len(df):
        raise ValueError(
            f"labels length ({len(lbl_arr)}) does not match DataFrame rows ({len(df)})."
        )

    work = df[required].copy()
    work["cluster"] = lbl_arr
    n_total = len(work)
    total_rev = float(work["monetary"].sum())

    agg_spec = {
        "customers": ("monetary", "count"),
        "mean_R": ("recency_days", "mean"),
        "median_R": ("recency_days", "median"),
        "mean_F": ("frequency", "mean"),
        "median_F": ("frequency", "median"),
        "mean_M": ("monetary", "mean"),
        "median_M": ("monetary", "median"),
        "total_revenue": ("monetary", "sum"),
    }
    if "aov" in work.columns:
        agg_spec["mean_aov"] = ("aov", "mean")
        agg_spec["median_aov"] = ("aov", "median")
    if "return_rate" in work.columns:
        agg_spec["mean_return_rate"] = ("return_rate", "mean")
        agg_spec["median_return_rate"] = ("return_rate", "median")
    if "is_uk" in work.columns:
        agg_spec["uk_share_pct"] = ("is_uk", lambda s: float(s.mean() * 100.0))

    profile_df = work.groupby("cluster").agg(**agg_spec)
    profile_df["cust_share_pct"] = (profile_df["customers"] / n_total) * 100.0
    profile_df["rev_share_pct"] = (
        (profile_df["total_revenue"] / total_rev) * 100.0 if total_rev > 0 else 0.0
    )
    profile_df = profile_df.sort_values("mean_M", ascending=False)

    ordered_cols = [
        "customers",
        "cust_share_pct",
        "mean_R",
        "median_R",
        "mean_F",
        "median_F",
        "mean_M",
        "median_M",
        "total_revenue",
        "rev_share_pct",
    ]
    for col in ("mean_aov", "median_aov", "mean_return_rate", "median_return_rate", "uk_share_pct"):
        if col in profile_df.columns:
            ordered_cols.append(col)
    profile_df = profile_df[ordered_cols]

    # Cluster means relative to global means (1.0 = cohort average)
    rel_cols = ["recency_days", "frequency", "monetary"]
    for col in ("aov", "return_rate"):
        if col in work.columns:
            rel_cols.append(col)

    global_means = work[rel_cols].mean()
    cluster_means = work.groupby("cluster")[rel_cols].mean().loc[profile_df.index]
    relative_df = cluster_means.divide(global_means.replace(0.0, np.nan), axis="columns")
    return profile_df, relative_df


def evaluate_external_validation(
    target_spend: pd.Series,
    segment_labels: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate how strongly unsupervised or rule segments separate unseen future 90d spend.

    Computes per-segment reactivation rate (target > 0), mean/median future spend, share
    of future spend, Kruskal-Wallis H/p-value, Kruskal rank eta-squared, and ANOVA
    eta-squared on ``log1p(target)``. Used post-hoc for validation only.
    """
    if len(target_spend) != len(segment_labels):
        raise ValueError("target_spend and segment_labels must have identical lengths.")
    if (target_spend < 0).any():
        raise ValueError("target_spend must be non-negative (gross spend) before log1p validation.")

    val_df = pd.DataFrame(
        {
            "target": target_spend.to_numpy(dtype=float),
            "log1p_target": np.log1p(target_spend.to_numpy(dtype=float)),
            "segment": segment_labels.to_numpy(),
        },
        index=target_spend.index,
    )

    n_total = len(val_df)
    total_future = float(val_df["target"].sum())

    per_seg = (
        val_df.groupby("segment")
        .agg(
            customers=("target", "count"),
            active_customers=("target", lambda s: int((s > 0).sum())),
            reactivation_rate_pct=("target", lambda s: float((s > 0).mean() * 100.0)),
            mean_target=("target", "mean"),
            median_target=("target", "median"),
            q90_target=("target", lambda s: float(s.quantile(0.90))),
            total_future_spend=("target", "sum"),
        )
        .assign(
            cust_share_pct=lambda d: (d["customers"] / n_total) * 100.0,
            future_spend_share_pct=lambda d: (
                (d["total_future_spend"] / total_future) * 100.0 if total_future > 0 else 0.0
            ),
        )
        .sort_values("mean_target", ascending=False)
    )

    ordered_cols = [
        "customers",
        "cust_share_pct",
        "active_customers",
        "reactivation_rate_pct",
        "mean_target",
        "median_target",
        "q90_target",
        "total_future_spend",
        "future_spend_share_pct",
    ]
    per_seg = per_seg[ordered_cols]

    groups_log = [grp["log1p_target"].to_numpy() for _, grp in val_df.groupby("segment")]
    k_groups = len(groups_log)
    kw_stat, kw_pvalue = kruskal(*groups_log)
    kruskal_eta2 = (
        max(0.0, (float(kw_stat) - k_groups + 1) / (n_total - k_groups))
        if n_total > k_groups
        else 0.0
    )

    # One-way ANOVA eta-squared on log1p(target): SS_between / SS_total
    grand_mean = float(val_df["log1p_target"].mean())
    ss_total = float(((val_df["log1p_target"] - grand_mean) ** 2).sum())
    ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups_log))
    anova_eta2_log1p = (ss_between / ss_total) if ss_total > 0 else 0.0
    f_stat, f_pvalue = f_oneway(*groups_log)

    stats_summary = {
        "n_segments": int(k_groups),
        "kruskal_h": float(kw_stat),
        "kruskal_pvalue": float(kw_pvalue),
        "kruskal_eta2": float(kruskal_eta2),
        "anova_f_log1p": float(f_stat),
        "anova_eta2_log1p": float(anova_eta2_log1p),
        "top_segment_spend_share_pct": float(per_seg["future_spend_share_pct"].max()),
        "top_segment_mean_target": float(per_seg["mean_target"].max()),
        "reactivation_spread_pp": float(
            per_seg["reactivation_rate_pct"].max() - per_seg["reactivation_rate_pct"].min()
        ),
    }
    return per_seg, stats_summary
