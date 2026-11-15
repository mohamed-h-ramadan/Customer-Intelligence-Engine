"""Unit tests for reusable RFM segmentation functions in src/segmentation.py."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.segmentation import (
    build_rfm_pipeline,
    build_rfm_preprocessor,
    evaluate_external_validation,
    evaluate_kmeans_k_range,
    evaluate_robustness_checks,
    map_rfm_segment,
    profile_clusters,
    score_rfm_rules,
    transform_rfm_features,
)


class TestRuleBasedRFMScoring:
    def test_frequency_ties_receive_identical_scores(self):
        # 5 out of 10 rows have frequency == 1 (50% ties would crash naive pd.qcut)
        df = pd.DataFrame(
            {
                "recency_days": [5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0],
                "frequency": [10, 8, 5, 3, 2, 1, 1, 1, 1, 1],
                "monetary": [5000.0, 3000.0, 1500.0, 900.0, 600.0, 100.0, 200.0, 150.0, 80.0, 50.0],
            },
            index=pd.Index(range(1, 11), name="Customer ID"),
        )
        scored = score_rfm_rules(df)

        assert list(scored.columns) == [
            "R_score",
            "F_score",
            "M_score",
            "RFM_score_sum",
            "RFM_code",
            "rule_segment",
        ]
        # Every customer with frequency == 1 must share the exact same F_score
        tied_f1_scores = scored.loc[df["frequency"] == 1, "F_score"].unique()
        assert len(tied_f1_scores) == 1
        assert int(tied_f1_scores[0]) == 2  # avg pct rank of bottom 5/10 = 0.30 -> bin (0.2, 0.4]

        # Most recent, highest freq, highest spend customer must be top Champion
        assert scored.loc[1, "R_score"] == 5
        assert scored.loc[1, "F_score"] == 5
        assert scored.loc[1, "M_score"] == 5
        assert scored.loc[1, "RFM_code"] == "555"
        assert scored.loc[1, "rule_segment"] == "Champions"

        # Most lapsed, low freq, lowest spend must be Hibernating / Lost
        assert scored.loc[10, "R_score"] == 1
        assert scored.loc[10, "M_score"] == 1
        assert scored.loc[10, "rule_segment"] == "Hibernating / Lost"

    def test_missing_columns_raises_error(self):
        bad = pd.DataFrame({"recency_days": [1.0], "frequency": [2]})
        with pytest.raises(ValueError, match="Missing required RFM columns"):
            score_rfm_rules(bad)

    @pytest.mark.parametrize(
        "r,f,m,expected",
        [
            (5, 5, 5, "Champions"),
            (5, 4, 2, "Loyal Customers"),
            (3, 5, 4, "Loyal Customers"),
            (4, 2, 3, "Potential Loyalists"),
            (5, 1, 5, "Recent / New Customers"),
            (3, 3, 3, "Need Attention"),
            (3, 1, 2, "About to Sleep"),
            (1, 5, 4, "At Risk (High Value)"),
            (2, 2, 5, "At Risk (High Value)"),
            (1, 2, 2, "Hibernating / Lost"),
        ],
    )
    def test_map_rfm_segment_exhaustive_branches(self, r, f, m, expected):
        assert map_rfm_segment(r, f, m) == expected


class TestPreprocessingPipeline:
    def test_transform_rfm_features_exact_math_and_extra_cols(self):
        toy = pd.DataFrame(
            {
                "recency_days": [0.0, 4.0, 9.0, 16.0],
                "frequency": [1, 3, 7, 15],
                "monetary": [np.e - 1.0, np.e**2 - 1.0, np.e**3 - 1.0, np.e**4 - 1.0],
                "unused_col": ["UK", "UK", "FR", "DE"],
            }
        )
        transformed = transform_rfm_features(toy, recency_transform="sqrt")
        assert transformed.shape == (4, 3)
        np.testing.assert_allclose(transformed[:, 0], [0.0, 2.0, 3.0, 4.0], rtol=1e-7)
        np.testing.assert_allclose(transformed[:, 1], np.log1p([1, 3, 7, 15]), rtol=1e-7)
        np.testing.assert_allclose(transformed[:, 2], [1.0, 2.0, 3.0, 4.0], rtol=1e-7)

    def test_preprocessor_standardization_and_pipeline_joblib_roundtrip(self, tmp_path):
        toy = pd.DataFrame(
            {
                "recency_days": [1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
                "frequency": [1, 2, 4, 8, 12, 20],
                "monetary": [50.0, 120.0, 300.0, 800.0, 2000.0, 5000.0],
            }
        )
        pre = build_rfm_preprocessor()
        X_scaled = pre.fit_transform(toy)
        assert X_scaled.shape == (6, 3)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(X_scaled.std(axis=0, ddof=0), 1.0, atol=1e-12)

        pipe = build_rfm_pipeline(n_clusters=2, n_init=20, random_state=42)
        labels_before = pipe.fit_predict(toy)
        assert set(labels_before) == {0, 1}

        model_file = tmp_path / "rfm_pipe.joblib"
        joblib.dump(pipe, model_file)
        loaded_pipe = joblib.load(model_file)
        labels_after = loaded_pipe.predict(toy)
        np.testing.assert_array_equal(labels_before, labels_after)

    def test_negative_values_and_low_n_init_raise(self):
        bad = pd.DataFrame({"recency_days": [-1.0], "frequency": [1], "monetary": [10.0]})
        with pytest.raises(ValueError, match="non-negative"):
            transform_rfm_features(bad)
        with pytest.raises(ValueError, match="n_init must be >= 20"):
            build_rfm_pipeline(n_clusters=3, n_init=10)


class TestKSelection:
    def test_evaluate_kmeans_k_range_on_well_separated_clusters(self):
        rng = np.random.default_rng(42)
        c1 = rng.normal(loc=[-5.0, -5.0, -5.0], scale=0.3, size=(12, 3))
        c2 = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.3, size=(12, 3))
        c3 = rng.normal(loc=[5.0, 5.0, 5.0], scale=0.3, size=(12, 3))
        X_blobs = np.vstack([c1, c2, c3])

        metrics_df, labels_by_k = evaluate_kmeans_k_range(
            X_blobs, k_values=range(2, 6), n_init=20, n_resamples=5, random_state=42
        )
        assert list(metrics_df["K"]) == [2, 3, 4, 5]
        assert set(labels_by_k.keys()) == {2, 3, 4, 5}
        # Inertia strictly decreases as K grows
        assert (np.diff(metrics_df["inertia"]) < 0).all()
        # K=3 achieves highest Silhouette, highest CH, lowest DB, and near-perfect ARI
        best_sil_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "K"])
        best_db_k = int(metrics_df.loc[metrics_df["davies_bouldin"].idxmin(), "K"])
        assert best_sil_k == 3
        assert best_db_k == 3
        k3_ari = float(metrics_df.loc[metrics_df["K"] == 3, "stability_ari_mean"].iloc[0])
        assert k3_ari > 0.95

    def test_evaluate_kmeans_k_range_invalid_inputs_raise(self):
        X_tiny = np.zeros((5, 3))
        with pytest.raises(ValueError, match="n_init must be >= 20"):
            evaluate_kmeans_k_range(X_tiny, n_init=5)
        with pytest.raises(ValueError, match="K must satisfy"):
            evaluate_kmeans_k_range(X_tiny, k_values=[1, 2], n_init=20)


class TestRobustnessChecks:
    def test_robustness_agrees_on_separated_blobs(self):
        rng = np.random.default_rng(42)
        c1 = rng.normal(loc=[-5.0, -5.0, -5.0], scale=0.25, size=(12, 3))
        c2 = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.25, size=(12, 3))
        c3 = rng.normal(loc=[5.0, 5.0, 5.0], scale=0.25, size=(12, 3))
        X_rfm = np.vstack([c1, c2, c3])
        # Extra 4th dimension reinforcing the same 3 regimes
        extra = np.concatenate([np.zeros(12), np.full(12, 3.0), np.full(12, 6.0)]).reshape(-1, 1)
        X_aov = np.hstack([X_rfm, extra])

        from sklearn.cluster import KMeans

        km_labels = KMeans(n_clusters=3, n_init=20, random_state=42).fit_predict(X_rfm)

        summary_df, assign_df = evaluate_robustness_checks(
            X_rfm, X_aov, km_labels, n_clusters=3, n_init=20, random_state=42
        )
        assert len(summary_df) == 4
        assert assign_df.shape == (36, 4)
        assert (summary_df["ari_vs_kmeans"] > 0.95).all()
        assert (summary_df["nmi_vs_kmeans"] > 0.95).all()

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="exact same row count"):
            evaluate_robustness_checks(
                np.zeros((10, 3)), np.zeros((9, 4)), np.zeros(10, dtype=int), n_clusters=2
            )


class TestClusterProfiling:
    def test_profile_clusters_exact_shares_and_relative_means(self):
        toy = pd.DataFrame(
            {
                "recency_days": [10.0, 20.0, 200.0, 300.0],
                "frequency": [10, 14, 1, 1],
                "monetary": [600.0, 1000.0, 150.0, 250.0],
                "aov": [60.0, 71.4, 150.0, 250.0],
                "return_rate": [0.10, 0.20, 0.00, 0.00],
                "is_uk": [1, 1, 0, 1],
            },
            index=pd.Index([1, 2, 3, 4], name="Customer ID"),
        )
        labels = [0, 0, 1, 1]
        prof, rel = profile_clusters(toy, labels)

        assert list(prof.index) == [0, 1]
        assert int(prof.loc[0, "customers"]) == 2
        assert np.isclose(prof.loc[0, "cust_share_pct"], 50.0)
        assert np.isclose(prof.loc[0, "mean_M"], 800.0)
        assert np.isclose(prof.loc[0, "rev_share_pct"], 80.0)  # 1600 / 2000
        assert np.isclose(prof.loc[1, "rev_share_pct"], 20.0)
        assert np.isclose(prof.loc[0, "mean_return_rate"], 0.15)
        assert np.isclose(prof["rev_share_pct"].sum(), 100.0)

        # Global mean monetary = 500; cluster 0 mean = 800 -> rel_monetary = 1.6x
        assert np.isclose(rel.loc[0, "monetary"], 1.6)
        assert np.isclose(rel.loc[1, "monetary"], 0.4)

    def test_profile_clusters_validates_columns_and_length(self):
        toy = pd.DataFrame(
            {"recency_days": [1.0, 2.0], "frequency": [1, 2], "monetary": [10.0, 20.0]}
        )
        with pytest.raises(ValueError, match="Missing required profiling columns"):
            profile_clusters(toy, [0, 1], extra_cols=("aov",))
        with pytest.raises(ValueError, match="does not match DataFrame rows"):
            profile_clusters(toy, [0], extra_cols=())


class TestExternalValidation:
    def test_evaluate_external_validation_strong_separation(self):
        target = pd.Series(
            [1000.0, 1200.0, 1500.0, 1100.0, 200.0, 150.0, 180.0, 220.0, 0.0, 0.0, 0.0, 5.0],
            index=pd.Index(range(1, 13), name="Customer ID"),
        )
        segments = pd.Series(
            ["VIP"] * 4 + ["Mid"] * 4 + ["Lost"] * 4,
            index=target.index,
        )
        per_seg, stats = evaluate_external_validation(target, segments)

        assert list(per_seg.index) == ["VIP", "Mid", "Lost"]
        assert np.isclose(per_seg.loc["VIP", "reactivation_rate_pct"], 100.0)
        assert np.isclose(per_seg.loc["Lost", "reactivation_rate_pct"], 25.0)
        assert np.isclose(per_seg["future_spend_share_pct"].sum(), 100.0)
        assert stats["kruskal_pvalue"] < 0.05
        assert stats["anova_eta2_log1p"] > 0.85
        assert stats["kruskal_eta2"] > 0.70

    def test_evaluate_external_validation_rejects_negative_or_mismatch(self):
        with pytest.raises(ValueError, match="non-negative"):
            evaluate_external_validation(pd.Series([-1.0, 10.0]), pd.Series(["A", "B"]))
        with pytest.raises(ValueError, match="identical lengths"):
            evaluate_external_validation(pd.Series([1.0, 2.0]), pd.Series(["A"]))
