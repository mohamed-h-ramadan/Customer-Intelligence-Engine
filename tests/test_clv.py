"""Unit tests for CLV evaluation metrics, calibration helpers, and modeling utilities."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from src.clv import (
    ALL_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CLV_FEATURE_GROUPS,
    CountryGrouper,
    HurdleCLVRegressor,
    Log1pSmearingRegressor,
    RuleBaselineRegressor,
    bootstrap_clv_metrics,
    build_candidate_models,
    build_linear_clv_preprocessor,
    build_tree_clv_preprocessor,
    calibration_by_predicted_decile,
    capture_and_lift_at_k,
    compute_coverage_ratio_breakdown,
    compute_quantile_pdp,
    compute_slice_error_table,
    compute_targeting_simulation,
    cross_validate_clv_estimator,
    cv_permutation_importance,
    evaluate_classifier_metrics,
    evaluate_clv_predictions,
    evaluate_whale_sensitivity_slices,
    make_stratification_bins,
    normalized_gini,
    ranking_capture_score,
    rmse_excluding_top_k,
)


class TestRankingAndCaptureMetrics:
    def test_perfect_and_inverse_normalized_gini(self):
        y_true = np.array([0.0, 0.0, 10.0, 20.0, 70.0])
        assert np.isclose(normalized_gini(y_true, y_true), 1.0)
        assert np.isclose(normalized_gini(y_true, -y_true), -1.0)
        # Constant predictions must have exact 0.0 Gini independent of array order
        const_pred = np.full_like(y_true, 50.0)
        assert np.isclose(normalized_gini(y_true, const_pred), 0.0)
        assert np.isclose(normalized_gini(y_true[::-1], const_pred), 0.0)

    def test_capture_and_lift_on_toy_ranking(self):
        # Total actual = 200 across 10 customers
        y_true = np.array([100.0, 50.0, 20.0, 10.0, 10.0, 5.0, 5.0, 0.0, 0.0, 0.0])
        # Exact ordering predicted
        y_pred = np.array([90.0, 45.0, 25.0, 12.0, 10.0, 6.0, 4.0, 1.0, 0.5, 0.0])

        cap10, lift10 = capture_and_lift_at_k(y_true, y_pred, top_frac=0.10)
        cap20, lift20 = capture_and_lift_at_k(y_true, y_pred, top_frac=0.20)
        assert np.isclose(cap10, 0.50)  # top 1 / 10 owns 100 / 200
        assert np.isclose(lift10, 5.0)
        assert np.isclose(cap20, 0.75)  # top 2 / 10 own 150 / 200
        assert np.isclose(lift20, 3.75)

        # Constant prediction must yield random lift == 1.0 regardless of row ordering
        cap10_const, lift10_const = capture_and_lift_at_k(y_true, np.ones(10), top_frac=0.10)
        assert np.isclose(cap10_const, 0.10)
        assert np.isclose(lift10_const, 1.0)

    def test_rmse_excluding_top_k(self):
        y_true = np.array([10.0, 20.0, 30.0, 1000.0])
        y_pred = np.array([12.0, 20.0, 30.0, 0.0])  # huge whale error on 1000 vs tiny error on rest
        rmse_all = np.sqrt(((2.0**2) + (1000.0**2)) / 4.0)
        assert np.isclose(rmse_excluding_top_k(y_true, y_pred, top_k=1), np.sqrt(4.0 / 3.0))
        assert rmse_all > 400.0

    def test_evaluate_clv_predictions_complete_suite(self):
        y_true = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 550.0])
        y_pred = np.array([1.0, 2.0, 12.0, 18.0, 25.0, 45.0, 55.0, 90.0, 220.0, 532.0])
        metrics = evaluate_clv_predictions(y_true, y_pred)

        expected_keys = {
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
            "rmse_excl_top1",
            "rmse_excl_top5",
        }
        assert set(metrics.keys()) == expected_keys
        assert metrics["spearman"] > 0.99
        assert np.isclose(metrics["normalized_gini"], 1.0)
        assert np.isclose(metrics["top_10_capture"], 0.55)  # 550 / 1000
        assert np.isclose(metrics["lift_at_10"], 5.5)
        assert np.isclose(metrics["total_rev_error_pct"], 0.0)  # sum_pred == 1000 == sum_true

        # Constant prediction safety
        const_metrics = evaluate_clv_predictions(y_true, np.full(10, 100.0))
        assert const_metrics["spearman"] == 0.0
        assert const_metrics["normalized_gini"] == 0.0
        assert np.isclose(const_metrics["lift_at_10"], 1.0)


class TestCalibrationAndStratification:
    def test_calibration_by_predicted_decile(self):
        y_true = np.arange(1.0, 21.0) * 10.0
        y_pred = np.arange(1.0, 21.0) * 10.0
        cal_df = calibration_by_predicted_decile(y_true, y_pred, n_bins=10)
        assert len(cal_df) == 10
        assert cal_df["n_customers"].sum() == 20
        np.testing.assert_allclose(cal_df["mean_pred"], cal_df["mean_actual"])
        np.testing.assert_allclose(cal_df["pred_to_actual_ratio"], 1.0)
        assert np.isclose(cal_df["actual_share_pct"].sum(), 100.0)

    def test_make_stratification_bins_separates_zeros_and_spreads_positives(self):
        y = np.array([0.0] * 10 + list(range(1, 21)))
        bins = make_stratification_bins(y, n_positive_bins=5)
        assert (bins[:10] == 0).all()
        assert set(bins[10:]) == {1, 2, 3, 4, 5}


class TestBaselinesAndCV:
    def test_rule_baselines_exact_values_and_cv_runner(self):
        X = pd.DataFrame(
            {
                "monetary": [300.0, 600.0, 900.0, 0.0, 1200.0, 150.0, 300.0, 450.0, 600.0, 750.0],
                "frequency": [1, 2, 3, 1, 4, 1, 2, 3, 2, 5],
            }
        )
        X["aov"] = X["monetary"] / X["frequency"]
        y = pd.Series([0.0, 0.0, 100.0, 0.0, 400.0, 50.0, 80.0, 120.0, 200.0, 250.0])

        est_rr = RuleBaselineRegressor(rule="run_rate").fit(X, y)
        est_fa = RuleBaselineRegressor(rule="freq_aov_run_rate").fit(X, y)
        np.testing.assert_allclose(est_rr.predict(X), X["monetary"] / 3.0)
        np.testing.assert_allclose(est_rr.predict(X), est_fa.predict(X))

        est_mean = RuleBaselineRegressor(rule="train_mean").fit(X, y)
        np.testing.assert_allclose(est_mean.predict(X), y.mean())

        strata = make_stratification_bins(y, n_positive_bins=2)
        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=42)
        summary, fold_df, oof = cross_validate_clv_estimator(est_rr, X, y, cv, strata)
        assert len(fold_df) == 4
        assert len(oof) == len(X)
        assert not np.isnan(oof).any()
        assert "spearman_mean" in summary and "top_10_capture_mean" in summary


class TestPipelinesSmearingAndHurdle:
    def test_country_grouper_maps_unseen_to_other(self):
        X_tr = pd.DataFrame(
            {"country": ["United Kingdom"] * 5 + ["Germany", "Germany", "France", "Spain"]}
        )
        X_te = pd.DataFrame({"country": ["United Kingdom", "Germany", "Mars", "France"]})
        grouper = CountryGrouper(top_n=1).fit(X_tr)
        assert grouper.kept_countries_ == ["United Kingdom", "Germany"]
        mapped = grouper.transform(X_te).ravel().tolist()
        assert mapped == ["United Kingdom", "Germany", "Other", "Other"]

    def test_duan_smearing_exact_formula_and_effect(self):
        from sklearn.dummy import DummyRegressor

        X = np.zeros((4, 1))
        # log1p(y) = [0, 1, 2, 3]; mean = 1.5 -> residuals [-1.5, -0.5, +0.5, +1.5]
        y_log = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.expm1(y_log)
        expected_smearing = float(np.mean(np.exp(y_log - 1.5)))

        smeared = Log1pSmearingRegressor(DummyRegressor(strategy="mean"), use_smearing=True).fit(
            X, y
        )
        naive = Log1pSmearingRegressor(DummyRegressor(strategy="mean"), use_smearing=False).fit(
            X, y
        )

        assert np.isclose(smeared.smearing_factor_, expected_smearing)
        assert np.isclose(naive.smearing_factor_, 1.0)
        np.testing.assert_allclose(smeared.predict(X), expected_smearing * np.exp(1.5) - 1.0)
        np.testing.assert_allclose(naive.predict(X), np.expm1(1.5))
        assert smeared.predict(X)[0] > naive.predict(X)[0]

    def test_hurdle_prediction_equals_p_times_mu_and_filters_zeros(self):
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import LogisticRegression

        X = pd.DataFrame({"x": [0.0, 0.1, 0.2, 5.0, 6.0, 7.0]})
        y = np.array([0.0, 0.0, 0.0, 100.0, 200.0, 300.0])

        hurdle = HurdleCLVRegressor(
            classifier=LogisticRegression(),
            regressor=DummyRegressor(strategy="mean"),
        ).fit(X, y)

        p = hurdle.predict_proba(X)
        mu = hurdle.predict_conditional(X)
        pred = hurdle.predict(X)
        # Stage 2 saw ONLY [100, 200, 300] -> mean == 200.0!
        np.testing.assert_allclose(mu, 200.0)
        np.testing.assert_allclose(pred, np.clip(p * mu, 0.0, None))
        assert pred[-1] > pred[0]

        clf_metrics = evaluate_classifier_metrics((y > 0).astype(int), p)
        assert clf_metrics["roc_auc"] > 0.95
        assert 0.0 <= clf_metrics["brier_score"] <= 0.25

    def test_preprocessors_and_candidate_models_fit_on_toy_schema(self):
        rng = np.random.default_rng(42)
        n = 50
        data = {col: rng.uniform(0.1, 100.0, size=n) for col in ALL_NUMERIC_FEATURES}
        data["mean_interpurchase_days"][:20] = np.nan
        data["is_single_invoice"] = np.array([1] * 20 + [0] * 30, dtype=int)
        data["country"] = np.where(
            np.arange(n) < 35, "United Kingdom", np.where(np.arange(n) < 43, "Germany", "Japan")
        )
        X_toy = pd.DataFrame(data)
        y_toy = pd.Series(
            np.where(np.arange(n) < 20, 0.0, rng.gamma(shape=2.0, scale=200.0, size=n))
        )

        lin_mat = build_linear_clv_preprocessor(top_n_countries=1).fit_transform(X_toy)
        assert not np.isnan(lin_mat).any()

        hgb_mat = build_tree_clv_preprocessor(
            top_n_countries=1, impute_missing=False
        ).fit_transform(X_toy)
        assert np.isnan(hgb_mat).sum() == 20  # structural NaNs preserved for HGB

        models = build_candidate_models(random_state=42, top_n_countries=1)
        assert len(models) == 8
        for m_name, est in models.items():
            est_fit = est.fit(X_toy, y_toy)
            preds = est_fit.predict(X_toy)
            assert preds.shape == (n,)
            assert not np.isnan(preds).any(), f"NaN predictions in {m_name}"
            assert (preds >= 0.0).all(), f"Negative predictions in {m_name}"

        # Verify ranking_capture_score and nested set_params compatibility for RandomizedSearchCV
        score_val = ranking_capture_score(
            y_toy, models["B: Tweedie GLM (power=1.5, log link)"].predict(X_toy)
        )
        assert 0.0 <= score_val <= 1.0
        models["D2: Hurdle (Logit Clf x Ridge Reg)"].set_params(
            classifier__clf__C=0.1,
            regressor__reg__regressor__alpha=50.0,
        )
        assert models["D2: Hurdle (Logit Clf x Ridge Reg)"].classifier.named_steps["clf"].C == 0.1

    def test_feature_groups_cover_all_26_features_cleanly(self):
        grouped_cols = [c for cols in CLV_FEATURE_GROUPS.values() for c in cols]
        expected_cols = ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES
        assert len(grouped_cols) == len(set(grouped_cols)) == 26
        assert set(grouped_cols) == set(expected_cols)

    def test_cv_permutation_importance_single_and_grouped_plus_pdp(self):
        rng = np.random.default_rng(7)
        n = 80
        sig_a = rng.uniform(1.0, 20.0, size=n)
        sig_b = sig_a * 2.0 + rng.normal(0.0, 0.2, size=n)
        noise_a = rng.uniform(0.0, 5.0, size=n)
        noise_b = rng.uniform(0.0, 5.0, size=n)
        X = pd.DataFrame({"sig_a": sig_a, "sig_b": sig_b, "noise_a": noise_a, "noise_b": noise_b})
        y = pd.Series(5.0 * sig_a + 1.0)
        strata = make_stratification_bins(y, n_positive_bins=4)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)

        # Single-feature permutation: signal columns outrank noise columns cleanly
        single_imp = cv_permutation_importance(
            Ridge(alpha=1.0), X, y, cv=cv, strata=strata, n_repeats=5, random_state=7
        )
        top_2 = set(single_imp.iloc[:2]["feature"])
        assert top_2 == {"sig_a", "sig_b"}
        assert float(single_imp.iloc[0]["gini_drop_mean"]) > 0.2
        assert (
            abs(float(single_imp[single_imp["feature"] == "noise_a"]["gini_drop_mean"].iloc[0]))
            < 0.05
        )

        # Grouped collinear permutation: destroying Signal Block drops Gini sharply vs Noise Block
        group_imp = cv_permutation_importance(
            Ridge(alpha=1.0),
            X,
            y,
            cv=cv,
            strata=strata,
            feature_groups={"Signal": ["sig_a", "sig_b"], "Noise": ["noise_a", "noise_b"]},
            n_repeats=5,
            random_state=7,
        )
        assert group_imp.iloc[0]["feature"] == "Signal"
        assert float(group_imp.iloc[0]["gini_drop_mean"]) > 0.8
        assert abs(float(group_imp.iloc[1]["gini_drop_mean"])) < 0.05

        # Quantile PDP on fitted estimator increases monotonically with sig_a
        fitted = Ridge(alpha=1.0).fit(X, y)
        pdp_df = compute_quantile_pdp(fitted, X, "sig_a", quantiles=(0.1, 0.5, 0.9))
        assert list(pdp_df.columns) == [
            "feature",
            "quantile",
            "grid_value",
            "mean_pred",
            "median_pred",
            "p10_pred",
            "p90_pred",
        ]
        assert (np.diff(pdp_df["mean_pred"].to_numpy()) > 0.0).all()
        assert (pdp_df["p10_pred"] <= pdp_df["median_pred"]).all()
        assert (pdp_df["median_pred"] <= pdp_df["p90_pred"]).all()

    def test_bootstrap_cis_and_whale_sensitivity_slices(self):
        rng = np.random.default_rng(123)
        n = 60
        yt = np.concatenate(
            [
                np.zeros(20),
                rng.uniform(50.0, 600.0, size=35),
                [2000.0, 2500.0, 3000.0, 4000.0, 10000.0],
            ]
        )
        yp_good = yt * 0.9 + rng.uniform(0.0, 40.0, size=n)
        yp_bad = np.full(n, yt.mean())

        boot = bootstrap_clv_metrics(
            yt, yp_good, ref_pred=yp_bad, n_bootstrap=100, ci=0.95, random_state=42
        )
        assert "normalized_gini" in boot.index
        assert (
            boot.loc["normalized_gini", "ci_low"] <= boot.loc["normalized_gini", "point_estimate"]
        )
        assert (
            boot.loc["normalized_gini", "point_estimate"] <= boot.loc["normalized_gini", "ci_high"]
        )
        assert boot.loc["normalized_gini", "delta_vs_ref"] > 0.5
        assert boot.loc["normalized_gini", "delta_ci_low"] > 0.5

        slices = evaluate_whale_sensitivity_slices(yt, yp_good, exclude_top_ks=(0, 1, 5))
        assert list(slices["n_customers"]) == [60, 59, 55]
        # Excluding largest whale matches rmse_excluding_top_k(k=1)
        assert np.isclose(slices.iloc[1]["rmse"], rmse_excluding_top_k(yt, yp_good, top_k=1))
        assert np.isclose(slices.iloc[2]["rmse"], rmse_excluding_top_k(yt, yp_good, top_k=5))
        assert slices.iloc[1]["actual_sum_gbp"] < slices.iloc[0]["actual_sum_gbp"]

    def test_compute_slice_error_table_exact_cohorts_and_bias(self):
        df = pd.DataFrame(
            {
                "segment": ["VIP", "VIP", "Low", "Low"],
                "target": [100.0, 300.0, 0.0, 20.0],
                "pred_frozen_tweedie": [110.0, 250.0, 10.0, 30.0],
                "pred_run_rate": [80.0, 200.0, 5.0, 15.0],
            }
        )
        tbl = compute_slice_error_table(df, group_col="segment")
        assert list(tbl.index) == ["VIP", "Low"]
        assert int(tbl.loc["VIP", "n_customers"]) == 2
        assert np.isclose(tbl.loc["VIP", "actual_sum_gbp"], 400.0)
        assert np.isclose(tbl.loc["VIP", "pred_sum_gbp"], 360.0)
        # VIP bias = mean([+10, -50]) = -20.0; rev error = -10.0%
        assert np.isclose(tbl.loc["VIP", "mean_bias_pred_minus_true"], -20.0)
        assert np.isclose(tbl.loc["VIP", "total_rev_error_pct"], -10.0)
        assert np.isclose(tbl.loc["Low", "zero_target_pct"], 50.0)
        assert np.isclose(tbl.loc["VIP", "run_rate_rev_err_pct"], -30.0)

        # sort_by=None preserves original category order cleanly (e.g. deciles 1..10)
        df_ord = df.copy()
        df_ord["bin"] = pd.Categorical([2, 2, 1, 1], categories=[1, 2], ordered=True)
        tbl_ord = compute_slice_error_table(df_ord, group_col="bin", sort_by=None)
        assert list(tbl_ord.index) == ["1", "2"]

    def test_targeting_simulation_and_coverage_ratio_breakdown(self):
        yt = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0])  # sum = 100.0
        perfect = np.arange(10, dtype=float)
        random_const = np.full(10, 5.0)
        sim = compute_targeting_simulation(
            yt, {"Perfect": perfect, "Random": random_const}, top_fracs=(0.10, 0.20)
        )
        top10_perf = sim[(sim["top_pct"] == 10) & (sim["model"] == "Perfect")].iloc[0]
        top10_rand = sim[(sim["top_pct"] == 10) & (sim["model"] == "Random")].iloc[0]
        assert np.isclose(top10_perf["actual_captured_gbp"], 40.0)
        assert np.isclose(top10_perf["pct_of_oracle_ceiling"], 100.0)
        assert np.isclose(top10_perf["incremental_vs_random_gbp"], 30.0)
        assert np.isclose(top10_rand["capture_share_pct"], 10.0)

        sales = pd.DataFrame(
            {
                "Customer ID": [1, 2, 3, 4],
                "InvoiceDate": pd.to_datetime(
                    ["2011-05-01", "2010-05-01", "2011-10-01", "2011-10-02"]
                ),
                "TotalPrice": [50.0, 30.0, 20.0, 10.0],
            }
        )
        # Add target-window sales for ID 1 (eligible) & ID 2 (dormant pre-lookback) + ID 4 (cold-start)
        sales = pd.concat(
            [
                sales,
                pd.DataFrame(
                    {
                        "Customer ID": [1, 2],
                        "InvoiceDate": pd.to_datetime(["2011-10-03", "2011-10-04"]),
                        "TotalPrice": [100.0, 40.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        # Note ID 3 already in target above (20.0, prior? None -> cold-start!); make returns-only ID 5
        returns = pd.DataFrame(
            {
                "Customer ID": [5],
                "InvoiceDate": pd.to_datetime(["2011-10-05"]),
                "TotalPrice": [-15.0],
            }
        )
        cov = compute_coverage_ratio_breakdown(
            sales, returns, eligible_ids=[1], cutoff="2011-09-10", horizon_days=90
        )
        assert np.isclose(
            cov.loc["Eligible (Model-Scored, Lookback Active)", "gross_spend_gbp"], 100.0
        )
        assert np.isclose(cov.loc["Out-of-Scope: Dormant-Reactivated", "gross_spend_gbp"], 40.0)
        assert np.isclose(cov.loc["Out-of-Scope: Cold-Start (New Buyers)", "gross_spend_gbp"], 30.0)
        assert np.isclose(cov.loc["Out-of-Scope: Returns-Only", "gross_spend_gbp"], 0.0)
        assert np.isclose(cov.loc["Out-of-Scope: Returns-Only", "net_spend_gbp"], -15.0)
        assert np.isclose(cov["gross_share_pct"].sum(), 100.0)

    def test_clv_pipeline_joblib_roundtrip_parity(self, tmp_path):
        rng = np.random.default_rng(99)
        n = 60
        data = {col: rng.uniform(0.5, 80.0, size=n) for col in ALL_NUMERIC_FEATURES}
        data["mean_interpurchase_days"][:20] = np.nan
        data["is_single_invoice"] = np.array([1] * 20 + [0] * 40, dtype=int)
        data["country"] = np.where(np.arange(n) < 45, "United Kingdom", "Germany")
        X = pd.DataFrame(data)
        y = pd.Series(np.where(np.arange(n) < 25, 0.0, rng.gamma(shape=2.0, scale=150.0, size=n)))

        models = build_candidate_models(random_state=42, top_n_countries=3)
        tweedie = (
            models["B: Tweedie GLM (power=1.5, log link)"]
            .set_params(model__power=1.65, model__alpha=0.5)
            .fit(X, y)
        )
        hurdle = models["D2: Hurdle (Logit Clf x Ridge Reg)"].fit(X, y)

        for name, fitted_est in [("tweedie.joblib", tweedie), ("hurdle.joblib", hurdle)]:
            out_file = tmp_path / name
            joblib.dump(fitted_est, out_file)
            loaded = joblib.load(out_file)
            np.testing.assert_allclose(
                loaded.predict(X), fitted_est.predict(X), rtol=1e-12, atol=1e-12
            )
