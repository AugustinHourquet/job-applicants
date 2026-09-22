"""End-to-end smoke tests.

Two jobs:

1. Prove the whole pipeline runs and writes every artifact the app expects.
   If this passes, `make pipeline` will not fall over on the real data for
   structural reasons — only for data reasons.
2. Pin the contracts the six of us rely on. A change that renames a column in
   the predictions table, or makes single-row encoding disagree with training,
   breaks somebody else's module silently. These tests make it loud.

Runs on synthetic data, so no Kaggle credentials and no network are needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import data, economics, fairness, models, stability
from src.pipeline import run_pipeline

# --------------------------------------------------------------------- data --


def test_clean_drops_index_and_validates_schema(raw_df, cfg):
    cleaned = data.clean(raw_df, cfg)
    assert not [c for c in cleaned.columns if c.startswith("Unnamed:")]
    assert cfg.data.target in cleaned.columns
    assert set(cleaned[cfg.data.target].unique()) <= {0, 1}


def test_clean_fails_loudly_on_a_missing_column(raw_df, cfg):
    """Silently modelling fewer features than configured is the bug to avoid."""
    with pytest.raises(KeyError, match="declared in config.yaml"):
        data.clean(raw_df.drop(columns=["Gender"]), cfg)


def test_feature_spec_is_fitted_on_training_data_only(raw_df, cfg):
    cleaned = data.clean(raw_df, cfg)
    train, _, test = data.split_frame(cleaned, cfg)
    spec = data.fit_feature_spec(train, cfg)
    # A level that appears only in test must fold into the rare bucket rather
    # than silently adding a column the model was never trained on.
    encoded_test = data.apply_feature_spec(test, spec)
    assert list(encoded_test.columns) == spec.feature_names


def test_single_row_encoding_matches_training(raw_df, cfg):
    """The app's form must produce exactly the matrix the models were trained on."""
    bundle = data.build_datasets(cfg, raw=raw_df)
    one = data.apply_feature_spec(raw_df.head(1), bundle.spec)
    assert list(one.columns) == bundle.feature_names
    assert one.shape == (1, len(bundle.feature_names))
    assert one.notna().all().all()


def test_splits_are_disjoint_and_stratified(raw_df, cfg):
    bundle = data.build_datasets(cfg, raw=raw_df)
    sizes = [len(bundle.X_train), len(bundle.X_val), len(bundle.X_test)]
    assert sum(sizes) == len(data.clean(raw_df, cfg))
    rates = [bundle.y_train.mean(), bundle.y_val.mean(), bundle.y_test.mean()]
    assert max(rates) - min(rates) < 0.05, "stratification should keep base rates close"


def test_proxy_check_is_not_fooled_by_cardinality(raw_df, cfg):
    """A near-unique free-text column must not flag as a proxy for everything.

    Plug-in mutual information is biased upward with cardinality; this is the
    regression test for the level cap that fixes it.
    """
    cleaned = data.clean(raw_df, cfg)
    checks = data.run_checks(cleaned, cfg)
    row = checks[checks["feature"] == "HaveWorkedWith"]
    assert not row.empty
    assert not bool(row["proxy_flag"].iloc[0])


# ------------------------------------------------------------------- models --


def test_multivariate_check_catches_a_combination_leak(raw_df, cfg):
    """A leak invisible to the per-column screen must still be caught.

    Regression test for the real finding: on the Kaggle data no single column
    exceeds 0.87, yet the HaveWorkedWith indicators together reach a held-out
    AUC of 1.000. Here we plant an equivalent leak — two columns that are
    individually uninformative but jointly reveal the target.
    """
    frame = raw_df.copy()
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 2, size=len(frame))
    # Individually each is a coin flip; their XOR is exactly the target.
    frame["YearsCode"] = noise
    frame["YearsCodePro"] = noise ^ frame[cfg.data.target].to_numpy()

    bundle = data.build_datasets(cfg, raw=frame)
    report = bundle.multivariate_checks
    assert not report.empty
    overall = report[report["scope"] == "ALL FEATURES"]["holdout_auc"].iloc[0]
    assert overall > 0.9, f"combination leak went undetected (AUC {overall})"


def test_multivariate_check_names_every_feature_group(raw_df, cfg):
    bundle = data.build_datasets(cfg, raw=raw_df)
    scopes = set(bundle.multivariate_checks["scope"])
    assert "ALL FEATURES" in scopes
    for column in cfg.data.categorical_features + cfg.data.numeric_features:
        assert column in scopes, f"{column} missing from the group breakdown"


def test_exclude_features_removes_columns_from_the_matrix(raw_df, cfg):
    """Switching to the honest feature set must be a config change only."""
    cfg.data.exclude_features = ["HaveWorkedWith", "ComputerSkills"]
    bundle = data.build_datasets(cfg, raw=raw_df)
    assert not [c for c in bundle.feature_names if c.startswith("HaveWorkedWith")]
    assert "ComputerSkills" not in bundle.feature_names
    # And the app's single-row encoding must follow the same exclusion.
    one = data.apply_feature_spec(raw_df.head(1), bundle.spec)
    assert list(one.columns) == bundle.feature_names


def test_registry_covers_every_configured_model(cfg):
    for name in cfg.enabled_models:
        assert name in models.available_models()


def test_predictions_table_schema(raw_df, cfg):
    """The contract every downstream module reads."""
    cfg.models["tabpfn"].enabled = False
    bundle = data.build_datasets(cfg, raw=raw_df)
    fitted = models.train_all(cfg, bundle)
    predictions = models.build_predictions_table(fitted, bundle)

    required = {"split", "model", "row_id", "y_true", "y_prob"}
    assert required <= set(predictions.columns)
    for attribute in cfg.fairness.protected_attributes:
        assert attribute in predictions.columns
    assert predictions["y_prob"].between(0, 1).all()
    assert set(predictions["split"]) == {"val", "test"}


def test_models_beat_chance(raw_df, cfg):
    """The synthetic data has real signal; a model that cannot find it is broken."""
    from sklearn.metrics import roc_auc_score

    cfg.models["tabpfn"].enabled = False
    bundle = data.build_datasets(cfg, raw=raw_df)
    fitted = models.train_all(cfg, bundle)
    for name, model in fitted.items():
        auc = roc_auc_score(bundle.y_test, model.predict_proba(bundle.X_test))
        assert auc > 0.55, f"{name} scored {auc:.3f}, barely above chance"


def test_save_and_load_round_trip(raw_df, cfg):
    cfg.models["tabpfn"].enabled = False
    bundle = data.build_datasets(cfg, raw=raw_df)
    fitted = models.train_all(cfg, bundle)
    original = fitted["logistic_regression"]
    models.save(original, cfg)
    restored = models.load("logistic_regression", cfg)
    np.testing.assert_allclose(
        original.predict_proba(bundle.X_test), restored.predict_proba(bundle.X_test)
    )


# ---------------------------------------------------------------- economics --


def test_profit_matches_the_cost_matrix_by_hand():
    cost = {
        "true_positive": 10.0,
        "false_positive": -5.0,
        "false_negative": -2.0,
        "true_negative": 0.0,
    }
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])  # one of each cell
    assert economics.profit(y_true, y_pred, cost) == 10.0 - 5.0 - 2.0 + 0.0


def test_threshold_is_chosen_on_validation_not_test(raw_df, cfg):
    """Choosing on test and reporting on test would be optimistic."""
    cfg.models["tabpfn"].enabled = False
    bundle = data.build_datasets(cfg, raw=raw_df)
    fitted = models.train_all(cfg, bundle)
    predictions = models.build_predictions_table(fitted, bundle)
    _, summary = economics.evaluate(cfg, predictions)

    assert (summary["threshold_split"] == "val").all()
    assert (summary["report_split"] == "test").all()
    # The oracle threshold can only ever do at least as well on test.
    assert (summary["cost_of_honest_threshold"] >= -1e-9).all()


def test_profit_curve_covers_the_grid(cfg):
    y_true = np.array([1, 0] * 50)
    y_prob = np.linspace(0.01, 0.99, 100)
    curve = economics.profit_curve(
        y_true, y_prob, cfg.economics.cost_matrix, np.arange(0.1, 1.0, 0.1)
    )
    assert len(curve) == 9
    assert (curve[["tp", "fp", "fn", "tn"]].sum(axis=1) == 100).all()


# ----------------------------------------------------------------- fairness --


def test_group_metrics_cover_every_group(raw_df, cfg):
    cfg.models["tabpfn"].enabled = False
    bundle = data.build_datasets(cfg, raw=raw_df)
    fitted = models.train_all(cfg, bundle)
    predictions = models.build_predictions_table(fitted, bundle)
    test = predictions[(predictions["split"] == "test") & (predictions["model"] == "xgboost")]

    metrics = fairness.group_metrics(test, "Gender", 0.5, cfg.fairness.min_group_size)
    assert set(metrics["group"]) == set(test["Gender"].unique())
    assert metrics["n"].sum() == len(test)
    for column in ("selection_rate", "true_positive_rate", "false_positive_rate"):
        finite = metrics[column].dropna()
        assert finite.between(0, 1).all()


def test_small_groups_are_flagged_not_dropped(raw_df, cfg):
    """Dropping them hides who the model fails."""
    cfg.models["tabpfn"].enabled = False
    bundle = data.build_datasets(cfg, raw=raw_df)
    fitted = models.train_all(cfg, bundle)
    predictions = models.build_predictions_table(fitted, bundle)
    test = predictions[(predictions["split"] == "test") & (predictions["model"] == "xgboost")]

    metrics = fairness.group_metrics(test, "Gender", 0.5, min_group_size=10_000)
    assert len(metrics) > 0
    assert not metrics["reliable"].any()


def test_reweighing_balances_group_and_label(raw_df, cfg):
    """After reweighing, group and label should be near-independent by construction."""
    cleaned = data.clean(raw_df, cfg)
    y = cleaned[cfg.data.target]
    protected = cleaned["Gender"]
    weights = fairness.reweighing_weights(y, protected)

    assert len(weights) == len(y)
    assert (weights > 0).all()
    overall = np.average(y, weights=weights)
    for group in protected.unique():
        mask = (protected == group).to_numpy()
        assert np.average(y[mask], weights=weights[mask]) == pytest.approx(overall, abs=1e-6)


def test_group_thresholds_move_rates_toward_the_reference(raw_df, cfg):
    cfg.models["tabpfn"].enabled = False
    bundle = data.build_datasets(cfg, raw=raw_df)
    fitted = models.train_all(cfg, bundle)
    predictions = models.build_predictions_table(fitted, bundle)
    test = predictions[(predictions["split"] == "test") & (predictions["model"] == "xgboost")]

    chosen = fairness.group_thresholds(test, "Gender", "true_positive_rate")
    assert set(chosen) == set(test["Gender"].unique())
    assert all(0 < t < 1 for t in chosen.values())
    decisions = fairness.apply_group_thresholds(test, "Gender", chosen)
    assert set(np.unique(decisions)) <= {0, 1}


# ---------------------------------------------------------------- stability --


def test_bootstrap_stability_reports_its_own_sample_size(raw_df, cfg):
    cfg.models["tabpfn"].enabled = False
    bundle = data.build_datasets(cfg, raw=raw_df)
    fitted = models.train_all(cfg, bundle)
    per_row, summary = stability.bootstrap_stability(cfg, bundle, "xgboost", fitted["xgboost"])

    assert not summary.empty
    assert summary["n_refits"].iloc[0] <= cfg.stability.n_boot
    assert per_row["flip_rate"].between(0, 1).all()
    assert len(per_row) == min(cfg.stability.eval_subsample, len(bundle.X_test))


def test_tabpfn_bootstrap_budget_is_capped(cfg):
    """TabPFN costs minutes per re-fit; the cap must actually bind."""
    cfg.stability.n_boot = 50
    assert stability._budget(cfg, "tabpfn") == 10
    assert stability._budget(cfg, "xgboost") == 50


# ----------------------------------------------------------------- pipeline --


@pytest.mark.slow
def test_full_pipeline_writes_every_artifact(raw_df, cfg):
    """The whole run, all three models, on synthetic data."""
    artefacts = run_pipeline(cfg, raw=raw_df)

    assert not artefacts["scorecard"].empty
    assert set(artefacts["scorecard"]["model"]) == set(cfg.enabled_models)

    expected = [
        "scorecard.csv",
        "statistical.csv",
        "economics.csv",
        "profit_curves.csv",
        "fairness.csv",
        "fairness_summary.csv",
        "stability.csv",
        "data_checks.csv",
        "feature_importance.csv",
        "predictions.parquet",
    ]
    for filename in expected:
        path = cfg.paths.results_dir / filename
        assert path.is_file(), f"pipeline did not write {filename}"
        assert path.stat().st_size > 0, f"{filename} is empty"

    for name in cfg.enabled_models:
        assert (cfg.paths.models_dir / f"{name}.joblib").is_file()

    for split in ("train", "val", "test"):
        assert (cfg.paths.processed_dir / f"{split}.parquet").is_file()
    assert (cfg.paths.processed_dir / "feature_spec.json").is_file()


@pytest.mark.slow
def test_scorecard_covers_all_four_dimensions(raw_df, cfg):
    """The deliverable is a trade-off table, not a leaderboard."""
    artefacts = run_pipeline(cfg, raw=raw_df)
    scorecard = artefacts["scorecard"]

    assert "roc_auc" in scorecard.columns  # statistical
    assert "profit" in scorecard.columns  # economic
    assert "bootstrap_flip_rate" in scorecard.columns  # stability
    assert "min_disparate_impact" in scorecard.columns  # fairness
    assert not artefacts["importance"].empty  # interpretability
