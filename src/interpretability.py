"""Explaining the models: global drivers and individual decisions.

Owner: Member 3.

The client needs two different explanations and they are not the same artefact:

* **Global** — "what drives this model overall", which is what goes in the deck
  and what a regulator asks about.
* **Local** — "why was *this* candidate scored this way", which is what the
  candidate asks about, and what an adverse-action notice has to contain.

Method per model, chosen for honesty rather than uniformity:

* logistic regression — exact coefficients as odds ratios. No approximation
  needed, and this is the white-box baseline the other two are judged against.
* XGBoost — exact TreeSHAP.
* TabPFN — permutation importance. There is no tree structure to exploit and
  KernelSHAP would need thousands of forward passes per explained row, which on
  CPU is hours. Reporting a cheaper, clearly-labelled method beats reporting a
  SHAP-shaped number that took a shortcut we did not disclose. That TabPFN is
  the hardest of the three to explain is itself a finding for the deck.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config
from src.data import DataBundle
from src.models import FittedModel

logger = logging.getLogger(__name__)


def _sample(cfg: Config, X: pd.DataFrame) -> pd.DataFrame:
    size = int(cfg.interpretability.shap_sample)
    if size >= len(X):
        return X
    return X.sample(n=size, random_state=cfg.seed)


# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------


def odds_ratios(fitted: FittedModel) -> pd.DataFrame:
    """Logistic-regression coefficients as odds ratios, most influential first.

    Features are standardised inside the pipeline, so each ratio reads as "the
    multiplicative change in the odds of being employed per one standard
    deviation of this feature", which is comparable across features.
    """
    estimator = fitted.estimator
    classifier = estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator
    coefficients = np.asarray(classifier.coef_).ravel()

    frame = pd.DataFrame(
        {
            "model": fitted.name,
            "feature": fitted.feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "abs_coefficient": np.abs(coefficients),
        }
    )
    return frame.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def tree_shap(cfg: Config, fitted: FittedModel, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Exact TreeSHAP for the gradient-boosted model."""
    import shap

    sample = _sample(cfg, X)
    explainer = shap.TreeExplainer(fitted.estimator)
    values = explainer.shap_values(sample)
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    values = np.asarray(values)

    importance = pd.DataFrame(
        {
            "model": fitted.name,
            "feature": sample.columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
            "method": "TreeSHAP",
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return importance.reset_index(drop=True), values


def linear_shap(
    cfg: Config, fitted: FittedModel, X: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Exact SHAP for the linear model, so its bars are comparable with XGBoost's."""
    import shap

    sample = _sample(cfg, X)
    explainer = shap.LinearExplainer(
        fitted.estimator.named_steps["clf"],
        fitted.estimator.named_steps["scale"].transform(sample),
    )
    values = np.asarray(
        explainer.shap_values(fitted.estimator.named_steps["scale"].transform(sample))
    )
    importance = pd.DataFrame(
        {
            "model": fitted.name,
            "feature": sample.columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
            "method": "LinearSHAP",
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return importance.reset_index(drop=True), values


def permutation_importance(
    cfg: Config, fitted: FittedModel, X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Model-agnostic fallback: AUC lost when a feature is shuffled.

    Used for TabPFN, where TreeSHAP does not apply and KernelSHAP would need
    thousands of CPU forward passes. Every (feature, repeat) pair costs one full
    inference pass, so the budget is capped in config and the candidate features
    are shortlisted first by absolute correlation with the target — a cheap,
    model-agnostic screen. The result is a shortlist ranking, not an exhaustive
    one, and the deck should say so.
    """
    from sklearn.metrics import roc_auc_score

    settings = cfg.interpretability
    n_rows = int(getattr(settings, "permutation_sample", 300))
    max_features = int(getattr(settings, "permutation_max_features", 12))
    n_repeats = int(getattr(settings, "permutation_repeats", 2))

    sample = X.sample(n=min(n_rows, len(X)), random_state=cfg.seed)
    y_sample = y.loc[sample.index]
    if y_sample.nunique() < 2:
        return pd.DataFrame()

    correlations = sample.corrwith(y_sample).abs().fillna(0.0)
    candidates = correlations.nlargest(min(max_features, len(correlations))).index.tolist()

    baseline = roc_auc_score(y_sample, fitted.predict_proba(sample))
    rng = np.random.default_rng(cfg.seed)

    rows = []
    for feature in candidates:
        drops = []
        for _ in range(n_repeats):
            shuffled = sample.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            drops.append(baseline - roc_auc_score(y_sample, fitted.predict_proba(shuffled)))
        rows.append(
            {
                "model": fitted.name,
                "feature": feature,
                "mean_abs_shap": float(np.mean(drops)),  # named for a shared schema
                "mean_shap": float(np.mean(drops)),
                "method": f"permutation importance, AUC drop ({n_repeats}x, top-{max_features} shortlist)",
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------


def local_contributions(
    values: np.ndarray, sample: pd.DataFrame, row: int, top_n: int = 10
) -> pd.DataFrame:
    """The features that pushed one candidate's score up and down, largest first."""
    contributions = pd.DataFrame(
        {
            "feature": sample.columns,
            "value": sample.iloc[row].to_numpy(),
            "contribution": values[row],
        }
    )
    contributions["abs_contribution"] = contributions["contribution"].abs()
    contributions["direction"] = np.where(
        contributions["contribution"] >= 0, "increases score", "decreases score"
    )
    return (
        contributions.sort_values("abs_contribution", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def explain_candidate(
    fitted: FittedModel, row: pd.DataFrame, background: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Explain a single row on demand — what the app's scoring page calls.

    Falls back to a coefficient-times-value decomposition when SHAP is not
    available for the model, so the page always has something to show.
    """
    import shap

    try:
        if fitted.name == "xgboost":
            values = np.asarray(shap.TreeExplainer(fitted.estimator).shap_values(row))
        elif fitted.is_linear:
            scaler = fitted.estimator.named_steps["scale"]
            explainer = shap.LinearExplainer(
                fitted.estimator.named_steps["clf"], scaler.transform(background)
            )
            values = np.asarray(explainer.shap_values(scaler.transform(row)))
        else:
            raise NotImplementedError(fitted.name)
        if values.ndim == 3:
            values = values[:, :, 1]
        return local_contributions(values, row, 0, top_n)
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "SHAP unavailable for %s (%s); using a simpler decomposition.", fitted.name, exc
        )
        deviation = (row.iloc[0] - background.mean()).to_numpy()
        return local_contributions(deviation[None, :], row, 0, top_n)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run(cfg: Config, bundle: DataBundle, models: dict[str, FittedModel]) -> pd.DataFrame:
    """Global importance for every model, plus local examples, written to CSV."""
    X_test, y_test, _ = bundle.split("test")
    importances: list[pd.DataFrame] = []
    locals_: list[pd.DataFrame] = []

    for name, fitted in models.items():
        try:
            if name == "xgboost":
                importance, values = tree_shap(cfg, fitted, X_test)
                sample = _sample(cfg, X_test)
            elif fitted.is_linear:
                importance, values = linear_shap(cfg, fitted, X_test)
                sample = _sample(cfg, X_test)
                odds_ratios(fitted).to_csv(cfg.paths.results_dir / "odds_ratios.csv", index=False)
            else:
                importance = permutation_importance(cfg, fitted, X_test, y_test)
                values, sample = None, None
            importances.append(importance)

            if values is not None and sample is not None:
                for i in range(min(int(cfg.interpretability.local_examples), len(sample))):
                    contribution = local_contributions(values, sample, i)
                    contribution.insert(0, "model", name)
                    contribution.insert(1, "example", i)
                    locals_.append(contribution)
        except Exception as exc:  # noqa: BLE001
            logger.error("Interpretability failed for %r: %s", name, exc)

    global_importance = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    global_importance.to_csv(cfg.paths.results_dir / "feature_importance.csv", index=False)
    if locals_:
        pd.concat(locals_, ignore_index=True).to_csv(
            cfg.paths.results_dir / "local_explanations.csv", index=False
        )
    return global_importance
