"""Group fairness metrics and two mitigation strategies.

Owner: Member 5.

Metrics are computed at the threshold the client would actually deploy — the
profit-maximising one chosen by :mod:`src.economics` — not at an arbitrary 0.5.
A fairness audit at a threshold nobody uses describes a system that does not
exist.

Four families are reported, because they disagree by construction and cannot
all be satisfied at once when base rates differ across groups:

    selection_rate      demographic parity   — who gets picked
    true_positive_rate  equal opportunity    — of those who would succeed,
                                               who gets picked
    false_positive_rate equalised odds       — of those who would not,
                                               who gets picked anyway
    precision           predictive parity    — of those picked, who succeeds

Which one the client should care about is an argument the deck has to make.
This module's job is to measure all four honestly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

METRIC_TO_CRITERION = {
    "selection_rate": "demographic parity",
    "true_positive_rate": "equal opportunity",
    "false_positive_rate": "equalised odds",
    "precision": "predictive parity",
}


def _rate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """The four group rates. NaN where a group has no cases to compute one from."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    positives = y_true == 1
    negatives = y_true == 0
    selected = y_pred == 1

    return {
        "selection_rate": float(selected.mean()) if len(y_pred) else np.nan,
        "true_positive_rate": float(y_pred[positives].mean()) if positives.any() else np.nan,
        "false_positive_rate": float(y_pred[negatives].mean()) if negatives.any() else np.nan,
        "precision": float(y_true[selected].mean()) if selected.any() else np.nan,
        "base_rate": float(positives.mean()) if len(y_true) else np.nan,
    }


def group_metrics(
    frame: pd.DataFrame, attribute: str, threshold: float, min_group_size: int = 0
) -> pd.DataFrame:
    """Per-group rates for one protected attribute at one threshold.

    Groups below ``min_group_size`` are kept but marked ``reliable=False``:
    dropping them hides who the model fails, while reporting a TPR from eleven
    people as if it were solid is its own kind of dishonesty.
    """
    rows = []
    for group_value, group in frame.groupby(attribute, sort=True):
        y_pred = (group["y_prob"].to_numpy() >= threshold).astype(int)
        rows.append(
            {
                "attribute": attribute,
                "group": str(group_value),
                "n": len(group),
                "reliable": len(group) >= min_group_size,
                **_rate_metrics(group["y_true"].to_numpy(), y_pred),
            }
        )
    return pd.DataFrame(rows)


def add_gaps(metrics: pd.DataFrame, reference: str | None, metric_names: list[str]) -> pd.DataFrame:
    """Add ``<metric>_gap`` columns measured against a reference group.

    ``reference=None`` means "use the largest group", which is the sensible
    default when no group is a natural baseline.
    """
    metrics = metrics.copy()
    if metrics.empty:
        return metrics

    if reference is None or reference not in set(metrics["group"]):
        reference = str(metrics.loc[metrics["n"].idxmax(), "group"])
    metrics["reference_group"] = reference

    reference_row = metrics[metrics["group"] == reference].iloc[0]
    for metric in metric_names:
        if metric in metrics.columns:
            metrics[f"{metric}_gap"] = metrics[metric] - reference_row[metric]

    # Disparate impact: the "four-fifths rule" familiar from employment law,
    # which makes the number immediately meaningful to an HR client.
    if "selection_rate" in metrics.columns and reference_row["selection_rate"]:
        metrics["disparate_impact_ratio"] = (
            metrics["selection_rate"] / reference_row["selection_rate"]
        )
    return metrics


def audit(
    cfg: Config, predictions: pd.DataFrame, thresholds: dict[str, float], split: str = "test"
) -> pd.DataFrame:
    """Full fairness table: every model x protected attribute x group.

    Args:
        thresholds: Per-model operating threshold, from :mod:`src.economics`.
    """
    subset = predictions[predictions["split"] == split]
    metric_names = cfg.fairness.metrics or list(METRIC_TO_CRITERION)
    frames: list[pd.DataFrame] = []

    for model_name, model_rows in subset.groupby("model", sort=False):
        threshold = float(thresholds.get(str(model_name), 0.5))
        for attribute in cfg.fairness.protected_attributes:
            if attribute not in model_rows.columns:
                logger.warning("Protected attribute %r absent from predictions", attribute)
                continue
            metrics = group_metrics(model_rows, attribute, threshold, cfg.fairness.min_group_size)
            metrics = add_gaps(metrics, cfg.fairness.reference_groups.get(attribute), metric_names)
            metrics.insert(0, "model", model_name)
            metrics.insert(1, "threshold", threshold)
            frames.append(metrics)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarise(fairness: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    """Worst gap per model and attribute — the headline number for the scorecard.

    The maximum absolute gap is the right summary: a model that treats five
    groups identically and one group badly is not a fair model, and an average
    would hide exactly that.
    """
    if fairness.empty:
        return pd.DataFrame()

    reliable = fairness[fairness["reliable"]]
    rows = []
    for (model_name, attribute), group in reliable.groupby(["model", "attribute"], sort=False):
        record: dict[str, Any] = {"model": model_name, "attribute": attribute}
        for metric in metric_names:
            column = f"{metric}_gap"
            if column in group.columns:
                record[f"max_abs_{metric}_gap"] = float(group[column].abs().max())
        if "disparate_impact_ratio" in group.columns:
            ratio = group["disparate_impact_ratio"].min()
            record["min_disparate_impact"] = float(ratio)
            # The four-fifths rule: a ratio below 0.8 is the conventional
            # threshold for adverse impact in employment screening.
            record["four_fifths_pass"] = bool(ratio >= 0.8)
        rows.append(record)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mitigation
# ---------------------------------------------------------------------------


def reweighing_weights(y: pd.Series, protected: pd.Series) -> np.ndarray:
    """Kamiran & Calders pre-processing weights.

    Each (group, label) cell is weighted by how far its observed frequency sits
    from the frequency it would have if group and label were independent. The
    model then sees a training set in which they are, without any row being
    dropped or any label being changed.
    """
    y = pd.Series(y).reset_index(drop=True)
    protected = pd.Series(protected).astype(str).reset_index(drop=True)
    n = len(y)

    weights = np.ones(n, dtype=float)
    for group_value in protected.unique():
        for label in y.unique():
            mask = (protected == group_value) & (y == label)
            observed = int(mask.sum())
            if observed == 0:
                continue
            expected = float((protected == group_value).sum()) * float((y == label).sum()) / n
            weights[mask.to_numpy()] = expected / observed
    return weights


def group_thresholds(
    frame: pd.DataFrame,
    attribute: str,
    target_metric: str = "true_positive_rate",
    grid: np.ndarray | None = None,
    reference: str | None = None,
) -> dict[str, float]:
    """Post-processing: one threshold per group, equalising ``target_metric``.

    The reference group keeps the threshold that is already in use; every other
    group gets the threshold bringing it closest to the reference's rate. This
    is the most transparent mitigation available — and also the one that makes
    the trade-off hardest to hide, since different cut-offs per group is
    precisely what the client has to decide whether it can defend.
    """
    grid = np.arange(0.01, 1.0, 0.01) if grid is None else grid
    groups = {str(g): sub for g, sub in frame.groupby(attribute, sort=True)}
    if not groups:
        return {}

    if reference is None or reference not in groups:
        reference = max(groups, key=lambda g: len(groups[g]))

    def rate_at(sub: pd.DataFrame, threshold: float) -> float:
        y_pred = (sub["y_prob"].to_numpy() >= threshold).astype(int)
        return _rate_metrics(sub["y_true"].to_numpy(), y_pred)[target_metric]

    base_threshold = 0.5
    target_rate = rate_at(groups[reference], base_threshold)

    chosen = {reference: base_threshold}
    for name, sub in groups.items():
        if name == reference:
            continue
        rates = np.array([rate_at(sub, t) for t in grid], dtype=float)
        if np.all(np.isnan(rates)):
            chosen[name] = base_threshold
            continue
        chosen[name] = float(grid[int(np.nanargmin(np.abs(rates - target_rate)))])
    return chosen


def apply_group_thresholds(
    frame: pd.DataFrame, attribute: str, thresholds: dict[str, float]
) -> np.ndarray:
    """Binary decisions using a per-group threshold, defaulting to 0.5."""
    per_row = frame[attribute].astype(str).map(thresholds).fillna(0.5).to_numpy(dtype=float)
    return (frame["y_prob"].to_numpy() >= per_row).astype(int)


def run(cfg: Config, predictions: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Pipeline entry point: audit, write both CSVs, return the per-group table."""
    metric_names = cfg.fairness.metrics or list(METRIC_TO_CRITERION)
    detail = audit(cfg, predictions, thresholds)
    detail.to_csv(cfg.paths.results_dir / "fairness.csv", index=False)

    summary = summarise(detail, metric_names)
    summary.to_csv(cfg.paths.results_dir / "fairness_summary.csv", index=False)

    if not summary.empty and "four_fifths_pass" in summary.columns:
        failing = summary[~summary["four_fifths_pass"]]
        for _, row in failing.iterrows():
            logger.warning(
                "%s fails the four-fifths rule on %s (disparate impact %.2f)",
                row["model"],
                row["attribute"],
                row["min_disparate_impact"],
            )
    return detail
