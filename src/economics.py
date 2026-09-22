"""Economic performance: what a model is worth to the client, in euros.

Owner: Member 2.

Statistical performance answers "is the ranking good". This module answers "at
what cut-off do we act, and what does acting there earn or cost". The two can
disagree sharply: the model with the best AUC is not always the most profitable
once the asymmetry between a wasted placement and a missed one is priced in.

Methodology note for the deck: the operating threshold is chosen on the
**validation** split and then applied unchanged to the **test** split. Picking
the threshold on test and reporting profit at that same threshold would be
optimistic, in exactly the way that gets a model deployed and then disappoints.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """True/false positive and negative counts, without sklearn's shape edge cases."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
    }


def profit(y_true: np.ndarray, y_pred: np.ndarray, cost_matrix: dict[str, float]) -> float:
    """Total euro value of a set of decisions under the client's cost matrix."""
    counts = confusion_counts(y_true, y_pred)
    return float(
        counts["tp"] * cost_matrix["true_positive"]
        + counts["fp"] * cost_matrix["false_positive"]
        + counts["fn"] * cost_matrix["false_negative"]
        + counts["tn"] * cost_matrix["true_negative"]
    )


def threshold_grid(cfg: Config) -> np.ndarray:
    grid = cfg.economics.threshold_grid
    return np.arange(grid["start"], grid["stop"] + grid["step"] / 2, grid["step"])


def profit_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_matrix: dict[str, float],
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Profit at every candidate threshold, plus the confusion counts behind it."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        counts = confusion_counts(y_true, y_pred)
        total = profit(y_true, y_pred, cost_matrix)
        rows.append(
            {
                "threshold": round(float(t), 4),
                **counts,
                "selection_rate": counts["tp"] + counts["fp"],
                "profit": total,
                "profit_per_candidate": total / n if n else 0.0,
            }
        )
    curve = pd.DataFrame(rows)
    curve["selection_rate"] = curve["selection_rate"] / n
    return curve


def optimal_threshold(curve: pd.DataFrame) -> float:
    """Threshold maximising profit. Ties break toward the more selective cut-off."""
    best = curve["profit"].max()
    return float(curve.loc[curve["profit"] == best, "threshold"].max())


def baseline_profits(y_true: np.ndarray, cost_matrix: dict[str, float]) -> dict[str, float]:
    """What the client earns without a model, as the bar every model must clear.

    ``accept_all`` is the status quo for a platform that coaches everyone;
    ``accept_none`` is doing nothing. A model that beats neither is not worth
    deploying regardless of its AUC.
    """
    y_true = np.asarray(y_true).astype(int)
    return {
        "accept_all": profit(y_true, np.ones_like(y_true), cost_matrix),
        "accept_none": profit(y_true, np.zeros_like(y_true), cost_matrix),
    }


def evaluate(cfg: Config, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profit curves and an operating-point summary for every model.

    Returns ``(curves, summary)``. ``curves`` is long-format over model and
    split, ready for a Plotly line chart. ``summary`` carries one row per model:
    the threshold chosen on validation, the profit it earns on test, and how far
    that sits above the best no-model baseline.
    """
    cost_matrix = cfg.economics.cost_matrix
    thresholds = threshold_grid(cfg)

    curve_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str]] = []

    for model_name, model_rows in predictions.groupby("model", sort=False):
        per_split: dict[str, pd.DataFrame] = {}
        for split_name, split_rows in model_rows.groupby("split", sort=False):
            curve = profit_curve(
                split_rows["y_true"].to_numpy(),
                split_rows["y_prob"].to_numpy(),
                cost_matrix,
                thresholds,
            )
            curve.insert(0, "model", model_name)
            curve.insert(1, "split", split_name)
            per_split[str(split_name)] = curve
            curve_frames.append(curve)

        # Choose on validation when it exists; fall back to test only if it does not.
        selection_split = "val" if "val" in per_split else next(iter(per_split))
        chosen = optimal_threshold(per_split[selection_split])

        report_split = "test" if "test" in per_split else selection_split
        report_curve = per_split[report_split]
        at_threshold = report_curve.iloc[(report_curve["threshold"] - chosen).abs().argmin()]

        report_rows = model_rows[model_rows["split"] == report_split]
        baselines = baseline_profits(report_rows["y_true"].to_numpy(), cost_matrix)
        best_baseline = max(baselines.values())

        # What the threshold would have earned if we had been allowed to cheat
        # and pick it on the test set: the size of the honest-methodology gap.
        oracle = float(report_curve["profit"].max())

        summary_rows.append(
            {
                "model": model_name,
                "threshold_split": selection_split,
                "optimal_threshold": chosen,
                "report_split": report_split,
                "profit": float(at_threshold["profit"]),
                "profit_per_candidate": float(at_threshold["profit_per_candidate"]),
                "selection_rate": float(at_threshold["selection_rate"]),
                "tp": int(at_threshold["tp"]),
                "fp": int(at_threshold["fp"]),
                "fn": int(at_threshold["fn"]),
                "tn": int(at_threshold["tn"]),
                "baseline_accept_all": baselines["accept_all"],
                "baseline_accept_none": baselines["accept_none"],
                "uplift_vs_best_baseline": float(at_threshold["profit"]) - best_baseline,
                "oracle_profit": oracle,
                "cost_of_honest_threshold": oracle - float(at_threshold["profit"]),
                "currency": cfg.economics.currency,
            }
        )

    curves = pd.concat(curve_frames, ignore_index=True)
    summary = (
        pd.DataFrame(summary_rows).sort_values("profit", ascending=False).reset_index(drop=True)
    )
    return curves, summary


def run(cfg: Config, predictions: pd.DataFrame) -> pd.DataFrame:
    """Pipeline entry point: evaluate, write both CSVs, return the summary."""
    curves, summary = evaluate(cfg, predictions)
    curves.to_csv(cfg.paths.results_dir / "profit_curves.csv", index=False)
    summary.to_csv(cfg.paths.results_dir / "economics.csv", index=False)
    best = summary.iloc[0]
    logger.info(
        "Economics: best model %r at threshold %.2f — %s %s on %s",
        best["model"],
        best["optimal_threshold"],
        cfg.economics.currency,
        f"{best['profit']:,.0f}",
        best["report_split"],
    )
    return summary
