"""Run the whole analysis in order: `python -m src.pipeline`.

This module orchestrates and nothing else. Every step lives in the module whose
owner is responsible for it, so adding a model or a metric means editing that
module — not this one. Keeping it that way is what lets six people work at once.

Order matters in one place: the operating threshold is chosen by
:mod:`src.economics` on the validation split, and :mod:`src.fairness` then audits
at that threshold. Auditing at 0.5 while deploying at 0.37 would describe a
system nobody is running.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import pandas as pd

from src import data, economics, fairness, interpretability, models, plots, stability
from src.config import Config, load_config

logger = logging.getLogger(__name__)

STEPS = ("data", "models", "economics", "fairness", "interpretability", "stability", "figures")


def build_master_scorecard(
    statistical: pd.DataFrame,
    economic: pd.DataFrame,
    stability_summary: pd.DataFrame,
    fairness_summary: pd.DataFrame,
) -> pd.DataFrame:
    """One row per model covering all four dimensions — the deck's centrepiece.

    Deliberately not collapsed into a single score. The whole point of the
    exercise is that the four dimensions trade off against each other and the
    client has to choose; a weighted average would hide exactly the judgement
    they are being asked to make.
    """
    board = statistical[["model", "roc_auc", "pr_auc", "brier", "accuracy", "f1"]].copy()

    if not economic.empty:
        board = board.merge(
            economic[
                [
                    "model",
                    "optimal_threshold",
                    "profit",
                    "profit_per_candidate",
                    "uplift_vs_best_baseline",
                ]
            ],
            on="model",
            how="left",
        )

    if not stability_summary.empty:
        bootstrap = stability_summary[stability_summary["probe"] == "bootstrap"]
        if not bootstrap.empty:
            board = board.merge(
                bootstrap[
                    ["model", "mean_flip_rate", "share_rows_flipped_often", "auc_std"]
                ].rename(columns={"mean_flip_rate": "bootstrap_flip_rate"}),
                on="model",
                how="left",
            )
        perturbation = stability_summary[stability_summary["probe"] == "perturbation"]
        if not perturbation.empty:
            board = board.merge(
                perturbation[["model", "mean_flip_rate"]].rename(
                    columns={"mean_flip_rate": "perturbation_flip_rate"}
                ),
                on="model",
                how="left",
            )

    if not fairness_summary.empty:
        worst = (
            fairness_summary.groupby("model")
            .agg(
                worst_selection_gap=("max_abs_selection_rate_gap", "max"),
                worst_tpr_gap=("max_abs_true_positive_rate_gap", "max"),
                min_disparate_impact=("min_disparate_impact", "min"),
            )
            .reset_index()
        )
        board = board.merge(worst, on="model", how="left")
        board["four_fifths_pass"] = board["min_disparate_impact"] >= 0.8

    return board.sort_values("roc_auc", ascending=False).reset_index(drop=True)


def export_figures(cfg: Config, artefacts: dict[str, Any]) -> int:
    """Write PNGs of the main charts for the slide deck. Never fails the run."""
    figures_dir = cfg.paths.figures_dir
    thresholds = artefacts.get("thresholds", {})

    def frame(key: str) -> pd.DataFrame | None:
        value = artefacts.get(key)
        return value if isinstance(value, pd.DataFrame) and not value.empty else None

    jobs: list[tuple[str, Any]] = []
    if (curves := frame("profit_curves")) is not None:
        jobs.append(("profit_curves.png", plots.profit_curve_chart(curves, chosen=thresholds)))
    if (predictions := frame("predictions")) is not None:
        jobs.append(("roc.png", plots.roc_chart(predictions)))
    if (importance := frame("importance")) is not None:
        jobs += [
            (f"importance_{name}.png", plots.importance_chart(importance, name))
            for name in importance["model"].unique()
        ]
    if (per_row := frame("stability_per_row")) is not None:
        jobs.append(("flip_rates.png", plots.flip_rate_chart(per_row)))

    written = sum(1 for name, figure in jobs if plots.save_png(figure, figures_dir / name))
    if written == 0:
        logger.warning(
            "No figures exported. PNG export needs kaleido; the app's charts are "
            "unaffected because they render in the browser."
        )
    return written


def run_pipeline(
    cfg: Config | None = None,
    raw: pd.DataFrame | None = None,
    skip: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Execute every stage and return the artefacts, which are also written to disk.

    Args:
        cfg: Project config; loaded from ``config.yaml`` when omitted.
        raw: Pre-loaded raw frame, used by the smoke test to avoid Kaggle.
        skip: Stage names to skip, for iterating on one part without paying for
            the expensive ones.
    """
    cfg = cfg or load_config()
    started = time.perf_counter()
    artefacts: dict[str, Any] = {}

    logger.info("[1/6] Data")
    bundle = data.build_datasets(cfg, raw=raw)
    data.persist(bundle, cfg)
    artefacts["bundle"] = bundle
    artefacts["checks"] = bundle.checks

    logger.info("[2/6] Models")
    fitted = models.train_all(cfg, bundle)
    for model in fitted.values():
        models.save(model, cfg)
    predictions = models.build_predictions_table(fitted, bundle)
    predictions.to_parquet(cfg.paths.results_dir / "predictions.parquet", index=False)
    statistical = models.scorecard(predictions, fitted)
    statistical.to_csv(cfg.paths.results_dir / "statistical.csv", index=False)
    artefacts.update(models=fitted, predictions=predictions, statistical=statistical)

    logger.info("[3/6] Economics")
    economic = pd.DataFrame()
    thresholds: dict[str, float] = dict.fromkeys(fitted, 0.5)
    if "economics" not in skip:
        economic = economics.run(cfg, predictions)
        thresholds = dict(zip(economic["model"], economic["optimal_threshold"], strict=False))
        artefacts["profit_curves"] = pd.read_csv(cfg.paths.results_dir / "profit_curves.csv")
    artefacts.update(economics=economic, thresholds=thresholds)

    logger.info("[4/6] Fairness")
    fairness_detail = pd.DataFrame()
    fairness_summary = pd.DataFrame()
    if "fairness" not in skip:
        fairness_detail = fairness.run(cfg, predictions, thresholds)
        fairness_summary = fairness.summarise(fairness_detail, cfg.fairness.metrics)
    artefacts.update(fairness=fairness_detail, fairness_summary=fairness_summary)

    logger.info("[5/6] Interpretability")
    importance = pd.DataFrame()
    if "interpretability" not in skip:
        importance = interpretability.run(cfg, bundle, fitted)
    artefacts["importance"] = importance

    logger.info("[6/6] Stability")
    stability_summary = pd.DataFrame()
    if "stability" not in skip:
        stability_summary = stability.run(cfg, bundle, fitted, raw_test=bundle.raw_test)
        per_row_path = cfg.paths.results_dir / "stability_per_row.csv"
        if per_row_path.is_file():
            artefacts["stability_per_row"] = pd.read_csv(per_row_path)
    artefacts["stability"] = stability_summary

    scorecard = build_master_scorecard(statistical, economic, stability_summary, fairness_summary)
    scorecard.to_csv(cfg.paths.results_dir / "scorecard.csv", index=False)
    artefacts["scorecard"] = scorecard

    if "figures" not in skip:
        export_figures(cfg, artefacts)

    elapsed = time.perf_counter() - started
    logger.info("Pipeline finished in %.1fs. Artifacts in %s", elapsed, cfg.paths.results_dir)
    return artefacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full scoring analysis.")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=STEPS,
        help="Stages to skip, for iterating without paying for the slow ones.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict to these models, overriding config.yaml.",
    )
    parser.add_argument("--config", default=None, help="Path to an alternative config file.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    if args.models:
        for name, spec in cfg.models.items():
            spec.enabled = name in args.models

    artefacts = run_pipeline(cfg, skip=tuple(args.skip))

    print("\n" + "=" * 78)
    print("MASTER SCORECARD".center(78))
    print("=" * 78)
    print(artefacts["scorecard"].round(4).to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    main()
