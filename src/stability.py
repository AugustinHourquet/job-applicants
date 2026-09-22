"""Stability: does the model say the same thing when the world wobbles slightly?

Owner: Member 4.

A model can be accurate and still be unfit to deploy. If re-fitting on a
slightly different sample, or nudging an input within measurement noise, flips
a candidate's decision, then the client cannot explain that decision to the
candidate — and cannot defend it if challenged.

Three probes, in increasing order of how much they worry a client:

1. **Resampling.** Re-fit on bootstrap samples of the training data. How often
   does an individual's decision flip purely because of which rows we happened
   to train on?
2. **Perturbation.** Add noise to the numeric inputs at the scale of plausible
   measurement error. How often does a decision flip?
3. **Shift.** Score held-out subgroups separately. Does performance hold up
   away from the bulk of the training distribution?

Cost control: re-fitting is cheap for logistic regression and XGBoost and
expensive for TabPFN, so ``stability.max_boot_per_model`` caps the budget per
model and the output records how many re-fits each number actually rests on.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config
from src.data import DataBundle
from src.models import FittedModel, train_one

logger = logging.getLogger(__name__)


def budget(cfg: Config, model_name: str, key: str, default_key: str | None = None) -> int:
    """One stability setting for one model, after any per-model reduction.

    Inference cost differs by orders of magnitude across the three models —
    logistic regression scores a thousand rows in milliseconds, TabPFN needs
    about seventy-five seconds. ``stability.budget_per_model`` lets the
    expensive one run a smaller probe rather than forcing everyone to wait for
    the slowest, and the reduced value is written into the output so a smaller
    sample is visible rather than hidden.

    A per-model value never *raises* the global setting, only lowers it.
    """
    overrides = (getattr(cfg.stability, "budget_per_model", {}) or {}).get(model_name, {})
    settings = cfg.stability
    fallback_key = default_key or key

    if fallback_key == "perturbation_repeats":
        global_value = int((settings.perturbation or {}).get("n_repeats", 10))
    elif fallback_key == "shift_eval_subsample":
        global_value = int((settings.shift or {}).get("eval_subsample", 6000))
    else:
        global_value = int(getattr(settings, fallback_key))

    return int(min(global_value, overrides.get(key, global_value)))


def _eval_slice(
    cfg: Config, bundle: DataBundle, model_name: str, key: str = "eval_subsample"
) -> tuple[pd.DataFrame, pd.Series]:
    """Fixed random slice of the test set that a probe is measured on.

    Seeded, so every model sees the same rows and flip rates stay comparable;
    a model on a reduced budget sees a subset of that same slice rather than a
    different draw.
    """
    X, y, _ = bundle.split("test")
    size = budget(cfg, model_name, key)
    if size >= len(X):
        return X, y
    rng = np.random.default_rng(cfg.seed)
    idx = np.sort(rng.choice(len(X), size=size, replace=False))
    return X.iloc[idx], y.iloc[idx]


# Kept for callers that only want the re-fit count.
def _budget(cfg: Config, model_name: str) -> int:
    return budget(cfg, model_name, "n_boot")


def bootstrap_stability(
    cfg: Config, bundle: DataBundle, model_name: str, reference: FittedModel
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-fit on bootstrap resamples and compare each re-fit to the reference.

    Returns ``(per_row, summary)``. ``per_row`` carries one row per evaluated
    observation with its probability spread and flip rate, which is what the
    app's flip-rate distribution chart plots.
    """
    from sklearn.metrics import roc_auc_score

    X_eval, y_eval = _eval_slice(cfg, bundle, model_name)
    threshold = float(cfg.stability.flip_threshold)
    reference_prob = reference.predict_proba(X_eval)
    reference_decision = (reference_prob >= threshold).astype(int)

    n_boot = _budget(cfg, model_name)
    frac = float(cfg.stability.bootstrap_frac)
    rng = np.random.default_rng(cfg.seed)

    probs: list[np.ndarray] = []
    aucs: list[float] = []
    n_train = len(bundle.X_train)
    size = max(2, int(frac * n_train))

    for i in range(n_boot):
        idx = rng.choice(n_train, size=size, replace=True)
        X_boot = bundle.X_train.iloc[idx]
        y_boot = bundle.y_train.iloc[idx]
        if y_boot.nunique() < 2:
            continue
        try:
            refit = train_one(model_name, cfg, X_boot, y_boot)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bootstrap %d for %s failed: %s", i, model_name, exc)
            continue
        p = refit.predict_proba(X_eval)
        probs.append(p)
        if y_eval.nunique() > 1:
            aucs.append(float(roc_auc_score(y_eval, p)))

    if not probs:
        return pd.DataFrame(), pd.DataFrame()

    matrix = np.vstack(probs)  # (n_boot, n_eval)
    decisions = (matrix >= threshold).astype(int)
    flips = (decisions != reference_decision[None, :]).mean(axis=0)

    per_row = pd.DataFrame(
        {
            "model": model_name,
            "row_id": np.arange(len(X_eval)),
            "y_true": y_eval.to_numpy(),
            "reference_prob": reference_prob,
            "mean_prob": matrix.mean(axis=0),
            "std_prob": matrix.std(axis=0),
            "prob_range": matrix.max(axis=0) - matrix.min(axis=0),
            "flip_rate": flips,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "probe": "bootstrap",
                "n_refits": len(probs),
                "n_evaluated": len(X_eval),
                "mean_flip_rate": float(flips.mean()),
                "share_rows_ever_flipped": float((flips > 0).mean()),
                "share_rows_flipped_often": float((flips >= 0.10).mean()),
                "mean_prob_std": float(matrix.std(axis=0).mean()),
                "auc_mean": float(np.mean(aucs)) if aucs else np.nan,
                "auc_std": float(np.std(aucs)) if aucs else np.nan,
            }
        ]
    )
    return per_row, summary


def perturbation_stability(
    cfg: Config, bundle: DataBundle, model_name: str, reference: FittedModel
) -> pd.DataFrame:
    """Flip rate under Gaussian noise on the numeric inputs.

    Noise is scaled to each feature's own standard deviation, so the probe asks
    a sensible question of a salary and of a years-of-experience count alike.
    """
    settings = cfg.stability.perturbation or {}
    noise_sd = float(settings.get("numeric_noise_sd", 0.05))
    n_repeats = budget(cfg, model_name, "perturbation_repeats")

    X_eval, y_eval = _eval_slice(cfg, bundle, model_name)
    threshold = float(cfg.stability.flip_threshold)
    reference_decision = (reference.predict_proba(X_eval) >= threshold).astype(int)

    numeric = [c for c in bundle.spec.numeric_features if c in X_eval.columns]
    scales = bundle.X_train[numeric].std().replace(0, 1.0)
    rng = np.random.default_rng(cfg.seed)

    flip_rates = []
    for _ in range(n_repeats):
        perturbed = X_eval.copy()
        noise = rng.normal(0.0, 1.0, size=(len(perturbed), len(numeric))) * (
            noise_sd * scales.to_numpy()
        )
        perturbed[numeric] = perturbed[numeric].to_numpy() + noise
        decision = (reference.predict_proba(perturbed) >= threshold).astype(int)
        flip_rates.append(float((decision != reference_decision).mean()))

    return pd.DataFrame(
        [
            {
                "model": model_name,
                "probe": "perturbation",
                "n_refits": 0,
                "n_evaluated": len(X_eval),
                "noise_sd_fraction": noise_sd,
                "n_repeats": n_repeats,
                "n_evaluated_rows": len(X_eval),
                "mean_flip_rate": float(np.mean(flip_rates)),
                "max_flip_rate": float(np.max(flip_rates)),
            }
        ]
    )


def shift_stability(
    cfg: Config, bundle: DataBundle, model_name: str, reference: FittedModel, raw_test: pd.DataFrame
) -> pd.DataFrame:
    """Per-subgroup performance, as a stand-in for distribution shift.

    Splitting the test set by a high-cardinality attribute such as country and
    scoring each slice separately shows whether a headline AUC is carried by the
    bulk of the data while smaller populations are served much worse.
    """
    from sklearn.metrics import roc_auc_score

    settings = cfg.stability.shift or {}
    by = settings.get("by")
    min_size = int(settings.get("min_group_size", 200))
    if not by or by not in raw_test.columns:
        return pd.DataFrame()

    X_test, y_test, _ = bundle.split("test")
    # Scoring the whole test set here was the single most expensive thing in the
    # pipeline: it asked the slowest model for ~14,700 predictions to answer a
    # question a sample answers just as well. Subgroups still have to clear
    # min_group_size after sampling, so undersized slices are dropped rather
    # than reported from a handful of rows.
    size = budget(cfg, model_name, "shift_eval_subsample")
    if size < len(X_test):
        rng = np.random.default_rng(cfg.seed)
        idx = np.sort(rng.choice(len(X_test), size=size, replace=False))
        X_test = X_test.iloc[idx]
        y_test = y_test.iloc[idx]
        raw_test = raw_test.iloc[idx]

    probs = reference.predict_proba(X_test)
    frame = pd.DataFrame(
        {"group": raw_test[by].astype(str).to_numpy(), "y_true": y_test.to_numpy(), "y_prob": probs}
    )

    rows = []
    for group_value, group in frame.groupby("group", sort=True):
        if len(group) < min_size or group["y_true"].nunique() < 2:
            continue
        rows.append(
            {
                "model": model_name,
                "probe": "shift",
                "attribute": by,
                "group": group_value,
                "n": len(group),
                "n_scored": len(frame),
                "base_rate": float(group["y_true"].mean()),
                "roc_auc": float(roc_auc_score(group["y_true"], group["y_prob"])),
                "mean_prob": float(group["y_prob"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run(
    cfg: Config,
    bundle: DataBundle,
    models: dict[str, FittedModel],
    raw_test: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pipeline entry point: run all three probes, write the CSVs, return the summary."""
    summaries: list[pd.DataFrame] = []
    per_rows: list[pd.DataFrame] = []
    shifts: list[pd.DataFrame] = []

    for name, fitted in models.items():
        logger.info("Stability: %s (%d bootstrap re-fits)", name, _budget(cfg, name))
        rows, summary = bootstrap_stability(cfg, bundle, name, fitted)
        if not rows.empty:
            per_rows.append(rows)
            summaries.append(summary)
        summaries.append(perturbation_stability(cfg, bundle, name, fitted))
        if raw_test is not None:
            shift = shift_stability(cfg, bundle, name, fitted, raw_test)
            if not shift.empty:
                shifts.append(shift)

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary.to_csv(cfg.paths.results_dir / "stability.csv", index=False)
    if per_rows:
        pd.concat(per_rows, ignore_index=True).to_csv(
            cfg.paths.results_dir / "stability_per_row.csv", index=False
        )
    if shifts:
        pd.concat(shifts, ignore_index=True).to_csv(
            cfg.paths.results_dir / "stability_shift.csv", index=False
        )
    return summary
