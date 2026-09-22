"""Model definitions, training, persistence and the shared predictions table.

Owners: Member 1 (logistic regression), Member 2 (XGBoost), Member 3 (TabPFN).

Models are added through the :func:`register` decorator, so a new model needs a
builder here and an entry in ``config.yaml`` — never an edit to ``pipeline.py``.

The important output of this module is not the fitted estimators but
:func:`build_predictions_table`. Economics, fairness, stability and every app
page that moves a threshold read that one table instead of re-running
inference, which is what makes the sliders in the app instant.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Config, load_config
from src.data import DataBundle

logger = logging.getLogger(__name__)

# Set before TabPFN is imported anywhere, so these hold no matter how the
# pipeline is launched (make, docker, pytest, a notebook).
#   - telemetry: this is coursework on personal data; nothing phones home.
#   - CPU limit: TabPFN refuses >1000 rows on CPU by default. We exceed that
#     deliberately and with a measured budget (see config.yaml).
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

Builder = Callable[[Config, "ModelSpecLike"], Any]
_REGISTRY: dict[str, Builder] = {}


class ModelSpecLike:
    """Structural stand-in for the model section of the config."""

    enabled: bool
    owner: str
    params: dict[str, Any]


def register(name: str) -> Callable[[Builder], Builder]:
    """Register a model builder under the key used in ``config.yaml``."""

    def decorator(builder: Builder) -> Builder:
        _REGISTRY[name] = builder
        return builder

    return decorator


def available_models() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class FittedModel:
    """A trained estimator plus what is needed to reproduce and explain it."""

    name: str
    estimator: Any
    feature_names: list[str]
    owner: str = ""
    train_seconds: float = 0.0
    n_train_rows: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class, as a 1-D array."""
        aligned = X.reindex(columns=self.feature_names, fill_value=0.0)
        return np.asarray(self.estimator.predict_proba(aligned))[:, 1]

    @property
    def is_linear(self) -> bool:
        """True when coefficients can be read directly as odds ratios."""
        return self.name == "logistic_regression"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


@register("logistic_regression")
def build_logistic_regression(cfg: Config, spec: Any) -> Pipeline:
    """White-box baseline. Scaled so coefficients are comparable across features."""
    params = dict(spec.params)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(random_state=cfg.seed, n_jobs=None, **params),
            ),
        ]
    )


@register("xgboost")
def build_xgboost(cfg: Config, spec: Any) -> Any:
    from xgboost import XGBClassifier

    params = dict(spec.params)
    return XGBClassifier(
        random_state=cfg.seed,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )


class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """sklearn-compatible TabPFN with subsampling and batched inference.

    TabPFN is a fixed-context transformer: it does not scale to 70k training
    rows on CPU. We fit on a stratified subsample of ``train_subsample`` rows
    but predict on the *full* test set, so its row in the scorecard is directly
    comparable with the other two models. The subsample size is reported in the
    metadata and must be stated in the deck.
    """

    def __init__(
        self,
        train_subsample: int = 10000,
        predict_batch_size: int = 2000,
        random_state: int = 42,
        **params: Any,
    ) -> None:
        self.train_subsample = train_subsample
        self.predict_batch_size = predict_batch_size
        self.random_state = random_state
        self.params = params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TabPFNWrapper:
        from tabpfn import TabPFNClassifier

        X = pd.DataFrame(X)
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        if len(X) > self.train_subsample:
            rng = np.random.default_rng(self.random_state)
            # Stratified subsample: keep the class balance of the full training set.
            keep: list[np.ndarray] = []
            for label in np.unique(y):
                idx = np.flatnonzero(y.to_numpy() == label)
                share = max(1, round(self.train_subsample * len(idx) / len(y)))
                keep.append(rng.choice(idx, size=min(share, len(idx)), replace=False))
            selected = np.sort(np.concatenate(keep))
            X, y = X.iloc[selected], y.iloc[selected]
            logger.info("TabPFN fitting on a %d-row stratified subsample", len(X))

        self.classes_ = np.unique(y)
        self.n_fit_rows_ = len(X)
        self._model = TabPFNClassifier(random_state=self.random_state, **self.params)
        self._model.fit(X.to_numpy(dtype=float), y.to_numpy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        values = pd.DataFrame(X).to_numpy(dtype=float)
        batches = [
            self._model.predict_proba(values[start : start + self.predict_batch_size])
            for start in range(0, len(values), self.predict_batch_size)
        ]
        return np.vstack(batches)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


@register("tabpfn")
def build_tabpfn(cfg: Config, spec: Any) -> TabPFNWrapper:
    return TabPFNWrapper(
        train_subsample=int(getattr(spec, "train_subsample", 10000)),
        predict_batch_size=int(getattr(spec, "predict_batch_size", 2000)),
        random_state=cfg.seed,
        **dict(spec.params),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one(
    name: str,
    cfg: Config,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> FittedModel:
    """Build and fit a single registered model."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {available_models()}")
    spec = cfg.models[name]
    estimator = _REGISTRY[name](cfg, spec)

    started = time.perf_counter()
    if sample_weight is not None and name != "tabpfn":
        # TabPFN has no sample-weight interface; reweighing mitigation is
        # reported as not applicable for it rather than silently ignored.
        fit_kwargs = (
            {"clf__sample_weight": sample_weight}
            if isinstance(estimator, Pipeline)
            else {"sample_weight": sample_weight}
        )
        estimator.fit(X, y, **fit_kwargs)
    else:
        estimator.fit(X, y)
    elapsed = time.perf_counter() - started

    fitted = FittedModel(
        name=name,
        estimator=estimator,
        feature_names=list(X.columns),
        owner=getattr(spec, "owner", ""),
        train_seconds=round(elapsed, 2),
        n_train_rows=int(getattr(estimator, "n_fit_rows_", len(X))),
        metadata={"params": dict(spec.params), "weighted": sample_weight is not None},
    )
    logger.info("Trained %s on %d rows in %.1fs", name, fitted.n_train_rows, fitted.train_seconds)
    return fitted


def train_all(
    cfg: Config | None = None,
    bundle: DataBundle | None = None,
    sample_weight: np.ndarray | None = None,
) -> dict[str, FittedModel]:
    """Train every model marked ``enabled: true`` in the config."""
    cfg = cfg or load_config()
    if bundle is None:
        from src.data import build_datasets

        bundle = build_datasets(cfg)

    fitted: dict[str, FittedModel] = {}
    for name in cfg.enabled_models:
        try:
            fitted[name] = train_one(name, cfg, bundle.X_train, bundle.y_train, sample_weight)
        except Exception as exc:  # noqa: BLE001 - one bad model must not sink the run
            logger.error("Model %r failed to train and was skipped: %s", name, exc)
    if not fitted:
        raise RuntimeError("No model trained successfully; check the log above.")
    return fitted


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save(fitted: FittedModel, cfg: Config) -> Path:
    path = cfg.paths.models_dir / f"{fitted.name}.joblib"
    joblib.dump(fitted, path)
    return path


def load(name: str, cfg: Config) -> FittedModel:
    path = cfg.paths.models_dir / f"{name}.joblib"
    if not path.is_file():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `make pipeline` (or `make docker-pipeline`) first."
        )
    return joblib.load(path)


def load_all(cfg: Config) -> dict[str, FittedModel]:
    """Load every model present on disk, skipping any that are missing."""
    found = {}
    for name in cfg.enabled_models:
        try:
            found[name] = load(name, cfg)
        except FileNotFoundError:
            logger.warning("Model %r not found on disk; skipping.", name)
    return found


# ---------------------------------------------------------------------------
# The shared predictions table
# ---------------------------------------------------------------------------


def build_predictions_table(
    models: dict[str, FittedModel], bundle: DataBundle, splits: tuple[str, ...] = ("val", "test")
) -> pd.DataFrame:
    """Long-format predictions: one row per (split, model, observation).

    This is the contract between the pipeline and everything downstream. Columns:
    ``split``, ``model``, ``row_id``, ``y_true``, ``y_prob``, plus one column per
    protected attribute. Any metric that depends only on scores and a threshold —
    profit curves, fairness gaps, flip rates — is a groupby over this frame.
    """
    frames: list[pd.DataFrame] = []
    for split_name in splits:
        X, y, protected = bundle.split(split_name)
        for model_name, fitted in models.items():
            frame = pd.DataFrame(
                {
                    "split": split_name,
                    "model": model_name,
                    "row_id": np.arange(len(X)),
                    "y_true": y.to_numpy(),
                    "y_prob": fitted.predict_proba(X),
                }
            )
            for column in protected.columns:
                frame[column] = protected[column].to_numpy()
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistical performance
# ---------------------------------------------------------------------------


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Threshold-free and thresholded statistical performance in one dict."""
    y_pred = (y_prob >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float("nan")
        if single_class
        else float(log_loss(y_true, y_prob, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
    }


def scorecard(
    predictions: pd.DataFrame, models: dict[str, FittedModel] | None = None, split: str = "test"
) -> pd.DataFrame:
    """Statistical performance of every model on one split, one row per model."""
    rows = []
    subset = predictions[predictions["split"] == split]
    for model_name, group in subset.groupby("model", sort=False):
        record: dict[str, Any] = {"model": model_name, "split": split, "n": len(group)}
        record.update(
            classification_metrics(group["y_true"].to_numpy(), group["y_prob"].to_numpy())
        )
        if models and model_name in models:
            record["train_seconds"] = models[model_name].train_seconds
            record["n_train_rows"] = models[model_name].n_train_rows
            record["owner"] = models[model_name].owner
        rows.append(record)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
