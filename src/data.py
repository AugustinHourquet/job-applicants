"""Data acquisition, cleaning, quality checks, feature building and splitting.

Owner: Member 1.

The public entry point is :func:`build_datasets`. Everything downstream — the
three models, and the economics, interpretability, stability and fairness
modules — consumes the :class:`DataBundle` it returns, so the contract here is
what keeps six people working in parallel.

Encoding is done in plain pandas against an explicit, serialised
:class:`FeatureSpec` rather than a fitted sklearn transformer. Two reasons:
the Streamlit "score a candidate" form can encode a single row with the exact
same rules, and the resulting column names stay human-readable, which matters
when SHAP and odds ratios are shown to a non-technical client.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import Config, load_config

logger = logging.getLogger(__name__)

SPEC_FILENAME = "feature_spec.json"


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """Everything needed to turn a cleaned raw row into a model-ready row.

    Fitted on the training split only — deriving category levels or a
    technology vocabulary from validation or test data would leak.
    """

    categorical_levels: dict[str, list[str]]
    reference_levels: dict[str, str]
    numeric_features: list[str]
    numeric_medians: dict[str, float]
    tech_vocabulary: dict[str, list[str]]
    rare_level_name: str
    feature_names: list[str]

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> FeatureSpec:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class DataBundle:
    """The split, encoded dataset plus what is needed to interpret it.

    ``X_*`` are numeric and model-ready. ``protected_*`` hold the *raw*
    categorical values of the protected attributes, aligned row-for-row with
    ``X_*``, so the fairness module can group by them without reverse-engineering
    one-hot columns.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    protected_train: pd.DataFrame
    protected_val: pd.DataFrame
    protected_test: pd.DataFrame
    spec: FeatureSpec
    checks: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Cleaned but un-encoded test rows. The stability module groups by them to
    # probe distribution shift, and the app shows real candidates in the form.
    raw_test: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)

    def split(self, name: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Return ``(X, y, protected)`` for ``"train"``, ``"val"`` or ``"test"``."""
        if name not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {name!r}; expected train, val or test.")
        return (
            getattr(self, f"X_{name}"),
            getattr(self, f"y_{name}"),
            getattr(self, f"protected_{name}"),
        )

    def describe(self) -> str:
        return (
            f"train={len(self.X_train):,}  val={len(self.X_val):,}  "
            f"test={len(self.X_test):,}  features={len(self.feature_names)}  "
            f"positive_rate={self.y_train.mean():.3f}"
        )


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def download_raw(cfg: Config, *, force: bool = False) -> Path:
    """Fetch the dataset from Kaggle into ``data/raw/``.

    Credentials come from ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` (see .env.example).
    Idempotent: does nothing if the file is already present.
    """
    destination = cfg.raw_path
    if destination.is_file() and not force:
        logger.info("Raw data already present at %s", destination)
        return destination

    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise RuntimeError(
            "Kaggle credentials not found. Copy .env.example to .env and fill in "
            "KAGGLE_USERNAME and KAGGLE_KEY (https://www.kaggle.com/settings -> "
            "API -> Create New Token), then re-run `make data`."
        )

    import kagglehub  # imported lazily: the smoke test must not need Kaggle

    logger.info("Downloading %s from Kaggle...", cfg.data.kaggle_dataset)
    cache_dir = Path(kagglehub.dataset_download(cfg.data.kaggle_dataset))

    matches = list(cache_dir.rglob(cfg.data.raw_filename))
    if not matches:
        available = sorted(p.name for p in cache_dir.rglob("*.csv"))
        raise FileNotFoundError(
            f"{cfg.data.raw_filename!r} not found in the downloaded dataset. "
            f"CSV files present: {available}. Update data.raw_filename in config.yaml."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], destination)
    logger.info("Raw data written to %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
    return destination


def load_raw(cfg: Config) -> pd.DataFrame:
    """Read the raw CSV, downloading it first if necessary."""
    path = cfg.raw_path
    if not path.is_file():
        path = download_raw(cfg)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Drop configured columns, validate the schema and coerce types.

    Fails loudly on a missing column rather than silently modelling fewer
    features than the config declares.
    """
    df = df.copy()

    present_drops = [c for c in cfg.data.drop_column_names if c in df.columns]
    df = df.drop(columns=present_drops)
    # The Kaggle export sometimes ships the row index unnamed.
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")

    missing = [c for c in cfg.data.declared_columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Columns declared in config.yaml are absent from the raw data: {missing}. "
            f"Available columns: {sorted(df.columns)}. Fix the data section of config.yaml."
        )

    target = cfg.data.target
    df[target] = pd.to_numeric(df[target], errors="coerce").astype("Int64")
    df = df[df[target].notna()].copy()
    df[target] = df[target].astype(int)
    if not set(df[target].unique()) <= {0, 1}:
        raise ValueError(f"Target {target!r} is not binary: found {sorted(df[target].unique())}.")

    for col in cfg.data.numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in cfg.data.categorical_features:
        df[col] = df[col].astype("string").fillna("Unknown").str.strip()

    for col in cfg.data.multilabel_columns:
        df[col] = df[col].astype("string").fillna("")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        logger.info("Dropped %d exact duplicate rows", before - len(df))

    return df


# ---------------------------------------------------------------------------
# Quality checks: leakage and proxies
# ---------------------------------------------------------------------------


def _single_feature_auc(series: pd.Series, y: pd.Series) -> float:
    """AUC of a single feature against the target, direction-agnostic."""
    if pd.api.types.is_numeric_dtype(series):
        values = series.fillna(series.median())
    else:
        # Encode categories by their target rate, which is the most generous
        # reading a single categorical can get. If even that separates the
        # target almost perfectly, the column is suspect.
        rates = y.groupby(series.astype(str)).mean()
        values = series.astype(str).map(rates).fillna(y.mean())
    if values.nunique() < 2:
        return 0.5
    return float(max(roc_auc_score(y, values), 1 - roc_auc_score(y, values)))


def _normalised_mutual_information(
    a: pd.Series, b: pd.Series, bins: int = 10, max_levels: int = 20
) -> float:
    """Normalised mutual information between two columns, numeric or not.

    Both columns are discretised to at most ``max_levels`` values first. Plug-in
    mutual information is biased upward with cardinality — a near-unique column
    such as a free-text skills list would otherwise score as a near-perfect
    "proxy" for every protected attribute purely as an artefact of the estimator.
    """

    def discretise(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            filled = s.fillna(s.median())
            if filled.nunique() <= bins:
                return filled.astype(str)
            return pd.qcut(filled, q=bins, duplicates="drop").astype(str)
        text = s.astype(str).fillna("Unknown")
        if text.nunique() > max_levels:
            keep = set(text.value_counts().head(max_levels).index)
            text = text.where(text.isin(keep), "__other__")
        return text

    da, db = discretise(a), discretise(b)
    mi = mutual_info_score(da, db)

    # Normalise by the smaller entropy so the score lands in [0, 1].
    def entropy(s: pd.Series) -> float:
        p = s.value_counts(normalize=True).to_numpy()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    denominator = min(entropy(da), entropy(db))
    return float(mi / denominator) if denominator > 0 else 0.0


def run_checks(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Screen every feature for target leakage and for proxying a protected attribute.

    Findings are reported, never auto-applied: dropping a column is a modelling
    decision the team makes and defends, not something a script does silently.
    Feeds the "Data checks" page of the app.
    """
    target = cfg.data.target
    y = df[target]
    protected = [p for p in cfg.fairness.protected_attributes if p in df.columns]
    features = [c for c in df.columns if c != target]

    rows: list[dict[str, Any]] = []
    for col in features:
        auc = _single_feature_auc(df[col], y) if col not in cfg.data.multilabel_columns else np.nan
        record: dict[str, Any] = {
            "feature": col,
            "single_feature_auc": round(auc, 4) if not np.isnan(auc) else None,
            "leakage_flag": bool(auc >= cfg.checks.leakage_auc_threshold)
            if not np.isnan(auc)
            else False,
        }
        worst_proxy, worst_nmi = None, 0.0
        for attribute in protected:
            if col == attribute:
                continue
            nmi = _normalised_mutual_information(df[col], df[attribute])
            record[f"nmi_{attribute}"] = round(nmi, 4)
            if nmi > worst_nmi:
                worst_proxy, worst_nmi = attribute, nmi
        record["strongest_proxy_for"] = worst_proxy
        record["strongest_proxy_nmi"] = round(worst_nmi, 4)
        record["proxy_flag"] = bool(worst_nmi >= cfg.checks.proxy_nmi_threshold)
        rows.append(record)

    report = pd.DataFrame(rows).sort_values(
        "single_feature_auc", ascending=False, na_position="last"
    )

    flagged_leak = report.loc[report["leakage_flag"], "feature"].tolist()
    flagged_proxy = report.loc[report["proxy_flag"], "feature"].tolist()
    if flagged_leak:
        logger.warning(
            "Possible target leakage (single-feature AUC >= %.2f): %s",
            cfg.checks.leakage_auc_threshold,
            flagged_leak,
        )
    if flagged_proxy:
        logger.warning(
            "Possible proxies for a protected attribute (NMI >= %.2f): %s",
            cfg.checks.proxy_nmi_threshold,
            flagged_proxy,
        )

    return report.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _split_tokens(series: pd.Series, separator: str) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.split(separator)
        .apply(lambda parts: [p.strip() for p in parts if p and p.strip()])
    )


def fit_feature_spec(train_df: pd.DataFrame, cfg: Config) -> FeatureSpec:
    """Derive the encoding rules from the training split only."""
    data_cfg = cfg.data
    max_levels = int(getattr(data_cfg, "max_categorical_levels", 20))
    rare_name = str(getattr(data_cfg, "rare_level_name", "Other"))

    categorical_levels: dict[str, list[str]] = {}
    reference_levels: dict[str, str] = {}
    for col in data_cfg.categorical_features:
        counts = train_df[col].astype(str).value_counts()
        kept = counts.head(max_levels).index.tolist()
        if len(counts) > max_levels:
            kept.append(rare_name)
        categorical_levels[col] = sorted(kept)
        # Most frequent level is the reference, dropped from the design matrix.
        # Gives logistic-regression odds ratios a natural "compared to" group.
        reference_levels[col] = str(counts.index[0])

    tech_vocabulary: dict[str, list[str]] = {}
    for col, settings in data_cfg.multilabel_columns.items():
        separator = settings.get("separator", ";")
        top_n = int(settings.get("top_n", 25))
        exploded = _split_tokens(train_df[col], separator).explode().dropna()
        tech_vocabulary[col] = exploded.value_counts().head(top_n).index.tolist()

    numeric_medians = {
        col: float(pd.to_numeric(train_df[col], errors="coerce").median())
        for col in data_cfg.numeric_features
    }

    spec = FeatureSpec(
        categorical_levels=categorical_levels,
        reference_levels=reference_levels,
        numeric_features=list(data_cfg.numeric_features),
        numeric_medians=numeric_medians,
        tech_vocabulary=tech_vocabulary,
        rare_level_name=rare_name,
        feature_names=[],
    )
    # Encode one row to fix the column order, then store it on the spec.
    spec.feature_names = list(apply_feature_spec(train_df.head(1), spec).columns)
    return spec


def apply_feature_spec(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Encode a cleaned frame into the model-ready numeric matrix.

    Works for a 70k-row training frame and for the single row the Streamlit
    form builds, which is what keeps app predictions consistent with training.
    """
    pieces: list[pd.DataFrame] = []

    numeric = pd.DataFrame(index=df.index)
    for col in spec.numeric_features:
        values = pd.to_numeric(df.get(col), errors="coerce")
        numeric[col] = values.fillna(spec.numeric_medians.get(col, 0.0))
    pieces.append(numeric)

    for col, levels in spec.categorical_levels.items():
        raw = df.get(col, pd.Series(index=df.index, dtype="object")).astype(str)
        known = set(levels)
        folded = raw.where(raw.isin(known), spec.rare_level_name)
        reference = spec.reference_levels.get(col)
        dummies = pd.DataFrame(index=df.index)
        for level in levels:
            if level == reference:
                continue  # reference level, deliberately omitted
            dummies[f"{col}={level}"] = (folded == level).astype(int)
        pieces.append(dummies)

    for col, vocabulary in spec.tech_vocabulary.items():
        tokens = _split_tokens(df.get(col, pd.Series(index=df.index, dtype="object")), ";")
        token_sets = tokens.apply(set)
        expanded = pd.DataFrame(index=df.index)
        expanded[f"{col}__n_skills"] = tokens.apply(len)
        for item in vocabulary:
            expanded[f"{col}={item}"] = token_sets.apply(lambda s, i=item: int(i in s))
        pieces.append(expanded)

    encoded = pd.concat(pieces, axis=1)
    if spec.feature_names:
        # Guarantee identical column order to training, filling anything absent.
        encoded = encoded.reindex(columns=spec.feature_names, fill_value=0)
    return encoded.astype(float)


# ---------------------------------------------------------------------------
# Splitting and assembly
# ---------------------------------------------------------------------------


def split_frame(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split."""
    target = cfg.data.target
    stratify = df[target] if cfg.data.split.stratify else None

    train_val, test = train_test_split(
        df,
        test_size=cfg.data.split.test_size,
        random_state=cfg.seed,
        stratify=stratify,
    )
    stratify_tv = train_val[target] if cfg.data.split.stratify else None
    train, val = train_test_split(
        train_val,
        test_size=cfg.data.split.val_size,
        random_state=cfg.seed,
        stratify=stratify_tv,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_datasets(cfg: Config | None = None, raw: pd.DataFrame | None = None) -> DataBundle:
    """Run the whole data stage and return the bundle everything else consumes.

    Args:
        cfg: Project config. Loaded from ``config.yaml`` when omitted.
        raw: Pre-loaded raw frame. Used by the smoke test to run the pipeline on
            synthetic data without touching Kaggle.
    """
    cfg = cfg or load_config()
    raw_df = raw.copy() if raw is not None else load_raw(cfg)
    logger.info("Loaded %d raw rows x %d columns", *raw_df.shape)

    cleaned = clean(raw_df, cfg)
    checks = run_checks(cleaned, cfg)

    train_df, val_df, test_df = split_frame(cleaned, cfg)
    spec = fit_feature_spec(train_df, cfg)

    target = cfg.data.target
    protected = [p for p in cfg.fairness.protected_attributes if p in cleaned.columns]

    bundle = DataBundle(
        X_train=apply_feature_spec(train_df, spec),
        y_train=train_df[target].astype(int),
        X_val=apply_feature_spec(val_df, spec),
        y_val=val_df[target].astype(int),
        X_test=apply_feature_spec(test_df, spec),
        y_test=test_df[target].astype(int),
        protected_train=train_df[protected].astype(str).reset_index(drop=True),
        protected_val=val_df[protected].astype(str).reset_index(drop=True),
        protected_test=test_df[protected].astype(str).reset_index(drop=True),
        spec=spec,
        checks=checks,
        raw_test=test_df.reset_index(drop=True),
    )
    logger.info("Built datasets: %s", bundle.describe())
    return bundle


def persist(bundle: DataBundle, cfg: Config) -> None:
    """Write processed splits, the feature spec and the checks report to disk."""
    processed = cfg.paths.processed_dir
    for name in ("train", "val", "test"):
        X, y, protected = bundle.split(name)
        frame = X.copy()
        frame[cfg.data.target] = y.to_numpy()
        for col in protected.columns:
            frame[f"protected__{col}"] = protected[col].to_numpy()
        frame.to_parquet(processed / f"{name}.parquet", index=False)

    if not bundle.raw_test.empty:
        bundle.raw_test.to_parquet(processed / "test_raw.parquet", index=False)

    bundle.spec.to_json(processed / SPEC_FILENAME)
    bundle.checks.to_csv(cfg.paths.results_dir / "data_checks.csv", index=False)
    logger.info("Processed splits written to %s", processed)


def main() -> None:
    """`make data` — download the raw dataset and report what is in it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    path = download_raw(cfg)
    df = pd.read_csv(path, nrows=5000)
    print(f"\nDownloaded: {path}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nFirst rows:\n{df.head()}")


if __name__ == "__main__":
    main()
