"""Typed access to ``config.yaml``.

Every module reads its settings through :func:`load_config` rather than opening
the YAML itself, so a renamed key fails once, loudly, in one place.

The models below are deliberately permissive (``extra="allow"``): a teammate
adding a key to their own section must never break someone else's import. Keys
that are declared get validation and autocomplete; keys that are not are still
reachable via attribute access.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Which config a bare `load_config()` picks up. The env var lets the Streamlit
# app point at a variant without a code change — `streamlit run` takes no
# pipeline-style flags of its own, and silently showing the wrong variant's
# artifacts is exactly the mistake worth engineering out.
CONFIG_ENV_VAR = "JOBAPP_CONFIG"


def default_config_path() -> Path:
    """Config file used when none is given: ``$JOBAPP_CONFIG`` or config.yaml."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        candidate = Path(override)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return PROJECT_ROOT / "config.yaml"


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class _Section(BaseModel):
    """Base for every config section: validated where declared, open elsewhere."""

    model_config = ConfigDict(extra="allow")


class ProjectSection(_Section):
    name: str = "job-applicants"
    seed: int = 42


class PathsSection(_Section):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("artifacts/models")
    results_dir: Path = Path("artifacts/results")
    figures_dir: Path = Path("artifacts/figures")

    def resolve(self, root: Path) -> None:
        """Rewrite every path as an absolute path under ``root``, in place."""
        for field in self.__class__.model_fields:
            value = getattr(self, field)
            if not value.is_absolute():
                setattr(self, field, root / value)

    def mkdirs(self) -> None:
        """Create every configured directory. Safe to call repeatedly."""
        for field in self.__class__.model_fields:
            getattr(self, field).mkdir(parents=True, exist_ok=True)


class SplitSection(_Section):
    test_size: float = 0.20
    val_size: float = 0.20
    stratify: bool = True


class DataSection(_Section):
    kaggle_dataset: str
    raw_filename: str
    target: str
    drop_columns: list[dict[str, str]] = Field(default_factory=list)
    multilabel_columns: dict[str, dict[str, Any]] = Field(default_factory=dict)
    exclude_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    numeric_features: list[str] = Field(default_factory=list)
    split: SplitSection = Field(default_factory=SplitSection)

    @property
    def declared_columns(self) -> list[str]:
        """Every column the config expects to find in the raw CSV.

        Includes excluded columns: they must still be present and readable, so
        that excluding one is a modelling choice rather than a way to paper over
        a schema mismatch.
        """
        return [
            *self.categorical_features,
            *self.numeric_features,
            *self.multilabel_columns,
            self.target,
        ]

    def modelling_columns(self, kind: str) -> list[str]:
        """Columns of one kind that survive ``exclude_features``."""
        source = {
            "categorical": self.categorical_features,
            "numeric": self.numeric_features,
            "multilabel": list(self.multilabel_columns),
        }[kind]
        excluded = set(self.exclude_features)
        return [c for c in source if c not in excluded]

    @property
    def drop_column_names(self) -> list[str]:
        return [entry["column"] for entry in self.drop_columns]


class ChecksSection(_Section):
    leakage_auc_threshold: float = 0.90
    proxy_nmi_threshold: float = 0.20
    multivariate_leakage_auc_threshold: float = 0.95


class ModelSpec(_Section):
    enabled: bool = True
    owner: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class EconomicsSection(_Section):
    currency: str = "EUR"
    cost_matrix: dict[str, float]
    threshold_grid: dict[str, float]


class InterpretabilitySection(_Section):
    shap_sample: int = 2000
    top_features: int = 20
    local_examples: int = 10


class StabilitySection(_Section):
    n_boot: int = 50
    bootstrap_frac: float = 0.8
    flip_threshold: float = 0.5
    perturbation: dict[str, Any] = Field(default_factory=dict)
    shift: dict[str, Any] = Field(default_factory=dict)


class FairnessSection(_Section):
    protected_attributes: list[str] = Field(default_factory=list)
    reference_groups: dict[str, str | None] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=list)
    min_group_size: int = 100
    mitigation: dict[str, bool] = Field(default_factory=dict)


class AppSection(_Section):
    title: str = "Trustworthy Hiring Scores"
    client_name: str = "TalentBridge"
    default_model: str = "xgboost"
    page_icon: str = "📊"


class Config(_Section):
    project: ProjectSection = Field(default_factory=ProjectSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    data: DataSection
    checks: ChecksSection = Field(default_factory=ChecksSection)
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    economics: EconomicsSection
    interpretability: InterpretabilitySection = Field(default_factory=InterpretabilitySection)
    stability: StabilitySection = Field(default_factory=StabilitySection)
    fairness: FairnessSection = Field(default_factory=FairnessSection)
    app: AppSection = Field(default_factory=AppSection)

    @property
    def seed(self) -> int:
        return self.project.seed

    @property
    def enabled_models(self) -> list[str]:
        """Names of the models to train, in config order."""
        return [name for name, spec in self.models.items() if spec.enabled]

    @property
    def raw_path(self) -> Path:
        return self.paths.raw_dir / self.data.raw_filename

    # Set by load_config so the app can show which variant is on screen.
    source_path: Path | None = None

    @property
    def variant(self) -> str:
        """Human-readable name of the config in use, for display."""
        if self.source_path is None:
            return "default"
        stem = self.source_path.stem
        return "default" if stem == "config" else stem.replace("config.", "")


def load_config(path: str | Path | None = None, *, mkdirs: bool = True) -> Config:
    """Load, validate and return the project configuration.

    Relative paths in the ``paths`` section are resolved against the project
    root, so the same config works from any working directory — a script run
    from ``src/``, a Streamlit app run from ``app/``, or pytest run from anywhere.

    Args:
        path: Config file to read. Defaults to ``config.yaml`` at the repo root.
        mkdirs: Create the configured directories after resolving them.
    """
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    cfg = Config.model_validate(raw)
    cfg.source_path = config_path.resolve()
    cfg.paths.resolve(config_path.resolve().parent)
    if mkdirs:
        cfg.paths.mkdirs()
    return cfg


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached config for callers that reload often, such as Streamlit reruns."""
    return load_config()
