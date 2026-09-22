"""Loading and layout helpers shared by every page.

The app is a *reader*. It never trains a model and never recomputes a metric
from scratch; the pipeline does that and writes artifacts, and everything here
loads them. The one exception is the scoring page, which loads the fitted models
to score a single candidate the user types in.

Anything that depends only on scores and a threshold — profit at a different
cut-off, fairness at a different threshold — is recomputed live from the
predictions table, which is why the sliders respond instantly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit runs this file directly, so the project root is not on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import plots  # noqa: E402
from src.config import Config, load_config  # noqa: E402

RESULT_FILES = {
    "scorecard": "scorecard.csv",
    "statistical": "statistical.csv",
    "economics": "economics.csv",
    "profit_curves": "profit_curves.csv",
    "fairness": "fairness.csv",
    "fairness_summary": "fairness_summary.csv",
    "stability": "stability.csv",
    "stability_per_row": "stability_per_row.csv",
    "stability_shift": "stability_shift.csv",
    "importance": "feature_importance.csv",
    "odds_ratios": "odds_ratios.csv",
    "local_explanations": "local_explanations.csv",
    "data_checks": "data_checks.csv",
    "data_checks_multivariate": "data_checks_multivariate.csv",
}


@st.cache_resource
def get_config() -> Config:
    return load_config()


@st.cache_data(show_spinner=False)
def load_result(name: str) -> pd.DataFrame:
    """Load one results CSV, or an empty frame if the pipeline has not run."""
    cfg = get_config()
    path = cfg.paths.results_dir / RESULT_FILES[name]
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    cfg = get_config()
    path = cfg.paths.results_dir / "predictions.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_raw_test() -> pd.DataFrame:
    cfg = get_config()
    path = cfg.paths.processed_dir / "test_raw.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource(show_spinner="Loading trained models...")
def load_models() -> dict:
    """Fitted models, loaded only by the page that scores a single candidate."""
    from src import models

    return models.load_all(get_config())


@st.cache_resource
def load_feature_spec():
    from src.data import SPEC_FILENAME, FeatureSpec

    path = get_config().paths.processed_dir / SPEC_FILENAME
    return FeatureSpec.from_json(path) if path.is_file() else None


def page_setup(title: str, icon: str = "📊") -> Config:
    """Standard page configuration. Call first in every page."""
    cfg = get_config()
    st.set_page_config(
        page_title=f"{title} — {cfg.app.title}",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    variant_banner(cfg)
    return cfg


def variant_banner(cfg: Config) -> None:
    """Say which feature set is on screen, on every page.

    Two variants of this analysis exist and they disagree wildly — one has a
    perfect AUC because a feature encodes the answer. Reading a number off the
    wrong one, and putting it in a slide, is the easiest mistake available here,
    so the app states which it is rather than leaving it to be inferred from a
    directory name.
    """
    excluded = list(cfg.data.exclude_features)
    with st.sidebar:
        if excluded:
            st.success(
                f"**Honest variant**\n\nExcluded: {', '.join(f'`{c}`' for c in excluded)}",
                icon="✅",
            )
        else:
            st.warning(
                "**Leaky variant** — all features, including the ones that leak "
                "the target. Numbers here are for the contrast, not for deployment.",
                icon="⚠️",
            )
        st.caption(f"config: `{cfg.variant}` · reading `{cfg.paths.results_dir.parent.name}/`")


def require_pipeline(*frames: pd.DataFrame) -> bool:
    """Show a clear instruction instead of a stack trace when artifacts are missing."""
    if all(frame is not None and not frame.empty for frame in frames):
        return True
    st.warning(
        "**No results yet.** This page reads artifacts written by the analysis "
        "pipeline, and they have not been generated.\n\n"
        "Run one of these from the project root, then reload:\n\n"
        "```bash\n"
        "make pipeline          # locally, with uv\n"
        "make docker-pipeline   # in Docker\n"
        "```"
    )
    return False


def model_selector(available: list[str], key: str, default: str | None = None) -> str:
    """Consistent model picker, showing friendly names but returning the key."""
    cfg = get_config()
    options = sorted(available)
    preferred = default or cfg.app.default_model
    index = options.index(preferred) if preferred in options else 0
    return st.selectbox("Model", options, index=index, format_func=plots.label, key=key)


def chart_with_table(figure, table: pd.DataFrame, caption: str = "") -> None:
    """Render a chart with its underlying numbers in an expander.

    Not decoration: one hue in the palette sits below the 3:1 contrast bar on a
    light surface, and shipping the table is the documented relief for that. It
    also happens to be what a jury asks for.
    """
    st.plotly_chart(figure, width="stretch")
    if caption:
        st.caption(caption)
    with st.expander("Show the numbers behind this chart"):
        st.dataframe(table, width="stretch", hide_index=True)


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """A row of KPI tiles: ``(label, value, help)``."""
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items, strict=False):
        column.metric(label, value, help=help_text)
