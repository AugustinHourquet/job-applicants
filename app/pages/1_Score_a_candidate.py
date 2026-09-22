"""Score one candidate with all three models and explain the result.

Owner: Member 6, with the explanation from Member 3's module.

The only page that loads the fitted models. The form builds a raw candidate row,
which is encoded through the very same FeatureSpec the models were trained on —
so what the client sees here is exactly what the pipeline would produce.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import pandas as pd
import streamlit as st

# Streamlit puts the entrypoint's directory on sys.path, so pages normally find
# _shared only because Home.py was launched first. Doing it explicitly means a
# page can also be run or tested on its own.
_APP_DIR = _Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in _sys.path:
    _sys.path.insert(0, str(_APP_DIR))

from _shared import load_feature_spec, load_models, load_raw_test, page_setup

from src import plots
from src.data import apply_feature_spec
from src.interpretability import explain_candidate

cfg = page_setup("Score a candidate", "🧮")
st.title("Score a candidate")
st.caption(
    "Fill in a profile and all three models score it side by side. Disagreement "
    "between them is the interesting case — it is where the choice of model "
    "changes someone's outcome."
)

spec = load_feature_spec()
models = load_models()
raw_test = load_raw_test()

if spec is None or not models:
    st.warning(
        "**Models not available.** Run `make pipeline` (or `make docker-pipeline`) "
        "first, then reload this page."
    )
    st.stop()

# ------------------------------------------------------------------- form ---

defaults = raw_test.iloc[0].to_dict() if not raw_test.empty else {}

with st.form("candidate"):
    st.subheader("Candidate profile")
    left, middle, right = st.columns(3)
    values: dict[str, object] = {}

    categorical = list(spec.categorical_levels)
    for i, column in enumerate(categorical):
        target = (left, middle, right)[i % 3]
        levels = spec.categorical_levels[column]
        current = str(defaults.get(column, levels[0]))
        index = levels.index(current) if current in levels else 0
        values[column] = target.selectbox(column, levels, index=index)

    st.divider()
    left, middle, right = st.columns(3)
    for i, column in enumerate(spec.numeric_features):
        target = (left, middle, right)[i % 3]
        median = float(spec.numeric_medians.get(column, 0.0))
        values[column] = target.number_input(
            column, value=float(defaults.get(column, median)), step=1.0
        )

    st.divider()
    for column, vocabulary in spec.tech_vocabulary.items():
        current = str(defaults.get(column, ""))
        preselected = [t for t in current.split(";") if t in vocabulary]
        chosen = st.multiselect(f"{column} — technologies used", vocabulary, default=preselected)
        values[column] = ";".join(chosen)

    submitted = st.form_submit_button("Score this candidate", type="primary")

# ------------------------------------------------------------------ score ---

if submitted:
    raw_row = pd.DataFrame([values])
    encoded = apply_feature_spec(raw_row, spec)

    economics = (
        pd.read_csv(cfg.paths.results_dir / "economics.csv")
        if (cfg.paths.results_dir / "economics.csv").is_file()
        else pd.DataFrame()
    )
    thresholds = (
        dict(zip(economics["model"], economics["optimal_threshold"], strict=False))
        if not economics.empty
        else {}
    )

    st.subheader("Scores")
    columns = st.columns(len(models))
    scores: dict[str, float] = {}
    for column, (name, fitted) in zip(columns, sorted(models.items()), strict=False):
        probability = float(fitted.predict_proba(encoded)[0])
        scores[name] = probability
        threshold = float(thresholds.get(name, 0.5))
        decision = "Sponsor" if probability >= threshold else "Decline"
        column.metric(
            plots.label(name),
            f"{probability:.1%}",
            delta=decision,
            delta_color="normal" if probability >= threshold else "inverse",
            help=f"Deployed threshold {threshold:.2f}",
        )

    spread = max(scores.values()) - min(scores.values())
    if spread > 0.15:
        st.warning(
            f"The models disagree by {spread:.0%} on this candidate. Cases like "
            "this are exactly where the choice of model is a decision about a "
            "person, not a technical preference."
        )

    st.divider()
    st.subheader("Why — what drove this score")
    chosen_model = st.selectbox(
        "Explain", sorted(models), format_func=plots.label, key="explain_model"
    )
    background = apply_feature_spec(raw_test, spec) if not raw_test.empty else encoded
    contributions = explain_candidate(models[chosen_model], encoded, background)
    figure = plots.gap_chart(contributions, "contribution", "feature", "Contribution to the score")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Blue pushes the score up, red pushes it down. This is the kind of "
        "statement an adverse-action notice has to be able to make."
    )
    with st.expander("Show the numbers behind this chart"):
        st.dataframe(contributions, width="stretch", hide_index=True)
else:
    st.info("Fill in the profile above and press **Score this candidate**.")
