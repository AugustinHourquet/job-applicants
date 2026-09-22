"""How much a decision moves when the world wobbles.

Owner: Member 4.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import streamlit as st

# Streamlit puts the entrypoint's directory on sys.path, so pages normally find
# _shared only because Home.py was launched first. Doing it explicitly means a
# page can also be run or tested on its own.
_APP_DIR = _Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in _sys.path:
    _sys.path.insert(0, str(_APP_DIR))

from _shared import chart_with_table, load_result, metric_row, page_setup, require_pipeline

from src import plots

cfg = page_setup("Stability", "🎯")
st.title("Stability")
st.caption(
    "A model can be accurate and still be unfit to deploy. If re-fitting on a "
    "slightly different sample flips a candidate's decision, the client cannot "
    "explain that decision to them — or defend it if challenged."
)

summary = load_result("stability")
per_row = load_result("stability_per_row")
shift = load_result("stability_shift")

if not require_pipeline(summary):
    st.stop()

bootstrap = summary[summary["probe"] == "bootstrap"]
perturbation = summary[summary["probe"] == "perturbation"]

if not bootstrap.empty:
    steadiest = bootstrap.loc[bootstrap["mean_flip_rate"].idxmin()]
    metric_row(
        [
            ("Steadiest model", plots.label(steadiest["model"]), "Lowest bootstrap flip rate"),
            (
                "Its flip rate",
                f"{steadiest['mean_flip_rate']:.1%}",
                "Decisions that change on re-fit",
            ),
            (
                "Never-stable share",
                f"{steadiest.get('share_rows_flipped_often', float('nan')):.1%}",
                "Candidates flipping in 10%+ of re-fits",
            ),
            ("AUC spread", f"±{steadiest.get('auc_std', 0):.3f}", "Std dev of AUC across re-fits"),
        ]
    )

st.divider()
st.subheader("1 · Resampling — does the training sample decide the outcome?")
if not per_row.empty:
    chart_with_table(
        plots.flip_rate_chart(per_row),
        bootstrap,
        "Each model re-fitted on bootstrap resamples of the training data. The "
        "right tail is what matters: candidates whose decision is settled by "
        "which rows we happened to draw, not by anything about them.",
    )
    st.warning(
        "Read a flip rate of 0.2 as: *this candidate would have been told "
        "something different in one run out of five.* A client cannot write that "
        "into a rejection letter."
    )
else:
    st.info("No per-row bootstrap results found. Re-run the pipeline without `--skip stability`.")

st.divider()
st.subheader("2 · Perturbation — does measurement noise decide the outcome?")
if not perturbation.empty:
    display = perturbation[
        ["model", "mean_flip_rate", "max_flip_rate", "noise_sd_fraction", "n_repeats"]
    ].copy()
    display["model"] = display["model"].map(plots.label)
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "mean_flip_rate": st.column_config.NumberColumn("Mean flip rate", format="%.3f"),
            "max_flip_rate": st.column_config.NumberColumn("Worst run", format="%.3f"),
            "noise_sd_fraction": st.column_config.NumberColumn(
                "Noise (× feature SD)", format="%.2f"
            ),
            "n_repeats": st.column_config.NumberColumn("Repeats"),
        },
    )
    st.caption(
        "Gaussian noise added to the numeric inputs at a fraction of each "
        "feature's own standard deviation — roughly the error you would expect "
        "from a self-reported salary or years-of-experience field."
    )

st.divider()
st.subheader("3 · Shift — does performance hold away from the bulk of the data?")
if not shift.empty:
    model = st.selectbox(
        "Model", sorted(shift["model"].unique()), format_func=plots.label, key="shift_model"
    )
    subset = shift[shift["model"] == model]
    chart_with_table(
        plots.shift_chart(shift, model),
        subset,
        "AUC computed separately within each subgroup. A wide spread means the "
        "headline number is carried by the largest populations while smaller "
        "ones are served materially worse.",
    )
    spread = subset["roc_auc"].max() - subset["roc_auc"].min()
    if spread > 0.10:
        st.warning(
            f"AUC varies by {spread:.2f} across subgroups. The single headline "
            "AUC is hiding real differences in who this model works for."
        )
else:
    st.info(
        "No subgroup met the minimum size in `stability.shift.min_group_size`. "
        "Lower it in `config.yaml` to see this analysis."
    )
