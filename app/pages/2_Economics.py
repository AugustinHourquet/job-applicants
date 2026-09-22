"""Cost matrix and threshold, recomputed live.

Owner: Member 2.

Every number here is recomputed in the browser from the predictions table, so
the client can rewrite the economics — change what a wasted sponsorship costs,
change the placement fee — and watch the optimal cut-off move. That the optimal
threshold is a *business* parameter rather than a modelling constant is the
single most useful thing this page teaches.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import streamlit as st

# Streamlit puts the entrypoint's directory on sys.path, so pages normally find
# _shared only because Home.py was launched first. Doing it explicitly means a
# page can also be run or tested on its own.
_APP_DIR = _Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in _sys.path:
    _sys.path.insert(0, str(_APP_DIR))

from _shared import chart_with_table, load_predictions, metric_row, page_setup, require_pipeline

from src import plots
from src.economics import baseline_profits, optimal_threshold, profit_curve

cfg = page_setup("Economics", "💶")
st.title("Economic performance")
st.caption(
    "Statistical performance ranks candidates. Economics decides where to cut. "
    "Change the costs below and every curve updates."
)

predictions = load_predictions()
if not require_pipeline(predictions):
    st.stop()

# ------------------------------------------------------------- the levers ---

defaults = cfg.economics.cost_matrix
with st.sidebar:
    st.subheader("Cost matrix (€)")
    st.caption("What each outcome is worth to the client.")
    cost_matrix = {
        "true_positive": st.number_input(
            "Sponsored, placed",
            value=float(defaults["true_positive"]),
            step=100.0,
            help="Placement fee earned",
        ),
        "false_positive": st.number_input(
            "Sponsored, not placed",
            value=float(defaults["false_positive"]),
            step=100.0,
            help="Coaching and recruiter time spent for nothing",
        ),
        "false_negative": st.number_input(
            "Declined, would have placed",
            value=float(defaults["false_negative"]),
            step=100.0,
            help="Margin lost on someone who placed without us",
        ),
        "true_negative": st.number_input(
            "Declined, would not have placed",
            value=float(defaults["true_negative"]),
            step=100.0,
        ),
    }

    denominator = (cost_matrix["true_negative"] - cost_matrix["false_positive"]) + (
        cost_matrix["true_positive"] - cost_matrix["false_negative"]
    )
    if denominator != 0:
        theoretical = (cost_matrix["true_negative"] - cost_matrix["false_positive"]) / denominator
        st.metric("Break-even threshold", f"{theoretical:.3f}")
        st.caption(
            "Closed form from the cost matrix alone. A model whose profit-maximising "
            "threshold sits far from this is telling you its probabilities are "
            "miscalibrated."
        )

thresholds = np.arange(0.01, 1.0, 0.01)
test = predictions[predictions["split"] == "test"]
validation = predictions[predictions["split"] == "val"]

curves, summary = [], []
for name, group in test.groupby("model", sort=True):
    curve = profit_curve(
        group["y_true"].to_numpy(), group["y_prob"].to_numpy(), cost_matrix, thresholds
    )
    curve.insert(0, "model", name)
    curve.insert(1, "split", "test")
    curves.append(curve)

    validation_group = validation[validation["model"] == name]
    source = validation_group if not validation_group.empty else group
    chosen = optimal_threshold(
        profit_curve(
            source["y_true"].to_numpy(), source["y_prob"].to_numpy(), cost_matrix, thresholds
        )
    )
    at = curve.iloc[(curve["threshold"] - chosen).abs().argmin()]
    baselines = baseline_profits(group["y_true"].to_numpy(), cost_matrix)
    summary.append(
        {
            "model": name,
            "threshold": chosen,
            "profit": at["profit"],
            "profit_per_candidate": at["profit_per_candidate"],
            "selection_rate": at["selection_rate"],
            "uplift_vs_accept_all": at["profit"] - baselines["accept_all"],
            "tp": int(at["tp"]),
            "fp": int(at["fp"]),
            "fn": int(at["fn"]),
            "tn": int(at["tn"]),
        }
    )

curves = pd.concat(curves, ignore_index=True)
summary = pd.DataFrame(summary).sort_values("profit", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------- display ---

best = summary.iloc[0]
metric_row(
    [
        ("Most profitable", plots.label(best["model"]), "At its own optimal threshold"),
        ("Profit on test", f"€{best['profit']:,.0f}", "At the threshold chosen on validation"),
        ("Per candidate", f"€{best['profit_per_candidate']:,.0f}", None),
        ("Sponsored", f"{best['selection_rate']:.1%}", "Share of candidates accepted"),
    ]
)

st.divider()
chart_with_table(
    plots.profit_curve_chart(
        curves, chosen=dict(zip(summary["model"], summary["threshold"], strict=False))
    ),
    summary,
    "Dotted lines mark each model's threshold, chosen on validation and applied "
    "to test. They sit slightly off each curve's own test-set peak — that gap is "
    "the honest cost of not tuning on the data you report.",
)

st.divider()
st.subheader("What the cut-off does to real people")
selected = st.selectbox(
    "Model", sorted(summary["model"]), format_func=plots.label, key="econ_model"
)
row = summary[summary["model"] == selected].iloc[0]
manual = st.slider(
    "Move the threshold by hand",
    0.01,
    0.99,
    float(row["threshold"]),
    0.01,
    help="The optimum maximises profit. It is not automatically the right choice.",
)

curve = curves[curves["model"] == selected]
at_manual = curve.iloc[(curve["threshold"] - manual).abs().argmin()]
metric_row(
    [
        ("Profit", f"€{at_manual['profit']:,.0f}", None),
        ("Sponsored", f"{at_manual['selection_rate']:.1%}", None),
        ("Wasted sponsorships", f"{int(at_manual['fp']):,}", "Paid for, no placement"),
        ("Missed candidates", f"{int(at_manual['fn']):,}", "Declined, would have placed"),
    ]
)
st.caption(
    "Raising the threshold cuts wasted spend and turns away people who would "
    "have succeeded. The model cannot decide which of those two errors matters "
    "more — the client has to."
)
