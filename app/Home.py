"""Landing page: the client's question, the dataset, the scorecard, the recommendation.

Owner: Member 6.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _shared import (
    chart_with_table,
    load_predictions,
    load_result,
    metric_row,
    page_setup,
    require_pipeline,
)

from src import plots

cfg = page_setup("Overview", "🏠")

st.title(cfg.app.title)
st.markdown(
    f"""
**Client:** {cfg.app.client_name}, a platform that sponsors job candidates —
paid coaching and recruiter time up front, a placement fee when they land a role.

Sponsoring is expensive, so the question is not "who is employable" but
**"who is worth sponsoring"**. We compare three models on the four dimensions
that decide whether a scoring system can actually be deployed: predictive
performance (statistical *and* economic), interpretability, stability, and fairness.
"""
)

scorecard = load_result("scorecard")
predictions = load_predictions()

if require_pipeline(scorecard):
    economics = load_result("economics")
    best_auc = scorecard.iloc[0]
    best_profit = (
        scorecard.sort_values("profit", ascending=False).iloc[0]
        if "profit" in scorecard.columns
        else best_auc
    )

    st.divider()
    st.subheader("Headline numbers")
    metric_row(
        [
            ("Models compared", str(len(scorecard)), "White-box, boosted trees, foundation model"),
            (
                "Best discrimination",
                f"{plots.label(best_auc['model'])} · {best_auc['roc_auc']:.3f}",
                "Highest ROC AUC on the held-out test set",
            ),
            (
                "Best economics",
                f"{plots.label(best_profit['model'])} · €{best_profit.get('profit', 0):,.0f}",
                "Profit on test at the threshold chosen on validation",
            ),
            (
                "Candidates scored",
                f"{len(predictions[predictions['split'] == 'test']) // max(len(scorecard), 1):,}"
                if not predictions.empty
                else "—",
                "Size of the held-out test set",
            ),
        ]
    )

    st.divider()
    st.subheader("Scorecard across all four dimensions")
    st.caption(
        "Deliberately not collapsed into a single score. The dimensions trade off "
        "against each other, and choosing between them is the client's decision — "
        "a weighted average would hide exactly the judgement being asked for."
    )

    display = scorecard.copy()
    display.insert(0, "Model", display.pop("model").map(plots.label))
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "roc_auc": st.column_config.NumberColumn("ROC AUC", format="%.3f"),
            "pr_auc": st.column_config.NumberColumn("PR AUC", format="%.3f"),
            "brier": st.column_config.NumberColumn("Brier", format="%.3f", help="Lower is better"),
            "profit": st.column_config.NumberColumn("Profit (€)", format="%.0f"),
            "optimal_threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
            "bootstrap_flip_rate": st.column_config.NumberColumn(
                "Flip rate", format="%.3f", help="Share of bootstrap re-fits that change a decision"
            ),
            "min_disparate_impact": st.column_config.NumberColumn(
                "Disparate impact", format="%.2f", help="Four-fifths rule: 0.80 or above passes"
            ),
            "four_fifths_pass": st.column_config.CheckboxColumn("4/5 rule"),
        },
    )

    if not predictions.empty:
        st.divider()
        st.subheader("Discrimination on the held-out test set")
        table = (
            load_result("statistical")[["model", "roc_auc", "pr_auc", "brier", "accuracy", "f1"]]
            if not load_result("statistical").empty
            else pd.DataFrame()
        )
        chart_with_table(
            plots.roc_chart(predictions),
            table,
            "ROC AUC measures ranking quality only. It says nothing about where to "
            "set the cut-off, which is what the Economics page decides.",
        )

    st.divider()
    st.subheader("Recommendation")
    if "four_fifths_pass" in scorecard.columns:
        passing = scorecard[scorecard["four_fifths_pass"].fillna(False)]
        st.markdown(
            f"""
Deployable models must clear every dimension, not just the leaderboard.

* **Highest AUC:** {plots.label(best_auc["model"])} ({best_auc["roc_auc"]:.3f})
* **Highest profit:** {plots.label(best_profit["model"])} (€{best_profit.get("profit", 0):,.0f})
* **Passes the four-fifths rule:** {", ".join(plots.label(m) for m in passing["model"]) or "_none_"}

Write the team's recommendation here once the results on the real dataset are in.
The interesting case for the jury is a model that wins on performance and loses
on fairness or stability — argue explicitly why the client should or should not
accept that trade.
"""
        )

with st.sidebar:
    st.markdown("### Pages")
    st.markdown(
        "- **Score a candidate** — three scores plus an explanation\n"
        "- **Economics** — cost matrix and threshold, live\n"
        "- **Fairness** — group metrics and mitigation\n"
        "- **Stability** — how much decisions move\n"
        "- **Data checks** — leakage and proxy findings"
    )
    st.divider()
    st.caption(f"Seed {cfg.seed} · {len(cfg.enabled_models)} models enabled")
