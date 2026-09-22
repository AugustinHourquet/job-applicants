"""Group fairness, measured at the deployed threshold, with mitigation on/off.

Owner: Member 5.
"""

from __future__ import annotations

import numpy as np
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
from src.fairness import (
    METRIC_TO_CRITERION,
    add_gaps,
    apply_group_thresholds,
    group_metrics,
    group_thresholds,
)

cfg = page_setup("Fairness", "⚖️")
st.title("Fairness audit")
st.caption(
    "Measured at the threshold the client would actually deploy, not at 0.5. "
    "Auditing a cut-off nobody uses describes a system that does not exist."
)

predictions = load_predictions()
economics = load_result("economics")
if not require_pipeline(predictions):
    st.stop()

deployed = (
    dict(zip(economics["model"], economics["optimal_threshold"], strict=False))
    if not economics.empty
    else {}
)

test = predictions[predictions["split"] == "test"]
available_attributes = [a for a in cfg.fairness.protected_attributes if a in test.columns]

left, right = st.columns([1, 1])
with left:
    model = st.selectbox(
        "Model", sorted(test["model"].unique()), format_func=plots.label, key="fair_model"
    )
with right:
    attribute = st.selectbox("Protected attribute", available_attributes, key="fair_attr")

threshold = float(deployed.get(model, 0.5))
frame = test[test["model"] == model]

st.sidebar.subheader("Mitigation")
mitigation = st.sidebar.radio(
    "Post-processing",
    ["None", "Per-group thresholds"],
    help=(
        "Per-group thresholds equalise the true positive rate by cutting at a "
        "different score for each group. It is the most transparent mitigation "
        "available and also the hardest to defend — different cut-offs by gender "
        "is exactly the thing a client has to decide whether it can justify."
    ),
)
target_metric = st.sidebar.selectbox(
    "Equalise",
    ["true_positive_rate", "selection_rate"],
    format_func=lambda m: f"{m.replace('_', ' ')} ({METRIC_TO_CRITERION[m]})",
)

st.info(
    f"Deployed threshold for **{plots.label(model)}**: **{threshold:.2f}** "
    f"(chosen on validation by the profit curve)."
)

# --------------------------------------------------------------- measure ----

metrics = group_metrics(frame, attribute, threshold, cfg.fairness.min_group_size)
metrics = add_gaps(metrics, cfg.fairness.reference_groups.get(attribute), cfg.fairness.metrics)

if mitigation == "Per-group thresholds":
    chosen = group_thresholds(
        frame,
        attribute,
        target_metric,
        reference=cfg.fairness.reference_groups.get(attribute),
    )
    decisions = apply_group_thresholds(frame, attribute, chosen)
    mitigated = frame.copy()
    # Rewrite the score so the shared metric code sees the mitigated decision.
    mitigated["y_prob"] = np.where(decisions == 1, 1.0, 0.0)
    metrics = group_metrics(mitigated, attribute, 0.5, cfg.fairness.min_group_size)
    metrics = add_gaps(metrics, cfg.fairness.reference_groups.get(attribute), cfg.fairness.metrics)
    st.sidebar.caption("Thresholds applied:")
    st.sidebar.json({k: round(v, 3) for k, v in chosen.items()})

reliable = metrics[metrics["reliable"]]
worst_selection = reliable["selection_rate_gap"].abs().max() if not reliable.empty else float("nan")
worst_tpr = reliable["true_positive_rate_gap"].abs().max() if not reliable.empty else float("nan")
impact = (
    reliable["disparate_impact_ratio"].min()
    if "disparate_impact_ratio" in reliable
    else float("nan")
)

metric_row(
    [
        ("Groups", str(len(metrics)), f"{int((~metrics['reliable']).sum())} below the size floor"),
        ("Worst selection gap", f"{worst_selection:.3f}", "Demographic parity"),
        ("Worst TPR gap", f"{worst_tpr:.3f}", "Equal opportunity"),
        (
            "Disparate impact",
            f"{impact:.2f}" if impact == impact else "—",
            "Four-fifths rule: 0.80 or above passes",
        ),
    ]
)

if impact == impact and impact < 0.8:
    st.error(
        f"**Fails the four-fifths rule** on {attribute} at {impact:.2f}. In employment "
        "screening this is the conventional threshold for adverse impact."
    )

st.divider()
metric_choice = st.selectbox(
    "Show gaps for",
    cfg.fairness.metrics,
    format_func=lambda m: f"{m.replace('_', ' ').title()} — {METRIC_TO_CRITERION.get(m, '')}",
    key="fair_metric",
)
gap_column = f"{metric_choice}_gap"
if gap_column in metrics.columns:
    reference = metrics["reference_group"].iloc[0] if "reference_group" in metrics else ""
    chart_with_table(
        plots.gap_chart(metrics, gap_column, "group", f"{metric_choice} gap vs {reference}"),
        metrics,
        f"Gaps against **{reference}**. Groups below {cfg.fairness.min_group_size} rows are "
        "kept but flagged unreliable — dropping them hides who the model fails, while "
        "reporting a rate from a handful of people would overstate what we know.",
    )

st.divider()
st.subheader("Why the four criteria disagree")
st.markdown(
    """
They are not alternative wordings of the same idea, and when base rates differ
across groups they **cannot all hold at once** — that is a theorem, not a
limitation of this analysis.

| Criterion | Equalises | Ignores |
|---|---|---|
| Demographic parity | who gets selected | whether they would have succeeded |
| Equal opportunity | selection among those who *would* succeed | the false-positive burden |
| Equalised odds | both error rates | differences in base rate |
| Predictive parity | precision among those selected | who never got selected |

Picking one is a decision about **which error the client is willing to make**,
and the deck should argue for a specific choice rather than report all four and
leave the jury to guess.
"""
)
