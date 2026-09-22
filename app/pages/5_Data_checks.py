"""Leakage and proxy findings — what we checked before modelling anything.

Owner: Member 1. Replaces the exploratory notebook as a deliverable.
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

from _shared import load_result, metric_row, page_setup, require_pipeline

cfg = page_setup("Data checks", "🔍")
st.title("Data checks")
st.caption(
    "Run before any model was fitted. Findings are reported, never applied "
    "automatically — dropping a column is a modelling decision we make and "
    "defend, not something a script does silently."
)

checks = load_result("data_checks")
if not require_pipeline(checks):
    st.stop()

leaking = checks[checks["leakage_flag"]] if "leakage_flag" in checks else checks.head(0)
proxies = checks[checks["proxy_flag"]] if "proxy_flag" in checks else checks.head(0)

metric_row(
    [
        ("Features screened", str(len(checks)), None),
        (
            "Leakage flags",
            str(len(leaking)),
            f"Single-feature AUC ≥ {cfg.checks.leakage_auc_threshold}",
        ),
        ("Proxy flags", str(len(proxies)), f"Normalised MI ≥ {cfg.checks.proxy_nmi_threshold}"),
        (
            "Protected attributes",
            str(len(cfg.fairness.protected_attributes)),
            ", ".join(cfg.fairness.protected_attributes),
        ),
    ]
)

st.divider()
st.subheader("Multivariate leakage — the headline finding")
st.markdown(
    """
The per-feature screen below is **necessary but not sufficient**. A leak can
live entirely in a *combination* of columns, where no individual column looks
unusual. This screen fits a linear model on the full encoded matrix, scores it
on held-out validation data, then repeats for each feature group so the source
is named rather than inferred.
"""
)

multivariate = load_result("data_checks_multivariate")
if multivariate.empty:
    st.info("Re-run the pipeline to generate this check.")
else:
    overall = multivariate[multivariate["scope"] == "ALL FEATURES"]
    culprits = multivariate[
        (multivariate["scope"] != "ALL FEATURES") & multivariate["leakage_flag"]
    ]
    if not overall.empty and bool(overall["leakage_flag"].iloc[0]):
        st.error(
            f"**Target leakage confirmed.** All features together reach a held-out "
            f"AUC of **{overall['holdout_auc'].iloc[0]:.4f}**. A model this good is "
            "not a good model — it is a model that has been told the answer."
        )
        if not culprits.empty:
            names = ", ".join(f"`{n}`" for n in culprits["scope"])
            st.warning(
                f"Isolated to: {names}. That group reaches the same AUC on its own, "
                "while every other group sits near chance."
            )
    else:
        st.success("No multivariate leakage detected.")

    st.dataframe(
        multivariate,
        width="stretch",
        hide_index=True,
        column_config={
            "scope": st.column_config.TextColumn("Feature group"),
            "n_columns": st.column_config.NumberColumn("Columns"),
            "holdout_auc": st.column_config.ProgressColumn(
                "Held-out AUC", min_value=0.5, max_value=1.0, format="%.4f"
            ),
            "leakage_flag": st.column_config.CheckboxColumn("Flagged"),
        },
    )
    st.caption(
        "Read this top-down: if the first row is near 1.0 and one group below it "
        "matches, that group is the leak. If the first row is near 1.0 and no "
        "single group is, the leak lives in the interaction between groups."
    )

st.divider()
st.subheader("Single-feature leakage")
st.markdown(
    f"""
Each feature is scored **on its own** against the target. Categoricals are
encoded by their target rate, which is the most generous reading a single
column can get — so if even that separates the target at AUC ≥
**{cfg.checks.leakage_auc_threshold}**, the column is almost certainly
recording the outcome rather than predicting it.
"""
)
if leaking.empty:
    st.success("No feature crossed the leakage threshold.")
else:
    st.error(f"**{len(leaking)} feature(s) flagged.** Decide and justify before modelling.")
    st.dataframe(leaking[["feature", "single_feature_auc"]], width="stretch", hide_index=True)

st.dataframe(
    checks[["feature", "single_feature_auc", "leakage_flag"]].sort_values(
        "single_feature_auc", ascending=False
    ),
    width="stretch",
    hide_index=True,
    column_config={
        "single_feature_auc": st.column_config.ProgressColumn(
            "Single-feature AUC", min_value=0.5, max_value=1.0, format="%.3f"
        ),
        "leakage_flag": st.column_config.CheckboxColumn("Flagged"),
    },
)

st.divider()
st.subheader("Proxies for protected attributes")
st.markdown(
    f"""
Removing gender from the feature set does not remove gender from the model if
another column carries the same information. Each feature is scored by
normalised mutual information against each protected attribute; anything at or
above **{cfg.checks.proxy_nmi_threshold}** is a usable proxy.

Both columns are capped at 20 levels before the calculation. Plug-in mutual
information is biased upward with cardinality, and without the cap a
near-unique free-text column flags as a proxy for everything — an artefact of
the estimator, not a finding.
"""
)
if proxies.empty:
    st.success("No feature crossed the proxy threshold.")
else:
    st.warning(f"**{len(proxies)} feature(s) flagged** as proxies.")

nmi_columns = [c for c in checks.columns if c.startswith("nmi_")]
if nmi_columns:
    st.dataframe(
        checks[
            ["feature", *nmi_columns, "strongest_proxy_for", "strongest_proxy_nmi", "proxy_flag"]
        ].sort_values("strongest_proxy_nmi", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "proxy_flag": st.column_config.CheckboxColumn("Flagged"),
            "strongest_proxy_nmi": st.column_config.NumberColumn("Strongest NMI", format="%.4f"),
        },
    )

st.divider()
with st.expander("Full checks table"):
    st.dataframe(checks, width="stretch", hide_index=True)
