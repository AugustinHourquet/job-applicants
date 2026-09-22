"""Shared chart style, so every figure in the app and the deck reads as one system.

Colour is assigned by the job it does, not by taste:

* **Categorical** (which model) — a fixed three-slot order, blue/orange/aqua.
  A model keeps its colour everywhere. Because the scorecard sorts by AUC and
  the app lets you filter models, colour must follow the *entity* and never its
  rank, or a filter would silently repaint the survivors.
* **Sequential** (magnitude: feature importance) — one blue hue, light to dark.
* **Diverging** (polarity: fairness gaps, SHAP contributions) — blue/red poles
  around a neutral grey midpoint, because the sign is the whole point.

The palette was validated against colour-vision-deficiency and contrast checks
for all three pairs. One caveat carried over: aqua sits at 2.74:1 on the light
surface, under the 3:1 bar, so every chart in the app is shipped **next to its
underlying table**. That is the documented relief for a contrast warning, and it
is good practice for a jury that will want the numbers anyway.

Never add a second y-axis. Two measures on different scales get two charts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# ----------------------------------------------------------------- tokens ---

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
GRID = "#e8e7e3"

# Fixed categorical order. Three slots is exactly the number of models, and
# these three validate against every pairwise check in both light and dark.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

DIVERGING_POSITIVE = "#2a78d6"
DIVERGING_NEGATIVE = "#e34948"
DIVERGING_MID = "#f0efec"

MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "xgboost": "XGBoost",
    "tabpfn": "TabPFN",
}


def model_colors(models: list[str]) -> dict[str, str]:
    """Stable model → colour map.

    Assigned once from a sorted model list so the mapping never depends on which
    models happen to be present in the frame being plotted.
    """
    return {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(sorted(models))}


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def apply_theme(fig: go.Figure, title: str = "", height: int = 420, **layout) -> go.Figure:
    """Recessive grid and axes, legend only where there is more than one series."""
    show_legend = len({trace.name for trace in fig.data if trace.name}) > 1
    # Caller overrides win, so a chart can ask for unified hover or a bar mode
    # without colliding with the defaults set below.
    defaults: dict = {"hovermode": "closest"}
    defaults.update(layout)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_PRIMARY)) if title else None,
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif",
            size=13,
            color=TEXT_SECONDARY,
        ),
        margin=dict(l=60, r=30, t=50 if title else 20, b=50),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SECONDARY),
        ),
        **defaults,
    )
    axis = dict(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=TEXT_MUTED),
        title_font=dict(color=TEXT_SECONDARY),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


# ------------------------------------------------------------------ charts --


def profit_curve_chart(
    curves: pd.DataFrame, split: str = "test", chosen: dict[str, float] | None = None
) -> go.Figure:
    """Profit against decision threshold, one line per model.

    Crosshair hover is on because the question the chart answers — "what happens
    if we move the cut-off" — is a read-off-the-line question.
    """
    subset = curves[curves["split"] == split]
    colors = model_colors(sorted(curves["model"].unique()))
    fig = go.Figure()

    for name, group in subset.groupby("model", sort=True):
        group = group.sort_values("threshold")
        fig.add_trace(
            go.Scatter(
                x=group["threshold"],
                y=group["profit"],
                name=label(str(name)),
                mode="lines",
                line=dict(color=colors[str(name)], width=2),
                hovertemplate="<b>%{fullData.name}</b><br>Threshold %{x:.2f}<br>Profit €%{y:,.0f}<extra></extra>",
            )
        )

    for name, threshold in (chosen or {}).items():
        if name in colors:
            fig.add_vline(
                x=threshold, line=dict(color=colors[name], width=1, dash="dot"), opacity=0.7
            )

    fig.add_hline(y=0, line=dict(color=TEXT_MUTED, width=1))
    apply_theme(fig, hovermode="x unified")
    fig.update_xaxes(title="Decision threshold")
    fig.update_yaxes(title="Total profit (€)", tickformat=",.0f")
    return fig


def roc_chart(predictions: pd.DataFrame, split: str = "test") -> go.Figure:
    """ROC curves with the chance diagonal for reference."""
    from sklearn.metrics import roc_auc_score, roc_curve

    subset = predictions[predictions["split"] == split]
    colors = model_colors(sorted(predictions["model"].unique()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            showlegend=False,
            line=dict(color=TEXT_MUTED, width=1, dash="dash"),
            hoverinfo="skip",
        )
    )
    for name, group in subset.groupby("model", sort=True):
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_prob"])
        auc = roc_auc_score(group["y_true"], group["y_prob"])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{label(str(name))} (AUC {auc:.3f})",
                line=dict(color=colors[str(name)], width=2),
                hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    apply_theme(fig)
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1], scaleanchor="x")
    return fig


def importance_chart(importance: pd.DataFrame, model: str, top_n: int = 15) -> go.Figure:
    """Feature importance as magnitude — one hue, darker means more important."""
    subset = (
        importance[importance["model"] == model]
        .nlargest(top_n, "mean_abs_shap")
        .sort_values("mean_abs_shap")
    )
    if subset.empty:
        return apply_theme(go.Figure(), "No importance available")

    peak = subset["mean_abs_shap"].max() or 1.0
    shades = [
        SEQUENTIAL_BLUE[min(len(SEQUENTIAL_BLUE) - 1, int(v / peak * (len(SEQUENTIAL_BLUE) - 1)))]
        for v in subset["mean_abs_shap"]
    ]
    fig = go.Figure(
        go.Bar(
            x=subset["mean_abs_shap"],
            y=subset["feature"],
            orientation="h",
            marker=dict(color=shades, cornerradius=4),
            hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
        )
    )
    method = subset["method"].iloc[0] if "method" in subset.columns else ""
    apply_theme(fig, height=max(360, 26 * len(subset)))
    fig.update_xaxes(title=f"Mean |contribution| — {method}")
    fig.update_yaxes(title="")
    return fig


def gap_chart(frame: pd.DataFrame, value_column: str, label_column: str, title_x: str) -> go.Figure:
    """Signed gaps as a diverging bar chart: blue above, red below, grey at zero."""
    subset = frame.dropna(subset=[value_column]).sort_values(value_column)
    if subset.empty:
        return apply_theme(go.Figure(), "No data")

    colors = [
        DIVERGING_POSITIVE if v > 0 else DIVERGING_NEGATIVE if v < 0 else DIVERGING_MID
        for v in subset[value_column]
    ]
    fig = go.Figure(
        go.Bar(
            x=subset[value_column],
            y=subset[label_column].astype(str),
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=[f"{v:+.3f}" for v in subset[value_column]],
            textposition="outside",
            textfont=dict(color=TEXT_SECONDARY, size=11),
            hovertemplate="<b>%{y}</b><br>%{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=TEXT_MUTED, width=1))
    apply_theme(fig, height=max(320, 34 * len(subset)))
    fig.update_xaxes(title=title_x)
    fig.update_yaxes(title="")
    return fig


def flip_rate_chart(per_row: pd.DataFrame) -> go.Figure:
    """Distribution of per-candidate flip rates, one histogram per model.

    The interesting part is the right tail: candidates whose decision is decided
    by which bootstrap sample we happened to draw.
    """
    colors = model_colors(sorted(per_row["model"].unique()))
    fig = go.Figure()
    for name, group in per_row.groupby("model", sort=True):
        fig.add_trace(
            go.Histogram(
                x=group["flip_rate"],
                name=label(str(name)),
                marker=dict(color=colors[str(name)], line=dict(color=SURFACE, width=2)),
                opacity=0.75,
                nbinsx=30,
                hovertemplate="Flip rate %{x:.2f}<br>%{y} candidates<extra>%{fullData.name}</extra>",
            )
        )
    apply_theme(fig, barmode="overlay")
    fig.update_xaxes(title="Share of bootstrap re-fits that flipped this decision")
    fig.update_yaxes(title="Candidates")
    return fig


def shift_chart(shift: pd.DataFrame, model: str) -> go.Figure:
    """Per-subgroup AUC as a dot plot, with the overall level marked."""
    subset = shift[shift["model"] == model].sort_values("roc_auc")
    if subset.empty:
        return apply_theme(go.Figure(), "No subgroups met the minimum size")

    fig = go.Figure(
        go.Scatter(
            x=subset["roc_auc"],
            y=subset["group"].astype(str),
            mode="markers",
            marker=dict(size=11, color=CATEGORICAL[0], line=dict(color=SURFACE, width=2)),
            hovertemplate="<b>%{y}</b><br>AUC %{x:.3f}<extra></extra>",
            name=label(model),
        )
    )
    fig.add_vline(
        x=float(subset["roc_auc"].mean()),
        line=dict(color=TEXT_MUTED, width=1, dash="dash"),
    )
    apply_theme(fig, height=max(320, 26 * len(subset)))
    fig.update_xaxes(title="ROC AUC within subgroup")
    fig.update_yaxes(title="")
    return fig


def save_png(fig: go.Figure, path: Path, scale: int = 2) -> Path | None:
    """Export a figure for the slide deck. Returns None if kaleido is unavailable."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path), scale=scale)
        return path
    except Exception:  # noqa: BLE001 — a missing export engine must not fail the pipeline
        return None
