"""Visualization report tools for the experience study skill."""

from __future__ import annotations

from html import escape
import math
from typing import Any, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from skills.experience_study_skill.io import (
    ToolExecutionContext,
    _ensure_output_dir,
    _error_result,
    _resolve_latest_sweep_path,
    _tabular_error_result,
    _tool_result,
)


_NEUTRAL_COLOR = "#1f4e79"
_FIGURE_CONFIG = {"displaylogo": False, "responsive": True}
_SCATTER_X_MAX = 3.0
_TREEMAP_COLOR_MAX = 2.0


def _validate_metric(metric: str) -> None:
    if metric not in {"count", "amount"}:
        raise ValueError("metric must be 'count' or 'amount'")


def _metric_columns(metric: str) -> dict[str, str]:
    _validate_metric(metric)
    if metric == "count":
        return {
            "ratio": "AE_Ratio_Count",
            "actual": "Sum_MAC",
            "expected": "Sum_MEC",
            "ci_low": "AE_Count_CI_Lower",
            "ci_high": "AE_Count_CI_Upper",
        }
    return {
        "ratio": "AE_Ratio_Amount",
        "actual": "Sum_MAF",
        "expected": "Sum_MEF",
        "ci_low": "AE_Amount_CI_Lower",
        "ci_high": "AE_Amount_CI_Upper",
    }


def _metric_label(metric: str) -> str:
    return "Count" if metric == "count" else "Amount"


def _ratio_label(metric: str) -> str:
    return f"A/E Ratio ({_metric_label(metric)})"


def _required_columns(df: pd.DataFrame, columns: Sequence[str], data_path: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")


def _finite_float(value: Any) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric_value) or not math.isfinite(numeric_value):
        return None
    return numeric_value


def _finite_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.map(lambda value: _finite_float(value) is not None))


def _format_number(value: Any, *, decimals: int = 2) -> str:
    numeric_value = _finite_float(value)
    if numeric_value is None:
        return "n/a"
    return f"{numeric_value:,.{decimals}f}"


def _format_percent(value: Any) -> str:
    numeric_value = _finite_float(value)
    if numeric_value is None:
        return "n/a"
    return f"{numeric_value:.1%}"


def _format_ci(lower: Any, upper: Any) -> str:
    lower_text = _format_percent(lower)
    upper_text = _format_percent(upper)
    if "n/a" in {lower_text, upper_text}:
        return "n/a"
    return f"{lower_text} - {upper_text}"


def _prepare_metric_frame(df: pd.DataFrame, metric: str, data_path: str) -> pd.DataFrame:
    metric_columns = _metric_columns(metric)
    required = [
        "Dimensions",
        "Sum_MOC",
        metric_columns["actual"],
        metric_columns["expected"],
        metric_columns["ratio"],
        metric_columns["ci_low"],
        metric_columns["ci_high"],
    ]
    _required_columns(df, required, data_path)
    prepared = df.copy()
    prepared["_ratio_numeric"] = _finite_series(prepared[metric_columns["ratio"]])
    prepared["_ci_low_numeric"] = _finite_series(prepared[metric_columns["ci_low"]])
    prepared["_ci_high_numeric"] = _finite_series(prepared[metric_columns["ci_high"]])
    return prepared.sort_values("_ratio_numeric", ascending=False, na_position="last")


def _build_scatter_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    metric_columns = _metric_columns(metric)
    prepared = _prepare_metric_frame(df, metric, data_path)
    prepared["display_ratio"] = prepared["_ratio_numeric"].clip(lower=0, upper=_SCATTER_X_MAX)
    prepared["display_ci_low"] = prepared["_ci_low_numeric"].clip(lower=0, upper=_SCATTER_X_MAX)
    prepared["display_ci_high"] = prepared["_ci_high_numeric"].clip(lower=0, upper=_SCATTER_X_MAX)
    prepared["display_ratio_text"] = prepared[metric_columns["ratio"]].map(_format_number)

    fig = go.Figure(
        go.Scatter(
            x=prepared["display_ratio"],
            y=prepared["Dimensions"],
            customdata=prepared["display_ratio_text"],
            mode="markers",
            marker={
                "color": _NEUTRAL_COLOR,
                "size": 14,
                "line": {"color": "#ffffff", "width": 1.2},
                "opacity": 0.92,
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (
                    prepared["display_ci_high"] - prepared["display_ratio"]
                ).clip(lower=0).fillna(0),
                "arrayminus": (
                    prepared["display_ratio"] - prepared["display_ci_low"]
                ).clip(lower=0).fillna(0),
                "visible": True,
                "color": _NEUTRAL_COLOR,
                "thickness": 1.4,
            },
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{_ratio_label(metric)}: "
                "%{customdata}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="#c05252", line_width=2)
    fig.update_xaxes(title_text=_ratio_label(metric), range=[0, _SCATTER_X_MAX])
    fig.update_yaxes(title_text="Cohort", autorange="reversed")
    fig.update_layout(
        title=f"Forest Plot ({_metric_label(metric)})",
        height=max(460, min(320 + max(len(prepared), 1) * 54, 920)),
        paper_bgcolor="#fbfaf7",
        plot_bgcolor="#ffffff",
        font={"color": "#1f2933", "family": "Avenir Next, Segoe UI, Arial, sans-serif"},
        margin={"l": 40, "r": 30, "t": 70, "b": 30},
        showlegend=False,
    )
    return fig


def _build_table_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    metric_columns = _metric_columns(metric)
    prepared = _prepare_metric_frame(df, metric, data_path)
    fig = go.Figure(
        go.Table(
            header={
                "values": [
                    "Cohort",
                    "Exposure (MOC)",
                    "Actual",
                    "Expected",
                    _ratio_label(metric),
                    "95% CI",
                ],
                "fill_color": "#e7eef5",
                "align": "left",
                "font": {"color": "#102a43", "size": 12},
            },
            cells={
                "values": [
                    prepared["Dimensions"],
                    prepared["Sum_MOC"].map(_format_number),
                    prepared[metric_columns["actual"]].map(_format_number),
                    prepared[metric_columns["expected"]].map(_format_number),
                    prepared[metric_columns["ratio"]].map(_format_number),
                    [
                        _format_ci(lower, upper)
                        for lower, upper in zip(
                            prepared[metric_columns["ci_low"]],
                            prepared[metric_columns["ci_high"]],
                        )
                    ],
                ],
                "align": "left",
                "fill_color": "#ffffff",
                "height": 30,
            },
        )
    )
    fig.update_layout(
        title=f"Filtered Cohort Detail ({_metric_label(metric)})",
        height=max(260, min(140 + max(len(prepared), 1) * 32, 1800)),
        paper_bgcolor="#fbfaf7",
        margin={"l": 40, "r": 30, "t": 70, "b": 30},
    )
    return fig


def _split_dimensions(label: str) -> list[str]:
    return [part.strip() for part in str(label).split("|") if part.strip()]


def _treemap_ratio(node: dict[str, Any], *, has_ratio_totals: bool) -> float | None:
    """Return the aggregate A/E ratio used to color a treemap node."""

    if has_ratio_totals:
        expected = _finite_float(node["expected"])
        if expected is not None and expected > 0:
            actual = _finite_float(node["actual"]) or 0.0
            return actual / expected

    weighted_ratio_weight = _finite_float(node["weighted_ratio_weight"]) or 0.0
    if weighted_ratio_weight > 0:
        return float(node["weighted_ratio_total"]) / weighted_ratio_weight

    ratio_count = int(node["ratio_count"])
    if ratio_count > 0:
        return float(node["ratio_sum"]) / ratio_count
    return None


def _build_treemap_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    metric_columns = _metric_columns(metric)
    value_column = "Sum_MOC" if metric == "count" else "Sum_MAF"
    actual_column = metric_columns["actual"]
    expected_column = metric_columns["expected"]
    required = ["Dimensions", metric_columns["ratio"], value_column]
    _required_columns(df, required, data_path)
    has_ratio_totals = actual_column in df.columns and expected_column in df.columns
    prepared = df.copy()
    prepared["_ratio_numeric"] = _finite_series(prepared[metric_columns["ratio"]])

    nodes: dict[str, dict[str, Any]] = {}
    for _, row in prepared.iterrows():
        parts = _split_dimensions(row["Dimensions"])
        if not parts:
            parts = [str(row["Dimensions"])]

        value = _finite_float(row[value_column]) or 0.0
        ratio_value = _finite_float(row["_ratio_numeric"])
        actual_value = _finite_float(row[actual_column]) if has_ratio_totals else None
        expected_value = _finite_float(row[expected_column]) if has_ratio_totals else None

        for depth in range(1, len(parts) + 1):
            node_id = " | ".join(parts[:depth])
            parent_id = " | ".join(parts[: depth - 1]) if depth > 1 else ""
            node = nodes.setdefault(
                node_id,
                {
                    "label": parts[depth - 1],
                    "parent": parent_id,
                    "value": 0.0,
                    "actual": 0.0,
                    "expected": 0.0,
                    "weighted_ratio_total": 0.0,
                    "weighted_ratio_weight": 0.0,
                    "ratio_sum": 0.0,
                    "ratio_count": 0,
                },
            )
            node["value"] += value
            node["actual"] += actual_value or 0.0
            node["expected"] += expected_value or 0.0
            if ratio_value is not None:
                if value > 0:
                    node["weighted_ratio_total"] += ratio_value * value
                    node["weighted_ratio_weight"] += value
                node["ratio_sum"] += ratio_value
                node["ratio_count"] += 1

    ids = list(nodes)
    labels = [nodes[node_id]["label"] for node_id in ids]
    parents = [nodes[node_id]["parent"] for node_id in ids]
    values = [nodes[node_id]["value"] for node_id in ids]
    ratios = [
        _treemap_ratio(nodes[node_id], has_ratio_totals=has_ratio_totals)
        for node_id in ids
    ]
    colors = [ratio if ratio is not None else 1.0 for ratio in ratios]
    ratio_text = [_format_number(ratio) for ratio in ratios]

    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            customdata=ratio_text,
            marker={
                "colors": colors,
                "colorscale": "RdYlGn_r",
                "cmid": 1.0,
                "cmin": 0.0,
                "cmax": _TREEMAP_COLOR_MAX,
                "line": {"width": 1, "color": "#ffffff"},
            },
            hovertemplate=(
                "<b>%{label}</b><br>"
                f"{_ratio_label(metric)}: "
                "%{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Risk Treemap ({_metric_label(metric)})",
        height=900,
        paper_bgcolor="#fbfaf7",
        margin={"l": 40, "r": 30, "t": 70, "b": 30},
    )
    return fig


def _figure_fragment(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_FIGURE_CONFIG)


def _build_report_html(
    *,
    title: str,
    metric: str,
    scatter_fragment: str,
    table_fragment: str,
    treemap_fragment: str,
) -> str:
    plotly_js = get_plotlyjs()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(title)}</title>
  <style>
    body {{
      margin: 0;
      padding: 32px 24px 48px;
      background:
        radial-gradient(circle at top left, rgba(31, 78, 121, 0.08), transparent 28%),
        linear-gradient(180deg, #faf7f2 0%, #f6f3ed 100%);
      color: #1f2933;
      font-family: "Avenir Next", "Segoe UI", Arial, sans-serif;
    }}
    .report-shell {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    .report-section {{
      margin-top: 26px;
      padding: 22px 22px 18px;
      border: 1px solid rgba(31, 41, 51, 0.12);
      border-radius: 24px;
      background: #ffffff;
      box-shadow: 0 18px 50px rgba(31, 41, 51, 0.08);
    }}
  </style>
  <script type="text/javascript">{plotly_js}</script>
</head>
<body>
  <main class="report-shell">
    <section class="report-section">{scatter_fragment}</section>
    <section class="report-section">{table_fragment}</section>
    <section class="report-section">{treemap_fragment}</section>
  </main>
</body>
</html>
"""


def generate_combined_report(
    *,
    context: ToolExecutionContext,
    metric: str = "amount",
    data_path: str | None = None,
) -> dict[str, Any]:
    source_path = _resolve_latest_sweep_path(data_path, context)
    if source_path is None:
        if data_path:
            return _error_result("validation_error", f"File not found: {data_path}")
        return _error_result(
            "missing_prerequisite",
            "Missing prerequisite. Run dimensional sweep first.",
        )
    if source_path.suffix.lower() != ".csv":
        return _error_result(
            "validation_error",
            f"Sweep summary must be a CSV artifact, got `{source_path.suffix or '<none>'}`.",
        )

    context.emit_status("Generating the combined visualization report.")
    try:
        df = pd.read_csv(source_path)
        scatter_fragment = _figure_fragment(_build_scatter_figure(df, metric, str(source_path)))
        table_fragment = _figure_fragment(_build_table_figure(df, metric, str(source_path)))
        treemap_fragment = _figure_fragment(_build_treemap_figure(df, metric, str(source_path)))
    except (
        FileNotFoundError,
        OSError,
        PermissionError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        return _tabular_error_result(source_path, exc)
    except ValueError as exc:
        return _error_result("validation_error", str(exc))
    report_html = _build_report_html(
        title=f"Combined A/E Visualization Report ({_metric_label(metric)})",
        metric=metric,
        scatter_fragment=scatter_fragment,
        table_fragment=table_fragment,
        treemap_fragment=treemap_fragment,
    )

    _ensure_output_dir(context)
    visualization_path = context.next_visualization_path()
    visualization_path.write_text(report_html, encoding="utf-8")
    return _tool_result(
        True,
        "visualization",
        f"Generated the combined visualization report at `{visualization_path}`.",
        artifacts={
            "sweep_summary_path": str(source_path),
            "visualization_path": str(visualization_path),
        },
        data={"metric": metric},
    )
