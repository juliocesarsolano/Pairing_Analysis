"""Plotly paired-sample comparison and spatial QA/QC figures."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from .statistics import PairMetrics, empirical_cdf, qq_values


# Corporate palette used across the technical applications.
CORPORATE_BLUE = "#03547C"  # RGB(3, 84, 124)
CORPORATE_GOLD = "#A39161"  # RGB(163, 145, 97)
CORPORATE_ORANGE = "#FDB813"  # RGB(253, 184, 19)
CORPORATE_GRAY = "#C7C8CA"  # RGB(199, 200, 202)

SOURCE_PALETTE = ("#1F77B4", "#FF7F0E")
CORPORATE_PALETTE = (CORPORATE_BLUE, CORPORATE_GOLD)
CORPORATE_GRID = "rgba(199,200,202,0.45)"
CORPORATE_NEUTRAL = "#F3F4F5"


@dataclass(frozen=True)
class PlotOptions:
    """User-adjustable plotting controls."""

    reference_label: str
    comparison_label: str
    variable_label: str
    mean_distance: float
    colors: tuple[str, str] = CORPORATE_PALETTE
    density_colorscale: str = "Viridis"
    scatter_limits: tuple[float, float] | None = None
    qq_limits: tuple[float, float] | None = None
    display_quantile: float | None = None
    width: int = 1000
    height: int = 1000


def build_multiplot(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
    metrics: PairMetrics,
    statistics_table: pd.DataFrame,
    options: PlotOptions,
) -> go.Figure:
    """Build the four-panel density/CDF/Q-Q/statistics figure."""
    y = np.asarray(reference_values, dtype=np.float64)
    x = np.asarray(comparison_values, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        raise ValueError("Cannot plot an empty paired sample.")

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
        horizontal_spacing=0.10,
        vertical_spacing=0.10,
        column_widths=[0.52, 0.48],
        row_heights=[0.52, 0.48],
    )

    ref_color, cmp_color = options.colors
    scatter_range = options.scatter_limits or _joint_axis_range(x, y, options.display_quantile)
    density = _kde_density(x, y)

    # [1] Top-left: density scatterplot.
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker={
                "size": 6,
                "color": density,
                "colorscale": options.density_colorscale,
                "showscale": False,
                "opacity": 0.92,
            },
            hovertemplate=(
                f"{options.comparison_label}: %{{x:.4g}}<br>"
                f"{options.reference_label}: %{{y:.4g}}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(scatter_range),
            y=list(scatter_range),
            mode="lines",
            line={"color": "black", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    mean_distance_text = (
        f"Mean distance = {options.mean_distance:.2f} m"
        if np.isfinite(options.mean_distance)
        else "Mean distance = n/a"
    )
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="x domain",
        yref="y domain",
        text=mean_distance_text,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"size": 13, "color": "black"},
        row=1,
        col=1,
    )

    annotation_position = _least_occupied_annotation_position(x, y, scatter_range)
    metric_text = (
        f"<i>n</i> : {metrics.n:,}<br>"
        f"<i>c</i> : {_fmt_coeff(metrics.rma_slope)}<br>"
        f"<i>ρ</i> : {_fmt_coeff(metrics.pearson_r)}<br>"
        f"<i>ρ<sub>s</sub></i> : {_fmt_coeff(metrics.spearman_r)}"
    )
    fig.add_annotation(
        x=annotation_position[0],
        y=annotation_position[1],
        xref="x domain",
        yref="y domain",
        text=metric_text,
        showarrow=False,
        xanchor=annotation_position[2],
        yanchor=annotation_position[3],
        align="left",
        bgcolor="rgba(255,255,255,0.80)",
        bordercolor="rgba(80,80,80,0.35)",
        borderwidth=1,
        borderpad=4,
        font={"size": 12, "color": "black"},
        row=1,
        col=1,
    )

    # [2] Top-right: empirical CDFs.
    ref_sorted, ref_prob = empirical_cdf(y)
    cmp_sorted, cmp_prob = empirical_cdf(x)
    fig.add_trace(
        go.Scatter(
            x=ref_sorted,
            y=ref_prob,
            mode="lines",
            name=f"{options.reference_label} {options.variable_label}",
            line={"color": ref_color, "width": 1.5},
            legendgroup="series",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=cmp_sorted,
            y=cmp_prob,
            mode="lines",
            name=f"{options.comparison_label} {options.variable_label}",
            line={"color": cmp_color, "width": 1.5},
            legendgroup="series",
        ),
        row=1,
        col=2,
    )

    cdf_range = _joint_axis_range(x, y, options.display_quantile, include_zero=False)

    # [3] Bottom-left: Q-Q plot.
    qq_x, qq_y = qq_values(y, x)
    qq_range = options.qq_limits or _joint_axis_range(qq_x, qq_y, options.display_quantile)
    fig.add_trace(
        go.Scatter(
            x=qq_x,
            y=qq_y,
            mode="lines",
            line={"color": cmp_color, "width": 1.4},
            hovertemplate="Comparison Q: %{x:.4g}<br>Reference Q: %{y:.4g}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(qq_range),
            y=list(qq_range),
            mode="lines",
            line={"color": ref_color, "width": 1.4},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # [4] Bottom-right: statistics table.
    table_values, has_flag = _format_statistics_for_plot(statistics_table)
    fig.add_trace(
        go.Table(
            columnwidth=[0.55, 1.35, 1.35, 1.00],
            header={
                "values": [
                    "",
                    f"{options.reference_label}<br>{options.variable_label}",
                    f"{options.comparison_label}<br>{options.variable_label}",
                    "Diff.(%)",
                ],
                "fill_color": [CORPORATE_NEUTRAL, ref_color, cmp_color, CORPORATE_NEUTRAL],
                "font": {"color": ["black", "white", "black", "black"], "size": 11},
                "align": ["left", "center", "center", "center"],
                "line_color": "black",
                "height": 36,
            },
            cells={
                "values": table_values,
                "fill_color": "white",
                "align": ["left", "center", "center", "center"],
                "font": {"size": 11, "color": "black"},
                "line_color": "black",
                "height": 34,
            },
        ),
        row=2,
        col=2,
    )

    # Axes and report-grade formatting.
    fig.update_xaxes(
        title_text=f"{options.comparison_label} {options.variable_label}",
        range=scatter_range,
        showgrid=False,
        zeroline=False,
        constrain="domain",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=f"{options.reference_label} {options.variable_label}",
        range=scatter_range,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text=options.variable_label,
        range=cdf_range,
        showgrid=True,
        gridcolor=CORPORATE_GRID,
        zeroline=False,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Probability",
        range=[0.0, 1.0],
        tickformat=".2f",
        dtick=0.2,
        showgrid=True,
        gridcolor=CORPORATE_GRID,
        zeroline=False,
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text=f"{options.comparison_label} {options.variable_label}",
        range=qq_range,
        showgrid=True,
        gridcolor=CORPORATE_GRID,
        zeroline=False,
        constrain="domain",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text=f"{options.reference_label} {options.variable_label}",
        range=qq_range,
        showgrid=True,
        gridcolor=CORPORATE_GRID,
        zeroline=False,
        scaleanchor="x3",
        scaleratio=1,
        row=2,
        col=1,
    )

    fig.update_layout(
        width=options.width,
        height=options.height,
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 12, "color": "black"},
        margin={"l": 70, "r": 35, "t": 25, "b": 80 if has_flag else 55},
        legend={
            "x": 0.985,
            "y": 0.985,
            "xanchor": "right",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.75)",
            "bordercolor": "rgba(120,120,120,0.35)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        hovermode="closest",
    )

    if has_flag:
        fig.add_annotation(
            x=0.75,
            y=-0.045,
            xref="paper",
            yref="paper",
            text="† Diff.(%) suppressed: reference denominator is near zero.",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font={"size": 10, "color": "#555555"},
        )

    return fig


def _kde_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a 2D Gaussian KDE at each observed point."""
    if x.size < 3:
        return np.ones(x.size, dtype=np.float64)
    values = np.vstack((x, y))
    try:
        kde = gaussian_kde(values)
        density = kde(values)
    except (np.linalg.LinAlgError, ValueError):
        density = np.ones(x.size, dtype=np.float64)
    order = np.argsort(density)
    ranked = np.empty_like(density)
    ranked[order] = density[order]
    return ranked


def _joint_axis_range(
    x: np.ndarray,
    y: np.ndarray,
    display_quantile: float | None,
    include_zero: bool = True,
) -> tuple[float, float]:
    combined = np.concatenate((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return (0.0, 1.0)

    if display_quantile is not None:
        if not 0.5 < display_quantile < 1.0:
            raise ValueError("display_quantile must be between 0.5 and 1.0.")
        lower_q = 1.0 - display_quantile
        lower = float(np.quantile(combined, lower_q))
        upper = float(np.quantile(combined, display_quantile))
    else:
        lower = float(np.min(combined))
        upper = float(np.max(combined))

    if include_zero and lower >= 0.0:
        lower = 0.0
    if math.isclose(lower, upper):
        span = max(abs(lower), 1.0)
        lower -= 0.05 * span
        upper += 0.05 * span
    else:
        span = upper - lower
        if lower < 0.0:
            lower -= 0.02 * span
        upper += 0.02 * span
    return (lower, upper)


def _least_occupied_annotation_position(
    x: np.ndarray,
    y: np.ndarray,
    axis_range: tuple[float, float],
) -> tuple[float, float, str, str]:
    """Choose a low-occupancy corner for the scatter metric annotation."""
    low, high = axis_range
    span = high - low
    if span <= 0:
        return (0.98, 0.05, "right", "bottom")
    xn = (x - low) / span
    yn = (y - low) / span

    candidates = [
        # x, y, xanchor, yanchor, xmin, xmax, ymin, ymax
        (0.98, 0.05, "right", "bottom", 0.68, 1.00, 0.00, 0.30),
        (0.98, 0.70, "right", "bottom", 0.68, 1.00, 0.66, 0.96),
        (0.02, 0.05, "left", "bottom", 0.00, 0.32, 0.00, 0.30),
    ]
    scored = []
    for candidate in candidates:
        _, _, _, _, xmin, xmax, ymin, ymax = candidate
        count = int(np.sum((xn >= xmin) & (xn <= xmax) & (yn >= ymin) & (yn <= ymax)))
        scored.append((count, candidate))
    _, best = min(scored, key=lambda item: item[0])
    return best[0], best[1], best[2], best[3]


def _format_statistics_for_plot(stats: pd.DataFrame) -> tuple[list[list[str]], bool]:
    metrics: list[str] = []
    ref_text: list[str] = []
    cmp_text: list[str] = []
    diff_text: list[str] = []
    has_flag = False

    for row in stats.itertuples(index=False):
        metric = str(row.metric)
        metrics.append(metric)
        is_count = metric.lower() in {"count", "n"}
        ref_text.append(_format_number(row.reference, is_count=is_count))
        cmp_text.append(_format_number(row.comparison, is_count=is_count))
        if bool(row.diff_flag):
            diff_text.append("n/a †")
            has_flag = True
        else:
            diff_text.append(_format_number(row.diff_pct, is_count=False))
    return [metrics, ref_text, cmp_text, diff_text], has_flag


def _format_number(value: float, is_count: bool) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    if is_count:
        return f"{float(value):,.0f}"
    return f"{float(value):,.2f}"


def _fmt_coeff(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.2f}"


@dataclass(frozen=True)
class PairLocationPlotOptions:
    """Display controls for the paired-sample spatial location plot."""

    projection: str = "XY"
    reference_label: str = "Reference"
    comparison_label: str = "Comparison"
    variable_label: str = "Variable"
    search_distance: float = math.nan
    colors: tuple[str, str] = CORPORATE_PALETTE
    height: int = 620


def build_pair_location_plot(
    reference_coordinates: np.ndarray,
    comparison_coordinates: np.ndarray,
    distances: np.ndarray,
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
    reference_ids: np.ndarray,
    comparison_ids: np.ndarray,
    options: PairLocationPlotOptions,
) -> go.Figure:
    """Plot the exact analysis-valid sample pairs in XY, XZ, or YZ projection.

    Each pair is represented by a light connector between its reference and
    comparison locations, with the two series overlaid as distinct markers.
    The plot uses the original mapped coordinates for visualization; the
    pairing engine may independently ignore Z when 2D search mode is enabled.
    """
    ref = np.asarray(reference_coordinates, dtype=np.float64)
    cmp = np.asarray(comparison_coordinates, dtype=np.float64)
    distance_values = np.asarray(distances, dtype=np.float64)
    ref_values = np.asarray(reference_values, dtype=np.float64)
    cmp_values = np.asarray(comparison_values, dtype=np.float64)
    ref_ids = np.asarray(reference_ids, dtype=object)
    cmp_ids = np.asarray(comparison_ids, dtype=object)

    if ref.ndim != 2 or cmp.ndim != 2 or ref.shape[1] != 3 or cmp.shape[1] != 3:
        raise ValueError("Pair location coordinates must be N x 3 arrays in X, Y, Z order.")
    n_pairs = ref.shape[0]
    if n_pairs == 0 or cmp.shape[0] != n_pairs:
        raise ValueError("Pair location plot requires a non-empty, aligned pair set.")
    aligned_sizes = {
        distance_values.size,
        ref_values.size,
        cmp_values.size,
        ref_ids.size,
        cmp_ids.size,
        n_pairs,
    }
    if len(aligned_sizes) != 1:
        raise ValueError("Pair location inputs must contain one aligned row per pair.")

    projection = str(options.projection).upper().strip()
    projection_axes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
    if projection not in projection_axes:
        raise ValueError("Projection must be one of XY, XZ, or YZ.")
    x_axis, y_axis = projection_axes[projection]
    axis_labels = ("X", "Y", "Z")

    visible = np.column_stack(
        [ref[:, x_axis], ref[:, y_axis], cmp[:, x_axis], cmp[:, y_axis]]
    )
    if not np.isfinite(visible).all():
        raise ValueError(
            f"Projection {projection} contains missing, non-numeric, or infinite coordinates."
        )

    # One batched line trace is much faster than one Plotly trace per pair.
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    for pair_index in range(n_pairs):
        line_x.extend(
            [float(ref[pair_index, x_axis]), float(cmp[pair_index, x_axis]), None]
        )
        line_y.extend(
            [float(ref[pair_index, y_axis]), float(cmp[pair_index, y_axis]), None]
        )

    pair_number = np.arange(1, n_pairs + 1, dtype=np.int64)
    reference_custom = np.column_stack(
        [pair_number, ref_ids, cmp_ids, distance_values, ref_values]
    )
    comparison_custom = np.column_stack(
        [pair_number, cmp_ids, ref_ids, distance_values, cmp_values]
    )

    ref_color, cmp_color = options.colors
    view_name = {
        "XY": "Plan View",
        "XZ": "X–Z Section",
        "YZ": "Y–Z Section",
    }[projection]
    threshold_text = (
        f" · strict threshold &lt; {options.search_distance:.2f} m"
        if np.isfinite(options.search_distance)
        else ""
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=line_x,
            y=line_y,
            mode="lines",
            line={"color": "rgba(68,84,106,0.28)", "width": 1.1},
            hoverinfo="skip",
            showlegend=False,
            name="Pair connection",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=ref[:, x_axis],
            y=ref[:, y_axis],
            mode="markers",
            name=options.reference_label,
            customdata=reference_custom,
            marker={
                "size": 8,
                "color": ref_color,
                "symbol": "circle",
                "opacity": 0.90,
                "line": {"color": "white", "width": 0.7},
            },
            hovertemplate=(
                "Pair %{customdata[0]:,.0f}<br>"
                f"{options.reference_label}: %{{customdata[1]}}<br>"
                f"{options.comparison_label}: %{{customdata[2]}}<br>"
                "Separation: %{customdata[3]:.3f} m<br>"
                f"{options.variable_label}: %{{customdata[4]:.4g}}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=cmp[:, x_axis],
            y=cmp[:, y_axis],
            mode="markers",
            name=options.comparison_label,
            customdata=comparison_custom,
            marker={
                "size": 8,
                "color": cmp_color,
                "symbol": "diamond",
                "opacity": 0.90,
                "line": {"color": "white", "width": 0.7},
            },
            hovertemplate=(
                "Pair %{customdata[0]:,.0f}<br>"
                f"{options.comparison_label}: %{{customdata[1]}}<br>"
                f"{options.reference_label}: %{{customdata[2]}}<br>"
                "Separation: %{customdata[3]:.3f} m<br>"
                f"{options.variable_label}: %{{customdata[4]:.4g}}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title={
            "text": (
                f"Paired Sample Locations — {projection} {view_name}"
                f"<br><sup>{n_pairs:,} analysis-valid pairs{threshold_text}</sup>"
            ),
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 18, "color": "#004967"},
        },
        height=options.height,
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font={"family": "Arial, sans-serif", "size": 12, "color": "#26323A"},
        margin={"l": 70, "r": 35, "t": 80, "b": 70},
        legend={
            "orientation": "h",
            "y": 1.02,
            "x": 1.0,
            "xanchor": "right",
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.78)",
            "bordercolor": "rgba(0,84,124,0.15)",
            "borderwidth": 1,
        },
        hovermode="closest",
    )
    fig.update_xaxes(
        title_text=f"{axis_labels[x_axis]} (m)",
        showgrid=True,
        gridcolor=CORPORATE_GRID,
        zeroline=False,
        constrain="domain",
    )
    fig.update_yaxes(
        title_text=f"{axis_labels[y_axis]} (m)",
        showgrid=True,
        gridcolor=CORPORATE_GRID,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )
    return fig
