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
MAP_BACKGROUND = "#F7F9FA"
MAP_GRID = "rgba(93,115,128,0.20)"
MAP_FRAME = "#526773"


@dataclass(frozen=True)
class PlotOptions:
    """User-adjustable plotting controls."""

    reference_label: str
    comparison_label: str
    variable_label: str
    mean_distance: float
    colors: tuple[str, str] = CORPORATE_PALETTE
    density_colorscale: str = "Viridis"
    scatter_x_limits: tuple[float, float] | None = None
    scatter_y_limits: tuple[float, float] | None = None
    qq_x_limits: tuple[float, float] | None = None
    qq_y_limits: tuple[float, float] | None = None
    link_xy_axes: bool = True
    display_quantile: float | None = None
    title: str | None = None
    subtitle: str | None = None
    footer_note: str | None = None
    width: int = 920
    height: int = 900


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
        vertical_spacing=0.20,
        column_widths=[0.52, 0.48],
        row_heights=[0.52, 0.48],
    )

    ref_color, cmp_color = options.colors

    if options.link_xy_axes:
        scatter_common_range = (
            options.scatter_x_limits
            or options.scatter_y_limits
            or _joint_axis_range(x, y, options.display_quantile)
        )
        scatter_x_range = scatter_common_range
        scatter_y_range = scatter_common_range
    else:
        scatter_x_range = options.scatter_x_limits or _single_axis_range(
            x, options.display_quantile
        )
        scatter_y_range = options.scatter_y_limits or _single_axis_range(
            y, options.display_quantile
        )

    scatter_reference_range = _reference_line_range(scatter_x_range, scatter_y_range)
    density = _kde_density(x, y)

    # [1] Top-left: density scatterplot.
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker={
                "size": 9.5,
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
            x=list(scatter_reference_range),
            y=list(scatter_reference_range),
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
        y=1.025,
        xref="x domain",
        yref="y domain",
        text=mean_distance_text,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 13, "color": "black"},
        row=1,
        col=1,
    )

    annotation_position = _least_occupied_annotation_position(
        x, y, scatter_reference_range
    )
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
        font={"size": 13, "color": "black"},
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

    cdf_range = _joint_axis_range(
        x, y, options.display_quantile, include_zero=False
    )

    # [3] Bottom-left: Q-Q plot.
    qq_x, qq_y = qq_values(y, x)
    if options.link_xy_axes:
        qq_common_range = (
            options.qq_x_limits
            or options.qq_y_limits
            or _joint_axis_range(qq_x, qq_y, options.display_quantile)
        )
        qq_x_range = qq_common_range
        qq_y_range = qq_common_range
    else:
        qq_x_range = options.qq_x_limits or _single_axis_range(
            qq_x, options.display_quantile
        )
        qq_y_range = options.qq_y_limits or _single_axis_range(
            qq_y, options.display_quantile
        )

    qq_reference_range = _reference_line_range(qq_x_range, qq_y_range)
    fig.add_trace(
        go.Scatter(
            x=qq_x,
            y=qq_y,
            mode="lines",
            line={"color": cmp_color, "width": 1.4},
            hovertemplate=(
                "Comparison Q: %{x:.4g}<br>Reference Q: %{y:.4g}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(qq_reference_range),
            y=list(qq_reference_range),
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
            columnwidth=[0.52, 1.28, 1.28, 0.95],
            header={
                "values": [
                    "",
                    f"{options.reference_label}<br>{options.variable_label}",
                    f"{options.comparison_label}<br>{options.variable_label}",
                    "Diff.(%)",
                ],
                "fill_color": [
                    CORPORATE_NEUTRAL,
                    ref_color,
                    cmp_color,
                    CORPORATE_NEUTRAL,
                ],
                "font": {
                    "color": ["black", "white", "black", "black"],
                    "size": 13,
                },
                "align": ["left", "center", "center", "center"],
                "line_color": "black",
                "height": 32,
            },
            cells={
                "values": table_values,
                "fill_color": "white",
                "align": ["left", "center", "center", "center"],
                "font": {"size": 13, "color": "black"},
                "line_color": "black",
                "height": 22,
            },
        ),
        row=2,
        col=2,
    )

    # Axes and report-grade formatting.
    scatter_axis_common = {
        "autorange": False,
        "showgrid": True,
        "gridcolor": "rgba(0,0,0,0.12)",
        "gridwidth": 1,
        "layer": "below traces",
        "zeroline": False,
        "constrain": "domain",
    }
    fig.update_xaxes(
        title_text=f"{options.comparison_label} {options.variable_label}",
        range=list(scatter_x_range),
        **scatter_axis_common,
        row=1,
        col=1,
    )
    scatter_y_axis = dict(scatter_axis_common)
    if options.link_xy_axes:
        scatter_y_axis.update({"scaleanchor": "x", "scaleratio": 1})
    fig.update_yaxes(
        title_text=f"{options.reference_label} {options.variable_label}",
        range=list(scatter_y_range),
        **scatter_y_axis,
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

    qq_axis_common = {
        "autorange": False,
        "showgrid": True,
        "gridcolor": CORPORATE_GRID,
        "gridwidth": 1,
        "layer": "below traces",
        "zeroline": False,
        "constrain": "domain",
    }
    fig.update_xaxes(
        title_text=f"{options.comparison_label} {options.variable_label}",
        range=list(qq_x_range),
        **qq_axis_common,
        row=2,
        col=1,
    )
    qq_y_axis = dict(qq_axis_common)
    if options.link_xy_axes:
        qq_y_axis.update({"scaleanchor": "x3", "scaleratio": 1})
    fig.update_yaxes(
        title_text=f"{options.reference_label} {options.variable_label}",
        range=list(qq_y_range),
        **qq_y_axis,
        row=2,
        col=1,
    )

    title_text = options.title or f"Paired-Sample Comparison — {options.variable_label}"
    if options.subtitle:
        title_text += f"<br><sup>{options.subtitle}</sup>"

    bottom_margin = 125
    if options.footer_note:
        bottom_margin += 48
    if has_flag:
        bottom_margin += 18

    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.01,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
            "font": {"size": 20, "color": "#004967"},
        },
        width=options.width,
        height=options.height,
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 13, "color": "black"},
        margin={"l": 70, "r": 35, "t": 95, "b": bottom_margin},
        legend={
            "x": 0.985,
            "y": 0.985,
            "xanchor": "right",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.75)",
            "bordercolor": "rgba(120,120,120,0.35)",
            "borderwidth": 1,
            "font": {"size": 13},
        },
        hovermode="closest",
    )

    # Preserve the original, roomier analytical-panel geometry. Only the
    # statistics table keeps a slightly extended vertical domain so its
    # final row remains fully visible in responsive rendering and exports.
    table_trace = next((trace for trace in fig.data if isinstance(trace, go.Table)), None)
    if table_trace is not None:
        table_trace.domain.y = [0.035, 0.485]

    footer_y = -0.075
    if options.footer_note:
        fig.add_annotation(
            x=0.0,
            y=footer_y,
            xref="paper",
            yref="paper",
            text=_format_footer_annotation(options.footer_note),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 12.5, "color": "#53636C"},
        )
        footer_y -= 0.065

    if has_flag:
        fig.add_annotation(
            x=0.0,
            y=footer_y,
            xref="paper",
            yref="paper",
            text="† Diff.(%) suppressed: reference denominator is near zero.",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={"size": 12, "color": "#555555"},
        )

    return fig


def _format_footer_annotation(text: str) -> str:
    """Wrap a long filter summary onto two readable lines."""
    parts = [part.strip() for part in str(text).split(" · ") if part.strip()]
    if len(parts) <= 3:
        return " · ".join(parts)
    split_at = max(2, (len(parts) + 1) // 2)
    return " · ".join(parts[:split_at]) + "<br>" + " · ".join(parts[split_at:])


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


def _single_axis_range(
    values: np.ndarray,
    display_quantile: float | None,
    include_zero: bool = True,
) -> tuple[float, float]:
    """Return a display range for one axis without coupling it to another axis."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)

    if display_quantile is not None:
        if not 0.5 < display_quantile < 1.0:
            raise ValueError("display_quantile must be between 0.5 and 1.0.")
        lower_q = 1.0 - display_quantile
        lower = float(np.quantile(finite, lower_q))
        upper = float(np.quantile(finite, display_quantile))
    else:
        lower = float(np.min(finite))
        upper = float(np.max(finite))

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


def _reference_line_range(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[float, float]:
    """Return a 1:1 line span covering the complete requested X/Y ranges."""
    return (min(x_range[0], y_range[0]), max(x_range[1], y_range[1]))


def _joint_axis_range(
    x: np.ndarray,
    y: np.ndarray,
    display_quantile: float | None,
    include_zero: bool = True,
) -> tuple[float, float]:
    combined = np.concatenate(
        (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    )
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
        count = int(
            np.sum((xn >= xmin) & (xn <= xmax) & (yn >= ymin) & (yn <= ymax))
        )
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
    height: int = 650


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
    """Plot exact analysis-valid pairs with a cartographic XY plan view.

    XY is rendered as a map-style technical plan with equal coordinate scale,
    north arrow, scale bar, frame, coordinate grid and legend. XZ and YZ retain
    the same visual language as technical sections. A web basemap is deliberately
    not used because the application accepts arbitrary local/projected mine grids
    and does not require a CRS definition.
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
            showlegend=True,
            name="Pair connector",
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
                "size": 10,
                "color": ref_color,
                "symbol": "circle",
                "opacity": 0.92,
                "line": {"color": "white", "width": 0.8},
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
                "size": 10,
                "color": cmp_color,
                "symbol": "diamond",
                "opacity": 0.92,
                "line": {"color": "white", "width": 0.8},
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

    x_values = np.concatenate((ref[:, x_axis], cmp[:, x_axis]))
    y_values = np.concatenate((ref[:, y_axis], cmp[:, y_axis]))
    x_range, y_range = _map_ranges(x_values, y_values)

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
        plot_bgcolor=MAP_BACKGROUND,
        paper_bgcolor="#FFFFFF",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#26323A"},
        margin={"l": 75, "r": 45, "t": 92, "b": 78},
        legend={
            "orientation": "v",
            "y": 0.985,
            "x": 0.015,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "rgba(0,84,124,0.22)",
            "borderwidth": 1,
            "font": {"size": 13},
        },
        hovermode="closest",
    )
    fig.update_xaxes(
        title_text=f"{axis_labels[x_axis]} (m)",
        range=x_range,
        showgrid=True,
        gridcolor=MAP_GRID,
        gridwidth=1,
        zeroline=False,
        constrain="domain",
        showline=True,
        mirror=True,
        linecolor=MAP_FRAME,
        linewidth=1.2,
        ticks="outside",
        tickcolor=MAP_FRAME,
        tickformat=",.0f",
    )
    fig.update_yaxes(
        title_text=f"{axis_labels[y_axis]} (m)",
        range=y_range,
        showgrid=True,
        gridcolor=MAP_GRID,
        gridwidth=1,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        showline=True,
        mirror=True,
        linecolor=MAP_FRAME,
        linewidth=1.2,
        ticks="outside",
        tickcolor=MAP_FRAME,
        tickformat=",.0f",
    )

    if projection == "XY":
        _add_north_arrow(fig, x_range, y_range)
        _add_scale_bar(fig, x_range, y_range)
    else:
        _add_section_orientation(fig, projection, x_range, y_range)

    return fig


def _map_ranges(
    x_values: np.ndarray,
    y_values: np.ndarray,
    padding_fraction: float = 0.06,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return padded display ranges while preserving meaningful metric extents."""
    xmin = float(np.min(x_values))
    xmax = float(np.max(x_values))
    ymin = float(np.min(y_values))
    ymax = float(np.max(y_values))
    x_span = xmax - xmin
    y_span = ymax - ymin
    if math.isclose(x_span, 0.0):
        x_span = max(abs(xmin), 1.0) * 0.10
    if math.isclose(y_span, 0.0):
        y_span = max(abs(ymin), 1.0) * 0.10
    return (
        (xmin - padding_fraction * x_span, xmax + padding_fraction * x_span),
        (ymin - padding_fraction * y_span, ymax + padding_fraction * y_span),
    )


def _nice_scale_length(target: float) -> float:
    """Round a target map scale length to a conventional 1/2/5 × 10^n value."""
    if not np.isfinite(target) or target <= 0:
        return 1.0
    exponent = math.floor(math.log10(target))
    fraction = target / (10**exponent)
    if fraction < 1.5:
        nice_fraction = 1.0
    elif fraction < 3.5:
        nice_fraction = 2.0
    elif fraction < 7.5:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10**exponent)


def _add_scale_bar(
    fig: go.Figure,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    """Add a metric scale bar to the XY plan view."""
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    length = _nice_scale_length(x_span * 0.18)
    x0 = x_range[0] + 0.065 * x_span
    x1 = x0 + length
    y0 = y_range[0] + 0.075 * y_span
    tick = 0.012 * y_span

    fig.add_shape(
        type="line",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y0,
        xref="x",
        yref="y",
        line={"color": "#26323A", "width": 5},
        layer="above",
    )
    for x_value in (x0, x1):
        fig.add_shape(
            type="line",
            x0=x_value,
            x1=x_value,
            y0=y0 - tick,
            y1=y0 + tick,
            xref="x",
            yref="y",
            line={"color": "#26323A", "width": 2},
            layer="above",
        )
    fig.add_annotation(
        x=(x0 + x1) / 2,
        y=y0 + 0.020 * y_span,
        xref="x",
        yref="y",
        text=f"<b>{length:,.0f} m</b>",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.78)",
        borderpad=2,
        font={"size": 11, "color": "#26323A"},
    )


def _add_north_arrow(
    fig: go.Figure,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    """Add a conventional north arrow in the upper-right map corner."""
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    north_x = x_range[1] - 0.07 * x_span
    north_y = y_range[1] - 0.06 * y_span
    fig.add_annotation(
        x=north_x,
        y=north_y,
        xref="x",
        yref="y",
        text="<b>N</b>",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.3,
        arrowwidth=2.2,
        arrowcolor="#26323A",
        ax=0,
        ay=48,
        font={"size": 16, "color": "#26323A"},
        bgcolor="rgba(255,255,255,0.72)",
        borderpad=2,
    )


def _add_section_orientation(
    fig: go.Figure,
    projection: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    """Add a compact orientation cue to XZ/YZ technical sections."""
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    label = "Elevation ↑" if projection in {"XZ", "YZ"} else ""
    if not label:
        return
    fig.add_annotation(
        x=x_range[1] - 0.03 * x_span,
        y=y_range[1] - 0.035 * y_span,
        xref="x",
        yref="y",
        text=f"<b>{label}</b>",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.80)",
        bordercolor="rgba(82,103,115,0.25)",
        borderwidth=1,
        borderpad=4,
        font={"size": 11, "color": "#526773"},
    )


def build_pairing_sensitivity_plot(
    sensitivity: pd.DataFrame,
    selected_distance: float,
    pairing_mode: str,
    search_geometry: str,
    colors: tuple[str, str] = CORPORATE_PALETTE,
) -> go.Figure:
    """Plot pair count and reference pairing rate versus search distance."""
    required = {
        "search_distance_m",
        "pairs",
        "pairing_rate_pct",
    }
    missing = required.difference(sensitivity.columns)
    if missing:
        raise ValueError(
            "Pairing sensitivity data are missing required columns: "
            + ", ".join(sorted(missing))
        )
    if sensitivity.empty:
        raise ValueError("Pairing sensitivity plot requires at least one search distance.")

    frame = sensitivity.sort_values("search_distance_m").reset_index(drop=True)
    ref_color, cmp_color = colors

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=frame["search_distance_m"],
            y=frame["pairs"],
            mode="lines+markers",
            name="Pairs",
            line={"color": cmp_color, "width": 2.2},
            marker={"size": 9},
            hovertemplate=(
                "Search distance: %{x:.2f} m<br>"
                "Pairs: %{y:,.0f}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["search_distance_m"],
            y=frame["pairing_rate_pct"],
            mode="lines+markers",
            name="Pairing rate",
            line={"color": ref_color, "width": 2.2},
            marker={"size": 9, "symbol": "diamond"},
            hovertemplate=(
                "Search distance: %{x:.2f} m<br>"
                "Pairing rate: %{y:.2f}%<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    if np.isfinite(selected_distance) and selected_distance > 0.0:
        fig.add_vline(
            x=float(selected_distance),
            line_width=1.4,
            line_dash="dash",
            line_color="#5B6570",
        )
        fig.add_annotation(
            x=float(selected_distance),
            y=1.02,
            xref="x",
            yref="paper",
            text=f"Selected = {selected_distance:.2f} m",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font={"size": 11, "color": "#4B5660"},
            bgcolor="rgba(255,255,255,0.86)",
        )

    fig.update_xaxes(
        title_text="Search distance (m)",
        showgrid=True,
        gridcolor=CORPORATE_GRID,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Pairs",
        rangemode="tozero",
        showgrid=True,
        gridcolor=CORPORATE_GRID,
        zeroline=False,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Pairing rate (%)",
        range=[0.0, 100.0],
        ticksuffix="%",
        showgrid=False,
        zeroline=False,
        secondary_y=True,
    )

    fig.update_layout(
        title={
            "text": (
                "Pairing Sensitivity — Search Distance"
                f"<br><sup>{search_geometry} · {pairing_mode}</sup>"
            ),
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 18, "color": "#004967"},
        },
        template="plotly_white",
        height=420,
        margin={"l": 65, "r": 65, "t": 80, "b": 60},
        font={"family": "Arial, sans-serif", "size": 13, "color": "#26323A"},
        legend={
            "orientation": "h",
            "x": 1.0,
            "xanchor": "right",
            "y": 1.02,
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.80)",
            "font": {"size": 13},
        },
        hovermode="x unified",
    )
    return fig

