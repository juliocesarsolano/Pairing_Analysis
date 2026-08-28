"""Premium Streamlit page for GETPAIRS-compatible paired-sample analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.core.config import (
    GetPairsParameters,
    export_getpairs_par,
    indices_for_uploaded_file,
    parse_getpairs_par,
)
from src.core.io import (
    LoadedTable,
    build_paired_dataframe,
    load_table,
    paired_to_gslib_bytes,
)
from src.core.pairing import (
    PairingResult,
    pair_records_kdtree,
    pairing_report,
    prepare_coordinate_arrays,
)
from src.core.plotting import (
    PairLocationPlotOptions,
    PlotOptions,
    build_multiplot,
    build_pair_location_plot,
    build_pairing_sensitivity_plot,
)
from src.core.statistics import (
    compute_pair_metrics,
    compute_statistics_table,
    extract_pair_values,
)


from src.ui.palettes import DEFAULT_FIGURE_PALETTE, FIGURE_PALETTES
from src.ui.theme import (
    pairing_mode_note,
    render_about,
    render_header,
    render_kpi_cards,
    sidebar_banner,
)


SENSITIVITY_RANGES_M = (1.0, 2.0, 3.0, 4.0, 5.0)


CANONICAL_VARIABLES = {
    "au_ppm": [
        "au_ppm",
        "au_ppm_d",
        "au",
        "gold",
        "au_gpt",
        "au_gt",
    ],
    "ag_ppm": [
        "ag_ppm",
        "ag_ppm_d",
        "ag",
        "silver",
        "ag_gpt",
        "ag_gt",
    ],
    "s_tot_pct": [
        "s_tot_pct",
        "stot_pct",
        "stot_pct_d",
        "s_tot_pct_d",
        "stot",
        "s_tot",
        "total_sulfur",
        "sulfur_total",
    ],
    "s2_pct": [
        "s2_pct",
        "s2_pct_d",
        "s2",
        "sulfide_sulfur",
        "sulfide_s",
    ],
    "c_tot_pct": [
        "c_tot_pct",
        "ctot_pct",
        "ctot_pct_d",
        "c_tot_pct_d",
        "ctot",
        "c_tot",
        "total_carbon",
    ],
    "c_org_pct": [
        "c_org_pct",
        "corg_pct",
        "oc_pct",
        "oc_pct_d",
        "corg",
        "c_org",
        "organic_carbon",
    ],
    "cu_pct": [
        "cu_pct",
        "cu_pct_d",
        "cu",
        "copper",
    ],
    "zn_pct": [
        "zn_pct",
        "zn_pct_d",
        "zn",
        "zinc",
    ],
    "cao_pct": [
        "cao_pct",
        "cao_pct_d",
        "cao",
    ],
    "sio2_pct": [
        "sio2_pct",
        "sio2_pct_d",
        "sio2",
    ],
    "Custom": [],
}


# Preferred fields are promoted to the top of the Primary Categorical Filter.
# Remaining valid categorical fields are still available below them.
PREFERRED_CATEGORICAL_VARIABLES = (
    "Destination",
    "metype_txt",
    "Mettype",
    "LithType",
    "ModelCode",
    "PI_ModelCode",
    "PI_MinAssemblage",
    "SAMPLETYPE",
    "PROJECTCODE",
    "YEAR",
    "STARTDATE",
)


# Fields that should not be proposed as categorical filters merely because
# they happen to have low cardinality in a small dataset.
NON_CATEGORICAL_FIELDS = {
    "holeid",
    "hole_id",
    "sampleid",
    "sample_id",
    "sampleid_2",
    "from",
    "to",
    "mid_x",
    "mid_y",
    "mid_z",
    "x",
    "y",
    "z",
    "easting",
    "northing",
    "elevation",
    "elev",
    "rl",
}


@st.cache_data(show_spinner=False)
def cached_load_table(payload: bytes, source_name: str) -> LoadedTable:
    """Cache file parsing by content and filename."""
    return load_table(payload, source_name)


@st.cache_data(show_spinner=False)
def cached_pairing(
    ref_coords: np.ndarray,
    cmp_coords: np.ndarray,
    dismax: float,
    keep_closest: bool,
) -> PairingResult:
    """Cache the expensive KD-tree search."""
    return pair_records_kdtree(ref_coords, cmp_coords, dismax, keep_closest)


@st.cache_data(show_spinner=False)
def cached_pairing_sensitivity(
    ref_coords: np.ndarray,
    cmp_coords: np.ndarray,
    ref_variable_values: np.ndarray,
    cmp_variable_values: np.ndarray,
    search_distances: tuple[float, ...],
    keep_closest: bool,
) -> pd.DataFrame:
    """Evaluate analysis-valid pairing response across search distances."""
    rows: list[dict[str, float | int]] = []
    n_reference = int(ref_coords.shape[0])

    for radius in search_distances:
        result = pair_records_kdtree(
            ref_coords, cmp_coords, float(radius), keep_closest
        )
        if result.n_pairs:
            ref_values = ref_variable_values[result.reference_indices]
            cmp_values = cmp_variable_values[result.comparison_indices]
            valid = np.isfinite(ref_values) & np.isfinite(cmp_values)
            ref_indices = result.reference_indices[valid]
            distances = result.distances[valid]
        else:
            ref_indices = np.empty(0, dtype=np.int64)
            distances = np.empty(0, dtype=np.float64)

        n_pairs = int(distances.size)
        n_paired_reference = int(np.unique(ref_indices).size) if n_pairs else 0
        pairing_rate = (
            100.0 * n_paired_reference / n_reference if n_reference else 0.0
        )
        rows.append(
            {
                "search_distance_m": float(radius),
                "pairs": n_pairs,
                "paired_reference": n_paired_reference,
                "pairing_rate_pct": pairing_rate,
                "mean_separation_m": (
                    float(np.mean(distances)) if n_pairs else np.nan
                ),
                "max_separation_m": (
                    float(np.max(distances)) if n_pairs else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_pair_location_plot(
    reference_coordinates: np.ndarray,
    comparison_coordinates: np.ndarray,
    distances: np.ndarray,
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
    reference_ids: np.ndarray,
    comparison_ids: np.ndarray,
    projection: str,
    reference_label: str,
    comparison_label: str,
    variable_label: str,
    search_distance: float,
    colors: tuple[str, str],
):
    """Cache construction of the paired-sample spatial location plot."""
    options = PairLocationPlotOptions(
        projection=projection,
        reference_label=reference_label,
        comparison_label=comparison_label,
        variable_label=variable_label,
        search_distance=search_distance,
        colors=colors,
    )
    return build_pair_location_plot(
        reference_coordinates=reference_coordinates,
        comparison_coordinates=comparison_coordinates,
        distances=distances,
        reference_values=reference_values,
        comparison_values=comparison_values,
        reference_ids=reference_ids,
        comparison_ids=comparison_ids,
        options=options,
    )


@st.cache_data(show_spinner=False)
def cached_multiplot(
    ref_values: np.ndarray,
    cmp_values: np.ndarray,
    distances: np.ndarray,
    stats_frame: pd.DataFrame,
    reference_label: str,
    comparison_label: str,
    variable_label: str,
    colors: tuple[str, str],
    density_colorscale: str,
    scatter_x_limits: tuple[float, float] | None,
    scatter_y_limits: tuple[float, float] | None,
    qq_x_limits: tuple[float, float] | None,
    qq_y_limits: tuple[float, float] | None,
    link_xy_axes: bool,
    display_quantile: float | None,
    figure_title: str,
    figure_subtitle: str,
    filter_note: str,
):
    """Cache KDE and Plotly figure construction."""
    metrics = compute_pair_metrics(ref_values, cmp_values)
    options = PlotOptions(
        reference_label=reference_label,
        comparison_label=comparison_label,
        variable_label=variable_label,
        mean_distance=float(np.mean(distances)) if distances.size else np.nan,
        colors=colors,
        density_colorscale=density_colorscale,
        scatter_x_limits=scatter_x_limits,
        scatter_y_limits=scatter_y_limits,
        qq_x_limits=qq_x_limits,
        qq_y_limits=qq_y_limits,
        link_xy_axes=link_xy_axes,
        display_quantile=display_quantile,
        title=figure_title,
        subtitle=figure_subtitle,
        footer_note=filter_note,
    )
    return build_multiplot(ref_values, cmp_values, metrics, stats_frame, options)


def render_app() -> None:
    """Render the complete premium pairing-analysis workflow."""
    render_header(
        [
            ("Engine", "GETPAIRS v2.000"),
            ("Search", "3D spherical · nearest default"),
            ("Platform", "Python · Streamlit"),
        ]
    )

    with st.sidebar:
        sidebar_banner("Data", "Input datasets")
        upload_a = st.file_uploader(
            "Dataset A",
            type=["dat", "out", "csv", "txt", "xlsx", "xls"],
            key="dataset_a",
        )
        upload_b = st.file_uploader(
            "Dataset B",
            type=["dat", "out", "csv", "txt", "xlsx", "xls"],
            key="dataset_b",
        )
        par_upload = st.file_uploader(
            "Optional getpairs.par", type=["par", "txt"], key="par"
        )

    render_about()

    par_params = _read_par_upload(par_upload)
    if upload_a is None or upload_b is None:
        st.info(
            "Upload Dataset A and Dataset B to start the spatial pairing analysis. "
            "GSLIB/GeoEAS, CSV and Excel are supported."
        )
        return

    try:
        table_a = cached_load_table(upload_a.getvalue(), upload_a.name)
        table_b = cached_load_table(upload_b.getvalue(), upload_b.name)
    except Exception as exc:
        st.error(str(exc))
        return

    # Nested categorical filters: always visible in the main banner area.
    # Layer 2 is evaluated on the subset produced by Layer 1.
    categorical_filters_cfg: list[dict[str, Any]] = []
    categorical_a = _categorical_candidates(table_a.frame, set())
    categorical_b = _categorical_candidates(table_b.frame, set())
    no_filter_label = "<No filter>"

    st.markdown(
        """
        <div style="margin:0.15rem 0 0.45rem 0;padding:0.72rem 0.95rem;
                    border-left:4px solid #A39161;border-radius:8px;
                    background:linear-gradient(90deg,rgba(3,84,124,0.10),rgba(255,255,255,0.88));
                    border-top:1px solid rgba(3,84,124,0.10);
                    border-right:1px solid rgba(3,84,124,0.10);
                    border-bottom:1px solid rgba(3,84,124,0.10);">
            <div style="font-size:0.68rem;font-weight:800;letter-spacing:0.10em;
                        color:#6C7881;text-transform:uppercase;">Primary Filters</div>
            <div style="font-size:1.02rem;font-weight:800;color:#004967;">
                Nested Categorical Filters
            </div>
            <div style="font-size:0.84rem;color:#52636D;margin-top:0.15rem;">
                Layer 2 is evaluated inside the subset selected by Layer 1. Multiple values within
                a layer use OR logic; active layers are combined with AND logic.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("#### Filter Layer 1")

        field_col_a, field_col_b = st.columns(2)
        options_a = [no_filter_label, *categorical_a]
        category_a = field_col_a.selectbox(
            "Dataset A field · Layer 1",
            options_a,
            index=0,
            key="categorical_l1_a",
            help=(
                "Choose the first categorical/domain field. Leave <No filter> to analyze all data."
            ),
        )

        options_b = [no_filter_label, *categorical_b]
        matched_b = (
            _matching_column(category_a, categorical_b)
            if category_a != no_filter_label
            else None
        )
        previous_category_a = st.session_state.get("_categorical_l1_a_previous")
        if previous_category_a != category_a:
            st.session_state["categorical_l1_b"] = matched_b or no_filter_label
            st.session_state["_categorical_l1_a_previous"] = category_a
            # Layer 2 depends on Layer 1; reset downstream field choices when
            # the upstream field changes so no stale selection survives.
            st.session_state.pop("categorical_l2_a", None)
            st.session_state.pop("categorical_l2_b", None)

        default_b_index = options_b.index(matched_b) if matched_b in options_b else 0
        category_b = field_col_b.selectbox(
            "Dataset B field · Layer 1",
            options_b,
            index=default_b_index,
            key="categorical_l1_b",
            help=(
                "Choose the equivalent Layer 1 field in Dataset B. Matching field names "
                "are selected automatically when available."
            ),
        )

        layer1_ready = False
        layer1_subset_a = table_a.frame
        layer1_subset_b = table_b.frame
        layer1_filter_a: tuple[str, list[Any]] | None = None
        layer1_filter_b: tuple[str, list[Any]] | None = None

        if category_a != no_filter_label and category_b != no_filter_label:
            values_a = _domain_values(table_a.frame[category_a])
            values_b = _domain_values(table_b.frame[category_b])
            value_col_a, value_col_b = st.columns(2)

            keep_a = _nested_multiselect(
                value_col_a,
                "Dataset A categories · Layer 1",
                values_a,
                key=f"categorical_l1_values_a__{_normalize(category_a)}",
            )
            keep_b = _nested_multiselect(
                value_col_b,
                "Dataset B categories · Layer 1",
                values_b,
                key=f"categorical_l1_values_b__{_normalize(category_b)}",
            )

            if not keep_a or not keep_b:
                st.warning(
                    "Layer 1 requires at least one category in both datasets. "
                    "Layer 2 remains unavailable until Layer 1 is valid."
                )
            else:
                layer1_ready = True
                layer1_filter_a = (category_a, keep_a)
                layer1_filter_b = (category_b, keep_b)
                layer1_subset_a = _apply_categorical_filters(
                    table_a.frame, [layer1_filter_a]
                )
                layer1_subset_b = _apply_categorical_filters(
                    table_b.frame, [layer1_filter_b]
                )

                if keep_a != values_a or keep_b != values_b:
                    categorical_filters_cfg.append(
                        {
                            "layer": 1,
                            "a_col": category_a,
                            "b_col": category_b,
                            "a_values": keep_a,
                            "b_values": keep_b,
                        }
                    )
                    st.caption(
                        "Layer 1 active · "
                        f"A[{category_a}] = {_format_filter_values(keep_a)} · "
                        f"B[{category_b}] = {_format_filter_values(keep_b)}"
                    )
                else:
                    st.caption(
                        "Layer 1 configured with all categories selected — no Layer 1 restriction."
                    )
        elif category_a != no_filter_label or category_b != no_filter_label:
            st.caption(
                "Select a Layer 1 field in both datasets to activate nested filtering."
            )
        else:
            st.caption("Layer 1: no categorical filter applied.")

        st.divider()
        st.markdown("#### Subfilter Layer 2")
        st.caption(
            "Available Layer 2 categories are recalculated from the Layer 1 subset."
        )

        if layer1_ready:
            layer2_candidates_a = _categorical_candidates(
                layer1_subset_a, {category_a}
            )
            layer2_candidates_b = _categorical_candidates(
                layer1_subset_b, {category_b}
            )
            layer2_options_a = [no_filter_label, *layer2_candidates_a]
            layer2_options_b = [no_filter_label, *layer2_candidates_b]

            # If an upstream Layer 1 selection removes a previously selected
            # Layer 2 field, return that widget to <No filter>.
            if st.session_state.get("categorical_l2_a") not in layer2_options_a:
                st.session_state["categorical_l2_a"] = no_filter_label
            if st.session_state.get("categorical_l2_b") not in layer2_options_b:
                st.session_state["categorical_l2_b"] = no_filter_label

            layer2_field_col_a, layer2_field_col_b = st.columns(2)
            category2_a = layer2_field_col_a.selectbox(
                "Dataset A field · Layer 2",
                layer2_options_a,
                index=0,
                key="categorical_l2_a",
                help=(
                    "Choose a second categorical/domain field. Its available categories "
                    "come only from records that passed Layer 1."
                ),
            )

            matched2_b = (
                _matching_column(category2_a, layer2_candidates_b)
                if category2_a != no_filter_label
                else None
            )
            previous_category2_a = st.session_state.get("_categorical_l2_a_previous")
            if previous_category2_a != category2_a:
                st.session_state["categorical_l2_b"] = matched2_b or no_filter_label
                st.session_state["_categorical_l2_a_previous"] = category2_a

            default2_b_index = (
                layer2_options_b.index(matched2_b)
                if matched2_b in layer2_options_b
                else 0
            )
            category2_b = layer2_field_col_b.selectbox(
                "Dataset B field · Layer 2",
                layer2_options_b,
                index=default2_b_index,
                key="categorical_l2_b",
                help=(
                    "Choose the equivalent Layer 2 field in Dataset B. Matching field names "
                    "are selected automatically when available."
                ),
            )

            if (
                category2_a != no_filter_label
                and category2_b != no_filter_label
            ):
                values2_a = _domain_values(layer1_subset_a[category2_a])
                values2_b = _domain_values(layer1_subset_b[category2_b])
                value2_col_a, value2_col_b = st.columns(2)

                keep2_a = _nested_multiselect(
                    value2_col_a,
                    "Dataset A categories · Layer 2",
                    values2_a,
                    key=f"categorical_l2_values_a__{_normalize(category2_a)}",
                )
                keep2_b = _nested_multiselect(
                    value2_col_b,
                    "Dataset B categories · Layer 2",
                    values2_b,
                    key=f"categorical_l2_values_b__{_normalize(category2_b)}",
                )

                if not keep2_a or not keep2_b:
                    st.warning(
                        "Layer 2 requires at least one category in both datasets. "
                        "Only Layer 1 will be used until Layer 2 is valid."
                    )
                elif keep2_a != values2_a or keep2_b != values2_b:
                    categorical_filters_cfg.append(
                        {
                            "layer": 2,
                            "a_col": category2_a,
                            "b_col": category2_b,
                            "a_values": keep2_a,
                            "b_values": keep2_b,
                        }
                    )
                    st.caption(
                        "Layer 2 active · "
                        f"A[{category2_a}] = {_format_filter_values(keep2_a)} · "
                        f"B[{category2_b}] = {_format_filter_values(keep2_b)}"
                    )
                else:
                    st.caption(
                        "Layer 2 configured with all categories selected — no additional restriction."
                    )
            elif (
                category2_a != no_filter_label
                or category2_b != no_filter_label
            ):
                st.caption(
                    "Select a Layer 2 field in both datasets to activate the subfilter."
                )
            else:
                st.caption("Layer 2: no subfilter applied.")
        else:
            st.info(
                "Configure a valid Layer 1 filter to enable the nested Layer 2 subfilter."
            )

        if categorical_filters_cfg:
            active_path = " → ".join(
                f"L{cfg['layer']} "
                f"A[{cfg['a_col']}]={_format_filter_values(cfg['a_values'])}; "
                f"B[{cfg['b_col']}]={_format_filter_values(cfg['b_values'])}"
                for cfg in categorical_filters_cfg
            )
            st.caption(f"Active nested path · {active_path}")
        else:
            st.caption("Active nested path · none")

    with st.sidebar:
        sidebar_banner("Role", "Reference direction")
        reference_choice = st.radio(
            "Reference dataset (Y axis)",
            ["Dataset A", "Dataset B"],
            index=_reference_default(upload_a.name, upload_b.name, par_params),
            help=(
                "The reference dataset drives the GETPAIRS outer loop and is plotted on Y. "
                "The comparison dataset is searched and plotted on X."
            ),
        )

        sidebar_banner("Spatial", "Coordinate mapping")
        map_a = _coordinate_mapping(
            "A", table_a, indices_for_uploaded_file(par_params, upload_a.name)
        )
        map_b = _coordinate_mapping(
            "B", table_b, indices_for_uploaded_file(par_params, upload_b.name)
        )

        sidebar_banner("Variable", "Analysis variable")
        canonical = st.selectbox("Variable", list(CANONICAL_VARIABLES), index=2)
        numeric_a = _numeric_candidates(table_a.frame)
        numeric_b = _numeric_candidates(table_b.frame)
        if not numeric_a or not numeric_b:
            st.error("Each dataset must contain at least one numeric analysis column.")
            return
        variable_a = st.selectbox(
            "Dataset A variable column",
            numeric_a,
            index=_default_variable_index(numeric_a, canonical),
        )
        variable_b = st.selectbox(
            "Dataset B variable column",
            numeric_b,
            index=_default_variable_index(numeric_b, canonical),
        )
        variable_label = st.text_input(
            "Figure variable label",
            value=canonical if canonical != "Custom" else variable_a,
        ).strip() or variable_a

        sidebar_banner("Pairing", "Search parameters")
        default_dismax = par_params.dismax if par_params is not None else 2.0
        dismax = st.number_input(
            "Maximum search distance (m)",
            min_value=0.0,
            value=float(default_dismax),
            step=0.25,
            format="%.3f",
            help="Pairs are retained only when distance < maximum distance; the boundary is excluded.",
        )
        if par_params is None:
            default_mode = 1
        else:
            default_mode = 1 if par_params.ikeepclose == 1 else 0
        mode = st.radio(
            "Pairing rule",
            ["All neighbors within radius", "Nearest neighbor only"],
            index=default_mode,
            help=(
                "Nearest neighbor is the default analysis mode and corresponds to one comparison "
                "sample per reference sample (GETPAIRS ikeepclose=1). All neighbors preserves "
                "the legacy ikeepclose=0 behavior."
            ),
        )
        keep_closest = mode == "Nearest neighbor only"
        pairing_mode_note(keep_closest)

        default_2d = False
        if par_params is not None:
            default_2d = (
                par_params.first_xyz[2] == 0 or par_params.second_xyz[2] == 0
            )
        with st.expander("Advanced search options", expanded=default_2d):
            geometry = st.radio(
                "Search geometry",
                ["3D spherical (XYZ)", "Plan-view (XY, ignore Z)"],
                index=1 if default_2d else 0,
                help=(
                    "3D spherical search uses X, Y and Z. Plan-view search removes vertical "
                    "separation from the distance calculation and can therefore produce many more pairs."
                ),
            )
            ignore_z = geometry == "Plan-view (XY, ignore Z)"
            if ignore_z:
                st.caption(
                    "Plan-view search sets Z to zero for both datasets. Samples that are close in XY "
                    "can pair even when they are far apart vertically."
                )
            else:
                st.caption(
                    "3D spherical search is the recommended default when valid X, Y and Z coordinates "
                    "are available in both datasets."
                )

        sidebar_banner("Filters", "Eligibility")
        valid_assays_only = st.checkbox(
            "Positive assays only (> 0)",
            value=True,
            help=(
                "Default mining-grade validity rule. Only finite values greater than zero are eligible "
                "for pairing, so negative sentinel values such as -99 and zero values are excluded."
            ),
        )
        use_trim = st.checkbox("Apply grade-range filter before pairing", value=False)
        trim_bounds: tuple[float, float] | None = None
        if use_trim:
            combined = pd.concat(
                [
                    pd.to_numeric(table_a.frame[variable_a], errors="coerce"),
                    pd.to_numeric(table_b.frame[variable_b], errors="coerce"),
                ],
                ignore_index=True,
            )
            finite = combined[np.isfinite(combined)]
            if valid_assays_only:
                finite = finite[finite > 0.0]
            if finite.empty:
                st.error(
                    "Selected variables contain no eligible positive finite values for grade filtering."
                )
                return
            col1, col2 = st.columns(2)
            lower = col1.number_input("Min", value=float(finite.min()), format="%.6g")
            upper = col2.number_input("Max", value=float(finite.max()), format="%.6g")
            if lower > upper:
                st.error("Minimum grade cannot exceed maximum grade.")
                return
            trim_bounds = (float(lower), float(upper))

        sidebar_banner("Labels", "Series names")
        series_name_help = (
            "Enter a short label that clearly identifies each dataset in the plots and "
            "statistics table. Examples: 'Historical RC', 'Recent RC', '< 2022', or '>= 2022'."
        )
        reference_label = st.text_input(
            "Reference series",
            value="",
            help=series_name_help,
        )
        comparison_label = st.text_input(
            "Comparison series",
            value="",
            help=series_name_help,
        )

        sidebar_banner("Display", "Figure style")
        palette_name = st.selectbox(
            "Colour palette",
            list(FIGURE_PALETTES),
            index=list(FIGURE_PALETTES).index(DEFAULT_FIGURE_PALETTE),
            help="Corporate is the default. The remaining options are standard scientific palettes.",
        )
        palette = FIGURE_PALETTES[palette_name]

        with st.expander("Advanced display", expanded=False):
            near_zero_fraction = st.number_input(
                "Near-zero Diff.(%) guard",
                min_value=0.0,
                value=0.001,
                step=0.0005,
                format="%.4f",
                help="Flags unstable percentage differences when the reference statistic is near zero.",
            )
            use_clip = st.checkbox("Display-only bilateral quantile clipping", value=False)
            display_quantile = None
            if use_clip:
                display_quantile = st.number_input(
                    "Upper display quantile",
                    min_value=0.900,
                    max_value=0.9999,
                    value=0.995,
                    step=0.001,
                    format="%.4f",
                )
            st.markdown("**Axis ranges**")
            link_xy_axes = st.checkbox(
                "Link X and Y axes",
                value=True,
                key="link_xy_axes",
                help=(
                    "When enabled, each scatter/Q-Q plot uses one common X/Y range with a 1:1 scale. "
                    "A custom X range is applied exactly to both axes. Disable this option to configure "
                    "X and Y independently."
                ),
            )

            st.caption("Density scatterplot")
            scatter_x_limits = _optional_limits("Scatter X", "scatter")
            if link_xy_axes:
                scatter_y_limits = scatter_x_limits
                if scatter_x_limits is not None:
                    st.caption(
                        f"Scatter Y is linked to X: {scatter_x_limits[0]:.4g} to "
                        f"{scatter_x_limits[1]:.4g}."
                    )
            else:
                scatter_y_limits = _optional_limits("Scatter Y", "scatter_y")

            st.caption("Q-Q plot")
            qq_x_limits = _optional_limits("Q-Q X", "qq")
            if link_xy_axes:
                qq_y_limits = qq_x_limits
                if qq_x_limits is not None:
                    st.caption(
                        f"Q-Q Y is linked to X: {qq_x_limits[0]:.4g} to "
                        f"{qq_x_limits[1]:.4g}."
                    )
            else:
                qq_y_limits = _optional_limits("Q-Q Y", "qq_y")

    if reference_choice == "Dataset A":
        ref_table, cmp_table = table_a, table_b
        ref_map, cmp_map = map_a, map_b
        ref_var, cmp_var = variable_a, variable_b
        ref_upload_name, cmp_upload_name = upload_a.name, upload_b.name
    else:
        ref_table, cmp_table = table_b, table_a
        ref_map, cmp_map = map_b, map_a
        ref_var, cmp_var = variable_b, variable_a
        ref_upload_name, cmp_upload_name = upload_b.name, upload_a.name

    # Series-name inputs intentionally start blank. If the user leaves them blank,
    # fall back to the uploaded file stems so figures and exports remain self-describing.
    reference_label = reference_label.strip() or Path(ref_upload_name).stem
    comparison_label = comparison_label.strip() or Path(cmp_upload_name).stem

    try:
        categorical_filters_a = [
            (cfg["a_col"], cfg["a_values"]) for cfg in categorical_filters_cfg
        ]
        categorical_filters_b = [
            (cfg["b_col"], cfg["b_values"]) for cfg in categorical_filters_cfg
        ]
        filtered_a = _apply_eligibility_filters(
            table_a.frame,
            variable_a,
            valid_assays_only,
            trim_bounds,
            categorical_filters_a,
        )
        filtered_b = _apply_eligibility_filters(
            table_b.frame,
            variable_b,
            valid_assays_only,
            trim_bounds,
            categorical_filters_b,
        )
        if reference_choice == "Dataset A":
            reference, comparison = filtered_a, filtered_b
        else:
            reference, comparison = filtered_b, filtered_a

        if reference.empty or comparison.empty:
            st.error("Eligibility filters leave one or both datasets empty.")
            return

        ref_cols = (ref_map["x"], ref_map["y"], ref_map["z"])
        cmp_cols = (cmp_map["x"], cmp_map["y"], cmp_map["z"])
        ref_coords, cmp_coords, active_axes = prepare_coordinate_arrays(
            reference,
            comparison,
            ref_cols,
            cmp_cols,
            force_ignore_z=ignore_z,
        )
        result = cached_pairing(ref_coords, cmp_coords, float(dismax), keep_closest)
    except Exception as exc:
        st.error(str(exc))
        return

    if result.n_pairs == 0:
        st.warning("No sample pairs satisfy the strict distance criterion distance < dismax.")
        _render_pairing_report(result, len(reference), len(comparison), active_axes)
        return

    ref_values, cmp_values, analysis_mask = extract_pair_values(
        reference,
        comparison,
        result.reference_indices,
        result.comparison_indices,
        ref_var,
        cmp_var,
    )
    analysis_distances = result.distances[analysis_mask]
    analysis_ref_indices = result.reference_indices[analysis_mask]
    analysis_cmp_indices = result.comparison_indices[analysis_mask]
    if ref_values.size == 0:
        st.error("Spatial pairs were found, but none have valid values for the selected variable.")
        return

    pair_ref_xyz = _paired_display_coordinates(reference, ref_map, analysis_ref_indices)
    pair_cmp_xyz = _paired_display_coordinates(comparison, cmp_map, analysis_cmp_indices)
    pair_ref_ids = _paired_sample_ids(
        reference, ref_map.get("id"), analysis_ref_indices, "Reference"
    )
    pair_cmp_ids = _paired_sample_ids(
        comparison, cmp_map.get("id"), analysis_cmp_indices, "Comparison"
    )
    spatial_projections = _available_pair_projections(pair_ref_xyz, pair_cmp_xyz)

    stats_frame = compute_statistics_table(
        ref_values,
        cmp_values,
        preset="study",
        near_zero_fraction=float(near_zero_fraction),
    )

    figure_title = f"Paired-Sample Comparison — {variable_label}"
    search_axes_label = _search_geometry_label(active_axes, ignore_z)
    figure_subtitle = (
        f"{comparison_label} vs {reference_label} · n = {ref_values.size:,} · "
        f"{search_axes_label} distance < {float(dismax):.2f} m · {mode}"
    )
    filter_note = _analysis_filter_note(
        valid_assays_only=valid_assays_only,
        trim_bounds=trim_bounds,
        categorical_filters_cfg=categorical_filters_cfg,
        search_axes_label=search_axes_label,
        dismax=float(dismax),
        pairing_mode=mode,
    )

    sensitivity_distances = _sensitivity_distances(float(dismax))
    sensitivity_ref_values = pd.to_numeric(
        reference[ref_var], errors="coerce"
    ).to_numpy(dtype=np.float64)
    sensitivity_cmp_values = pd.to_numeric(
        comparison[cmp_var], errors="coerce"
    ).to_numpy(dtype=np.float64)
    sensitivity_frame = cached_pairing_sensitivity(
        ref_coords,
        cmp_coords,
        sensitivity_ref_values,
        sensitivity_cmp_values,
        sensitivity_distances,
        keep_closest,
    )
    sensitivity_fig = build_pairing_sensitivity_plot(
        sensitivity_frame,
        selected_distance=float(dismax),
        pairing_mode=mode,
        search_geometry=search_axes_label,
        colors=palette.series,
    )

    fig = cached_multiplot(
        ref_values,
        cmp_values,
        analysis_distances,
        stats_frame,
        reference_label,
        comparison_label,
        variable_label,
        palette.series,
        palette.density_scale,
        scatter_x_limits,
        scatter_y_limits,
        qq_x_limits,
        qq_y_limits,
        link_xy_axes,
        display_quantile,
        figure_title,
        figure_subtitle,
        filter_note,
    )

    tabs = st.tabs(["Analysis", "Pairing Report", "Exports"])

    with tabs[0]:
        st.markdown("### Spatial Pair Locations")
        if spatial_projections:
            projection_labels = {
                "XY": "XY · Plan View",
                "XZ": "XZ · Section",
                "YZ": "YZ · Section",
            }
            projection_options = [projection_labels[key] for key in spatial_projections]
            default_projection_index = (
                spatial_projections.index("XY") if "XY" in spatial_projections else 0
            )
            selected_projection_label = st.radio(
                "Pair location view",
                projection_options,
                index=default_projection_index,
                horizontal=True,
                key="pair_location_projection",
                help=(
                    "Shows the exact analysis-valid paired records used in the statistics and comparison figure. "
                    "Each light line connects one reference/comparison pair."
                ),
            )
            projection = next(
                key
                for key, label in projection_labels.items()
                if label == selected_projection_label
            )
            spatial_fig = cached_pair_location_plot(
                pair_ref_xyz,
                pair_cmp_xyz,
                analysis_distances,
                ref_values,
                cmp_values,
                pair_ref_ids,
                pair_cmp_ids,
                projection,
                reference_label,
                comparison_label,
                variable_label,
                float(dismax),
                palette.series,
            )
            st.plotly_chart(
                spatial_fig,
                use_container_width=True,
                config={"displaylogo": False, "scrollZoom": True},
            )
            if ignore_z and "Z" in projection:
                st.caption(
                    "Z is shown for spatial context only. The selected pairing search is 2D, "
                    "so Z did not contribute to pair separation or threshold acceptance."
                )
            st.caption(
                f"Displayed pairs: **{len(analysis_distances):,}** · Strict criterion: "
                f"**distance < {float(dismax):.2f} m** · Connector length represents the paired-sample separation."
            )
        else:
            spatial_fig = None
            st.info(
                "A two-axis spatial view is unavailable because fewer than two mapped coordinate "
                "axes contain valid values in both paired datasets."
            )

        st.markdown("### Paired-Sample Comparison")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        with st.expander("Statistics table — full-precision calculations", expanded=False):
            st.dataframe(stats_frame, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown("### Pairing Report")
        _render_pairing_report(result, len(reference), len(comparison), active_axes)

        st.markdown("### Pairing Sensitivity")
        st.caption(
            "Search-distance sensitivity using the same filtered datasets, search geometry and pairing rule. "
            "The standard diagnostic evaluates 1–5 m; the currently selected distance is also included when needed."
        )
        st.plotly_chart(
            sensitivity_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )
        with st.expander("Sensitivity table", expanded=False):
            sensitivity_display = sensitivity_frame.rename(
                columns={
                    "search_distance_m": "Search distance (m)",
                    "pairs": "Pairs",
                    "paired_reference": "Paired reference",
                    "pairing_rate_pct": "Pairing rate (%)",
                    "mean_separation_m": "Mean separation (m)",
                    "max_separation_m": "Max separation (m)",
                }
            )
            st.dataframe(
                sensitivity_display.style.format(
                    {
                        "Search distance (m)": "{:.2f}",
                        "Pairing rate (%)": "{:.2f}",
                        "Mean separation (m)": "{:.2f}",
                        "Max separation (m)": "{:.2f}",
                    },
                    na_rep="n/a",
                ),
                use_container_width=True,
                hide_index=True,
            )

        if analysis_mask.sum() != result.n_pairs:
            st.caption(
                f"Analysis-valid pairs: {int(analysis_mask.sum()):,} of {result.n_pairs:,} spatial pairs. "
                "Invalid selected-variable values are excluded from the figure and statistics."
            )
        st.caption(
            f"Input records before eligibility filters — Dataset A: {len(table_a.frame):,}; "
            f"Dataset B: {len(table_b.frame):,}."
        )

    with tabs[2]:
        st.markdown("### Exports")
        paired_csv = build_paired_dataframe(
            reference,
            comparison,
            analysis_ref_indices,
            analysis_cmp_indices,
            analysis_distances,
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.download_button(
            "Paired CSV",
            data=paired_csv.to_csv(index=False).encode("utf-8"),
            file_name=f"paired_{variable_label}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        try:
            gslib_bytes = paired_to_gslib_bytes(
                reference,
                comparison,
                analysis_ref_indices,
                analysis_cmp_indices,
                ref_table.title,
                cmp_table.title,
            )
            col2.download_button(
                "Paired GSLIB",
                data=gslib_bytes,
                file_name=f"paired_{variable_label}.out",
                mime="text/plain",
                use_container_width=True,
            )
        except ValueError as exc:
            col2.button("Paired GSLIB unavailable", disabled=True, use_container_width=True)
            st.caption(str(exc))

        col3.download_button(
            "Statistics CSV",
            data=stats_frame.to_csv(index=False).encode("utf-8"),
            file_name=f"statistics_{variable_label}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        col4.download_button(
            "Figure HTML",
            data=fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
            file_name=f"multiplot_{variable_label}.html",
            mime="text/html",
            use_container_width=True,
        )

        st.download_button(
            "Pairing Sensitivity CSV",
            data=sensitivity_frame.to_csv(index=False).encode("utf-8"),
            file_name=f"pairing_sensitivity_{variable_label}.csv",
            mime="text/csv",
        )

        if spatial_fig is not None:
            spatial_col1, spatial_col2 = st.columns(2)
            spatial_col1.download_button(
                "Pair Locations HTML",
                data=spatial_fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"),
                file_name=f"pair_locations_{variable_label}.html",
                mime="text/html",
                use_container_width=True,
            )
            try:
                spatial_png = spatial_fig.to_image(
                    format="png", width=1500, height=900, scale=2
                )
                spatial_col2.download_button(
                    "Pair Locations PNG",
                    data=spatial_png,
                    file_name=f"pair_locations_{variable_label}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except Exception:
                spatial_col2.button(
                    "Pair Locations PNG unavailable",
                    disabled=True,
                    use_container_width=True,
                )

        try:
            png = fig.to_image(format="png", width=1500, height=1250, scale=2)
            st.download_button(
                "Presentation-ready PNG",
                data=png,
                file_name=f"multiplot_{variable_label}.png",
                mime="image/png",
            )
        except Exception:
            st.info("PNG export requires the Kaleido package listed in requirements.txt.")

        par_current = _current_par(
            ref_upload_name,
            cmp_upload_name,
            reference,
            comparison,
            ref_map,
            cmp_map,
            ignore_z,
            float(dismax),
            keep_closest,
        )
        st.download_button(
            "Export getpairs.par",
            data=export_getpairs_par(par_current).encode("utf-8"),
            file_name="getpairs.par",
            mime="text/plain",
        )
        if ref_table.source_format != "gslib" or cmp_table.source_format != "gslib":
            st.caption(
                "The exported .par is syntactically valid, but legacy GETPAIRS can directly read only "
                "GSLIB/GeoEAS numeric input files."
            )

def _read_par_upload(upload: Any) -> GetPairsParameters | None:
    if upload is None:
        return None
    try:
        return parse_getpairs_par(upload.getvalue().decode("utf-8-sig"))
    except Exception as exc:
        st.sidebar.error(f"Could not import parameter file: {exc}")
        return None


def _reference_default(name_a: str, name_b: str, params: GetPairsParameters | None) -> int:
    if params is None:
        return 0
    first = Path(params.first_file).name.lower()
    if Path(name_b).name.lower() == first:
        return 1
    return 0


def _coordinate_mapping(
    label: str,
    table: LoadedTable,
    imported_indices: tuple[int, int, int] | None,
) -> dict[str, str | None]:
    columns = list(table.frame.columns)
    optional_columns: list[str | None] = [None, *columns]
    st.markdown(f"**Dataset {label}**")

    mappings: dict[str, str | None] = {}
    id_default = _find_column(
        columns,
        ["sample_id", "sampleid", "hole_id", "holeid", "dhid", "bhid", "id"],
    )
    id_index = optional_columns.index(id_default) if id_default in optional_columns else 0
    mappings["id"] = st.selectbox(
        "Sample / hole ID column",
        optional_columns,
        index=id_index,
        format_func=lambda value: "<not mapped>" if value is None else str(value),
        key=f"coord_{label}_id",
    )

    for axis, candidates in (
        ("x", ["x", "easting", "east", "midx", "xcoord"]),
        ("y", ["y", "northing", "north", "midy", "ycoord"]),
        ("z", ["z", "elevation", "elev", "rl", "midz", "zcoord"]),
    ):
        imported = None
        if imported_indices is not None:
            idx = imported_indices[("x", "y", "z").index(axis)]
            imported = columns[idx - 1] if 1 <= idx <= len(columns) else None
        default_name = imported or _find_column(columns, candidates)
        default_index = (
            optional_columns.index(default_name) if default_name in optional_columns else 0
        )
        mappings[axis] = st.selectbox(
            f"{axis.upper()} column",
            optional_columns,
            index=default_index,
            format_func=lambda value: "<disabled>" if value is None else str(value),
            key=f"coord_{label}_{axis}",
        )
    # X and Y are normally required by the modern UI; allowing None preserves the
    # general legacy axis-collapse behavior for imported/special workflows.
    return mappings


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize(col): col for col in columns}
    for candidate in candidates:
        if _normalize(candidate) in normalized:
            return normalized[_normalize(candidate)]
    return None


def _normalize(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _numeric_candidates(frame: pd.DataFrame) -> list[str]:
    candidates = []
    for col in frame.columns:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().any():
            candidates.append(str(col))
    return candidates


def _default_variable_index(columns: list[str], canonical: str) -> int:
    if canonical == "Custom":
        return 0
    aliases = CANONICAL_VARIABLES[canonical]
    normalized = [_normalize(col) for col in columns]
    for alias in [canonical, *aliases]:
        target = _normalize(alias)
        if target in normalized:
            return normalized.index(target)
    return 0


def _domain_values(series: pd.Series) -> list[Any]:
    values = series.dropna().drop_duplicates().tolist()
    try:
        return sorted(values)
    except TypeError:
        return values


def _categorical_candidates(
    frame: pd.DataFrame,
    excluded: set[str],
    max_unique: int = 60,
) -> list[str]:
    """Return ranked categorical fields suitable for the Primary Filter.

    Text/category/bool fields and low-cardinality numeric codes are eligible.
    Known IDs, coordinates, interval limits and analytical variables are
    excluded so that small datasets do not accidentally promote continuous
    measurements such as ``from``, ``to`` or assay columns.
    """
    candidates: list[str] = []
    n_rows = max(len(frame), 1)

    excluded_normalized = {_normalize(name) for name in excluded}
    excluded_normalized.update(_normalize(name) for name in NON_CATEGORICAL_FIELDS)

    # All canonical analytical names and aliases are measurements, not
    # categorical domains, even when a particular file contains few values.
    for canonical, aliases in CANONICAL_VARIABLES.items():
        if canonical == "Custom":
            continue
        excluded_normalized.add(_normalize(canonical))
        excluded_normalized.update(_normalize(alias) for alias in aliases)

    for column in frame.columns:
        name = str(column)
        if _normalize(name) in excluded_normalized:
            continue

        series = frame[column]
        unique_count = int(series.nunique(dropna=True))
        if unique_count == 0 or unique_count > max_unique:
            continue

        is_text_like = (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        )
        is_low_cardinality_code = (
            pd.api.types.is_numeric_dtype(series)
            and unique_count <= max(12, int(n_rows * 0.10))
        )
        if is_text_like or is_low_cardinality_code:
            candidates.append(name)

    # Promote mining-domain fields such as Destination, MetType and LithType
    # while preserving source-column order for any remaining candidates.
    preferred_rank = {
        _normalize(name): rank
        for rank, name in enumerate(PREFERRED_CATEGORICAL_VARIABLES)
    }
    original_rank = {name: rank for rank, name in enumerate(candidates)}
    candidates.sort(
        key=lambda name: (
            0 if _normalize(name) in preferred_rank else 1,
            preferred_rank.get(_normalize(name), original_rank[name]),
            original_rank[name],
        )
    )
    return candidates


def _matching_column(source_column: str, candidates: list[str]) -> str | None:
    """Find the most direct normalized-name match in a second dataset."""
    source = _normalize(source_column)
    for candidate in candidates:
        if _normalize(candidate) == source:
            return candidate
    return None


def _format_filter_values(values: list[Any], max_items: int = 5) -> str:
    """Return a compact, report-friendly categorical selection label."""
    if not values:
        return "none"
    text = [str(value) for value in values]
    if len(text) <= max_items:
        return ", ".join(text)
    return ", ".join(text[:max_items]) + f", +{len(text) - max_items} more"


def _analysis_filter_note(
    *,
    valid_assays_only: bool,
    trim_bounds: tuple[float, float] | None,
    categorical_filters_cfg: list[dict[str, Any]],
    search_axes_label: str,
    dismax: float,
    pairing_mode: str,
) -> str:
    """Build the concise filter footnote printed inside the comparison figure."""
    parts = [
        f"Search: {search_axes_label} distance < {dismax:.2f} m",
        f"Pairing: {pairing_mode}",
        "Positive assays only (> 0)" if valid_assays_only else "Non-positive assays allowed",
    ]
    if trim_bounds is not None:
        parts.append(f"Grade range: {trim_bounds[0]:.4g} to {trim_bounds[1]:.4g}")
    else:
        parts.append("Grade range: not applied")

    if categorical_filters_cfg:
        for cfg in categorical_filters_cfg:
            parts.append(
                f"Categorical L{cfg['layer']}: "
                f"A[{cfg['a_col']}] = {_format_filter_values(cfg['a_values'])}; "
                f"B[{cfg['b_col']}] = {_format_filter_values(cfg['b_values'])}"
            )
    else:
        parts.append("Categorical filters: not applied")
    return "Filters — " + " · ".join(parts)


def _sensitivity_distances(selected_distance: float) -> tuple[float, ...]:
    """Return standard 1–5 m sensitivity ranges plus the active search distance."""
    values = set(SENSITIVITY_RANGES_M)
    if np.isfinite(selected_distance) and selected_distance > 0.0:
        values.add(float(selected_distance))
    return tuple(sorted(values))


def _search_geometry_label(
    active_axes: tuple[bool, bool, bool],
    ignore_z: bool,
) -> str:
    """Return a concise description of the actual search geometry."""
    if active_axes == (True, True, True) and not ignore_z:
        return "3D XYZ spherical"
    if active_axes[0] and active_axes[1] and not active_axes[2]:
        return "Plan-view XY (Z ignored)"
    axes = ",".join(
        axis
        for axis, active in zip(("X", "Y", "Z"), active_axes, strict=True)
        if active
    )
    return f"Active axes: {axes or 'none'}"


def _paired_display_coordinates(
    frame: pd.DataFrame,
    mapping: dict[str, str | None],
    indices: np.ndarray,
) -> np.ndarray:
    """Return original mapped XYZ coordinates for the exact analysis pair rows."""
    coordinates = np.full((indices.size, 3), np.nan, dtype=np.float64)
    for axis_index, axis in enumerate(("x", "y", "z")):
        column = mapping.get(axis)
        if column is None or column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        coordinates[:, axis_index] = values[indices]
    return coordinates


def _paired_sample_ids(
    frame: pd.DataFrame,
    id_column: str | None,
    indices: np.ndarray,
    prefix: str,
) -> np.ndarray:
    """Return mapped sample IDs, falling back to filtered-row identifiers."""
    if id_column is not None and id_column in frame.columns:
        values = frame[id_column].astype("string").fillna("<missing>").to_numpy(dtype=object)
        return values[indices]
    return np.asarray(
        [f"{prefix} row {int(index) + 1}" for index in indices],
        dtype=object,
    )


def _available_pair_projections(
    reference_coordinates: np.ndarray,
    comparison_coordinates: np.ndarray,
) -> list[str]:
    """List projections that can display every analysis-valid pair without dropping rows."""
    axes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
    available: list[str] = []
    for name, (first_axis, second_axis) in axes.items():
        values = np.column_stack(
            [
                reference_coordinates[:, first_axis],
                reference_coordinates[:, second_axis],
                comparison_coordinates[:, first_axis],
                comparison_coordinates[:, second_axis],
            ]
        )
        if values.size and np.isfinite(values).all():
            available.append(name)
    return available


def _apply_categorical_filters(
    frame: pd.DataFrame,
    categorical_filters: list[tuple[str, list[Any]]],
) -> pd.DataFrame:
    """Apply categorical layers sequentially using AND logic between layers."""
    mask = pd.Series(True, index=frame.index)
    for column, allowed in categorical_filters:
        mask &= frame[column].isin(allowed)
    return frame.loc[mask].copy()


def _nested_multiselect(
    container: Any,
    label: str,
    options: list[Any],
    *,
    key: str,
) -> list[Any]:
    """Render a cascading multiselect and remove stale upstream-dependent values."""
    if key in st.session_state:
        current = list(st.session_state[key])
        cleaned = [value for value in current if value in options]
        if cleaned != current:
            st.session_state[key] = cleaned
    return container.multiselect(
        label,
        options,
        default=options if key not in st.session_state else None,
        key=key,
    )


def _apply_eligibility_filters(
    frame: pd.DataFrame,
    variable: str,
    valid_only: bool,
    trim_bounds: tuple[float, float] | None,
    categorical_filters: list[tuple[str, list[Any]]] | None,
) -> pd.DataFrame:
    """Apply analytical eligibility rules and all active categorical layers."""
    mask = pd.Series(True, index=frame.index)
    numeric_variable = pd.to_numeric(frame[variable], errors="coerce")
    if valid_only:
        mask &= np.isfinite(numeric_variable) & (numeric_variable > 0.0)
    if trim_bounds is not None:
        lower, upper = trim_bounds
        mask &= np.isfinite(numeric_variable) & numeric_variable.between(
            lower, upper, inclusive="both"
        )
    for column, allowed in categorical_filters or []:
        mask &= frame[column].isin(allowed)
    return frame.loc[mask].copy().reset_index(drop=True)


def _optional_limits(label: str, key: str) -> tuple[float, float] | None:
    """Render an optional exact axis-range control with strict min/max validation."""
    enabled = st.checkbox(
        f"Custom {label} axis limits",
        value=False,
        key=f"{key}_limits_on",
    )
    if not enabled:
        return None
    col1, col2 = st.columns(2)
    low = col1.number_input(f"{label} min", value=0.0, key=f"{key}_min")
    high = col2.number_input(f"{label} max", value=15.0, key=f"{key}_max")
    if low >= high:
        st.error(f"{label} minimum must be smaller than maximum.")
        return None
    return float(low), float(high)


def _render_pairing_report(
    result: PairingResult,
    n_reference: int,
    n_comparison: int,
    active_axes: tuple[bool, bool, bool],
) -> None:
    """Render report-grade pairing KPIs and search diagnostics."""
    report = pairing_report(result, n_reference, n_comparison)
    render_kpi_cards(
        [
            ("Reference records", f"{report['reference_records']:,}"),
            ("Comparison records", f"{report['comparison_records']:,}"),
            ("Pairs", f"{report['pairs']:,}"),
            ("Pairing rate", f"{report['pairing_rate_pct']:.2f}%"),
            ("Unpaired reference", f"{report['unpaired_reference_records']:,}"),
            ("Reused comparison", f"{report['reused_comparison_records']:,}"),
            ("Mean separation", _distance_text(report["mean_distance"])),
            ("Median separation", _distance_text(report["median_distance"])),
            ("Max separation", _distance_text(report["max_distance"])),
            ("Extra reuse assignments", f"{report['extra_comparison_reuses']:,}"),
        ]
    )
    axes = ", ".join(
        axis
        for axis, active in zip(("X", "Y", "Z"), active_axes, strict=True)
        if active
    )
    st.caption(
        f"Active search axes: **{axes}**. Comparison-sample reuse is allowed, matching GETPAIRS."
    )

def _distance_text(value: Any) -> str:
    return "n/a" if not np.isfinite(float(value)) else f"{float(value):.2f} m"


def _column_position(frame: pd.DataFrame, column: str | None) -> int:
    if column is None:
        return 0
    return int(frame.columns.get_loc(column)) + 1


def _current_par(
    ref_name: str,
    cmp_name: str,
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    ref_map: dict[str, str | None],
    cmp_map: dict[str, str | None],
    ignore_z: bool,
    dismax: float,
    keep_closest: bool,
) -> GetPairsParameters:
    ref_xyz = (
        _column_position(reference, ref_map["x"]),
        _column_position(reference, ref_map["y"]),
        0 if ignore_z else _column_position(reference, ref_map["z"]),
    )
    cmp_xyz = (
        _column_position(comparison, cmp_map["x"]),
        _column_position(comparison, cmp_map["y"]),
        0 if ignore_z else _column_position(comparison, cmp_map["z"]),
    )
    return GetPairsParameters(
        first_file=ref_name,
        first_xyz=ref_xyz,
        second_file=cmp_name,
        second_xyz=cmp_xyz,
        output_file="getpairs.out",
        dismax=dismax,
        ikeepclose=1 if keep_closest else 0,
    )

