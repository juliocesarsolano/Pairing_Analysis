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


CANONICAL_VARIABLES = {
    "au_ppm": ["au_ppm", "au", "gold", "au_gpt", "au_gt"],
    "ag_ppm": ["ag_ppm", "ag", "silver", "ag_gpt", "ag_gt"],
    "s_tot_pct": ["s_tot_pct", "stot", "s_tot", "total_sulfur", "sulfur_total"],
    "s2_pct": ["s2_pct", "s2", "sulfide_sulfur", "sulfide_s"],
    "c_tot_pct": ["c_tot_pct", "ctot", "c_tot", "total_carbon"],
    "c_org_pct": ["c_org_pct", "corg", "c_org", "organic_carbon"],
    "cu_pct": ["cu_pct", "cu", "copper"],
    "zn_pct": ["zn_pct", "zn", "zinc"],
    "cao_pct": ["cao_pct", "cao"],
    "sio2_pct": ["sio2_pct", "sio2"],
    "Custom": [],
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
    scatter_limits: tuple[float, float] | None,
    qq_limits: tuple[float, float] | None,
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
        scatter_limits=scatter_limits,
        qq_limits=qq_limits,
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
            ("Search", "3D / 2D spherical"),
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

        default_2d = False
        if par_params is not None:
            default_2d = par_params.first_xyz[2] == 0 or par_params.second_xyz[2] == 0
        ignore_z = st.checkbox(
            "2D search (ignore Z)",
            value=default_2d,
            help="When enabled, Z is forced to zero in both datasets and distance is calculated in XY only.",
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
        default_mode = 1 if par_params is not None and par_params.ikeepclose == 1 else 0
        mode = st.radio(
            "Pairing rule",
            ["All neighbors within radius", "Nearest neighbor only"],
            index=default_mode,
            help=(
                "All neighbors corresponds to GETPAIRS ikeepclose=0. Nearest neighbor corresponds "
                "to ikeepclose=1."
            ),
        )
        keep_closest = mode == "Nearest neighbor only"
        pairing_mode_note(keep_closest)

        sidebar_banner("Filters", "Eligibility")
        valid_assays_only = st.checkbox(
            "Valid assay only",
            value=True,
            help=(
                "Recommended for variable-specific paired analysis. Records with invalid values for the selected "
                "variable are removed before spatial pairing."
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
            if finite.empty:
                st.error("Selected variables contain no finite values for grade filtering.")
                return
            col1, col2 = st.columns(2)
            lower = col1.number_input("Min", value=float(finite.min()), format="%.6g")
            upper = col2.number_input("Max", value=float(finite.max()), format="%.6g")
            if lower > upper:
                st.error("Minimum grade cannot exceed maximum grade.")
                return
            trim_bounds = (float(lower), float(upper))

        use_category_filter = st.checkbox(
            "Categorical Variable Filter",
            value=False,
            help=(
                "Optionally restrict the analysis to selected categories such as lithology, "
                "estimation domain, alteration, weathering, sample type, phase, or year. "
                "Choose the corresponding field independently in each dataset."
            ),
        )
        categorical_cfg: dict[str, Any] | None = None
        if use_category_filter:
            excluded_a = {variable_a, *(value for value in map_a.values() if value is not None)}
            excluded_b = {variable_b, *(value for value in map_b.values() if value is not None)}
            categorical_a = _categorical_candidates(table_a.frame, excluded_a)
            categorical_b = _categorical_candidates(table_b.frame, excluded_b)
            if not categorical_a or not categorical_b:
                st.warning(
                    "No suitable categorical fields were detected in one or both datasets. "
                    "Categorical fields may be text/category columns or low-cardinality coded fields."
                )
            else:
                category_a = st.selectbox(
                    "Dataset A categorical field",
                    categorical_a,
                    key="categorical_a",
                    help="Select the categorical field used to restrict Dataset A.",
                )
                matched_b = _matching_column(category_a, categorical_b)
                category_b = st.selectbox(
                    "Dataset B categorical field",
                    categorical_b,
                    index=categorical_b.index(matched_b) if matched_b in categorical_b else 0,
                    key="categorical_b",
                    help="Select the equivalent categorical field in Dataset B.",
                )
                values_a = _domain_values(table_a.frame[category_a])
                values_b = _domain_values(table_b.frame[category_b])
                keep_a = st.multiselect(
                    "Dataset A categories",
                    values_a,
                    default=values_a,
                    key="categorical_values_a",
                )
                keep_b = st.multiselect(
                    "Dataset B categories",
                    values_b,
                    default=values_b,
                    key="categorical_values_b",
                )
                categorical_cfg = {
                    "a_col": category_a,
                    "b_col": category_b,
                    "a_values": keep_a,
                    "b_values": keep_b,
                }

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
            scatter_limits = _optional_limits("Scatter", "scatter")
            qq_limits = _optional_limits("Q-Q", "qq")

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
        filtered_a = _apply_eligibility_filters(
            table_a.frame,
            variable_a,
            valid_assays_only,
            trim_bounds,
            None
            if categorical_cfg is None
            else (categorical_cfg["a_col"], categorical_cfg["a_values"]),
        )
        filtered_b = _apply_eligibility_filters(
            table_b.frame,
            variable_b,
            valid_assays_only,
            trim_bounds,
            None
            if categorical_cfg is None
            else (categorical_cfg["b_col"], categorical_cfg["b_values"]),
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
    search_axes_label = "XY" if ignore_z else "XYZ"
    figure_subtitle = (
        f"{comparison_label} vs {reference_label} · n = {ref_values.size:,} · "
        f"{search_axes_label} distance < {float(dismax):.2f} m · {mode}"
    )
    filter_note = _analysis_filter_note(
        valid_assays_only=valid_assays_only,
        trim_bounds=trim_bounds,
        categorical_cfg=categorical_cfg,
        search_axes_label=search_axes_label,
        dismax=float(dismax),
        pairing_mode=mode,
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
        scatter_limits,
        qq_limits,
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
            png = fig.to_image(format="png", width=1400, height=1400, scale=2)
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
    """Return text/category and low-cardinality coded fields suitable for filtering."""
    candidates: list[str] = []
    n_rows = max(len(frame), 1)
    for column in frame.columns:
        name = str(column)
        if name in excluded:
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
    categorical_cfg: dict[str, Any] | None,
    search_axes_label: str,
    dismax: float,
    pairing_mode: str,
) -> str:
    """Build the concise filter footnote printed inside the comparison figure."""
    parts = [
        f"Search: {search_axes_label} distance < {dismax:.2f} m",
        f"Pairing: {pairing_mode}",
        "Valid assays only" if valid_assays_only else "Invalid assays not pre-filtered",
    ]
    if trim_bounds is not None:
        parts.append(f"Grade range: {trim_bounds[0]:.4g} to {trim_bounds[1]:.4g}")
    else:
        parts.append("Grade range: not applied")
    if categorical_cfg is not None:
        parts.append(
            "Categorical: "
            f"A[{categorical_cfg['a_col']}] = {_format_filter_values(categorical_cfg['a_values'])}; "
            f"B[{categorical_cfg['b_col']}] = {_format_filter_values(categorical_cfg['b_values'])}"
        )
    else:
        parts.append("Categorical filter: not applied")
    return "Filters — " + " · ".join(parts)


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


def _apply_eligibility_filters(
    frame: pd.DataFrame,
    variable: str,
    valid_only: bool,
    trim_bounds: tuple[float, float] | None,
    categorical_filter: tuple[str, list[Any]] | None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    numeric_variable = pd.to_numeric(frame[variable], errors="coerce")
    if valid_only:
        mask &= np.isfinite(numeric_variable)
    if trim_bounds is not None:
        lower, upper = trim_bounds
        mask &= np.isfinite(numeric_variable) & numeric_variable.between(
            lower, upper, inclusive="both"
        )
    if categorical_filter is not None:
        column, allowed = categorical_filter
        mask &= frame[column].isin(allowed)
    return frame.loc[mask].copy().reset_index(drop=True)


def _optional_limits(label: str, key: str) -> tuple[float, float] | None:
    enabled = st.checkbox(f"Custom {label} axis limits", value=False, key=f"{key}_limits_on")
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

