"""GETPAIRS-compatible spatial pairing algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PairingResult:
    """Spatial pairing result in legacy output order."""

    reference_indices: np.ndarray
    comparison_indices: np.ndarray
    squared_distances: np.ndarray
    distances: np.ndarray

    @property
    def n_pairs(self) -> int:
        """Number of output pair rows."""
        return int(self.reference_indices.size)


def prepare_coordinate_arrays(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    reference_columns: Sequence[str | None],
    comparison_columns: Sequence[str | None],
    force_ignore_z: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[bool, bool, bool]]:
    """Build coordinate arrays with the GETPAIRS shared-axis quirk.

    An axis is populated only if its coordinate mapping is enabled in BOTH datasets.
    Otherwise that axis is forced to 0.0 in both arrays. ``force_ignore_z`` provides
    the explicit modern 2D toggle requested for the Streamlit UI.
    """
    if len(reference_columns) != 3 or len(comparison_columns) != 3:
        raise ValueError("Coordinate mappings must contain exactly X, Y, and Z entries.")

    active = [
        reference_columns[i] is not None and comparison_columns[i] is not None
        for i in range(3)
    ]
    if force_ignore_z:
        active[2] = False
    if not any(active):
        raise ValueError("At least one coordinate axis must be enabled in both datasets.")

    ref_coords = np.zeros((len(reference), 3), dtype=np.float64)
    cmp_coords = np.zeros((len(comparison), 3), dtype=np.float64)

    for axis, is_active in enumerate(active):
        if not is_active:
            continue
        ref_name = reference_columns[axis]
        cmp_name = comparison_columns[axis]
        assert ref_name is not None and cmp_name is not None
        ref_values = pd.to_numeric(reference[ref_name], errors="coerce").to_numpy(dtype=np.float64)
        cmp_values = pd.to_numeric(comparison[cmp_name], errors="coerce").to_numpy(
            dtype=np.float64
        )
        _validate_coordinates(ref_values, f"reference {('X', 'Y', 'Z')[axis]}")
        _validate_coordinates(cmp_values, f"comparison {('X', 'Y', 'Z')[axis]}")
        ref_coords[:, axis] = ref_values
        cmp_coords[:, axis] = cmp_values

    return ref_coords, cmp_coords, tuple(active)  # type: ignore[return-value]


def _validate_coordinates(values: np.ndarray, label: str) -> None:
    if values.size and not np.isfinite(values).all():
        bad_count = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"Selected {label} coordinate contains {bad_count:,} non-numeric, missing, "
            "or infinite value(s)."
        )


def pair_records_kdtree(
    reference_coordinates: np.ndarray,
    comparison_coordinates: np.ndarray,
    dismax: float,
    keep_closest: bool,
) -> PairingResult:
    """Pair records with cKDTree while preserving GETPAIRS logical behavior.

    Key compatibility safeguards:
    - strict squared-distance predicate ``dis < dismax**2`` is re-evaluated after
      every KD-tree candidate search;
    - all-pairs output is ordered by reference row then comparison row, matching the
      nested FORTRAN loops;
    - closest-mode exact-distance ties choose the earliest comparison row.
    """
    ref = _validate_coordinate_matrix(reference_coordinates, "reference")
    cmp = _validate_coordinate_matrix(comparison_coordinates, "comparison")
    if dismax < 0 or not np.isfinite(dismax):
        raise ValueError("Maximum search distance must be a finite non-negative value.")
    if ref.shape[1] != cmp.shape[1]:
        raise ValueError("Reference and comparison coordinates must have the same dimension.")
    if ref.shape[0] == 0 or cmp.shape[0] == 0 or dismax == 0:
        return _empty_result()

    dsqd = np.float64(dismax) ** 2
    tree = cKDTree(cmp)

    if keep_closest:
        return _pair_closest(tree, ref, cmp, float(dismax), dsqd)
    return _pair_all(tree, ref, cmp, float(dismax), dsqd)


def _pair_all(
    tree: cKDTree,
    ref: np.ndarray,
    cmp: np.ndarray,
    dismax: float,
    dsqd: np.float64,
) -> PairingResult:
    neighborhoods = tree.query_ball_point(ref, r=dismax)
    ref_out: list[int] = []
    cmp_out: list[int] = []
    d2_out: list[float] = []

    for ref_idx, candidates in enumerate(neighborhoods):
        if not candidates:
            continue
        ordered = np.asarray(sorted(candidates), dtype=np.int64)
        delta = cmp[ordered] - ref[ref_idx]
        d2 = np.einsum("ij,ij->i", delta, delta, dtype=np.float64)
        keep = d2 < dsqd  # strict legacy predicate; exact radius is excluded
        for cmp_idx, squared in zip(ordered[keep], d2[keep], strict=True):
            ref_out.append(ref_idx)
            cmp_out.append(int(cmp_idx))
            d2_out.append(float(squared))

    return _make_result(ref_out, cmp_out, d2_out)


def _pair_closest(
    tree: cKDTree,
    ref: np.ndarray,
    cmp: np.ndarray,
    dismax: float,
    dsqd: np.float64,
) -> PairingResult:
    # query() supplies the fast nearest-neighbor seed requested for closest mode.
    # A tiny tie-neighborhood search then restores the FORTRAN first-occurrence rule.
    nearest_dist, nearest_idx = tree.query(
        ref,
        k=1,
        distance_upper_bound=dismax,
        workers=-1,
    )

    ref_out: list[int] = []
    cmp_out: list[int] = []
    d2_out: list[float] = []
    n_cmp = cmp.shape[0]

    for ref_idx, (distance, seed_idx) in enumerate(zip(nearest_dist, nearest_idx, strict=True)):
        if not np.isfinite(distance) or int(seed_idx) >= n_cmp:
            continue
        seed_idx = int(seed_idx)
        seed_delta = cmp[seed_idx] - ref[ref_idx]
        seed_d2 = float(np.dot(seed_delta, seed_delta))
        if not seed_d2 < dsqd:
            continue

        # Include all candidates at the seed radius, then compare exact squared
        # distances in original dataset-2 order. np.nextafter prevents sqrt rounding
        # from accidentally omitting an equal-distance tie.
        tie_radius = float(np.nextafter(np.sqrt(seed_d2), np.inf))
        candidates = sorted(tree.query_ball_point(ref[ref_idx], r=tie_radius))
        if not candidates:
            candidates = [seed_idx]
        candidate_idx = np.asarray(candidates, dtype=np.int64)
        delta = cmp[candidate_idx] - ref[ref_idx]
        exact_d2 = np.einsum("ij,ij->i", delta, delta, dtype=np.float64)
        min_d2 = float(exact_d2.min())
        tied_positions = np.flatnonzero(exact_d2 == min_d2)
        chosen = int(candidate_idx[int(tied_positions[0])])

        if min_d2 < dsqd:
            ref_out.append(ref_idx)
            cmp_out.append(chosen)
            d2_out.append(min_d2)

    return _make_result(ref_out, cmp_out, d2_out)


def brute_force_pairs(
    reference_coordinates: np.ndarray,
    comparison_coordinates: np.ndarray,
    dismax: float,
    keep_closest: bool,
) -> PairingResult:
    """Literal O(n1*n2) reference implementation mirroring the FORTRAN loops."""
    ref = _validate_coordinate_matrix(reference_coordinates, "reference")
    cmp = _validate_coordinate_matrix(comparison_coordinates, "comparison")
    if dismax < 0 or not np.isfinite(dismax):
        raise ValueError("Maximum search distance must be a finite non-negative value.")
    if ref.shape[1] != cmp.shape[1]:
        raise ValueError("Reference and comparison coordinates must have the same dimension.")

    dsqd = np.float64(dismax) ** 2
    ref_out: list[int] = []
    cmp_out: list[int] = []
    d2_out: list[float] = []

    for j in range(ref.shape[0]):
        pair_found = False
        closest_index = -1
        closest_d2 = float(dsqd)
        for i in range(cmp.shape[0]):
            delta = ref[j] - cmp[i]
            squared = float(np.dot(delta, delta))
            if squared < dsqd:
                if not keep_closest:
                    ref_out.append(j)
                    cmp_out.append(i)
                    d2_out.append(squared)
                elif squared < closest_d2:  # strict: ties retain first occurrence
                    pair_found = True
                    closest_index = i
                    closest_d2 = squared
        if keep_closest and pair_found:
            ref_out.append(j)
            cmp_out.append(closest_index)
            d2_out.append(closest_d2)

    return _make_result(ref_out, cmp_out, d2_out)


def pairing_report(
    result: PairingResult, n_reference: int, n_comparison: int
) -> dict[str, float | int]:
    """Compute the requested pairing diagnostics."""
    unique_ref = np.unique(result.reference_indices).size if result.n_pairs else 0
    counts_cmp = (
        np.bincount(result.comparison_indices, minlength=n_comparison)
        if result.n_pairs
        else np.zeros(n_comparison, dtype=np.int64)
    )
    reused_records = int(np.sum(counts_cmp > 1))
    extra_reuses = int(np.maximum(counts_cmp - 1, 0).sum())

    return {
        "reference_records": int(n_reference),
        "comparison_records": int(n_comparison),
        "pairs": result.n_pairs,
        "paired_reference_records": int(unique_ref),
        "pairing_rate_pct": (100.0 * unique_ref / n_reference) if n_reference else 0.0,
        "unpaired_reference_records": int(max(n_reference - unique_ref, 0)),
        "reused_comparison_records": reused_records,
        "extra_comparison_reuses": extra_reuses,
        "mean_distance": float(np.mean(result.distances)) if result.n_pairs else np.nan,
        "median_distance": float(np.median(result.distances)) if result.n_pairs else np.nan,
        "max_distance": float(np.max(result.distances)) if result.n_pairs else np.nan,
    }


def _validate_coordinate_matrix(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{label.title()} coordinates must be a 2D array.")
    if values.shape[1] == 0:
        raise ValueError(f"{label.title()} coordinates must contain at least one axis.")
    if values.size and not np.isfinite(values).all():
        raise ValueError(f"{label.title()} coordinates contain non-finite values.")
    return values


def _make_result(
    ref_indices: Sequence[int],
    cmp_indices: Sequence[int],
    squared_distances: Sequence[float],
) -> PairingResult:
    ref_array = np.asarray(ref_indices, dtype=np.int64)
    cmp_array = np.asarray(cmp_indices, dtype=np.int64)
    d2_array = np.asarray(squared_distances, dtype=np.float64)
    return PairingResult(
        reference_indices=ref_array,
        comparison_indices=cmp_array,
        squared_distances=d2_array,
        distances=np.sqrt(d2_array, dtype=np.float64),
    )


def _empty_result() -> PairingResult:
    empty_i = np.asarray([], dtype=np.int64)
    empty_f = np.asarray([], dtype=np.float64)
    return PairingResult(empty_i, empty_i.copy(), empty_f, empty_f.copy())
