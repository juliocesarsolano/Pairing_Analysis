"""Paired-sample statistics for the comparison figure."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class PairMetrics:
    """Annotations used in the density scatterplot."""

    n: int
    rma_slope: float
    pearson_r: float
    spearman_r: float


def extract_pair_values(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    reference_indices: np.ndarray,
    comparison_indices: np.ndarray,
    reference_variable: str,
    comparison_variable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite paired assay values and a mask into the spatial pair set."""
    ref_values = pd.to_numeric(
        reference.iloc[reference_indices][reference_variable], errors="coerce"
    ).to_numpy(dtype=np.float64)
    cmp_values = pd.to_numeric(
        comparison.iloc[comparison_indices][comparison_variable], errors="coerce"
    ).to_numpy(dtype=np.float64)
    valid = np.isfinite(ref_values) & np.isfinite(cmp_values)
    return ref_values[valid], cmp_values[valid], valid


def compute_pair_metrics(
    reference_values: np.ndarray, comparison_values: np.ndarray
) -> PairMetrics:
    """Compute n, reduced-major-axis slope, Pearson r, and Spearman rho."""
    y = np.asarray(reference_values, dtype=np.float64)
    x = np.asarray(comparison_values, dtype=np.float64)
    if y.shape != x.shape:
        raise ValueError("Reference and comparison arrays must have identical shapes.")
    n = int(y.size)
    if n < 2:
        return PairMetrics(n=n, rma_slope=np.nan, pearson_r=np.nan, spearman_r=np.nan)

    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    if sx == 0.0 or sy == 0.0:
        pearson = np.nan
        spearman = np.nan
        rma = np.nan
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spearman = float(stats.spearmanr(x, y).statistic)
        rma = float(np.sign(pearson) * sy / sx) if np.isfinite(pearson) else np.nan

    return PairMetrics(n=n, rma_slope=rma, pearson_r=pearson, spearman_r=spearman)


def compute_statistics_table(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
    preset: str = "study",
    near_zero_fraction: float = 1e-3,
) -> pd.DataFrame:
    """Compute the report table at full floating-point precision.

    ``diff_pct`` is (comparison-reference)/reference*100. Rows whose reference
    denominator is near zero relative to the reference mean scale are explicitly
    flagged and assigned NaN rather than a misleading percentage.
    """
    ref = np.asarray(reference_values, dtype=np.float64)
    cmp = np.asarray(comparison_values, dtype=np.float64)
    if ref.shape != cmp.shape:
        raise ValueError("Reference and comparison arrays must have identical shapes.")
    if ref.size == 0:
        raise ValueError("Cannot compute statistics for an empty paired sample.")
    if near_zero_fraction < 0:
        raise ValueError("near_zero_fraction must be non-negative.")

    metric_functions = _metric_functions(preset)
    ref_stats = {name: float(func(ref)) for name, func in metric_functions}
    cmp_stats = {name: float(func(cmp)) for name, func in metric_functions}

    reference_std = float(np.std(ref, ddof=1)) if ref.size > 1 else 0.0
    characteristic_scale = max(
        abs(float(np.mean(ref))), abs(reference_std), np.finfo(np.float64).eps
    )
    rows: list[dict[str, object]] = []
    for metric, _ in metric_functions:
        ref_value = ref_stats[metric]
        cmp_value = cmp_stats[metric]
        metric_key = metric.lower()
        metric_scale = 1.0 if metric_key == "cv" else characteristic_scale
        threshold = near_zero_fraction * metric_scale
        near_zero = (
            metric_key not in {"count", "n"}
            and np.isfinite(ref_value)
            and abs(ref_value) <= threshold
        )
        if near_zero or not np.isfinite(ref_value) or ref_value == 0.0:
            diff = np.nan
            flagged = True
        else:
            diff = (cmp_value - ref_value) / ref_value * 100.0
            flagged = False
        rows.append(
            {
                "metric": metric,
                "reference": ref_value,
                "comparison": cmp_value,
                "diff_pct": float(diff) if np.isfinite(diff) else np.nan,
                "diff_flag": bool(flagged),
            }
        )
    return pd.DataFrame(rows)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted values and empirical cumulative probabilities."""
    data = np.sort(np.asarray(values, dtype=np.float64))
    if data.size == 0:
        return data, data.copy()
    probabilities = np.arange(1, data.size + 1, dtype=np.float64) / data.size
    return data, probabilities


def qq_values(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return comparison quantiles on X and reference quantiles on Y."""
    ref = np.asarray(reference_values, dtype=np.float64)
    cmp = np.asarray(comparison_values, dtype=np.float64)
    if ref.size == 0 or cmp.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if ref.size == cmp.size:
        return np.sort(cmp), np.sort(ref)
    n_quantiles = min(ref.size, cmp.size)
    probabilities = np.linspace(0.0, 1.0, n_quantiles, dtype=np.float64)
    return np.quantile(cmp, probabilities), np.quantile(ref, probabilities)


def _metric_functions(preset: str):
    study = [
        ("count", lambda a: a.size),
        ("mean", np.mean),
        ("stdev", lambda a: np.std(a, ddof=1) if a.size > 1 else np.nan),
        ("cv", _coefficient_of_variation),
        ("min", np.min),
        ("P10", lambda a: np.quantile(a, 0.10)),
        ("P50", lambda a: np.quantile(a, 0.50)),
        ("P90", lambda a: np.quantile(a, 0.90)),
        ("max", np.max),
    ]
    if preset == "study":
        return study
    raise ValueError("Unknown statistics preset. Use 'study'.")


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if mean == 0.0 or values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1) / mean)
