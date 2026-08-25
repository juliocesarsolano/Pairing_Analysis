"""Input/output helpers for GSLIB, CSV, Excel, and paired-data exports."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LoadedTable:
    """A loaded dataset plus source metadata."""

    frame: pd.DataFrame
    title: str
    source_name: str
    source_format: str


def _validate_column_names(columns: Iterable[object]) -> list[str]:
    """Return normalized string column names and reject blanks/duplicates."""
    names = [str(col).strip() for col in columns]
    if any(not name for name in names):
        raise ValueError("Column names must not be blank.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicated column name(s): {', '.join(duplicates)}")
    return names


def read_gslib_bytes(payload: bytes, source_name: str = "uploaded.dat") -> LoadedTable:
    """Read a numeric GSLIB/GeoEAS free-format file from bytes.

    Expected layout: title, nvari, nvari variable-name lines, then whitespace-
    separated numeric rows. Data are stored as float64.
    """
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = payload.decode("latin-1")

    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("Malformed GSLIB file: missing title or variable-count line.")

    title = lines[0].strip() or Path(source_name).stem
    try:
        nvari = int(lines[1].strip().split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError("Malformed GSLIB header: nvari must be an integer.") from exc
    if nvari <= 0:
        raise ValueError("Malformed GSLIB header: nvari must be greater than zero.")

    header_end = 2 + nvari
    if len(lines) < header_end:
        raise ValueError("Malformed GSLIB header: not enough variable-name lines.")

    names = _validate_column_names(lines[2:header_end])
    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(lines[header_end:], start=header_end + 1):
        if not raw_line.strip():
            continue
        tokens = raw_line.split()
        if len(tokens) != nvari:
            raise ValueError(
                f"Malformed GSLIB data at line {line_number}: expected {nvari} "
                f"values, found {len(tokens)}."
            )
        try:
            row = [float(token.replace("D", "E").replace("d", "e")) for token in tokens]
        except ValueError as exc:
            raise ValueError(
                f"Malformed GSLIB data at line {line_number}: non-numeric value found."
            ) from exc
        rows.append(row)

    frame = pd.DataFrame(np.asarray(rows, dtype=np.float64), columns=names)
    return LoadedTable(frame=frame, title=title, source_name=source_name, source_format="gslib")


def _sniff_csv_delimiter(text: str) -> str:
    """Infer a CSV delimiter with a conservative comma fallback."""
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_csv_bytes(payload: bytes, source_name: str) -> LoadedTable:
    """Read a delimited text file while preserving duplicate-header validation."""
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = payload.decode("latin-1")
    delimiter = _sniff_csv_delimiter(text)

    header_frame = pd.read_csv(StringIO(text), sep=delimiter, header=None, nrows=1)
    names = _validate_column_names(header_frame.iloc[0].tolist())
    frame = pd.read_csv(StringIO(text), sep=delimiter, header=0)
    frame.columns = names
    return LoadedTable(
        frame=frame,
        title=Path(source_name).stem,
        source_name=source_name,
        source_format="csv",
    )


def read_excel_bytes(payload: bytes, source_name: str) -> LoadedTable:
    """Read the first worksheet from an Excel workbook."""
    header_frame = pd.read_excel(BytesIO(payload), header=None, nrows=1)
    names = _validate_column_names(header_frame.iloc[0].tolist())
    frame = pd.read_excel(BytesIO(payload), header=0)
    frame.columns = names
    return LoadedTable(
        frame=frame,
        title=Path(source_name).stem,
        source_name=source_name,
        source_format="excel",
    )


def load_table(payload: bytes, source_name: str) -> LoadedTable:
    """Dispatch an uploaded dataset to the appropriate parser."""
    suffix = Path(source_name).suffix.lower()
    if suffix in {".dat", ".out"}:
        return read_gslib_bytes(payload, source_name)
    if suffix in {".csv", ".txt"}:
        return read_csv_bytes(payload, source_name)
    if suffix in {".xlsx", ".xls"}:
        return read_excel_bytes(payload, source_name)
    raise ValueError(
        f"Unsupported file extension '{suffix}'. Use GSLIB (.dat/.out), CSV, or Excel."
    )


def build_paired_dataframe(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    ref_indices: np.ndarray,
    cmp_indices: np.ndarray,
    distances: np.ndarray,
) -> pd.DataFrame:
    """Build a paired CSV-friendly dataframe with unambiguous column prefixes."""
    ref_part = reference.iloc[ref_indices].reset_index(drop=True).copy()
    cmp_part = comparison.iloc[cmp_indices].reset_index(drop=True).copy()
    ref_part.columns = [f"ref__{col}" for col in ref_part.columns]
    cmp_part.columns = [f"cmp__{col}" for col in cmp_part.columns]
    paired = pd.concat([ref_part, cmp_part], axis=1)
    paired["pair_distance_m"] = np.asarray(distances, dtype=np.float64)
    return paired


def paired_to_gslib_bytes(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    ref_indices: np.ndarray,
    cmp_indices: np.ndarray,
    reference_title: str,
    comparison_title: str,
) -> bytes:
    """Write legacy-style paired GSLIB output.

    The output declares nvari1 + nvari2 variables and writes the complete original
    reference record followed by the complete comparison record, with no distance
    column, matching GETPAIRS. GSLIB data must be numeric.
    """
    ref_selected = reference.iloc[ref_indices].reset_index(drop=True)
    cmp_selected = comparison.iloc[cmp_indices].reset_index(drop=True)

    for role, frame in (("reference", ref_selected), ("comparison", cmp_selected)):
        non_numeric = [
            col
            for col in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[col])
        ]
        if non_numeric:
            raise ValueError(
                f"Legacy GSLIB export requires numeric columns. {role.title()} dataset "
                f"contains non-numeric column(s): {', '.join(map(str, non_numeric))}."
            )

    buffer = StringIO()
    buffer.write(f"Paired data from {reference_title} and {comparison_title}\n")
    buffer.write(f"{reference.shape[1] + comparison.shape[1]}\n")
    for col in reference.columns:
        buffer.write(f"{col}\n")
    for col in comparison.columns:
        buffer.write(f"{col}\n")

    ref_values = ref_selected.to_numpy(dtype=np.float64, copy=False)
    cmp_values = cmp_selected.to_numpy(dtype=np.float64, copy=False)
    for left, right in zip(ref_values, cmp_values, strict=True):
        values = np.concatenate((left, right))
        buffer.write(" ".join(_format_gslib_number(value) for value in values))
        buffer.write("\n")
    return buffer.getvalue().encode("utf-8")


def _format_gslib_number(value: float) -> str:
    """Format a float for round-trippable free-format GSLIB output."""
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return format(float(value), ".17g")
