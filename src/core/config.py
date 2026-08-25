"""GETPAIRS parameter-file import/export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class GetPairsParameters:
    """Parameters in legacy getpairs.par order."""

    first_file: str
    first_xyz: tuple[int, int, int]
    second_file: str
    second_xyz: tuple[int, int, int]
    output_file: str
    dismax: float
    ikeepclose: int


def _legacy_filename(line: str) -> str:
    """Approximate the FORTRAN chknam behavior for parameter-file filenames."""
    value = line.lstrip()
    double_blank = value.find("  ")
    if double_blank >= 0:
        value = value[:double_blank]
    for marker in ("-fi", "\\fi"):
        marker_pos = value.find(marker)
        if marker_pos >= 0:
            value = value[:marker_pos]
    return value.strip()


def parse_getpairs_par(text: str) -> GetPairsParameters:
    """Parse a GETPAIRS v2.000 parameter file."""
    lines = text.splitlines()
    start_index = next(
        (i for i, line in enumerate(lines) if line.lstrip().upper().startswith("STAR")),
        None,
    )
    if start_index is None:
        raise ValueError("Parameter file does not contain 'START OF PARAMETERS'.")

    payload = lines[start_index + 1 :]
    if len(payload) < 7:
        raise ValueError("Parameter file ends before all seven parameter entries are present.")

    first_file = _legacy_filename(payload[0])
    first_xyz = _parse_xyz(payload[1], "first dataset")
    second_file = _legacy_filename(payload[2])
    second_xyz = _parse_xyz(payload[3], "second dataset")
    output_file = _legacy_filename(payload[4])

    try:
        dismax = float(payload[5].split()[0])
        ikeepclose = int(payload[6].split()[0])
    except (ValueError, IndexError) as exc:
        raise ValueError(
            "Invalid maximum distance or ikeepclose value in parameter file."
        ) from exc

    if dismax < 0:
        raise ValueError("Maximum search distance must be non-negative.")
    if ikeepclose not in {0, 1}:
        raise ValueError("ikeepclose must be 0 (all pairs) or 1 (closest only).")

    return GetPairsParameters(
        first_file=first_file,
        first_xyz=first_xyz,
        second_file=second_file,
        second_xyz=second_xyz,
        output_file=output_file,
        dismax=dismax,
        ikeepclose=ikeepclose,
    )


def _parse_xyz(line: str, label: str) -> tuple[int, int, int]:
    numbers = re.findall(r"[-+]?\d+", line)
    if len(numbers) < 3:
        raise ValueError(f"Invalid X/Y/Z column indices for {label}.")
    xyz = tuple(int(value) for value in numbers[:3])
    if any(value < 0 for value in xyz):
        raise ValueError("Legacy coordinate indices must be zero or positive.")
    return xyz  # type: ignore[return-value]


def export_getpairs_par(params: GetPairsParameters) -> str:
    """Render a syntactically valid legacy GETPAIRS parameter file."""
    return "\n".join(
        [
            "                  Parameters for GETPAIRS",
            "                  ***********************",
            "",
            "START OF PARAMETERS:",
            f"{params.first_file:<30} -first data file",
            f"{params.first_xyz[0]:d}  {params.first_xyz[1]:d}  {params.first_xyz[2]:d}"
            "                       -   columns for X, Y, Z",
            f"{params.second_file:<30} -second data file",
            f"{params.second_xyz[0]:d}  {params.second_xyz[1]:d}  {params.second_xyz[2]:d}"
            "                       -   columns for X, Y, Z",
            f"{params.output_file:<30} -output file with pairs",
            f"{params.dismax:.17g}                          -maximum distance",
            f" {params.ikeepclose:d}                            "
            "-keep all pairs or closest only (0=all, 1=closest)",
            "",
        ]
    )


def indices_for_uploaded_file(
    params: GetPairsParameters | None,
    uploaded_name: str,
) -> tuple[int, int, int] | None:
    """Return matching PAR coordinate indices for an uploaded filename, if found."""
    if params is None:
        return None
    name = Path(uploaded_name).name.lower()
    if Path(params.first_file).name.lower() == name:
        return params.first_xyz
    if Path(params.second_file).name.lower() == name:
        return params.second_xyz
    return None
