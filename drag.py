"""Drag-coefficient lookup and interpolation for standard ballistic tables.

When no checked-in ``*_fine.csv`` is present, the source G1/G7 table is
interpolated onto a 200-point Mach grid with a cubic spline and cached locally.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


SUPPORTED_PROFILES: Sequence[str] = ("g1", "g7")
FINE_TABLE_POINT_COUNT: int = 200


def _read_table(path: Path) -> List[Tuple[float, float]]:
    """Read and validate Mach/Cd pairs from a CSV table."""
    if not path.exists():
        raise FileNotFoundError(f"Drag table not found: {path}")

    table: List[Tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV is missing a header row: {path}")
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        if "mach" not in field_map or "cd" not in field_map:
            raise ValueError(f"CSV must include 'mach' and 'cd' columns: {path}")
        for row in reader:
            mach_value = row.get(field_map["mach"])
            cd_value = row.get(field_map["cd"])
            if mach_value is None or cd_value is None or not mach_value.strip() or not cd_value.strip():
                continue
            table.append((float(mach_value), float(cd_value)))

    table.sort(key=lambda item: item[0])
    if len(table) < 2:
        raise ValueError(f"Drag table must contain at least two rows: {path}")
    if any(table[i][0] >= table[i + 1][0] for i in range(len(table) - 1)):
        raise ValueError(f"Mach values must be unique: {path}")
    return table


def _generate_fine_table(source_path: Path, fine_path: Path) -> None:
    """Generate a 200-point cubic-spline CSV unless it already exists.

    Args:
        source_path: Original G1/G7 source CSV.
        fine_path: Destination fine CSV.

    Returns:
        None.
    """
    if fine_path.exists():
        return
    source = _read_table(source_path)
    mach = np.asarray([point[0] for point in source], dtype=float)
    cd = np.asarray([point[1] for point in source], dtype=float)
    spline = CubicSpline(mach, cd)
    fine_mach = np.linspace(float(mach[0]), float(mach[-1]), FINE_TABLE_POINT_COUNT)
    fine_cd = spline(fine_mach)

    fine_path.parent.mkdir(parents=True, exist_ok=True)
    with fine_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("mach", "cd"))
        writer.writerows((f"{m:.8f}", f"{c:.10f}") for m, c in zip(fine_mach, fine_cd))


def _resolve_csv_path(profile: str, csv_path: str | Path | None = None) -> Path:
    """Resolve explicit table or prefer the fine table, generating it if needed."""
    normalized = profile.strip().lower()
    if normalized not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported drag profile '{profile}'.")
    if csv_path is not None:
        return Path(csv_path)

    data_dir = Path(__file__).resolve().parent / "data"
    fine_path = data_dir / f"{normalized}_fine.csv"
    source_path = data_dir / f"{normalized}.csv"
    if fine_path.exists():
        return fine_path
    if not source_path.exists():
        raise FileNotFoundError(f"Drag table not found: {source_path}")
    # Generate the fine CSV from the original table when it is absent; the
    # original is preserved and remains the fallback if generation fails.
    try:
        _generate_fine_table(source_path, fine_path)
    except (OSError, ValueError):
        return source_path
    return fine_path if fine_path.exists() else source_path


@lru_cache(maxsize=4)
def _load_cached(path_string: str) -> tuple[tuple[float, float], ...]:
    """Cache parsed table contents so solver RHS evaluations avoid repeated I/O."""
    return tuple(_read_table(Path(path_string)))


def load_drag_table(
    profile: str,
    csv_path: str | Path | None = None,
) -> List[Tuple[float, float]]:
    """Load a preferred fine table or an explicitly supplied source table.

    Args:
        profile: Table name, such as "g1" or "g7".
        csv_path: Optional explicit file path override.

    Returns:
        Sorted list of (Mach, Cd) pairs.
    """
    path = _resolve_csv_path(profile, csv_path)
    return list(_load_cached(str(path.resolve())))


def interpolate_cd(
    mach: float,
    profile: str,
    csv_path: str | Path | None = None,
) -> float:
    """Return Cd by linear interpolation on the selected Mach/Cd table.

    Args:
        mach: Mach number.
        profile: Table name, such as "g1" or "g7".
        csv_path: Optional explicit file path override.

    Returns:
        Interpolated Cd; values outside the table domain use endpoint values.
    """
    if mach < 0.0:
        raise ValueError("Mach number must be non-negative.")
    table = load_drag_table(profile, csv_path)
    mach_values = np.asarray([point[0] for point in table], dtype=float)
    cd_values = np.asarray([point[1] for point in table], dtype=float)
    return float(np.interp(mach, mach_values, cd_values))


if __name__ == "__main__":
    g1_table = load_drag_table("g1")
    g7_table = load_drag_table("g7")

    assert len(g1_table) == FINE_TABLE_POINT_COUNT
    assert len(g7_table) == FINE_TABLE_POINT_COUNT
    assert abs(interpolate_cd(1.0, "g1") - 0.4805) < 0.05
    assert abs(interpolate_cd(1.0, "g7") - 0.3803) < 0.05
    assert abs(interpolate_cd(0.5, "g1") - 0.2032) < 0.05
    assert abs(interpolate_cd(0.5, "g7") - 0.1194) < 0.05

    print(f"G1 points={len(g1_table)}")
    print(f"G1 Cd@Mach 1.0 = {interpolate_cd(1.0, 'g1'):.4f}")
    print(f"G7 points={len(g7_table)}")
    print(f"G7 Cd@Mach 1.0 = {interpolate_cd(1.0, 'g7'):.4f}")
