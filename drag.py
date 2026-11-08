"""Drag-coefficient lookup and interpolation for standard ballistic tables.

This module loads the G1 and G7 reference drag tables from CSV files and returns
an interpolated Cd value at the requested Mach number.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


SUPPORTED_PROFILES: Sequence[str] = ("g1", "g7")


def _resolve_csv_path(
    profile: str,
    csv_path: str | Path | None = None,
) -> Path:
    """Resolve the path to the drag-table CSV for the requested profile.

    Args:
        profile: Table name, such as "g1" or "g7".
        csv_path: Optional explicit file path override.

    Returns:
        Path to the drag-table CSV file.
    """
    normalized = profile.strip().lower()
    if normalized not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported drag profile '{profile}'.")

    if csv_path is not None:
        return Path(csv_path)

    repo_root = Path(__file__).resolve().parent
    return repo_root / "data" / f"{normalized}.csv"


def load_drag_table(
    profile: str,
    csv_path: str | Path | None = None,
) -> List[Tuple[float, float]]:
    """Load a drag table from a CSV file.

    Args:
        profile: Table name, such as "g1" or "g7".
        csv_path: Optional explicit file path override.

    Returns:
        A sorted list of (Mach, Cd) pairs.
    """
    path = _resolve_csv_path(profile, csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Drag table not found: {path}")

    table: List[Tuple[float, float]] = []
    with path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV is missing a header row: {path}")

        normalized_fieldnames = {name.strip().lower() for name in reader.fieldnames}
        if "mach" not in normalized_fieldnames or "cd" not in normalized_fieldnames:
            raise ValueError(
                f"CSV must include 'mach' and 'cd' columns: {path}"
            )

        for row in reader:
            mach_value = row.get("mach") or row.get("Mach")
            cd_value = row.get("cd") or row.get("Cd")
            if mach_value is None or cd_value is None:
                continue
            table.append((float(mach_value), float(cd_value)))

    if not table:
        raise ValueError(f"Drag table is empty: {path}")

    table.sort(key=lambda item: item[0])
    return table


def interpolate_cd(
    mach: float,
    profile: str,
    csv_path: str | Path | None = None,
) -> float:
    """Return the drag coefficient at a Mach number by linear interpolation.

    Args:
        mach: Mach number.
        profile: Table name, such as "g1" or "g7".
        csv_path: Optional explicit file path override.

    Returns:
        Interpolated Cd value.
    """
    if mach < 0.0:
        raise ValueError("Mach number must be non-negative.")

    table = load_drag_table(profile, csv_path)
    mach_values = np.array([point[0] for point in table], dtype=float)
    cd_values = np.array([point[1] for point in table], dtype=float)
    return float(np.interp(mach, mach_values, cd_values))


if __name__ == "__main__":
    g1_table = load_drag_table("g1")
    g7_table = load_drag_table("g7")

    assert len(g1_table) > 0
    assert len(g7_table) > 0
    assert abs(interpolate_cd(1.0, "g1") - 0.4805) < 0.05
    assert abs(interpolate_cd(1.0, "g7") - 0.3803) < 0.05
    assert abs(interpolate_cd(0.5, "g1") - 0.2032) < 0.05
    assert abs(interpolate_cd(0.5, "g7") - 0.1194) < 0.05

    print(f"G1 rows={len(g1_table)}")
    print(f"G1 Cd@Mach 1.0 = {interpolate_cd(1.0, 'g1'):.4f}")
    print(f"G7 rows={len(g7_table)}")
    print(f"G7 Cd@Mach 1.0 = {interpolate_cd(1.0, 'g7'):.4f}")
