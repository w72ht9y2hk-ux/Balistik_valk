"""Monte Carlo impact uncertainty estimates for the trajectory solver."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

import numpy as np

DIRECTION_SIGMA_DEG: float = 10.0


def monte_carlo_trajectory(
    params: Mapping[str, Any],
    n_samples: int = 10000,
    mv_sigma: float = 0.01,
    wind_sigma: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Perturb muzzle velocity and wind, then summarize impact uncertainty.

    ``params`` contains keyword arguments accepted by ``solver.solve_trajectory``
    and also requires ``impact_range_m`` and ``zero_height_m``. Vertical impact
    is reported as drop below the supplied zero-line height; lateral impact is
    the solver's z-axis position. ``mv_sigma`` is a fractional standard
    deviation, while ``wind_sigma`` is an absolute m/s standard deviation.

    Args:
        params: Solver arguments plus target impact range and zero-line height.
        n_samples: Number of independent trajectories to simulate.
        mv_sigma: Fractional Gaussian muzzle-velocity standard deviation.
        wind_sigma: Gaussian wind-speed standard deviation in m/s.
        seed: Seed for reproducible random sampling.

    Returns:
        Dict with means, standard deviations, 5th/50th/95th percentiles,
        95-percentile WEZ radius, individual impacts, mean TOF, and elapsed time.
    """
    from solver import solve_trajectory

    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if not math.isfinite(mv_sigma) or mv_sigma < 0.0:
        raise ValueError("mv_sigma must be finite and non-negative.")
    if not math.isfinite(wind_sigma) or wind_sigma < 0.0:
        raise ValueError("wind_sigma must be finite and non-negative.")

    base = dict(params)
    try:
        target_range_m = float(base.pop("impact_range_m"))
        zero_height_m = float(base.pop("zero_height_m"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("params must include numeric impact_range_m and zero_height_m.") from error
    if not math.isfinite(target_range_m) or target_range_m <= 0.0:
        raise ValueError("impact_range_m must be positive and finite.")
    if not math.isfinite(zero_height_m):
        raise ValueError("zero_height_m must be finite.")
    if "launch_speed_m_s" not in base or "wind_speed_m_s" not in base:
        raise ValueError("params must include launch_speed_m_s and wind_speed_m_s.")

    rng = np.random.default_rng(seed)
    base_velocity = float(base["launch_speed_m_s"])
    base_wind = float(base["wind_speed_m_s"])
    if base_velocity <= 0.0 or base_wind < 0.0:
        raise ValueError("Base muzzle velocity must be positive and wind speed non-negative.")

    start_time = time.perf_counter()
    vertical = np.empty(n_samples, dtype=float)
    lateral = np.empty(n_samples, dtype=float)
    impact_times = np.empty(n_samples, dtype=float)

    for index in range(n_samples):
        sample = dict(base)
        sample_velocity = max(float(rng.normal(base_velocity, base_velocity * mv_sigma)), 1e-6)
        sample_wind = max(float(rng.normal(base_wind, wind_sigma)), 0.0)
        sample_direction = (float(base.get("wind_direction_deg", 0.0)) + float(
            rng.normal(0.0, DIRECTION_SIGMA_DEG)
        )) % 360.0
        sample["launch_speed_m_s"] = sample_velocity
        sample["wind_speed_m_s"] = sample_wind
        sample["wind_direction_deg"] = sample_direction
        trajectory = solve_trajectory(**sample)
        if trajectory.x_m[-1] < target_range_m:
            raise ValueError(
                f"Sample {index + 1} did not reach impact_range_m; increase time_end_s."
            )
        impact_y = float(np.interp(target_range_m, trajectory.x_m, trajectory.y_m))
        impact_z = float(np.interp(target_range_m, trajectory.x_m, trajectory.z_m))
        impact_time = float(np.interp(target_range_m, trajectory.x_m, trajectory.time_s))
        vertical[index] = zero_height_m - impact_y
        lateral[index] = impact_z
        impact_times[index] = impact_time

    elapsed_s = time.perf_counter() - start_time
    mean_vertical = float(np.mean(vertical))
    mean_lateral = float(np.mean(lateral))
    radial_offsets = np.hypot(vertical - mean_vertical, lateral - mean_lateral)

    def percentile_triplet(values: np.ndarray) -> dict[str, float]:
        """Return 5th, median, and 95th percentile values."""
        p5, p50, p95 = np.percentile(values, (5.0, 50.0, 95.0))
        return {"5": float(p5), "50": float(p50), "95": float(p95)}

    vertical_percentiles = percentile_triplet(vertical)
    lateral_percentiles = percentile_triplet(lateral)
    return {
        "mean_impact": {"vertical_m": mean_vertical, "lateral_m": mean_lateral},
        "std": {"vertical_m": float(np.std(vertical, ddof=1 if n_samples > 1 else 0)),
                "lateral_m": float(np.std(lateral, ddof=1 if n_samples > 1 else 0))},
        "percentiles": {"vertical_m": vertical_percentiles, "lateral_m": lateral_percentiles},
        "wez_radius_m": float(np.percentile(radial_offsets, 95.0)),
        "impact_points": {"vertical_m": vertical, "lateral_m": lateral},
        "mean_tof_s": float(np.mean(impact_times)),
        "elapsed_s": elapsed_s,
        "n_samples": n_samples,
    }


if __name__ == "__main__":
    from solver import solve_trajectory

    params = {
        "mass_kg": 0.010,
        "diameter_m": 0.008,
        "drag_profile": "g7",
        "launch_speed_m_s": 300.0,
        "launch_elevation_rad": math.radians(12.0),
        "pressure_pa": 101325.0,
        "temperature_c": 15.0,
        "relative_humidity": 0.0,
        "time_end_s": 5.0,
        "max_step_s": 0.02,
        "wind_speed_m_s": 2.0,
        "wind_direction_deg": 0.0,
        "impact_range_m": 500.0,
        "zero_height_m": 0.0,
    }
    result = monte_carlo_trajectory(params, n_samples=5000, mv_sigma=0.01, wind_sigma=0.5, seed=42)
    assert result["n_samples"] == 5000
    assert math.isfinite(result["wez_radius_m"]) and result["wez_radius_m"] >= 0.0
    assert len(result["impact_points"]["vertical_m"]) == 5000
    print(f"Monte Carlo self-test: PASS; 5000 samples in {result['elapsed_s']:.2f} s")
    print(f"WEZ radius (95%): {result['wez_radius_m']:.3f} m")
