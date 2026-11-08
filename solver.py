"""Three-dimensional trajectory solver with adaptive range sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from atmosphere import air_density, speed_of_sound
from drag import interpolate_cd


GRAVITY_M_S2: float = 9.80665
DEFAULT_ZERO_RANGE_M: float = 100.0
NEAR_ZERO_STEP_M: float = 10.0
MID_RANGE_STEP_M: float = 25.0
FAR_RANGE_STEP_M: float = 50.0


@dataclass
class TrajectoryResult:
    """Container for solved projectile trajectory samples."""

    time_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    vx_m_s: np.ndarray
    vy_m_s: np.ndarray
    vz_m_s: np.ndarray


@dataclass
class ShotParams:
    """Normalized solver inputs, including optional ballistic coefficient."""

    mass_kg: float
    diameter_m: float
    drag_profile: str
    launch_speed_m_s: float
    launch_elevation_rad: float
    launch_azimuth_rad: float = 0.0
    pressure_pa: float = 101325.0
    temperature_c: float = 15.0
    relative_humidity: float = 0.0
    time_end_s: float = 10.0
    initial_altitude_m: float = 0.0
    max_step_s: float = 0.01
    ballistic_coefficient: float | None = None
    zero_range_m: float = DEFAULT_ZERO_RANGE_M


def _adaptive_range_samples(max_range_m: float, zero_range_m: float) -> np.ndarray:
    """Generate increasing range samples at 10/25/50 m regional intervals.

    Args:
        max_range_m: Maximum downrange distance reached by the integration.
        zero_range_m: User's zero distance in meters.

    Returns:
        Range samples including zero and the exact zero and double-zero boundaries
        when those boundaries lie inside the simulated range.
    """
    if max_range_m <= 0.0:
        return np.array([0.0], dtype=float)

    positions = [0.0]
    boundaries = (zero_range_m, 2.0 * zero_range_m, max_range_m)
    steps = (NEAR_ZERO_STEP_M, MID_RANGE_STEP_M, FAR_RANGE_STEP_M)
    current = 0.0
    for boundary, step in zip(boundaries, steps):
        end = min(boundary, max_range_m)
        if end <= current:
            continue
        while current + step < end - 1e-10:
            current += step
            positions.append(current)
        if end > positions[-1] + 1e-10:
            positions.append(end)
        current = end
        if current >= max_range_m - 1e-10:
            break
    return np.asarray(positions, dtype=float)


def projectile_derivative(
    time_s: float,
    state: Sequence[float],
    mass_kg: float,
    diameter_m: float,
    drag_profile: str,
    pressure_pa: float,
    temperature_c: float,
    relative_humidity: float,
    azimuth_rad: float = 0.0,
    ballistic_coefficient: float | None = None,
) -> Sequence[float]:
    """Compute the derivative for [x, y, z, vx, vy, vz]."""
    del time_s  # The force model is autonomous; solve_ivp still supplies time.
    _, _, _, vx, vy, vz = state
    velocity = np.array([vx, vy, vz], dtype=float)
    speed = float(np.linalg.norm(velocity))

    if speed <= 1e-12:
        return np.array([vx, vy, vz, 0.0, -GRAVITY_M_S2, 0.0], dtype=float)

    rho = air_density(pressure_pa, temperature_c, relative_humidity)
    sound_speed = speed_of_sound(temperature_c, pressure_pa, relative_humidity)
    mach = speed / sound_speed
    cd = interpolate_cd(mach, drag_profile)
    if ballistic_coefficient is not None:
        if not math.isfinite(ballistic_coefficient) or ballistic_coefficient <= 0.0:
            raise ValueError("ballistic_coefficient must be positive and finite.")
        cd *= 1.0 / ballistic_coefficient

    area_m2 = math.pi * diameter_m**2 / 4.0
    drag_force_mag = 0.5 * rho * speed**2 * cd * area_m2
    drag_accel = (drag_force_mag / mass_kg) * (-velocity / speed)
    drag_accel[1] -= GRAVITY_M_S2
    return np.array([vx, vy, vz, drag_accel[0], drag_accel[1], drag_accel[2]], dtype=float)


def solve_trajectory(
    mass_kg: float,
    diameter_m: float,
    drag_profile: str,
    launch_speed_m_s: float,
    launch_elevation_rad: float,
    launch_azimuth_rad: float = 0.0,
    pressure_pa: float = 101325.0,
    temperature_c: float = 15.0,
    relative_humidity: float = 0.0,
    time_end_s: float = 10.0,
    initial_altitude_m: float = 0.0,
    max_step_s: float = 0.01,
    ballistic_coefficient: float | None = None,
    zero_range_m: float = DEFAULT_ZERO_RANGE_M,
    wind_speed_m_s: float = 0.0,
    wind_direction_deg: float = 0.0,
) -> TrajectoryResult:
    """Integrate a trajectory and resample it adaptively by downrange distance.

    The adaptive sample spacing is 10 m up to zero range, 25 m from zero to
    twice zero range, and 50 m beyond twice zero range. ``solve_ivp`` dense
    output is parameterized by time, so sampled x positions are mapped to their
    corresponding times before evaluating its interpolant.

    Args:
        mass_kg: Projectile mass in kg.
        diameter_m: Projectile diameter in meters.
        drag_profile: Reference table name, "g1" or "g7".
        launch_speed_m_s: Launch speed in m/s.
        launch_elevation_rad: Elevation above horizontal in radians.
        launch_azimuth_rad: Launch azimuth in radians.
        pressure_pa: Ambient pressure in Pa.
        temperature_c: Ambient temperature in degrees Celsius.
        relative_humidity: Relative humidity in [0, 1].
        time_end_s: End time in seconds.
        initial_altitude_m: Initial altitude in meters.
        max_step_s: Maximum RK45 step size.
        ballistic_coefficient: Optional positive BC used to scale drag.
        zero_range_m: Zeroing distance controlling adaptive sample spacing.
        wind_speed_m_s: Wind speed in m/s.
        wind_direction_deg: Meteorological wind direction FROM which it blows.

    Returns:
        TrajectoryResult sampled at adaptive downrange positions.
    """
    numeric_values = (
        mass_kg, diameter_m, launch_speed_m_s, launch_elevation_rad,
        launch_azimuth_rad, pressure_pa, temperature_c, relative_humidity,
        time_end_s, initial_altitude_m, max_step_s, zero_range_m,
        wind_speed_m_s, wind_direction_deg,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("Solver inputs must be finite.")
    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive.")
    if diameter_m <= 0.0:
        raise ValueError("diameter_m must be positive.")
    if time_end_s <= 0.0:
        raise ValueError("time_end_s must be positive.")
    if max_step_s <= 0.0:
        raise ValueError("max_step_s must be positive.")
    if zero_range_m <= 0.0:
        raise ValueError("zero_range_m must be positive.")
    if wind_speed_m_s < 0.0:
        raise ValueError("wind_speed_m_s must be non-negative.")
    if ballistic_coefficient is not None and (
        not math.isfinite(ballistic_coefficient) or ballistic_coefficient <= 0.0
    ):
        raise ValueError("ballistic_coefficient must be positive and finite.")
    if not 0.0 <= relative_humidity <= 1.0:
        raise ValueError("relative_humidity must be between 0 and 1 inclusive.")

    # The solver's x axis follows the bore; initial crossrange speed is zero.
    vx0 = launch_speed_m_s * math.cos(launch_elevation_rad)
    vy0 = launch_speed_m_s * math.sin(launch_elevation_rad)
    vz0 = 0.0
    initial_state = np.array([0.0, initial_altitude_m, 0.0, vx0, vy0, vz0], dtype=float)

    def rhs(time_s: float, state: Sequence[float]) -> Sequence[float]:
        return projectile_derivative(
            time_s=time_s,
            state=state,
            mass_kg=mass_kg,
            diameter_m=diameter_m,
            drag_profile=drag_profile,
            pressure_pa=pressure_pa,
            temperature_c=temperature_c,
            relative_humidity=relative_humidity,
            azimuth_rad=launch_azimuth_rad,
            ballistic_coefficient=ballistic_coefficient,
        )

    solution = solve_ivp(
        fun=rhs,
        t_span=(0.0, time_end_s),
        y0=initial_state,
        method="RK45",
        max_step=max_step_s,
        dense_output=True,
        rtol=1e-7,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(f"Trajectory integration failed: {solution.message}")
    if solution.sol is None:
        raise RuntimeError("Trajectory integration did not provide dense output.")

    # Dense output is indexed by time, not range. The projectile's downrange
    # position must be monotonic to map each requested x sample to a unique t.
    integrated_x = solution.y[0]
    if np.any(np.diff(integrated_x) < -1e-9):
        raise ValueError("Adaptive range sampling requires monotonically increasing downrange position.")
    max_range_m = max(0.0, float(integrated_x[-1]))
    x_positions = _adaptive_range_samples(max_range_m, zero_range_m)
    sample_times = np.interp(x_positions, integrated_x, solution.t)
    sampled_state = solution.sol(sample_times)

    return TrajectoryResult(
        time_s=np.asarray(sample_times),
        x_m=np.asarray(sampled_state[0]),
        y_m=np.asarray(sampled_state[1]),
        z_m=np.asarray(sampled_state[2]),
        vx_m_s=np.asarray(sampled_state[3]),
        vy_m_s=np.asarray(sampled_state[4]),
        vz_m_s=np.asarray(sampled_state[5]),
    )


if __name__ == "__main__":
    result = solve_trajectory(
        mass_kg=0.010,
        diameter_m=0.008,
        drag_profile="g7",
        launch_speed_m_s=300.0,
        launch_elevation_rad=math.radians(12.0),
        pressure_pa=101325.0,
        temperature_c=15.0,
        relative_humidity=0.0,
        time_end_s=5.0,
        max_step_s=0.01,
        ballistic_coefficient=0.35,
        zero_range_m=100.0,
    )

    assert result.time_s.size > 0
    assert np.all(np.isfinite(result.x_m))
    assert np.all(np.isfinite(result.y_m))
    assert np.all(np.isfinite(result.vx_m_s))
    assert result.y_m.max() > 0.0
    assert result.x_m[-1] > 0.0
    assert result.x_m.size >= 5
    assert np.allclose(np.diff(result.x_m[:5]), NEAR_ZERO_STEP_M)

    print(f"Output points: {result.x_m.size}")
    print(f"First five range samples: {result.x_m[:5].tolist()}")
    print(f"Peak height: {result.y_m.max():.2f} m")
    print(f"Final range: {result.x_m[-1]:.2f} m")
    final_speed = np.linalg.norm([result.vx_m_s[-1], result.vy_m_s[-1], result.vz_m_s[-1]])
    print(f"Final speed: {final_speed:.2f} m/s")
    print("Adaptive sampling self-test: PASS (first four intervals are 10 m)")
