"""Three-dimensional trajectory solver with wind-relative aerodynamic drag.

The state is [x, y, z, vx, vy, vz]. Coordinates are downrange, vertical, and
crossrange respectively. Inputs and returned trajectory values use SI units.
Wind direction is meteorological: degrees FROM which the wind blows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from atmosphere import air_density, speed_of_sound
from drag import interpolate_cd


GRAVITY_M_S2: float = 9.80665


@dataclass
class TrajectoryResult:
    """Container for solved projectile trajectory samples, all in SI units."""

    time_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    vx_m_s: np.ndarray
    vy_m_s: np.ndarray
    vz_m_s: np.ndarray


def _wind_components(
    wind_speed_m_s: float,
    wind_direction_deg: float,
    azimuth_rad: float,
) -> tuple[float, float]:
    """Resolve meteorological wind into downrange and crossrange components.

    Args:
        wind_speed_m_s: Wind speed in m/s.
        wind_direction_deg: Direction FROM which wind blows, clockwise degrees.
        azimuth_rad: Shooting azimuth in radians.

    Returns:
        (wx, wz), the wind components along downrange and crossrange axes in m/s.
    """
    relative_to_angle_deg = wind_direction_deg + 180.0 - math.degrees(azimuth_rad)
    angle_rad = math.radians(relative_to_angle_deg)
    return (
        wind_speed_m_s * math.cos(angle_rad),
        wind_speed_m_s * math.sin(angle_rad),
    )


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
    wind_speed_m_s: float = 0.0,
    wind_direction_deg: float = 0.0,
) -> Sequence[float]:
    """Compute derivatives for [x, y, z, vx, vy, vz] with wind-relative drag.

    Args:
        time_s: Current time in seconds (retained for the integrator interface).
        state: Current state [x, y, z, vx, vy, vz].
        mass_kg: Projectile mass in kg.
        diameter_m: Projectile diameter in meters.
        drag_profile: Drag table name, e.g. "g1" or "g7".
        pressure_pa: Ambient pressure in Pa.
        temperature_c: Ambient temperature in degrees Celsius.
        relative_humidity: Relative humidity in [0, 1].
        azimuth_rad: Shooting azimuth in radians.
        wind_speed_m_s: Wind speed in m/s.
        wind_direction_deg: Meteorological direction FROM which wind blows.

    Returns:
        Time derivative of the six-component state vector.
    """
    del time_s  # The force model is autonomous; time is required by solve_ivp.
    _, _, _, vx, vy, vz = state
    wx, wz = _wind_components(wind_speed_m_s, wind_direction_deg, azimuth_rad)
    rvx = vx - wx
    rvy = vy
    rvz = vz - wz
    relative_velocity = np.array([rvx, rvy, rvz], dtype=float)
    relative_speed = float(np.linalg.norm(relative_velocity))

    if relative_speed <= 1e-12:
        return np.array([vx, vy, vz, 0.0, -GRAVITY_M_S2, 0.0], dtype=float)

    rho = air_density(
        pressure_pa=pressure_pa,
        temperature_c=temperature_c,
        relative_humidity=relative_humidity,
    )
    sound_speed = speed_of_sound(
        temperature_c=temperature_c,
        pressure_pa=pressure_pa,
        relative_humidity=relative_humidity,
    )
    mach = relative_speed / sound_speed
    cd = interpolate_cd(mach, drag_profile)

    area_m2 = math.pi * diameter_m**2 / 4.0
    drag_coeff = 0.5 * rho * cd * area_m2 / mass_kg
    drag_accel = -drag_coeff * relative_speed * relative_velocity
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
    wind_speed_m_s: float = 0.0,
    wind_direction_deg: float = 0.0,
) -> TrajectoryResult:
    """Solve projectile flight with RK45 and wind-relative drag.

    Args:
        mass_kg: Projectile mass in kg.
        diameter_m: Projectile diameter in meters.
        drag_profile: Drag table name, e.g. "g1" or "g7".
        launch_speed_m_s: Launch speed in m/s.
        launch_elevation_rad: Elevation above horizontal in radians.
        launch_azimuth_rad: Shooting azimuth in radians, used to orient wind.
        pressure_pa: Ambient pressure in Pa.
        temperature_c: Ambient temperature in degrees Celsius.
        relative_humidity: Relative humidity in [0, 1].
        time_end_s: End time for integration in seconds.
        initial_altitude_m: Initial altitude in meters.
        max_step_s: Maximum RK45 step size in seconds.
        wind_speed_m_s: Wind speed in m/s.
        wind_direction_deg: Meteorological direction FROM which wind blows.

    Returns:
        TrajectoryResult containing positions in meters and velocities in m/s.
    """
    finite_values = {
        "mass_kg": mass_kg,
        "diameter_m": diameter_m,
        "launch_speed_m_s": launch_speed_m_s,
        "launch_elevation_rad": launch_elevation_rad,
        "launch_azimuth_rad": launch_azimuth_rad,
        "pressure_pa": pressure_pa,
        "temperature_c": temperature_c,
        "relative_humidity": relative_humidity,
        "time_end_s": time_end_s,
        "initial_altitude_m": initial_altitude_m,
        "max_step_s": max_step_s,
        "wind_speed_m_s": wind_speed_m_s,
        "wind_direction_deg": wind_direction_deg,
    }
    for name, value in finite_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive.")
    if diameter_m <= 0.0:
        raise ValueError("diameter_m must be positive.")
    if launch_speed_m_s < 0.0:
        raise ValueError("launch_speed_m_s must be non-negative.")
    if pressure_pa <= 0.0:
        raise ValueError("pressure_pa must be positive.")
    if time_end_s <= 0.0:
        raise ValueError("time_end_s must be positive.")
    if max_step_s <= 0.0:
        raise ValueError("max_step_s must be positive.")
    if wind_speed_m_s < 0.0:
        raise ValueError("wind_speed_m_s must be non-negative.")
    if not 0.0 <= relative_humidity <= 1.0:
        raise ValueError("relative_humidity must be between 0 and 1 inclusive.")

    # x is the bore/downrange axis; the projectile has no initial crossrange speed.
    vx0 = launch_speed_m_s * math.cos(launch_elevation_rad)
    vy0 = launch_speed_m_s * math.sin(launch_elevation_rad)
    vz0 = 0.0
    initial_state = np.array(
        [0.0, initial_altitude_m, 0.0, vx0, vy0, vz0],
        dtype=float,
    )

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
            wind_speed_m_s=wind_speed_m_s,
            wind_direction_deg=wind_direction_deg,
        )

    solution = solve_ivp(
        fun=rhs,
        t_span=(0.0, time_end_s),
        y0=initial_state,
        method="RK45",
        max_step=max_step_s,
        dense_output=False,
        rtol=1e-7,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(f"Trajectory integration failed: {solution.message}")

    return TrajectoryResult(
        time_s=solution.t,
        x_m=solution.y[0],
        y_m=solution.y[1],
        z_m=solution.y[2],
        vx_m_s=solution.y[3],
        vy_m_s=solution.y[4],
        vz_m_s=solution.y[5],
    )


def _wind_drift_self_test() -> bool:
    """Check that a 10 m/s left crosswind gives plausible drift at 1 km.

    Returns:
        True when absolute crossrange drift lies between 0.30 and 0.50 m.
    """
    mass_168_gr = 168.0 * 0.00006479891
    diameter_308_in = 0.308 * 0.0254
    result = solve_trajectory(
        mass_kg=mass_168_gr,
        diameter_m=diameter_308_in,
        drag_profile="g7",
        launch_speed_m_s=800.0,
        launch_elevation_rad=0.0,
        pressure_pa=101325.0,
        temperature_c=15.0,
        relative_humidity=0.0,
        time_end_s=5.0,
        max_step_s=0.005,
        wind_speed_m_s=10.0,
        wind_direction_deg=90.0,
    )
    indices = np.flatnonzero(result.x_m >= 1000.0)
    if indices.size == 0:
        print("Wind drift self-test: FAIL (trajectory did not reach 1000 m)")
        return False

    upper = int(indices[0])
    if upper == 0:
        drift_m = float(result.z_m[0])
    else:
        lower = upper - 1
        drift_m = float(
            np.interp(1000.0, result.x_m[lower : upper + 1], result.z_m[lower : upper + 1])
        )
    passed = 0.30 <= abs(drift_m) <= 0.50
    status = "PASS" if passed else "FAIL"
    print(f"Wind drift self-test: {status} ({drift_m:.3f} m at 1000 m; expected 0.30-0.50 m)")
    return passed


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
    )
    assert result.time_s.size > 0
    assert np.all(np.isfinite(result.x_m))
    assert np.all(np.isfinite(result.y_m))
    assert np.all(np.isfinite(result.vx_m_s))
    assert result.y_m.max() > 0.0
    assert result.x_m[-1] > 0.0
    print(f"Time samples: {result.time_s.size}")
    print(f"Peak height: {result.y_m.max():.2f} m")
    print(f"Final range: {result.x_m[-1]:.2f} m")
    final_speed = np.linalg.norm([result.vx_m_s[-1], result.vy_m_s[-1], result.vz_m_s[-1]])
    print(f"Final speed: {final_speed:.2f} m/s")
    if not _wind_drift_self_test():
        raise SystemExit(1)
