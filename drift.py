"""Additional empirical spin-drift and Coriolis estimates for the UI.

These corrections are displayed separately from the solver trajectory and are
not applied to its positions. The spin estimate uses the Litz time-of-flight
relationship with a nominal Sg because this API does not include bullet length,
diameter, mass, or twist direction. Positive spin_rate_rps denotes right-hand
spin; negative values denote left-hand spin.
"""

from __future__ import annotations

import math

EARTH_ROTATION_RAD_S = 7.2921159e-5
NOMINAL_STABILITY_FACTOR = 1.5
INCHES_TO_CM = 2.54


def spin_drift(t: float, spin_rate_rps: float, v0: float, v: float) -> float:
    """Estimate spin drift in cm using the Litz TOF empirical relationship.

    Litz's commonly used relationship is 1.25 * (Sg + 1.2) * TOF**1.83
    inches. This compact UI helper uses nominal Sg=1.5; spin sign chooses
    right/left direction. Speeds are validated as trajectory context but this
    approximation does not independently infer Sg from them.

    Args:
        t: Time of flight in seconds.
        spin_rate_rps: Signed projectile spin rate in revolutions per second.
        v0: Muzzle velocity in m/s.
        v: Current velocity in m/s.

    Returns:
        Signed spin drift estimate in centimeters.
    """
    if not all(math.isfinite(value) for value in (t, spin_rate_rps, v0, v)):
        raise ValueError("Spin-drift inputs must be finite.")
    if t < 0.0:
        raise ValueError("time of flight must be non-negative.")
    if v0 <= 0.0 or v <= 0.0:
        raise ValueError("velocities must be positive.")
    if spin_rate_rps == 0.0 or t == 0.0:
        return 0.0
    drift_inches = 1.25 * (NOMINAL_STABILITY_FACTOR + 1.2) * t**1.83
    return math.copysign(drift_inches * INCHES_TO_CM, spin_rate_rps)


def coriolis_deflection(
    t: float,
    v: float,
    lat_deg: float,
    az_deg: float,
) -> tuple[float, float]:
    """Estimate Coriolis east-west and north-south displacement in meters.

    Uses constant horizontal velocity over the flight time and integrates the
    local horizontal Coriolis acceleration. Azimuth is clockwise from north.

    Args:
        t: Time of flight in seconds.
        v: Representative projectile speed in m/s.
        lat_deg: Latitude in degrees, north positive.
        az_deg: Azimuth in degrees clockwise from north.

    Returns:
        (east_west_m, north_south_m), with east and north positive.
    """
    if not all(math.isfinite(value) for value in (t, v, lat_deg, az_deg)):
        raise ValueError("Coriolis inputs must be finite.")
    if t < 0.0 or v < 0.0:
        raise ValueError("time and speed must be non-negative.")
    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError("latitude must be between -90 and 90 degrees.")

    latitude = math.radians(lat_deg)
    azimuth = math.radians(az_deg)
    # Local east/north acceleration for horizontal motion, a = -2 Ω × v.
    acceleration_east = 2.0 * EARTH_ROTATION_RAD_S * math.sin(latitude) * v * math.cos(azimuth)
    acceleration_north = -2.0 * EARTH_ROTATION_RAD_S * math.sin(latitude) * v * math.sin(azimuth)
    factor = 0.5 * t * t
    return acceleration_east * factor, acceleration_north * factor


if __name__ == "__main__":
    assert spin_drift(0.0, 40.0, 800.0, 600.0) == 0.0
    assert spin_drift(1.0, 40.0, 800.0, 600.0) > 0.0
    assert spin_drift(1.0, -40.0, 800.0, 600.0) < 0.0
    east, north = coriolis_deflection(1.0, 800.0, 45.0, 0.0)
    assert east > 0.0 and abs(north) < 1e-12
    _, south = coriolis_deflection(1.0, 800.0, 45.0, 90.0)
    assert south < 0.0
    print(f"Spin drift at 1 s (nominal Sg): {spin_drift(1.0, 40.0, 800.0, 600.0):+.2f} cm")
    print(f"Coriolis at 1 s, 45N northbound: EW={east:+.6f} m, NS={north:+.6f} m")
