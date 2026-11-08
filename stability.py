"""Miller-rule gyroscopic stability estimate for a projectile.

The original empirical Miller rule uses grain, inch, and calibers-per-turn
units. This module accepts SI inputs and performs those conversions internally.
"""

from __future__ import annotations

import math

GRAINS_PER_KILOGRAM: float = 1.0 / 0.00006479891
INCHES_PER_METER: float = 1.0 / 0.0254
STANDARD_VELOCITY_M_S: float = 2800.0 * 0.3048
STANDARD_TEMPERATURE_C: float = 15.0
STANDARD_PRESSURE_PA: float = 101325.0
STANDARD_PRESSURE_IN_HG: float = 29.92
PASCALS_PER_IN_HG: float = 3386.389


def stability_factor(
    bullet_mass_kg: float,
    bullet_diameter_m: float,
    bullet_length_m: float,
    twist_m_per_turn: float,
    velocity_m_s: float = STANDARD_VELOCITY_M_S,
    temperature_c: float = STANDARD_TEMPERATURE_C,
    pressure_pa: float = STANDARD_PRESSURE_PA,
) -> float:
    """Estimate gyroscopic stability factor using the Miller twist rule.

    Args:
        bullet_mass_kg: Projectile mass in kilograms.
        bullet_diameter_m: Projectile diameter in meters.
        bullet_length_m: Projectile length in meters.
        twist_m_per_turn: Rifling twist length in meters per turn.
        velocity_m_s: Muzzle velocity in meters per second.
        temperature_c: Ambient temperature in degrees Celsius.
        pressure_pa: Ambient pressure in pascals.

    Returns:
        Dimensionless gyroscopic stability factor Sg.
    """
    if bullet_mass_kg <= 0.0:
        raise ValueError("bullet_mass_kg must be positive.")
    if bullet_diameter_m <= 0.0:
        raise ValueError("bullet_diameter_m must be positive.")
    if bullet_length_m <= 0.0:
        raise ValueError("bullet_length_m must be positive.")
    if twist_m_per_turn <= 0.0:
        raise ValueError("twist_m_per_turn must be positive.")
    if velocity_m_s <= 0.0:
        raise ValueError("velocity_m_s must be positive.")
    if pressure_pa <= 0.0:
        raise ValueError("pressure_pa must be positive.")
    if temperature_c <= -273.15:
        raise ValueError("temperature_c must be above absolute zero.")

    mass_grains = bullet_mass_kg * GRAINS_PER_KILOGRAM
    diameter_in = bullet_diameter_m * INCHES_PER_METER
    length_in = bullet_length_m * INCHES_PER_METER
    twist_in_per_turn = twist_m_per_turn * INCHES_PER_METER

    length_calibers = length_in / diameter_in
    twist_calibers_per_turn = twist_in_per_turn / diameter_in

    base_sg = (30.0 * mass_grains) / (
        twist_calibers_per_turn**2
        * diameter_in**3
        * length_calibers
        * (1.0 + length_calibers**2)
    )

    velocity_correction = (velocity_m_s / STANDARD_VELOCITY_M_S) ** (1.0 / 3.0)
    temperature_f = temperature_c * (9.0 / 5.0) + 32.0
    temperature_correction = (temperature_f + 460.0) / (59.0 + 460.0)
    pressure_in_hg = pressure_pa / PASCALS_PER_IN_HG
    pressure_correction = STANDARD_PRESSURE_IN_HG / pressure_in_hg

    return base_sg * velocity_correction * temperature_correction * pressure_correction


if __name__ == "__main__":
    # Representative .308-caliber, 175-grain, 1.24-inch projectile in a 1:12 twist.
    sg_standard = stability_factor(
        bullet_mass_kg=175.0 / GRAINS_PER_KILOGRAM,
        bullet_diameter_m=0.308 / INCHES_PER_METER,
        bullet_length_m=1.24 / INCHES_PER_METER,
        twist_m_per_turn=12.0 / INCHES_PER_METER,
    )
    sg_faster_twist = stability_factor(
        bullet_mass_kg=175.0 / GRAINS_PER_KILOGRAM,
        bullet_diameter_m=0.308 / INCHES_PER_METER,
        bullet_length_m=1.24 / INCHES_PER_METER,
        twist_m_per_turn=10.0 / INCHES_PER_METER,
    )

    assert math.isfinite(sg_standard) and sg_standard > 0.0
    assert sg_faster_twist > sg_standard
    assert abs(
        stability_factor(
            bullet_mass_kg=175.0 / GRAINS_PER_KILOGRAM,
            bullet_diameter_m=0.308 / INCHES_PER_METER,
            bullet_length_m=1.24 / INCHES_PER_METER,
            twist_m_per_turn=12.0 / INCHES_PER_METER,
            velocity_m_s=STANDARD_VELOCITY_M_S,
            temperature_c=STANDARD_TEMPERATURE_C,
            pressure_pa=STANDARD_PRESSURE_PA,
        )
        - sg_standard
    ) < 1e-12

    print(f"Sg (1:12 twist) = {sg_standard:.3f}")
    print(f"Sg (1:10 twist) = {sg_faster_twist:.3f}")
