"""Air-property utilities for the Balistik_valk ballistic solver.

This module provides a compact CIPM-2007 density model and Arden Buck vapor
pressure relations for moist air.
"""

from __future__ import annotations

import math

AIR_GAS_CONSTANT: float = 287.058
WATER_GAS_CONSTANT: float = 461.495
GAMMA_DRY_AIR: float = 1.4
KELVIN_OFFSET: float = 273.15
STANDARD_PRESSURE_PA: float = 101325.0


def _saturation_vapor_pressure_pa(temperature_c: float) -> float:
    """Compute the saturation vapor pressure of water in Pa.

    Args:
        temperature_c: Air temperature in degrees Celsius.

    Returns:
        Saturation vapor pressure in pascals using the Arden Buck approximation.
    """
    if temperature_c < -50.0 or temperature_c > 60.0:
        raise ValueError("Temperature is outside the supported Arden Buck range.")

    exponent = (18.678 - (temperature_c / 234.5)) * (
        temperature_c / (temperature_c + 257.14)
    )
    saturation_hpa = 6.1121 * math.exp(exponent)
    return saturation_hpa * 100.0


def air_density(
    pressure_pa: float,
    temperature_c: float,
    relative_humidity: float = 0.0,
) -> float:
    """Compute moist-air density using the CIPM-2007 formulation.

    Args:
        pressure_pa: Ambient pressure in pascals.
        temperature_c: Ambient temperature in degrees Celsius.
        relative_humidity: Relative humidity in the range [0, 1].

    Returns:
        Air density in kg/m^3.
    """
    if pressure_pa <= 0.0:
        raise ValueError("pressure_pa must be positive.")
    if not 0.0 <= relative_humidity <= 1.0:
        raise ValueError("relative_humidity must be between 0 and 1 inclusive.")

    temperature_k = temperature_c + KELVIN_OFFSET
    vapor_pressure_pa = _saturation_vapor_pressure_pa(temperature_c) * relative_humidity
    dry_air_pressure_pa = max(pressure_pa - vapor_pressure_pa, 0.0)

    dry_air_density = dry_air_pressure_pa / (AIR_GAS_CONSTANT * temperature_k)
    water_vapor_density = vapor_pressure_pa / (WATER_GAS_CONSTANT * temperature_k)
    return dry_air_density + water_vapor_density


def speed_of_sound(
    temperature_c: float,
    pressure_pa: float = STANDARD_PRESSURE_PA,
    relative_humidity: float = 0.0,
) -> float:
    """Compute the local speed of sound in humid air.

    Args:
        temperature_c: Ambient temperature in degrees Celsius.
        pressure_pa: Ambient pressure in pascals.
        relative_humidity: Relative humidity in the range [0, 1].

    Returns:
        Speed of sound in m/s.
    """
    if pressure_pa <= 0.0:
        raise ValueError("pressure_pa must be positive.")
    if not 0.0 <= relative_humidity <= 1.0:
        raise ValueError("relative_humidity must be between 0 and 1 inclusive.")

    temperature_k = temperature_c + KELVIN_OFFSET
    vapor_pressure_pa = _saturation_vapor_pressure_pa(temperature_c) * relative_humidity
    dry_air_pressure_pa = max(pressure_pa - vapor_pressure_pa, 0.0)
    effective_gas_constant = (
        dry_air_pressure_pa * AIR_GAS_CONSTANT + vapor_pressure_pa * WATER_GAS_CONSTANT
    ) / pressure_pa

    return math.sqrt(GAMMA_DRY_AIR * effective_gas_constant * temperature_k)


if __name__ == "__main__":
    rho_dry = air_density(STANDARD_PRESSURE_PA, 15.0, 0.0)
    rho_humid = air_density(STANDARD_PRESSURE_PA, 15.0, 0.5)
    sonic_speed = speed_of_sound(15.0, STANDARD_PRESSURE_PA, 0.0)

    assert rho_dry > 0.0
    assert rho_humid > 0.0
    assert abs(rho_dry - 1.225) < 0.15
    assert 330.0 < sonic_speed < 350.0

    print(f"rho_dry={rho_dry:.4f} kg/m^3")
    print(f"rho_humid={rho_humid:.4f} kg/m^3")
    print(f"speed_of_sound={sonic_speed:.2f} m/s")
