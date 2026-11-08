"""Regression and edge-case tests for the ballistics calculation modules."""

from __future__ import annotations

import math

import pytest

from app import _display_units, _si_inputs
from atmosphere import air_density, speed_of_sound
from drag import interpolate_cd, load_drag_table
from solver import solve_trajectory
from stability import stability_factor


@pytest.mark.parametrize("humidity", [0.0, 0.5, 1.0])
def test_air_properties_are_finite_for_valid_humidity(humidity: float) -> None:
    density = air_density(101325.0, 15.0, humidity)
    sound_speed = speed_of_sound(15.0, 101325.0, humidity)
    assert math.isfinite(density) and density > 0.0
    assert math.isfinite(sound_speed) and 330.0 < sound_speed < 350.0


def test_atmosphere_rejects_invalid_pressure_and_humidity() -> None:
    with pytest.raises(ValueError, match="pressure_pa must be positive"):
        air_density(0.0, 15.0)
    with pytest.raises(ValueError, match="relative_humidity"):
        air_density(101325.0, 15.0, 1.1)


def test_drag_tables_are_sorted_and_interpolatable() -> None:
    for profile in ("g1", "g7"):
        table = load_drag_table(profile)
        assert len(table) > 1
        assert all(
            table[index][0] < table[index + 1][0]
            for index in range(len(table) - 1)
        )
        assert math.isfinite(interpolate_cd(1.0, profile))
        assert interpolate_cd(1.0, profile) > 0.0


def test_drag_profile_and_mach_validation() -> None:
    with pytest.raises(ValueError, match="Unsupported drag profile"):
        load_drag_table("unknown")
    with pytest.raises(ValueError, match="Mach number must be non-negative"):
        interpolate_cd(-0.1, "g1")


def test_stability_factor_is_finite_and_faster_twist_increases_sg() -> None:
    mass_kg = 175.0 * 0.00006479891
    diameter_m = 0.308 * 0.0254
    length_m = 1.24 * 0.0254
    sg_1_in_12 = stability_factor(mass_kg, diameter_m, length_m, 12.0 * 0.0254)
    sg_1_in_10 = stability_factor(mass_kg, diameter_m, length_m, 10.0 * 0.0254)
    assert math.isfinite(sg_1_in_12) and sg_1_in_12 > 0.0
    assert sg_1_in_10 > sg_1_in_12


def test_solver_returns_finite_trajectory() -> None:
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
    assert result.y_m.max() > 0.0
    assert result.x_m[-1] > 0.0
    assert all(math.isfinite(float(value)) for value in result.x_m)
    assert all(math.isfinite(float(value)) for value in result.y_m)


def test_solver_rejects_invalid_core_inputs() -> None:
    base_args = {
        "mass_kg": 0.010,
        "diameter_m": 0.008,
        "drag_profile": "g7",
        "launch_speed_m_s": 300.0,
        "launch_elevation_rad": math.radians(12.0),
        "time_end_s": 1.0,
    }
    with pytest.raises(ValueError, match="mass_kg must be positive"):
        solve_trajectory(**{**base_args, "mass_kg": 0.0})
    with pytest.raises(ValueError, match="diameter_m must be positive"):
        solve_trajectory(**{**base_args, "diameter_m": -0.001})
    with pytest.raises(ValueError, match="relative_humidity"):
        solve_trajectory(**{**base_args, "relative_humidity": 1.5})


def test_metric_and_imperial_inputs_normalize_to_same_si_values() -> None:
    metric = _si_inputs(
        11.34, "g", 7.82, 31.496, 254.0, "mm", 800.0, "m/s",
        1013.25, "hPa", 15.0, "°C", 50.0,
    )
    imperial = _si_inputs(
        175.0, "gr", 0.308, 1.24, 10.0, "in", 2624.67, "ft/s",
        29.92, "inHg", 59.0, "°F", 50.0,
    )
    for key in (
        "mass_kg", "diameter_m", "length_m", "twist_m_per_turn",
        "velocity_m_s", "pressure_pa", "temperature_c", "relative_humidity",
    ):
        assert math.isclose(metric[key], imperial[key], rel_tol=0.003, abs_tol=1e-8)


def test_ui_rejects_invalid_normalized_inputs() -> None:
    with pytest.raises(ValueError, match="mass_kg must be positive"):
        _si_inputs(
            0.0, "g", 7.82, 31.5, 254.0, "mm", 800.0, "m/s",
            1013.25, "hPa", 15.0, "°C", 50.0,
        )
    with pytest.raises(ValueError, match="Relative humidity"):
        _si_inputs(
            11.34, "g", 7.82, 31.5, 254.0, "mm", 800.0, "m/s",
            1013.25, "hPa", 15.0, "°C", 101.0,
        )


def test_ui_unit_labels() -> None:
    assert _display_units("Metric") == ("g", "mm", "m/s", "°C", "hPa")
    assert _display_units("Imperial") == ("gr", "in", "ft/s", "°F", "inHg")
    with pytest.raises(ValueError, match="Unsupported unit system"):
        _display_units("unknown")
