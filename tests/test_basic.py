"""Basic regression tests for the projectile simulation utility modules."""

from __future__ import annotations

import math

from atmosphere import air_density, speed_of_sound
from drag import interpolate_cd, load_drag_table
from solver import solve_trajectory
from stability import stability_factor


def test_air_density_and_sound_speed() -> None:
    density = air_density(101325.0, 15.0, 0.0)
    sound_speed = speed_of_sound(15.0, 101325.0, 0.0)
    assert density > 0.0
    assert 330.0 < sound_speed < 350.0


def test_drag_tables_and_interpolation() -> None:
    g1 = load_drag_table("g1")
    g7 = load_drag_table("g7")
    assert len(g1) > 1
    assert len(g7) > 1
    assert interpolate_cd(1.0, "g1") > 0.0
    assert interpolate_cd(1.0, "g7") > 0.0


def test_stability_factor() -> None:
    factor = stability_factor(
        bullet_mass_kg=175.0 / (1.0 / 0.00006479891),
        bullet_diameter_m=(0.308 / (1.0 / 0.0254)),
        bullet_length_m=(1.24 / (1.0 / 0.0254)),
        twist_m_per_turn=(12.0 / (1.0 / 0.0254)),
    )
    assert math.isfinite(factor)
    assert factor > 0.0


def test_solver_returns_trajectory() -> None:
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
