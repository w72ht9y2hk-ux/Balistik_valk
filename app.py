"""Streamlit interface for the Balistik_valk ballistics tools.

This final app shell exposes atmospheric conditions, gyroscopic stability,
reference drag tables, and a working point-mass trajectory calculation.
"""

from __future__ import annotations

import math
import sys
from typing import Tuple


APP_TITLE: str = "Balistik_valk Ballistics"
DEFAULT_PRESSURE_PA: float = 101325.0
DEFAULT_TEMPERATURE_C: float = 15.0
DEFAULT_RELATIVE_HUMIDITY: float = 0.50
DEFAULT_VELOCITY_M_S: float = 800.0
DEFAULT_MASS_KG: float = 0.01134
DEFAULT_DIAMETER_M: float = 0.00782
DEFAULT_LENGTH_M: float = 0.03150
DEFAULT_TWIST_M_PER_TURN: float = 0.254
FEET_PER_METER: float = 1.0 / 0.3048
INCHES_PER_METER: float = 1.0 / 0.0254
GRAINS_PER_KILOGRAM: float = 1.0 / 0.00006479891
PASCALS_PER_HPA: float = 100.0
PASCALS_PER_IN_HG: float = 3386.389


def _mass_to_kg(value: float, unit: str) -> float:
    """Convert a displayed mass value to kilograms.

    Args:
        value: Mass in the selected display unit.
        unit: Either "g", "gr", or "kg".

    Returns:
        Mass in kilograms.
    """
    factors = {"kg": 1.0, "g": 0.001, "gr": 1.0 / GRAINS_PER_KILOGRAM}
    if unit not in factors:
        raise ValueError(f"Unsupported mass unit: {unit}")
    return value * factors[unit]


def _length_to_m(value: float, unit: str) -> float:
    """Convert a displayed length to meters.

    Args:
        value: Length in the selected display unit.
        unit: Either "m", "mm", or "in".

    Returns:
        Length in meters.
    """
    factors = {"m": 1.0, "mm": 0.001, "in": 1.0 / INCHES_PER_METER}
    if unit not in factors:
        raise ValueError(f"Unsupported length unit: {unit}")
    return value * factors[unit]


def _velocity_to_m_s(value: float, unit: str) -> float:
    """Convert a displayed velocity to meters per second.

    Args:
        value: Velocity in the selected display unit.
        unit: Either "m/s" or "ft/s".

    Returns:
        Velocity in meters per second.
    """
    factors = {"m/s": 1.0, "ft/s": 0.3048}
    if unit not in factors:
        raise ValueError(f"Unsupported velocity unit: {unit}")
    return value * factors[unit]


def _temperature_to_c(value: float, unit: str) -> float:
    """Convert a displayed temperature to degrees Celsius.

    Args:
        value: Temperature in the selected display unit.
        unit: Either "°C" or "°F".

    Returns:
        Temperature in degrees Celsius.
    """
    if unit == "°C":
        return value
    if unit == "°F":
        return (value - 32.0) * (5.0 / 9.0)
    raise ValueError(f"Unsupported temperature unit: {unit}")


def _pressure_to_pa(value: float, unit: str) -> float:
    """Convert a displayed pressure to pascals.

    Args:
        value: Pressure in the selected display unit.
        unit: Either "Pa", "hPa", or "inHg".

    Returns:
        Pressure in pascals.
    """
    factors = {"Pa": 1.0, "hPa": PASCALS_PER_HPA, "inHg": PASCALS_PER_IN_HG}
    if unit not in factors:
        raise ValueError(f"Unsupported pressure unit: {unit}")
    return value * factors[unit]


def _display_units(system: str) -> Tuple[str, str, str, str, str]:
    """Return the mass, length, velocity, temperature, and pressure units.

    Args:
        system: Either "Metric" or "US customary".

    Returns:
        A tuple of unit labels for the user interface.
    """
    if system == "Metric":
        return "g", "mm", "m/s", "°C", "hPa"
    if system == "US customary":
        return "gr", "in", "ft/s", "°F", "inHg"
    raise ValueError(f"Unsupported unit system: {system}")


def _render_atmosphere_tab(
    st: object,
    pressure_pa: float,
    temperature_c: float,
    relative_humidity: float,
) -> None:
    """Show estimated air properties for the selected conditions.

    Args:
        st: Streamlit module instance.
        pressure_pa: Ambient pressure in pascals.
        temperature_c: Ambient temperature in degrees Celsius.
        relative_humidity: Relative humidity as a fraction from 0 to 1.

    Returns:
        None.
    """
    from atmosphere import air_density, speed_of_sound

    st.subheader("Atmosphere")
    st.caption("Air-property estimates from the atmosphere module.")
    density = air_density(pressure_pa, temperature_c, relative_humidity)
    sound_speed = speed_of_sound(temperature_c, pressure_pa, relative_humidity)

    col_density, col_sound = st.columns(2)
    col_density.metric("Air density", f"{density:.4f} kg/m³")
    col_sound.metric("Speed of sound", f"{sound_speed:.2f} m/s")

    st.caption(
        f"Inputs: {pressure_pa:.0f} Pa, {temperature_c:.2f} °C, "
        f"{relative_humidity * 100.0:.1f}% RH"
    )


def _render_stability_tab(
    st: object,
    mass_kg: float,
    diameter_m: float,
    length_m: float,
    twist_m_per_turn: float,
    velocity_m_s: float,
    temperature_c: float,
    pressure_pa: float,
) -> None:
    """Show the Miller-rule stability estimate for the selected projectile.

    Args:
        st: Streamlit module instance.
        mass_kg: Projectile mass in kilograms.
        diameter_m: Projectile diameter in meters.
        length_m: Projectile length in meters.
        twist_m_per_turn: Rifling twist length in meters per turn.
        velocity_m_s: Muzzle velocity in meters per second.
        temperature_c: Ambient temperature in degrees Celsius.
        pressure_pa: Ambient pressure in pascals.

    Returns:
        None.
    """
    from stability import stability_factor

    st.subheader("Gyroscopic stability")
    sg = stability_factor(
        bullet_mass_kg=mass_kg,
        bullet_diameter_m=diameter_m,
        bullet_length_m=length_m,
        twist_m_per_turn=twist_m_per_turn,
        velocity_m_s=velocity_m_s,
        temperature_c=temperature_c,
        pressure_pa=pressure_pa,
    )
    st.metric("Estimated Miller Sg", f"{sg:.3f}")
    st.caption(
        "This is an empirical estimate based on the Miller twist rule, "
        "not a guarantee of real-world in-flight behavior."
    )


def _render_drag_tab(st: object, profile: str, mach: float) -> None:
    """Show the selected drag table and interpolated value.

    Args:
        st: Streamlit module instance.
        profile: Drag model name, "g1" or "g7".
        mach: Mach number at which to display the interpolated coefficient.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from drag import interpolate_cd, load_drag_table

    table = load_drag_table(profile)
    mach_values = [point[0] for point in table]
    cd_values = [point[1] for point in table]
    cd = interpolate_cd(mach, profile)

    st.subheader(f"{profile.upper()} reference drag table")
    st.metric(f"Interpolated Cd at Mach {mach:.3f}", f"{cd:.5f}")

    figure, axis = plt.subplots()
    axis.plot(mach_values, cd_values, label=f"{profile.upper()} table")
    axis.scatter([mach], [cd], color="tab:red", zorder=3, label="Selected Mach")
    axis.set_xlabel("Mach")
    axis.set_ylabel("Cd")
    axis.grid(True, alpha=0.3)
    axis.legend()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

    with st.expander("View table data"):
        frame = pd.DataFrame(table, columns=["Mach", "Cd"])
        st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_trajectory_tab(
    st: object,
    mass_kg: float,
    diameter_m: float,
    profile: str,
    launch_speed_m_s: float,
    launch_angle_deg: float,
    pressure_pa: float,
    temperature_c: float,
    relative_humidity: float,
    time_end_s: float,
) -> None:
    """Compute and render the projectile trajectory.

    Args:
        st: Streamlit module instance.
        mass_kg: Projectile mass in kg.
        diameter_m: Projectile diameter in m.
        profile: Drag model name.
        launch_speed_m_s: Launch speed in m/s.
        launch_angle_deg: Launch elevation in degrees.
        pressure_pa: Ambient pressure in Pa.
        temperature_c: Ambient temperature in degC.
        relative_humidity: Relative humidity fraction.
        time_end_s: End time of the simulation in seconds.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from solver import solve_trajectory

    result = solve_trajectory(
        mass_kg=mass_kg,
        diameter_m=diameter_m,
        drag_profile=profile,
        launch_speed_m_s=launch_speed_m_s,
        launch_elevation_rad=math.radians(launch_angle_deg),
        pressure_pa=pressure_pa,
        temperature_c=temperature_c,
        relative_humidity=relative_humidity,
        time_end_s=time_end_s,
    )

    peak_height_m = float(result.y_m.max())
    downrange_m = float(result.x_m[-1])
    final_speed_m_s = float(
        math.sqrt(
            result.vx_m_s[-1] ** 2 + result.vy_m_s[-1] ** 2 + result.vz_m_s[-1] ** 2
        )
    )

    st.subheader("Trajectory")
    col_height, col_range, col_speed = st.columns(3)
    col_height.metric("Peak height", f"{peak_height_m:.2f} m")
    col_range.metric("Downrange", f"{downrange_m:.2f} m")
    col_speed.metric("Final speed", f"{final_speed_m_s:.2f} m/s")

    figure, axis = plt.subplots()
    axis.plot(result.x_m, result.y_m, label="Projectile path")
    axis.set_xlabel("Downrange (m)")
    axis.set_ylabel("Altitude (m)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

    summary = pd.DataFrame(
        {
            "time_s": result.time_s,
            "x_m": result.x_m,
            "y_m": result.y_m,
            "z_m": result.z_m,
            "vx_m_s": result.vx_m_s,
            "vy_m_s": result.vy_m_s,
            "vz_m_s": result.vz_m_s,
        }
    )
    with st.expander("View trajectory data"):
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.caption(
        f"Launch: {launch_speed_m_s:.1f} m/s at {launch_angle_deg:.1f}°, "
        f"profile={profile.upper()}, time_end={time_end_s:.1f} s"
    )


def main() -> None:
    """Run the Streamlit application."""
    import streamlit as st

    from drag import SUPPORTED_PROFILES

    st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
    st.title(APP_TITLE)
    st.write(
        "Explore atmospheric conditions, estimate gyroscopic stability, "
        "inspect the built-in G1/G7 drag tables, and calculate a projectile path."
    )

    unit_system = st.sidebar.selectbox("Input units", options=("Metric", "US customary"))
    mass_unit, length_unit, velocity_unit, temperature_unit, pressure_unit = _display_units(
        unit_system
    )

    with st.sidebar:
        st.header("Environment")
        pressure_value = st.number_input(
            f"Pressure ({pressure_unit})",
            min_value=0.01,
            value=1013.25 if pressure_unit == "hPa" else 29.92,
            step=1.0,
        )
        temperature_value = st.number_input(
            f"Temperature ({temperature_unit})",
            value=15.0 if temperature_unit == "°C" else 59.0,
            step=1.0,
        )
        humidity_percent = st.slider("Relative humidity (%)", 0, 100, 50)

        st.header("Projectile")
        mass_value = st.number_input(
            f"Mass ({mass_unit})",
            min_value=0.001,
            value=11.34 if mass_unit == "g" else 175.0,
            step=0.1,
        )
        diameter_value = st.number_input(
            f"Diameter ({length_unit})",
            min_value=0.001,
            value=7.82 if length_unit == "mm" else 0.308,
            step=0.01,
        )
        length_value = st.number_input(
            f"Length ({length_unit})",
            min_value=0.001,
            value=31.50 if length_unit == "mm" else 1.24,
            step=0.1,
        )
        twist_value = st.number_input(
            f"Twist length ({length_unit}/turn)",
            min_value=0.001,
            value=254.0 if length_unit == "mm" else 10.0,
            step=1.0,
        )
        velocity_value = st.number_input(
            f"Muzzle velocity ({velocity_unit})",
            min_value=0.001,
            value=800.0 if velocity_unit == "m/s" else 2625.0,
            step=10.0,
        )
        launch_angle_deg = st.slider("Launch angle (deg)", 1.0, 80.0, 15.0, step=0.5)
        time_end_s = st.slider("Simulation time (s)", 1.0, 30.0, 10.0, step=0.5)

        st.header("Drag table")
        profile = st.selectbox("Reference profile", options=SUPPORTED_PROFILES)
        mach = st.slider("Mach number", min_value=0.0, max_value=5.0, value=1.0, step=0.01)

    pressure_pa = _pressure_to_pa(pressure_value, pressure_unit)
    temperature_c = _temperature_to_c(temperature_value, temperature_unit)
    relative_humidity = humidity_percent / 100.0
    mass_kg = _mass_to_kg(mass_value, mass_unit)
    diameter_m = _length_to_m(diameter_value, length_unit)
    length_m = _length_to_m(length_value, length_unit)
    twist_m_per_turn = _length_to_m(twist_value, length_unit)
    velocity_m_s = _velocity_to_m_s(velocity_value, velocity_unit)

    atmosphere_tab, stability_tab, drag_tab, trajectory_tab = st.tabs(
        ("Atmosphere", "Stability", "Drag tables", "Trajectory")
    )

    with atmosphere_tab:
        _render_atmosphere_tab(st, pressure_pa, temperature_c, relative_humidity)

    with stability_tab:
        _render_stability_tab(
            st,
            mass_kg,
            diameter_m,
            length_m,
            twist_m_per_turn,
            velocity_m_s,
            temperature_c,
            pressure_pa,
        )

    with drag_tab:
        _render_drag_tab(st, profile, mach)

    with trajectory_tab:
        _render_trajectory_tab(
            st,
            mass_kg,
            diameter_m,
            profile,
            velocity_m_s,
            launch_angle_deg,
            pressure_pa,
            temperature_c,
            relative_humidity,
            time_end_s,
        )


def _self_test() -> None:
    """Run lightweight checks for conversions and existing calculation modules."""
    from atmosphere import air_density, speed_of_sound
    from drag import interpolate_cd, load_drag_table
    from solver import solve_trajectory
    from stability import stability_factor

    assert math.isclose(_mass_to_kg(1000.0, "g"), 1.0)
    assert math.isclose(_length_to_m(25.4, "mm"), 0.0254)
    assert math.isclose(_velocity_to_m_s(1.0, "ft/s"), 0.3048)
    assert math.isclose(_temperature_to_c(32.0, "°F"), 0.0)
    assert math.isclose(_pressure_to_pa(1.0, "hPa"), 100.0)
    assert _display_units("Metric") == ("g", "mm", "m/s", "°C", "hPa")

    assert air_density(DEFAULT_PRESSURE_PA, DEFAULT_TEMPERATURE_C) > 0.0
    assert speed_of_sound(DEFAULT_TEMPERATURE_C) > 0.0
    assert len(load_drag_table("g1")) > 1
    assert len(load_drag_table("g7")) > 1
    assert interpolate_cd(1.0, "g1") > 0.0
    assert interpolate_cd(1.0, "g7") > 0.0
    assert stability_factor(
        DEFAULT_MASS_KG,
        DEFAULT_DIAMETER_M,
        DEFAULT_LENGTH_M,
        DEFAULT_TWIST_M_PER_TURN,
        DEFAULT_VELOCITY_M_S,
    ) > 0.0

    trajectory = solve_trajectory(
        mass_kg=0.010,
        diameter_m=0.008,
        drag_profile="g7",
        launch_speed_m_s=300.0,
        launch_elevation_rad=math.radians(12.0),
        pressure_pa=DEFAULT_PRESSURE_PA,
        temperature_c=DEFAULT_TEMPERATURE_C,
        relative_humidity=0.0,
        time_end_s=5.0,
    )
    assert trajectory.time_s.size > 0
    assert trajectory.y_m.max() > 0.0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
        print("app.py self-test passed")
    else:
        main()
