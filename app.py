"""Streamlit interface for the Balistik_valk ballistics tools.

The UI is a thin presentation layer: user-entered values are converted to SI
units here, then passed to the atmosphere, drag, stability, and solver modules.
"""

from __future__ import annotations

import math
import platform
import sys
from typing import Any, Tuple


APP_TITLE: str = "Balistik_valk Ballistics"
DEFAULT_PRESSURE_PA: float = 101325.0
DEFAULT_TEMPERATURE_C: float = 15.0
DEFAULT_RELATIVE_HUMIDITY: float = 0.50
DEFAULT_VELOCITY_M_S: float = 800.0
DEFAULT_MASS_KG: float = 0.01134
DEFAULT_DIAMETER_M: float = 0.00782
DEFAULT_LENGTH_M: float = 0.03150
DEFAULT_TWIST_M_PER_TURN: float = 0.254
INCHES_PER_METER: float = 1.0 / 0.0254
GRAINS_PER_KILOGRAM: float = 1.0 / 0.00006479891
PASCALS_PER_HPA: float = 100.0
PASCALS_PER_IN_HG: float = 3386.389


def _mass_to_kg(value: float, unit: str) -> float:
    """Convert a displayed mass value to kilograms.

    Args:
        value: Mass in the selected display unit.
        unit: One of "g", "gr", or "kg".

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
        unit: One of "m", "mm", or "in".

    Returns:
        Length in meters.
    """
    factors = {"m": 1.0, "mm": 0.001, "in": 0.0254}
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
        unit: One of "Pa", "hPa", or "inHg".

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


def _si_inputs(
    mass_value: float,
    mass_unit: str,
    diameter_value: float,
    length_value: float,
    twist_value: float,
    length_unit: str,
    velocity_value: float,
    velocity_unit: str,
    pressure_value: float,
    pressure_unit: str,
    temperature_value: float,
    temperature_unit: str,
    humidity_percent: float,
) -> dict[str, float]:
    """Convert and validate all user-facing inputs into SI values.

    Args:
        mass_value: Projectile mass in its selected UI unit.
        mass_unit: Mass unit label.
        diameter_value: Projectile diameter in its selected UI unit.
        length_value: Projectile length in its selected UI unit.
        twist_value: Rifling twist length in its selected UI unit per turn.
        length_unit: Length unit label.
        velocity_value: Launch speed in its selected UI unit.
        velocity_unit: Velocity unit label.
        pressure_value: Atmospheric pressure in its selected UI unit.
        pressure_unit: Pressure unit label.
        temperature_value: Air temperature in its selected UI unit.
        temperature_unit: Temperature unit label.
        humidity_percent: Relative humidity percentage.

    Returns:
        Validated normalized SI inputs in a dictionary.
    """
    inputs = {
        "mass_kg": _mass_to_kg(mass_value, mass_unit),
        "diameter_m": _length_to_m(diameter_value, length_unit),
        "length_m": _length_to_m(length_value, length_unit),
        "twist_m_per_turn": _length_to_m(twist_value, length_unit),
        "velocity_m_s": _velocity_to_m_s(velocity_value, velocity_unit),
        "pressure_pa": _pressure_to_pa(pressure_value, pressure_unit),
        "temperature_c": _temperature_to_c(temperature_value, temperature_unit),
        "relative_humidity": humidity_percent / 100.0,
    }
    for key, value in inputs.items():
        if not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
    for key in ("mass_kg", "diameter_m", "length_m", "twist_m_per_turn", "velocity_m_s", "pressure_pa"):
        if inputs[key] <= 0.0:
            raise ValueError(f"{key} must be positive.")
    if inputs["temperature_c"] <= -273.15:
        raise ValueError("Temperature must be above absolute zero.")
    if not 0.0 <= inputs["relative_humidity"] <= 1.0:
        raise ValueError("Relative humidity must be between 0 and 100%.")
    return inputs


def _show_error(st: Any, title: str, error: Exception, debug: bool) -> None:
    """Display a user-friendly error and optional traceback.

    Args:
        st: Streamlit module or container.
        title: Short context for the failed operation.
        error: Exception raised by the calculation.
        debug: Whether to show detailed exception information.

    Returns:
        None.
    """
    st.error(f"{title}: {error}")
    if debug:
        st.exception(error)


def _render_atmosphere_tab(st: Any, inputs: dict[str, float], debug: bool) -> None:
    """Show air properties for the selected atmospheric conditions.

    Args:
        st: Streamlit module instance.
        inputs: Validated SI inputs.
        debug: Whether to show traceback details on errors.

    Returns:
        None.
    """
    from atmosphere import air_density, speed_of_sound

    st.subheader("Atmosphere")
    try:
        density = air_density(
            inputs["pressure_pa"], inputs["temperature_c"], inputs["relative_humidity"]
        )
        sound_speed = speed_of_sound(
            inputs["temperature_c"], inputs["pressure_pa"], inputs["relative_humidity"]
        )
    except (ValueError, OverflowError) as error:
        _show_error(st, "Could not calculate atmospheric properties", error, debug)
        return

    density_col, sound_col = st.columns(2)
    density_col.metric("Air density", f"{density:.4f} kg/m³")
    sound_col.metric("Speed of sound", f"{sound_speed:.2f} m/s")
    st.caption(
        f"{inputs['pressure_pa']:.0f} Pa · {inputs['temperature_c']:.2f} °C · "
        f"{inputs['relative_humidity'] * 100.0:.1f}% RH"
    )


def _render_stability_tab(st: Any, inputs: dict[str, float], debug: bool) -> None:
    """Show the Miller-rule stability estimate for the selected projectile.

    Args:
        st: Streamlit module instance.
        inputs: Validated SI inputs.
        debug: Whether to show traceback details on errors.

    Returns:
        None.
    """
    from stability import stability_factor

    st.subheader("Gyroscopic stability")
    try:
        sg = stability_factor(
            bullet_mass_kg=inputs["mass_kg"],
            bullet_diameter_m=inputs["diameter_m"],
            bullet_length_m=inputs["length_m"],
            twist_m_per_turn=inputs["twist_m_per_turn"],
            velocity_m_s=inputs["velocity_m_s"],
            temperature_c=inputs["temperature_c"],
            pressure_pa=inputs["pressure_pa"],
        )
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        _show_error(st, "Could not calculate stability", error, debug)
        return

    st.metric("Estimated Miller Sg", f"{sg:.3f}")
    st.caption(
        "Empirical Miller-rule estimate; it does not guarantee real-world "
        "stability for a specific projectile or operating condition."
    )


def _render_drag_tab(st: Any, profile: str, mach: float, debug: bool) -> None:
    """Show the selected drag table and interpolated drag coefficient.

    Args:
        st: Streamlit module instance.
        profile: Drag model name, "g1" or "g7".
        mach: Mach number for interpolation.
        debug: Whether to show traceback details on errors.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from drag import interpolate_cd, load_drag_table

    st.subheader(f"{profile.upper()} reference drag table")
    try:
        table = load_drag_table(profile)
        cd = interpolate_cd(mach, profile)
    except (ValueError, FileNotFoundError, OSError) as error:
        _show_error(st, "Could not load or interpolate the drag table", error, debug)
        return

    st.metric(f"Interpolated Cd at Mach {mach:.3f}", f"{cd:.5f}")
    mach_values = [point[0] for point in table]
    cd_values = [point[1] for point in table]
    figure, axis = plt.subplots()
    try:
        axis.plot(mach_values, cd_values, label=f"{profile.upper()} table")
        axis.scatter([mach], [cd], color="tab:red", zorder=3, label="Selected Mach")
        axis.set(xlabel="Mach", ylabel="Cd", title=f"{profile.upper()} drag curve")
        axis.grid(True, alpha=0.3)
        axis.legend()
        st.pyplot(figure, clear_figure=True)
    finally:
        plt.close(figure)

    with st.expander("Inspect drag-table data"):
        st.dataframe(
            pd.DataFrame(table, columns=["Mach", "Cd"]),
            use_container_width=True,
            hide_index=True,
        )


def _run_trajectory(inputs: dict[str, float], profile: str, angle_deg: float, duration_s: float) -> Any:
    """Run the trajectory calculation for normalized inputs.

    Args:
        inputs: Validated SI inputs.
        profile: Drag model name.
        angle_deg: Launch elevation in degrees.
        duration_s: Integration duration in seconds.

    Returns:
        TrajectoryResult from the solver module.
    """
    from solver import solve_trajectory

    return solve_trajectory(
        mass_kg=inputs["mass_kg"],
        diameter_m=inputs["diameter_m"],
        drag_profile=profile,
        launch_speed_m_s=inputs["velocity_m_s"],
        launch_elevation_rad=math.radians(angle_deg),
        pressure_pa=inputs["pressure_pa"],
        temperature_c=inputs["temperature_c"],
        relative_humidity=inputs["relative_humidity"],
        time_end_s=duration_s,
    )


def _render_trajectory_tab(
    st: Any,
    inputs: dict[str, float],
    profile: str,
    angle_deg: float,
    duration_s: float,
    debug: bool,
) -> None:
    """Run on demand and display trajectory metrics, plot, and samples.

    Args:
        st: Streamlit module instance.
        inputs: Validated SI inputs.
        profile: Drag model name.
        angle_deg: Launch elevation in degrees.
        duration_s: Integration duration in seconds.
        debug: Whether to show traceback details on errors.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    st.subheader("Trajectory")
    st.caption(
        "Point-mass RK45 integration over the requested time window. "
        "The solver does not yet stop automatically at ground impact."
    )
    if not st.button("Run / update trajectory", type="primary", key="run_trajectory"):
        st.info("Adjust the inputs, then run the trajectory calculation.")
        return

    try:
        result = _run_trajectory(inputs, profile, angle_deg, duration_s)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        st.session_state.pop("trajectory_result", None)
        _show_error(st, "Trajectory calculation failed", error, debug)
        return

    st.session_state["trajectory_result"] = result
    peak_index = int(result.y_m.argmax())
    final_speed = math.sqrt(
        float(result.vx_m_s[-1]) ** 2
        + float(result.vy_m_s[-1]) ** 2
        + float(result.vz_m_s[-1]) ** 2
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Peak height", f"{float(result.y_m.max()):.2f} m")
    metric_cols[1].metric("Downrange at end", f"{float(result.x_m[-1]):.2f} m")
    metric_cols[2].metric("Final speed", f"{final_speed:.2f} m/s")
    metric_cols[3].metric("Time to peak", f"{float(result.time_s[peak_index]):.2f} s")

    figure, axis = plt.subplots()
    try:
        axis.plot(result.x_m, result.y_m, label="Projectile path")
        axis.scatter(
            [result.x_m[peak_index]],
            [result.y_m[peak_index]],
            color="tab:red",
            label="Peak",
            zorder=3,
        )
        axis.set(xlabel="Downrange (m)", ylabel="Altitude (m)", title="Trajectory")
        axis.grid(True, alpha=0.3)
        axis.legend()
        st.pyplot(figure, clear_figure=True)
    finally:
        plt.close(figure)

    data = pd.DataFrame(
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
    with st.expander("Inspect trajectory samples"):
        st.dataframe(data, use_container_width=True, hide_index=True)
        st.download_button(
            "Download trajectory CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="trajectory.csv",
            mime="text/csv",
        )


def main() -> None:
    """Run the Streamlit application."""
    import streamlit as st

    from drag import SUPPORTED_PROFILES

    st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
    st.title(APP_TITLE)
    st.write(
        "Explore atmospheric conditions, estimate gyroscopic stability, "
        "inspect G1/G7 drag tables, and calculate a point-mass trajectory."
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
            f"Mass ({mass_unit})", min_value=0.001,
            value=11.34 if mass_unit == "g" else 175.0, step=0.1,
        )
        diameter_value = st.number_input(
            f"Diameter ({length_unit})", min_value=0.001,
            value=7.82 if length_unit == "mm" else 0.308, step=0.01,
        )
        length_value = st.number_input(
            f"Length ({length_unit})", min_value=0.001,
            value=31.50 if length_unit == "mm" else 1.24, step=0.1,
        )
        twist_value = st.number_input(
            f"Twist length ({length_unit}/turn)", min_value=0.001,
            value=254.0 if length_unit == "mm" else 10.0, step=1.0,
        )
        velocity_value = st.number_input(
            f"Muzzle velocity ({velocity_unit})", min_value=0.001,
            value=800.0 if velocity_unit == "m/s" else 2625.0, step=10.0,
        )
        launch_angle_deg = st.slider("Launch elevation (deg)", 0.1, 80.0, 15.0, step=0.5)
        duration_s = st.slider("Integration duration (s)", 1.0, 30.0, 10.0, step=0.5)

        st.header("Drag table")
        profile = st.selectbox("Reference profile", options=SUPPORTED_PROFILES)
        mach = st.slider("Mach for Cd lookup", 0.0, 5.0, 1.0, step=0.01)
        debug = st.checkbox("Show detailed errors (debug)", value=False)

    try:
        inputs = _si_inputs(
            mass_value, mass_unit, diameter_value, length_value, twist_value, length_unit,
            velocity_value, velocity_unit, pressure_value, pressure_unit,
            temperature_value, temperature_unit, float(humidity_percent),
        )
    except ValueError as error:
        st.error(f"Please check the input values: {error}")
        if debug:
            st.exception(error)
        return

    atmosphere_tab, stability_tab, drag_tab, trajectory_tab = st.tabs(
        ("Atmosphere", "Stability", "Drag tables", "Trajectory")
    )
    with atmosphere_tab:
        _render_atmosphere_tab(st, inputs, debug)
    with stability_tab:
        _render_stability_tab(st, inputs, debug)
    with drag_tab:
        _render_drag_tab(st, profile, mach, debug)
    with trajectory_tab:
        _render_trajectory_tab(st, inputs, profile, launch_angle_deg, duration_s, debug)

    if debug:
        with st.expander("Debug: normalized inputs and runtime"):
            st.json(
                {
                    "si_inputs": inputs,
                    "drag_profile": profile,
                    "drag_lookup_mach": mach,
                    "launch_angle_deg": launch_angle_deg,
                    "integration_duration_s": duration_s,
                    "python": sys.version,
                    "platform": platform.platform(),
                }
            )


def _self_test() -> None:
    """Run quick checks for UI conversions and the project's calculation modules."""
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

    normalized = _si_inputs(
        175.0, "gr", 0.308, 1.24, 10.0, "in", 2625.0, "ft/s",
        29.92, "inHg", 59.0, "°F", 50.0,
    )
    assert math.isclose(normalized["mass_kg"], 175.0 / GRAINS_PER_KILOGRAM)
    assert air_density(DEFAULT_PRESSURE_PA, DEFAULT_TEMPERATURE_C) > 0.0
    assert speed_of_sound(DEFAULT_TEMPERATURE_C) > 0.0
    assert len(load_drag_table("g1")) > 1 and len(load_drag_table("g7")) > 1
    assert interpolate_cd(1.0, "g1") > 0.0 and interpolate_cd(1.0, "g7") > 0.0
    assert stability_factor(
        DEFAULT_MASS_KG, DEFAULT_DIAMETER_M, DEFAULT_LENGTH_M,
        DEFAULT_TWIST_M_PER_TURN, DEFAULT_VELOCITY_M_S,
    ) > 0.0
    result = solve_trajectory(
        mass_kg=0.010, diameter_m=0.008, drag_profile="g7",
        launch_speed_m_s=300.0, launch_elevation_rad=math.radians(12.0),
        pressure_pa=DEFAULT_PRESSURE_PA, temperature_c=DEFAULT_TEMPERATURE_C,
        relative_humidity=0.0, time_end_s=5.0,
    )
    assert result.time_s.size > 0 and result.y_m.max() > 0.0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
        print("app.py self-test passed")
    else:
        main()
