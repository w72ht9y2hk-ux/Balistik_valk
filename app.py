from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Tuple


APP_TITLE: str = "Balistik_valk Ballistics"
DEFAULT_PRESSURE_PA: float = 101325.0
DEFAULT_TEMPERATURE_C: float = 15.0
DEFAULT_MASS_KG: float = 0.01134
DEFAULT_DIAMETER_M: float = 0.00782
DEFAULT_LENGTH_M: float = 0.03150
DEFAULT_TWIST_M_PER_TURN: float = 0.254
DEFAULT_VELOCITY_M_S: float = 800.0
INCHES_PER_METER: float = 1.0 / 0.0254
FEET_PER_METER: float = 1.0 / 0.3048
GRAINS_PER_KILOGRAM: float = 1.0 / 0.00006479891
PASCALS_PER_HPA: float = 100.0
PASCALS_PER_IN_HG: float = 3386.389
KG_M3_TO_LB_FT3: float = 0.0624279606
BULLET_DATA_PATH = Path(__file__).resolve().parent / "data" / "bullets.json"


def _mass_to_kg(value: float, unit: str) -> float:
    """Convert displayed projectile mass to kilograms.

    Args:
        value: Mass value in the displayed unit.
        unit: One of "kg", "g", or "gr" (grains).

    Returns:
        Mass in kilograms.
    """
    factors = {"kg": 1.0, "g": 0.001, "gr": 1.0 / GRAINS_PER_KILOGRAM}
    if unit not in factors:
        raise ValueError(f"Unsupported mass unit: {unit}")
    return value * factors[unit]


def _length_to_m(value: float, unit: str) -> float:
    """Convert displayed length to meters.

    Args:
        value: Length value in the displayed unit.
        unit: One of "m", "mm", or "in".

    Returns:
        Length in meters.
    """
    factors = {"m": 1.0, "mm": 0.001, "in": 0.0254}
    if unit not in factors:
        raise ValueError(f"Unsupported length unit: {unit}")
    return value * factors[unit]


def _velocity_to_m_s(value: float, unit: str) -> float:
    """Convert displayed velocity to meters per second.

    Args:
        value: Velocity value in the displayed unit.
        unit: Either "m/s" or "ft/s".

    Returns:
        Velocity in meters per second.
    """
    factors = {"m/s": 1.0, "ft/s": 0.3048}
    if unit not in factors:
        raise ValueError(f"Unsupported velocity unit: {unit}")
    return value * factors[unit]


def _temperature_to_c(value: float, unit: str) -> float:
    """Convert displayed temperature to degrees Celsius.

    Args:
        value: Temperature value in the displayed unit.
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
    """Convert displayed pressure to pascals.

    Args:
        value: Pressure value in the displayed unit.
        unit: One of "Pa", "hPa", or "inHg".

    Returns:
        Pressure in pascals.
    """
    factors = {"Pa": 1.0, "hPa": PASCALS_PER_HPA, "inHg": PASCALS_PER_IN_HG}
    if unit not in factors:
        raise ValueError(f"Unsupported pressure unit: {unit}")
    return value * factors[unit]


def _distance_from_m(value_m: float, system: str) -> float:
    """Convert an SI distance to the chosen display system.

    Args:
        value_m: Distance in meters.
        system: Either "Metric" or "Imperial".

    Returns:
        Distance in meters for Metric or feet for Imperial.
    """
    if system == "Metric":
        return value_m
    if system == "Imperial":
        return value_m * FEET_PER_METER
    raise ValueError(f"Unsupported unit system: {system}")


def _speed_from_m_s(value_m_s: float, system: str) -> float:
    """Convert SI speed to the chosen display system.

    Args:
        value_m_s: Speed in meters per second.
        system: Either "Metric" or "Imperial".

    Returns:
        Speed in m/s for Metric or ft/s for Imperial.
    """
    if system == "Metric":
        return value_m_s
    if system == "Imperial":
        return value_m_s * FEET_PER_METER
    raise ValueError(f"Unsupported unit system: {system}")


def _density_from_kg_m3(value_kg_m3: float, system: str) -> float:
    """Convert SI air density to the chosen display system.

    Args:
        value_kg_m3: Density in kg/m^3.
        system: Either "Metric" or "Imperial".

    Returns:
        Density in kg/m^3 for Metric or lb/ft^3 for Imperial.
    """
    if system == "Metric":
        return value_kg_m3
    if system == "Imperial":
        return value_kg_m3 * KG_M3_TO_LB_FT3
    raise ValueError(f"Unsupported unit system: {system}")


def _pressure_from_pa(value_pa: float, system: str) -> float:
    """Convert SI pressure to the chosen display system.

    Args:
        value_pa: Pressure in pascals.
        system: Either "Metric" or "Imperial".

    Returns:
        Pressure in hPa for Metric or inHg for Imperial.
    """
    if system == "Metric":
        return value_pa / PASCALS_PER_HPA
    if system == "Imperial":
        return value_pa / PASCALS_PER_IN_HG
    raise ValueError(f"Unsupported unit system: {system}")


def _temperature_from_c(value_c: float, system: str) -> float:
    """Convert SI temperature to the chosen display system.

    Args:
        value_c: Temperature in degrees Celsius.
        system: Either "Metric" or "Imperial".

    Returns:
        Temperature in °C for Metric or °F for Imperial.
    """
    if system == "Metric":
        return value_c
    if system == "Imperial":
        return value_c * (9.0 / 5.0) + 32.0
    raise ValueError(f"Unsupported unit system: {system}")


def _display_units(system: str) -> Tuple[str, str, str, str, str]:
    """Return input labels for mass, length, velocity, temperature, and pressure.

    Args:
        system: Either "Metric" or "Imperial".

    Returns:
        A tuple of unit labels.
    """
    if system == "Metric":
        return "g", "mm", "m/s", "°C", "hPa"
    if system == "Imperial":
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
    """Convert and validate all UI values into the SI inputs used by physics.

    Args:
        mass_value: Projectile mass in its UI unit.
        mass_unit: Mass unit label.
        diameter_value: Projectile diameter in its UI unit.
        length_value: Projectile length in its UI unit.
        twist_value: Twist length per turn in its UI unit.
        length_unit: Length unit label.
        velocity_value: Launch speed in its UI unit.
        velocity_unit: Velocity unit label.
        pressure_value: Ambient pressure in its UI unit.
        pressure_unit: Pressure unit label.
        temperature_value: Ambient temperature in its UI unit.
        temperature_unit: Temperature unit label.
        humidity_percent: Relative humidity percentage.

    Returns:
        Validated normalized values in SI units.
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
            raise ValueError(f"{key} must be finite.")
    for key in (
        "mass_kg", "diameter_m", "length_m", "twist_m_per_turn",
        "velocity_m_s", "pressure_pa",
    ):
        if inputs[key] <= 0.0:
            raise ValueError(f"{key} must be positive.")
    if inputs["temperature_c"] <= -273.15:
        raise ValueError("Temperature must be above absolute zero.")
    if not 0.0 <= inputs["relative_humidity"] <= 1.0:
        raise ValueError("Relative humidity must be between 0 and 100%.")
    return inputs


def _show_error(st: Any, title: str, error: Exception, debug: bool) -> None:
    """Display a readable error with an optional traceback.

    Args:
        st: Streamlit module or container.
        title: Context for the failed operation.
        error: Exception raised by a calculation.
        debug: Whether to display detailed traceback information.

    Returns:
        None.
    """
    st.error(f"{title}: {error}")
    if debug:
        st.exception(error)


def _load_bullet_database() -> list[dict[str, Any]]:
    """Load the KGW-labs bullet bundle from the repo's data directory."""
    if not BULLET_DATA_PATH.exists():
        return []
    payload = json.loads(BULLET_DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Bullet database must be a JSON object.")
    bullets = payload.get("bullets")
    if not isinstance(bullets, list):
        raise ValueError("Bullet database is missing a 'bullets' array.")
    records: list[dict[str, Any]] = []
    for bullet in bullets:
        try:
            if (
                bullet.get("id")
                and bullet.get("maker")
                and bullet.get("name")
                and float(bullet["diameter_in"]) > 0.0
                and float(bullet["weight_gr"]) > 0.0
            ):
                records.append(bullet)
        except (KeyError, TypeError, ValueError):
            continue
    return records


def _bullet_bc_value(bullet: dict[str, Any]) -> tuple[float | None, str | None]:
    """Return the preferred ballistic coefficient and corresponding profile name."""
    if not isinstance(bullet, dict):
        return None, None
    coefficients = bullet.get("bc") or {}
    if coefficients.get("g7") is not None:
        return float(coefficients["g7"]), "g7"
    if coefficients.get("g1") is not None:
        return float(coefficients["g1"]), "g1"
    return None, None


def _bullet_label(bullet: dict[str, Any]) -> str:
    """Format the picker label for a bullet record from the KGW bundle."""
    diameter_in = float(bullet["diameter_in"])
    weight_gr = float(bullet["weight_gr"])
    return f"{bullet['maker']} — .{diameter_in:.3f} — {weight_gr:g} gr ({bullet['name']})"


def _render_atmosphere_tab(st: Any, inputs: dict[str, float], system: str, debug: bool) -> None:
    """Display atmosphere results in the selected measurement system.

    Args:
        st: Streamlit module instance.
        inputs: Validated SI inputs.
        system: Output system, "Metric" or "Imperial".
        debug: Whether to show traceback details on errors.

    Returns:
        None.
    """
    from atmosphere import air_density, speed_of_sound

    try:
        density_si = air_density(inputs["pressure_pa"], inputs["temperature_c"], inputs["relative_humidity"])
        sound_si = speed_of_sound(inputs["temperature_c"], inputs["pressure_pa"], inputs["relative_humidity"])
    except (ValueError, OverflowError) as error:
        _show_error(st, "Could not calculate atmospheric properties", error, debug)
        return
    density_unit = "kg/m³" if system == "Metric" else "lb/ft³"
    speed_unit = "m/s" if system == "Metric" else "ft/s"
    pressure_unit = "hPa" if system == "Metric" else "inHg"
    temperature_unit = "°C" if system == "Metric" else "°F"
    density_col, sound_col = st.columns(2)
    density_col.metric("Air density", f"{_density_from_kg_m3(density_si, system):.5f} {density_unit}")
    sound_col.metric("Speed of sound", f"{_speed_from_m_s(sound_si, system):.2f} {speed_unit}")
    st.caption(
        f"Conditions: {_pressure_from_pa(inputs['pressure_pa'], system):.2f} {pressure_unit} · "
        f"{_temperature_from_c(inputs['temperature_c'], system):.1f} {temperature_unit} · "
        f"{inputs['relative_humidity'] * 100.0:.1f}% RH"
    )


def _render_stability_tab(st: Any, inputs: dict[str, float], debug: bool) -> None:
    """Display Miller-rule stability from SI-normalized inputs.

    Args:
        st: Streamlit module instance.
        inputs: Validated SI inputs.
        debug: Whether to show traceback details on errors.

    Returns:
        None.
    """
    from stability import stability_factor

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
    st.caption("Dimensionless empirical estimate; it is not a guarantee of real-world stability.")
    if debug:
        with st.expander("Stability inputs (SI)"):
            st.json({key: inputs[key] for key in (
                "mass_kg", "diameter_m", "length_m", "twist_m_per_turn",
                "velocity_m_s", "temperature_c", "pressure_pa",
            )})


def _render_drag_tab(st: Any, profile: str, mach: float, debug: bool) -> None:
    """Display a standard drag curve and interpolated Cd value.

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

    try:
        table = load_drag_table(profile)
        cd = interpolate_cd(mach, profile)
    except (ValueError, FileNotFoundError, OSError) as error:
        _show_error(st, "Could not load or interpolate drag table", error, debug)
        return
    st.subheader(f"{profile.upper()} reference drag table")
    st.metric(f"Interpolated Cd at Mach {mach:.3f}", f"{cd:.5f}")
    figure, axis = plt.subplots()
    try:
        axis.plot([point[0] for point in table], [point[1] for point in table], label=profile.upper())
        axis.scatter([mach], [cd], color="tab:red", zorder=3, label="Selected Mach")
        axis.set(xlabel="Mach", ylabel="Cd", title=f"{profile.upper()} drag curve")
        axis.grid(True, alpha=0.3)
        axis.legend()
        st.pyplot(figure, clear_figure=True)
    finally:
        plt.close(figure)
    with st.expander("Inspect drag-table data"):
        st.dataframe(pd.DataFrame(table, columns=["Mach", "Cd"]), use_container_width=True, hide_index=True)


def _run_trajectory(
    inputs: dict[str, float],
    profile: str,
    angle_deg: float,
    duration_s: float,
    ballistic_coefficient: float | None = None,
) -> Any:
    """Run the SI solver using normalized input values.

    Args:
        inputs: Validated SI inputs.
        profile: Drag model name.
        angle_deg: Launch elevation in degrees.
        duration_s: Integration duration in seconds.
        ballistic_coefficient: Optional bullet BC used to scale drag.

    Returns:
        Solver trajectory result with positions in meters and velocities in m/s.
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
        ballistic_coefficient=ballistic_coefficient,
    )


def _trajectory_frame(result: Any, system: str) -> Any:
    """Build a trajectory dataframe in the selected output system.

    Args:
        result: TrajectoryResult whose values are SI.
        system: Either "Metric" or "Imperial".

    Returns:
        Pandas dataframe; time is seconds and the remaining columns use the chosen system.
    """
    import pandas as pd

    if system not in ("Metric", "Imperial"):
        raise ValueError(f"Unsupported unit system: {system}")
    distance_unit = "m" if system == "Metric" else "ft"
    speed_unit = "m/s" if system == "Metric" else "ft/s"
    return pd.DataFrame({
        "time_s": result.time_s,
        f"x_{distance_unit}": [_distance_from_m(float(value), system) for value in result.x_m],
        f"y_{distance_unit}": [_distance_from_m(float(value), system) for value in result.y_m],
        f"z_{distance_unit}": [_distance_from_m(float(value), system) for value in result.z_m],
        f"vx_{speed_unit.replace('/', '_per_')}": [_speed_from_m_s(float(value), system) for value in result.vx_m_s],
        f"vy_{speed_unit.replace('/', '_per_')}": [_speed_from_m_s(float(value), system) for value in result.vy_m_s],
        f"vz_{speed_unit.replace('/', '_per_')}": [_speed_from_m_s(float(value), system) for value in result.vz_m_s],
    })


def _render_trajectory_tab(
    st: Any, inputs: dict[str, float], profile: str, angle_deg: float,
    duration_s: float, system: str, debug: bool,
    ballistic_coefficient: float | None = None,
) -> None:
    """Run and display a trajectory in the selected unit system."""
    import matplotlib.pyplot as plt

    st.subheader("Trajectory")
    distance_unit = "m" if system == "Metric" else "ft"
    speed_unit = "m/s" if system == "Metric" else "ft/s"
    st.caption("Point-mass RK45 result. Solver inputs and internal state remain SI; this view converts outputs only.")
    if not st.button("Run / update trajectory", type="primary", key="run_trajectory"):
        st.info("Adjust inputs, then run the trajectory calculation.")
        return
    try:
        result = _run_trajectory(inputs, profile, angle_deg, duration_s, ballistic_coefficient=ballistic_coefficient)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        st.session_state.pop("trajectory_result", None)
        _show_error(st, "Trajectory calculation failed", error, debug)
        return
    st.session_state["trajectory_result"] = result
    peak_index = int(result.y_m.argmax())
    peak_height = _distance_from_m(float(result.y_m.max()), system)
    downrange = _distance_from_m(float(result.x_m[-1]), system)
    final_speed_si = math.sqrt(float(result.vx_m_s[-1]) ** 2 + float(result.vy_m_s[-1]) ** 2 + float(result.vz_m_s[-1]) ** 2)
    final_speed = _speed_from_m_s(final_speed_si, system)
    cols = st.columns(4)
    cols[0].metric("Peak height", f"{peak_height:.2f} {distance_unit}")
    cols[1].metric(f"Downrange at {duration_s:g} s", f"{downrange:.2f} {distance_unit}")
    cols[2].metric("Final speed", f"{final_speed:.2f} {speed_unit}")
    cols[3].metric("Time to peak", f"{float(result.time_s[peak_index]):.2f} s")

    figure, axis = plt.subplots()
    try:
        plot_x = [_distance_from_m(float(value), system) for value in result.x_m]
        plot_y = [_distance_from_m(float(value), system) for value in result.y_m]
        axis.plot(plot_x, plot_y, label="Projectile path")
        axis.scatter([plot_x[peak_index]], [plot_y[peak_index]], color="tab:red", label="Peak", zorder=3)
        axis.set(xlabel=f"Downrange ({distance_unit})", ylabel=f"Altitude ({distance_unit})", title="Trajectory")
        axis.grid(True, alpha=0.3)
        axis.legend()
        st.pyplot(figure, clear_figure=True)
    finally:
        plt.close(figure)

    data = _trajectory_frame(result, system)
    with st.expander("Inspect trajectory samples"):
        st.dataframe(data, use_container_width=True, hide_index=True)
        st.download_button(
            f"Download {system.lower()} trajectory CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name=f"trajectory_{system.lower()}.csv",
            mime="text/csv",
        )


def main() -> None:
    """Run the Streamlit application."""
    import streamlit as st

    from drag import SUPPORTED_PROFILES

    st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
    st.title(APP_TITLE)
    st.write("A SI-based point-mass ballistics calculator with Metric and Imperial UI units.")
    system = st.sidebar.selectbox("Measurement system", options=("Metric", "Imperial"))
    mass_unit, length_unit, velocity_unit, temperature_unit, pressure_unit = _display_units(system)

    bullet_records: list[dict[str, Any]] = []
    try:
        bullet_records = _load_bullet_database()
    except (OSError, ValueError, json.JSONDecodeError):
        bullet_records = []

    bullet_choices = ["Custom"]
    bullet_ids = ["__custom__"]
    bullet_map: dict[str, dict[str, Any]] = {}
    for bullet in bullet_records:
        bullet_id = str(bullet["id"])
        bullet_ids.append(bullet_id)
        bullet_choices.append(_bullet_label(bullet))
        bullet_map[bullet_id] = bullet

    selected_bullet_id = st.sidebar.selectbox(
        "Bullet preset",
        options=bullet_ids,
        format_func=lambda value: "Custom" if value == "__custom__" else bullet_map[value].get("name", value),
        index=0,
        key="selected_bullet_id",
    )

    selected_bullet = bullet_map.get(selected_bullet_id)
    selected_bc_value: float | None = None
    selected_drag_profile: str | None = None
    if selected_bullet is not None:
        selected_bc_value, selected_drag_profile = _bullet_bc_value(selected_bullet)
        st.sidebar.caption(f"Selected bullet: {_bullet_label(selected_bullet)}")
        if selected_bc_value is not None:
            st.sidebar.caption(f"Using bullet BC: {selected_bc_value:.3f} on {selected_drag_profile.upper()} reference.")
        else:
            st.sidebar.caption("Selected bullet has no published BC.")

    with st.sidebar:
        st.header("Environment")
        pressure_value = st.number_input(f"Pressure ({pressure_unit})", min_value=0.01,
            value=1013.25 if system == "Metric" else 29.92, step=1.0)
        temperature_value = st.number_input(f"Temperature ({temperature_unit})",
            value=15.0 if system == "Metric" else 59.0, step=1.0)
        humidity_percent = st.slider("Relative humidity (%)", 0, 100, 50)
        st.header("Projectile")
        mass_value = st.number_input(f"Mass ({mass_unit})", min_value=0.001,
            value=11.34 if system == "Metric" else 175.0, step=0.1)
        diameter_value = st.number_input(f"Diameter ({length_unit})", min_value=0.001,
            value=7.82 if system == "Metric" else 0.308, step=0.01)
        length_value = st.number_input(f"Projectile length ({length_unit})", min_value=0.001,
            value=31.50 if system == "Metric" else 1.24, step=0.1)
        twist_value = st.number_input(f"Twist length ({length_unit}/turn)", min_value=0.001,
            value=254.0 if system == "Metric" else 10.0, step=1.0)
        velocity_value = st.number_input(f"Muzzle velocity ({velocity_unit})", min_value=0.001,
            value=800.0 if system == "Metric" else 2625.0, step=10.0)
        launch_angle_deg = st.slider("Launch elevation (deg)", 0.1, 80.0, 15.0, step=0.5)
        duration_s = st.slider("Integration duration (s)", 1.0, 30.0, 10.0, step=0.5)
        st.header("Drag table")
        profile = st.selectbox("Reference profile", options=SUPPORTED_PROFILES)
        if selected_drag_profile is not None:
            profile = selected_drag_profile
        mach = st.slider("Mach for Cd lookup", 0.0, 5.0, 1.0, step=0.01)
        debug = st.checkbox("Show detailed errors and SI values", value=False)

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
        _render_atmosphere_tab(st, inputs, system, debug)
    with stability_tab:
        _render_stability_tab(st, inputs, debug)
    with drag_tab:
        _render_drag_tab(st, profile, mach, debug)
    with trajectory_tab:
        _render_trajectory_tab(st, inputs, profile, launch_angle_deg, duration_s, system, debug,
                              ballistic_coefficient=selected_bc_value)

    if debug:
        with st.expander("Debug: normalized SI inputs and runtime"):
            st.json({"si_inputs": inputs, "display_system": system, "drag_profile": profile,
                "lookup_mach": mach, "launch_angle_deg": launch_angle_deg,
                "integration_duration_s": duration_s, "python": sys.version,
                "platform": platform.platform(), "bullet_bc": selected_bc_value,
                "bullet_drag_profile": selected_drag_profile})


def _self_test() -> None:
    """Run unit-conversion and calculation smoke tests without launching Streamlit."""
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
    assert _display_units("Imperial") == ("gr", "in", "ft/s", "°F", "inHg")
    assert math.isclose(_distance_from_m(1.0, "Imperial"), FEET_PER_METER)
    assert math.isclose(_speed_from_m_s(1.0, "Imperial"), FEET_PER_METER)
    assert math.isclose(_pressure_from_pa(101325.0, "Metric"), 1013.25)
    assert math.isclose(_temperature_from_c(0.0, "Imperial"), 32.0)

    metric = _si_inputs(11.34, "g", 7.82, 31.5, 254.0, "mm", 800.0, "m/s",
        1013.25, "hPa", 15.0, "°C", 50.0)
    imperial = _si_inputs(175.0, "gr", 0.308, 1.24, 10.0, "in", 2624.67, "ft/s",
        29.92, "inHg", 59.0, "°F", 50.0)
    for key in ("mass_kg", "diameter_m", "length_m", "pressure_pa", "temperature_c"):
        assert math.isclose(metric[key], imperial[key], rel_tol=0.01)

    assert air_density(DEFAULT_PRESSURE_PA, DEFAULT_TEMPERATURE_C) > 0.0
    assert speed_of_sound(DEFAULT_TEMPERATURE_C) > 0.0
    assert len(load_drag_table("g1")) > 1 and len(load_drag_table("g7")) > 1
    assert interpolate_cd(1.0, "g1") > 0.0 and interpolate_cd(1.0, "g7") > 0.0
    assert stability_factor(DEFAULT_MASS_KG, DEFAULT_DIAMETER_M, DEFAULT_LENGTH_M,
        DEFAULT_TWIST_M_PER_TURN, DEFAULT_VELOCITY_M_S) > 0.0
    result = solve_trajectory(mass_kg=0.010, diameter_m=0.008, drag_profile="g7",
        launch_speed_m_s=300.0, launch_elevation_rad=math.radians(12.0),
        pressure_pa=DEFAULT_PRESSURE_PA, temperature_c=DEFAULT_TEMPERATURE_C,
        relative_humidity=0.0, time_end_s=2.0)
    assert result.time_s.size > 0 and result.y_m.max() > 0.0
    assert math.isclose(_distance_from_m(float(result.x_m[-1]), "Imperial"),
        float(result.x_m[-1]) * FEET_PER_METER)

    result_with_bc = solve_trajectory(
        mass_kg=0.010,
        diameter_m=0.008,
        drag_profile="g7",
        launch_speed_m_s=300.0,
        launch_elevation_rad=math.radians(12.0),
        pressure_pa=DEFAULT_PRESSURE_PA,
        temperature_c=DEFAULT_TEMPERATURE_C,
        relative_humidity=0.0,
        time_end_s=2.0,
        ballistic_coefficient=0.5,
    )
    assert result_with_bc.time_s.size > 0
    assert math.isfinite(result_with_bc.x_m[-1])


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
        print("app.py self-test passed")
    else:
        main()
