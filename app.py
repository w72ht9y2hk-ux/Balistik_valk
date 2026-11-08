"""Streamlit application for the Balistik_valk trajectory tools.

The UI accepts Metric or Imperial values, converts them to SI, and uses the
repository's SI physics modules. Optional drift/Monte Carlo modules are loaded
when present so the app remains usable while those modules are developed.
"""

from __future__ import annotations

import json
import math
from typing import Any

import requests

from atmosphere import air_density, speed_of_sound
from drag import SUPPORTED_PROFILES, interpolate_cd
from solver import TrajectoryResult, solve_trajectory
from stability import stability_factor

try:
    import drift as drift_module
except ImportError:  # The current repository does not yet include drift.py.
    drift_module = None

try:
    import monte_carlo as monte_carlo_module
except ImportError:  # The current repository does not yet include monte_carlo.py.
    monte_carlo_module = None


APP_TITLE = "Balistik_valk Ballistics"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
METERS_PER_YARD = 0.9144
METERS_PER_INCH = 0.0254
METERS_PER_FOOT = 0.3048
GRAINS_PER_KG = 1.0 / 0.00006479891
PA_PER_INHG = 3386.389

# Persisted UI settings use these keys so JSON presets can be safely restored.
CONFIG_KEYS = {
    "units": "units",
    "profile": "profile",
    "muzzle_velocity": "muzzle_velocity",
    "mass": "mass",
    "diameter": "diameter",
    "bc": "bc",
    "zero_range": "zero_range",
    "max_range": "max_range",
    "pressure": "pressure",
    "temperature": "temperature",
    "humidity": "humidity",
    "wind_speed": "wind_speed",
    "wind_direction": "wind_direction",
    "latitude": "latitude",
}


def _to_si(value: float, units: str, kind: str) -> float:
    """Convert a supported UI value to SI.

    Args:
        value: User-entered quantity.
        units: Selected display system, Metric or Imperial.
        kind: Quantity name: speed, mass, length, pressure, or temperature.

    Returns:
        Quantity in SI units.
    """
    if kind == "speed":
        return value if units == "Metric" else value * METERS_PER_FOOT
    if kind == "mass":
        return value / 1000.0 if units == "Metric" else value / GRAINS_PER_KG
    if kind == "length":
        return value if units == "Metric" else value * METERS_PER_YARD
    if kind == "diameter":
        return value / 1000.0 if units == "Metric" else value * METERS_PER_INCH
    if kind == "pressure":
        return value * 100.0 if units == "Metric" else value * PA_PER_INHG
    if kind == "temperature":
        return value if units == "Metric" else (value - 32.0) * (5.0 / 9.0)
    raise ValueError(f"Unsupported conversion kind: {kind}")


def _from_si(value: float, units: str, kind: str) -> float:
    """Convert an SI value for display in the selected system."""
    if kind == "speed":
        return value if units == "Metric" else value / METERS_PER_FOOT
    if kind == "length":
        return value if units == "Metric" else value / METERS_PER_YARD
    if kind == "pressure":
        return value / 100.0 if units == "Metric" else value / PA_PER_INHG
    if kind == "temperature":
        return value if units == "Metric" else value * (9.0 / 5.0) + 32.0
    raise ValueError(f"Unsupported conversion kind: {kind}")


def _weather(latitude: float, longitude: float) -> dict[str, float | str]:
    """Fetch current Open-Meteo conditions and return normalized values.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.

    Returns:
        Temperature (C), humidity (percent), pressure (hPa), wind speed (m/s),
        wind direction (degrees FROM), and provider timestamp.
    """
    if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError("Invalid latitude")
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise ValueError("Invalid longitude")
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,pressure_msl,"
                "wind_speed_10m,wind_direction_10m"
            ),
            "timezone": "auto",
        },
        timeout=10,
    )
    response.raise_for_status()
    current = response.json()["current"]
    result: dict[str, float | str] = {
        "temperature_c": float(current["temperature_2m"]),
        "humidity_percent": float(current["relative_humidity_2m"]),
        "pressure_hpa": float(current["pressure_msl"]),
        # Open-Meteo defaults wind_speed_10m to km/h.
        "wind_speed_m_s": float(current["wind_speed_10m"]) / 3.6,
        "wind_direction_deg": float(current["wind_direction_10m"]),
        "timestamp": str(current["time"]),
    }
    if any(not math.isfinite(float(value)) for key, value in result.items() if key != "timestamp"):
        raise ValueError("Non-finite weather response")
    return result


def _trajectory_at_angle(
    *, mass_kg: float, diameter_m: float, profile: str, muzzle_velocity_m_s: float,
    angle_rad: float, pressure_pa: float, temperature_c: float, humidity: float,
    wind_speed_m_s: float, wind_direction_deg: float, max_range_m: float,
) -> TrajectoryResult:
    """Run the existing solver long enough to attempt the requested range."""
    forward_speed = max(muzzle_velocity_m_s * math.cos(angle_rad), 1.0)
    duration_s = min(max(max_range_m / forward_speed * 2.5 + 2.0, 2.0), 120.0)
    return solve_trajectory(
        mass_kg=mass_kg,
        diameter_m=diameter_m,
        drag_profile=profile,
        launch_speed_m_s=muzzle_velocity_m_s,
        launch_elevation_rad=angle_rad,
        pressure_pa=pressure_pa,
        temperature_c=temperature_c,
        relative_humidity=humidity,
        wind_speed_m_s=wind_speed_m_s,
        wind_direction_deg=wind_direction_deg,
        time_end_s=duration_s,
        max_step_s=0.02,
    )


def _height_at_range(result: TrajectoryResult, range_m: float) -> float:
    """Interpolate trajectory height at a requested downrange distance."""
    if len(result.x_m) == 0 or float(result.x_m[-1]) < range_m:
        raise ValueError(
            "The simulated trajectory did not reach the zero/max range. "
            "Try a shorter range or higher muzzle velocity."
        )
    return float(__import__("numpy").interp(range_m, result.x_m, result.y_m))


def _solve_zeroed_trajectory(
    *, mass_kg: float, diameter_m: float, profile: str, muzzle_velocity_m_s: float,
    zero_range_m: float, max_range_m: float, pressure_pa: float,
    temperature_c: float, humidity: float, wind_speed_m_s: float,
    wind_direction_deg: float,
) -> tuple[TrajectoryResult, float]:
    """Find launch elevation that crosses the sight line at the zero range.

    Uses bracketed bisection and the existing point-mass solver. The sight line
    is taken to pass through the muzzle; scope height/offset is not modeled.
    """
    def evaluate(angle: float) -> tuple[float, TrajectoryResult]:
        trajectory = _trajectory_at_angle(
            mass_kg=mass_kg, diameter_m=diameter_m, profile=profile,
            muzzle_velocity_m_s=muzzle_velocity_m_s, angle_rad=angle,
            pressure_pa=pressure_pa, temperature_c=temperature_c, humidity=humidity,
            wind_speed_m_s=wind_speed_m_s, wind_direction_deg=wind_direction_deg,
            max_range_m=max_range_m,
        )
        return _height_at_range(trajectory, zero_range_m), trajectory

    low, high = -0.05, 0.35
    low_height, _ = evaluate(low)
    high_height, high_result = evaluate(high)
    if low_height * high_height > 0.0:
        raise ValueError("Could not find a zeroing angle for these inputs and zero range.")

    best = high_result
    for _ in range(18):
        middle = (low + high) / 2.0
        middle_height, best = evaluate(middle)
        if abs(middle_height) < 0.005:
            return best, middle
        if low_height * middle_height <= 0.0:
            high = middle
        else:
            low = middle
            low_height = middle_height
    return best, (low + high) / 2.0


def _make_results(result: TrajectoryResult, zero_range_m: float, max_range_m: float) -> Any:
    """Create a table of drop and drift, truncated at max range."""
    import numpy as np
    import pandas as pd

    zero_height = _height_at_range(result, zero_range_m)
    mask = result.x_m <= max_range_m
    if not np.any(mask):
        raise ValueError("No trajectory samples fall within the selected maximum range.")
    ranges = result.x_m[mask]
    return pd.DataFrame(
        {
            "time_s": result.time_s[mask],
            "range_m": ranges,
            # Positive drop means below the zero line.
            "drop_m": zero_height - result.y_m[mask],
            "drift_m": result.z_m[mask],
            "velocity_m_s": np.sqrt(
                result.vx_m_s[mask] ** 2
                + result.vy_m_s[mask] ** 2
                + result.vz_m_s[mask] ** 2
            ),
        }
    )


def _apply_config(config: dict[str, Any], st: Any) -> None:
    """Load supported JSON config keys into Streamlit's session state."""
    for key in CONFIG_KEYS.values():
        if key in config:
            st.session_state[key] = config[key]


def main() -> None:
    """Run the Streamlit ballistics application."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
    st.title(APP_TITLE)
    st.caption("Point-mass trajectory estimate. Solver calculations use SI units.")

    with st.sidebar:
        st.header("Configuration")
        uploaded_config = st.file_uploader("Load settings (JSON)", type=("json",), key="config_upload")
        if st.button("Load JSON settings", disabled=uploaded_config is None, key="load_config"):
            try:
                loaded = json.loads(uploaded_config.getvalue().decode("utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("JSON root must be an object")
                _apply_config(loaded, st)
                st.success("Settings loaded. Review the inputs below.")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                st.error("Could not load JSON settings")

        units = st.selectbox("Units", ("Metric", "Imperial"), key=CONFIG_KEYS["units"])
        profile = st.selectbox(
            "Drag table", SUPPORTED_PROFILES, key=CONFIG_KEYS["profile"],
            help="The solver uses the selected G1/G7 reference table; BC is retained as a user setting but is not consumed by the current solver.",
        )
        st.header("Location")
        latitude = st.number_input("Latitude (°)", min_value=-90.0, max_value=90.0, value=0.0, step=0.01, key=CONFIG_KEYS["latitude"])
        longitude = st.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=0.0, step=0.01, key="longitude")
        if st.button("Get Current Weather", key="get_weather"):
            try:
                weather = _weather(float(latitude), float(longitude))
                st.session_state["weather_temperature_c"] = float(weather["temperature_c"])
                st.session_state["weather_pressure_hpa"] = float(weather["pressure_hpa"])
                st.session_state["weather_humidity"] = float(weather["humidity_percent"])
                st.session_state["weather_wind_m_s"] = float(weather["wind_speed_m_s"])
                st.session_state["weather_wind_direction"] = float(weather["wind_direction_deg"])
                st.session_state["weather_timestamp"] = str(weather["timestamp"])
                st.session_state.pop("weather_error", None)
            except (requests.RequestException, KeyError, TypeError, ValueError, OverflowError):
                st.session_state["weather_error"] = True
        if st.session_state.get("weather_error"):
            st.error("Could not fetch weather data")
        elif st.session_state.get("weather_timestamp"):
            st.caption(f"Updated: {st.session_state['weather_timestamp']} from Open-Meteo")

        st.header("Environment")
        pressure_default = st.session_state.get("weather_pressure_hpa", 1013.25)
        temp_default = st.session_state.get("weather_temperature_c", 15.0)
        if units == "Imperial":
            pressure_default = pressure_default * 100.0 / PA_PER_INHG
            temp_default = temp_default * 9.0 / 5.0 + 32.0
        pressure = st.number_input(
            f"Pressure ({'hPa' if units == 'Metric' else 'inHg'})", min_value=0.01,
            value=float(pressure_default), step=1.0, key=f"pressure_{units}",
        )
        temperature = st.number_input(
            f"Temperature ({'°C' if units == 'Metric' else '°F'})",
            value=float(temp_default), step=1.0, key=f"temperature_{units}",
        )
        humidity = st.slider(
            "Humidity (%)", min_value=0, max_value=100,
            value=int(st.session_state.get("weather_humidity", 50)), key="humidity",
        )
        wind_default = float(st.session_state.get("weather_wind_m_s", 0.0))
        if units == "Imperial":
            wind_default /= 0.44704
        wind_speed = st.number_input(
            f"Wind speed ({'m/s' if units == 'Metric' else 'mph'})", min_value=0.0,
            value=wind_default, step=0.1, key=f"wind_speed_{units}",
        )
        wind_direction = st.number_input(
            "Wind direction FROM (°)", min_value=0.0, max_value=360.0,
            value=float(st.session_state.get("weather_wind_direction", 0.0)),
            step=1.0, key="wind_direction",
        )

        st.header("Shot setup")
        velocity = st.number_input(
            f"Muzzle velocity ({'m/s' if units == 'Metric' else 'ft/s'})",
            min_value=1.0, value=800.0 if units == "Metric" else 2625.0,
            step=10.0, key=CONFIG_KEYS["muzzle_velocity"],
        )
        mass = st.number_input(
            f"Projectile mass ({'g' if units == 'Metric' else 'gr'})",
            min_value=0.01, value=10.0 if units == "Metric" else 168.0,
            step=0.1, key=CONFIG_KEYS["mass"],
        )
        diameter = st.number_input(
            f"Diameter ({'mm' if units == 'Metric' else 'in'})",
            min_value=0.01, value=7.82 if units == "Metric" else 0.308,
            step=0.01, key=CONFIG_KEYS["diameter"],
        )
        bc = st.number_input(
            "Ballistic coefficient (BC)", min_value=0.001, value=0.462,
            step=0.001, format="%.3f", key=CONFIG_KEYS["bc"],
            help="Saved with the run settings for reference; the current solver uses G1/G7 Cd tables and does not apply BC.",
        )
        zero_range = st.number_input(
            f"Zero range ({'m' if units == 'Metric' else 'yd'})",
            min_value=1.0, value=100.0, step=10.0, key=CONFIG_KEYS["zero_range"],
        )
        max_range = st.number_input(
            f"Maximum range ({'m' if units == 'Metric' else 'yd'})",
            min_value=10.0, value=1000.0 if units == "Metric" else 1094.0,
            step=50.0, key=CONFIG_KEYS["max_range"],
        )

    st.caption("BC is included in saved settings, but the available solver does not currently use BC; it calculates drag from the selected G1/G7 table.")
    if drift_module is None or monte_carlo_module is None:
        unavailable = [name for name, module in (("drift.py", drift_module), ("monte_carlo.py", monte_carlo_module)) if module is None]
        st.info("Optional modules not present yet: " + ", ".join(unavailable) + ". Core trajectory calculations remain available.")

    if st.button("Calculate", type="primary", key="calculate"):
        try:
            if not all(math.isfinite(float(value)) for value in (
                velocity, mass, diameter, bc, zero_range, max_range, pressure,
                temperature, wind_speed, wind_direction, latitude,
            )):
                raise ValueError("All numeric inputs must be finite.")
            muzzle_velocity_m_s = _to_si(float(velocity), units, "speed")
            mass_kg = _to_si(float(mass), units, "mass")
            diameter_m = _to_si(float(diameter), units, "diameter")
            zero_range_m = _to_si(float(zero_range), units, "length")
            max_range_m = _to_si(float(max_range), units, "length")
            pressure_pa = _to_si(float(pressure), units, "pressure")
            temperature_c = _to_si(float(temperature), units, "temperature")
            wind_speed_m_s = float(wind_speed) if units == "Metric" else float(wind_speed) * 0.44704
            if mass_kg <= 0 or diameter_m <= 0 or muzzle_velocity_m_s <= 0:
                raise ValueError("Muzzle velocity, mass, and diameter must be positive.")
            if zero_range_m <= 0 or max_range_m <= zero_range_m:
                raise ValueError("Maximum range must be greater than the positive zero range.")
            humidity_fraction = float(humidity) / 100.0
            # Validate the selected atmosphere before starting the integrations.
            air_density(pressure_pa, temperature_c, humidity_fraction)
            sound_speed = speed_of_sound(temperature_c, pressure_pa, humidity_fraction)
            result, zero_angle_rad = _solve_zeroed_trajectory(
                mass_kg=mass_kg, diameter_m=diameter_m, profile=profile,
                muzzle_velocity_m_s=muzzle_velocity_m_s, zero_range_m=zero_range_m,
                max_range_m=max_range_m, pressure_pa=pressure_pa,
                temperature_c=temperature_c, humidity=humidity_fraction,
                wind_speed_m_s=wind_speed_m_s, wind_direction_deg=float(wind_direction),
            )
            result_frame = _make_results(result, zero_range_m, max_range_m)
            range_label = "range_m" if units == "Metric" else "range_yd"
            drop_label = "drop_m" if units == "Metric" else "drop_in"
            drift_label = "drift_m" if units == "Metric" else "drift_in"
            speed_label = "velocity_m_s" if units == "Metric" else "velocity_ft_s"
            output = result_frame.copy()
            if units == "Imperial":
                output[range_label] = output.pop("range_m") / METERS_PER_YARD
                output[drop_label] = output.pop("drop_m") / METERS_PER_INCH
                output[drift_label] = output.pop("drift_m") / METERS_PER_INCH
                output[speed_label] = output.pop("velocity_m_s") / METERS_PER_FOOT
            else:
                output = output.rename(columns={
                    "range_m": range_label, "drop_m": drop_label,
                    "drift_m": drift_label, "velocity_m_s": speed_label,
                })

            zero_drop = float(output[drop_label].iloc[(output[range_label] - zero_range).abs().argmin()])
            peak_height_m = float(result.y_m.max())
            final_drift_m = float(result_frame["drift_m"].iloc[-1])
            st.session_state["last_result_json"] = {
                "units": units, "profile": profile, "muzzle_velocity": velocity,
                "mass": mass, "diameter": diameter, "bc": bc,
                "zero_range": zero_range, "max_range": max_range,
                "pressure": pressure, "temperature": temperature,
                "humidity": humidity, "wind_speed": wind_speed,
                "wind_direction": wind_direction, "latitude": latitude,
                "zero_angle_deg": math.degrees(zero_angle_rad),
            }
            metrics = st.columns(4)
            metrics[0].metric("Zero angle", f"{math.degrees(zero_angle_rad):.3f}°")
            metrics[1].metric("Peak height", f"{_from_si(peak_height_m, units, 'length'):.2f} {'m' if units == 'Metric' else 'yd'}")
            metrics[2].metric("Drift at max range", f"{final_drift_m / (1.0 if units == 'Metric' else METERS_PER_INCH):.2f} {'m' if units == 'Metric' else 'in'}")
            metrics[3].metric("Speed of sound", f"{_from_si(sound_speed, units, 'speed'):.1f} {'m/s' if units == 'Metric' else 'ft/s'}")

            figure, axis = plt.subplots()
            axis.plot(output[range_label], output[drop_label], label="Drop from zero line")
            axis.set_xlabel(f"Range ({'m' if units == 'Metric' else 'yd'})")
            axis.set_ylabel(f"Drop ({'m' if units == 'Metric' else 'in'})")
            axis.set_title("Trajectory drop vs range")
            axis.grid(True, alpha=0.3)
            axis.legend()
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)
            st.dataframe(output, use_container_width=True, hide_index=True)
            st.download_button(
                "Download trajectory CSV", output.to_csv(index=False).encode("utf-8"),
                file_name="trajectory.csv", mime="text/csv",
            )
            if abs(zero_drop) > (0.02 if units == "Metric" else 0.8):
                st.caption("The sampled table may not include a point exactly at the zero range; the solver zeroing uses interpolation.")
        except (ValueError, RuntimeError, requests.RequestException, OSError) as error:
            st.error(f"Calculation failed: {error}")

    with st.sidebar:
        config = {key: st.session_state.get(key) for key in CONFIG_KEYS.values()}
        config["longitude"] = longitude
        config["schema_version"] = 1
        st.download_button(
            "Save settings (JSON)", data=json.dumps(config, indent=2),
            file_name="balistik_settings.json", mime="application/json",
        )
