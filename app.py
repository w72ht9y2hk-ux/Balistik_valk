"""Streamlit app for Balistik_valk ballistic trajectory calculations.

This app:
- uses the repo's existing atmosphere.py, drag.py, solver.py, and stability.py
- accepts Metric or Imperial inputs
- fetches current weather from Open-Meteo
- computes a trajectory with the current solver API
- displays a drop-vs-range plot, trajectory table, CSV export, and JSON config I/O

The current repository does not yet include drift.py or monte_carlo.py, so they are
imported defensively and treated as optional.
"""

from __future__ import annotations

import json
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from atmosphere import air_density, speed_of_sound
from drag import SUPPORTED_PROFILES
from solver import solve_trajectory
from stability import stability_factor

try:
    import drift as drift_module  # optional module; may not exist in this repo
except ImportError:  # pragma: no cover - optional dependency
    drift_module = None

try:
    import monte_carlo as monte_carlo_module  # optional module; may not exist
except ImportError:  # pragma: no cover - optional dependency
    monte_carlo_module = None


APP_TITLE = "Balistik_valk Ballistics"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Conversion constants
METERS_PER_FOOT = 0.3048
METERS_PER_YARD = 0.9144
METERS_PER_INCH = 0.0254
GRAINS_PER_KG = 1.0 / 0.00006479891
PA_PER_INHG = 3386.389

# Saved settings keys
CONFIG_KEYS = {
    "units": "units",
    "profile": "profile",
    "mass": "mass",
    "diameter": "diameter",
    "bc": "bc",
    "zero_range": "zero_range",
    "max_range": "max_range",
    "muzzle_velocity": "muzzle_velocity",
    "pressure": "pressure",
    "temperature": "temperature",
    "humidity": "humidity",
    "wind_speed": "wind_speed",
    "wind_direction": "wind_direction",
    "latitude": "latitude",
    "longitude": "longitude",
}


def _to_si(value: float, units: str, kind: str) -> float:
    """Convert a UI value to SI units."""
    if kind == "mass":
        if units == "Metric":
            return value / 1000.0
        return value / GRAINS_PER_KG
    if kind == "diameter":
        if units == "Metric":
            return value / 1000.0
        return value * METERS_PER_INCH
    if kind == "length":
        if units == "Metric":
            return value
        return value * METERS_PER_YARD
    if kind == "speed":
        if units == "Metric":
            return value
        return value * METERS_PER_FOOT
    if kind == "pressure":
        if units == "Metric":
            return value * 100.0
        return value * PA_PER_INHG
    if kind == "temperature":
        if units == "Metric":
            return value
        return (value - 32.0) * (5.0 / 9.0)
    if kind == "wind_speed":
        if units == "Metric":
            return value
        return value * 0.44704  # mph -> m/s
    raise ValueError(f"Unsupported conversion kind: {kind}")


def _from_si(value: float, units: str, kind: str) -> float:
    """Convert SI values back to display units for labels."""
    if kind == "length":
        if units == "Metric":
            return value
        return value / METERS_PER_YARD
    if kind == "speed":
        if units == "Metric":
            return value
        return value / METERS_PER_FOOT
    if kind == "pressure":
        if units == "Metric":
            return value / 100.0
        return value / PA_PER_INHG
    if kind == "temperature":
        if units == "Metric":
            return value
        return value * (9.0 / 5.0) + 32.0
    if kind == "wind_speed":
        if units == "Metric":
            return value
        return value / 0.44704
    raise ValueError(f"Unsupported conversion kind: {kind}")


def _fetch_weather(latitude: float, longitude: float) -> dict[str, float | str]:
    """Fetch the current weather from Open-Meteo and normalize values."""
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("Longitude must be between -180 and 180.")

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
    payload = response.json()
    current = payload.get("current")
    if not isinstance(current, dict):
        raise ValueError("No current weather data returned.")

    return {
        "temperature_c": float(current["temperature_2m"]),
        "humidity_percent": float(current["relative_humidity_2m"]),
        "pressure_hpa": float(current["pressure_msl"]),
        "wind_speed_m_s": float(current["wind_speed_10m"]) / 3.6,
        "wind_direction_deg": float(current["wind_direction_10m"]),
        "timestamp": str(current["time"]),
    }


def _solve_zero_angle(
    *,
    muzzle_velocity_m_s: float,
    mass_kg: float,
    diameter_m: float,
    drag_profile: str,
    zero_range_m: float,
    max_range_m: float,
    pressure_pa: float,
    temperature_c: float,
    relative_humidity: float,
    wind_speed_m_s: float,
    wind_direction_deg: float,
) -> float:
    """Find a launch angle that reaches the desired zero range."""

    def eval_angle(angle_deg: float) -> float:
        effective_duration = max(3.0, (max_range_m / max(muzzle_velocity_m_s, 1.0)) * 1.5 + 3.0)
        result = solve_trajectory(
            mass_kg=mass_kg,
            diameter_m=diameter_m,
            drag_profile=drag_profile,
            launch_speed_m_s=muzzle_velocity_m_s,
            launch_elevation_rad=math.radians(angle_deg),
            pressure_pa=pressure_pa,
            temperature_c=temperature_c,
            relative_humidity=relative_humidity,
            time_end_s=effective_duration,
            max_step_s=0.02,
            wind_speed_m_s=wind_speed_m_s,
            wind_direction_deg=wind_direction_deg,
        )
        index = np.searchsorted(result.x_m, zero_range_m, side="left")
        if index >= len(result.x_m):
            return -1e6
        if index == 0:
            return float(result.y_m[0])
        left_x = result.x_m[index - 1]
        left_y = result.y_m[index - 1]
        right_x = result.x_m[index]
        right_y = result.y_m[index]
        if right_x == left_x:
            return float(right_y)
        alpha = (zero_range_m - left_x) / (right_x - left_x)
        return float(left_y + alpha * (right_y - left_y))

    low, high = 0.1, 45.0
    low_val = eval_angle(low)
    high_val = eval_angle(high)

    for _ in range(25):
        mid = (low + high) / 2.0
        mid_val = eval_angle(mid)
        if low_val == 0.0 or abs(mid_val) < 1e-6:
            return mid
        if low_val * mid_val <= 0.0:
            high = mid
            high_val = mid_val
        else:
            low = mid
            low_val = mid_val

    return (low + high) / 2.0


def _build_table(result) -> pd.DataFrame:
    """Create a compact trajectory DataFrame for export and display."""
    data = pd.DataFrame(
        {
            "time_s": result.time_s,
            "range_m": result.x_m,
            "height_m": result.y_m,
            "drift_m": result.z_m,
            "vx_m_s": result.vx_m_s,
            "vy_m_s": result.vy_m_s,
            "vz_m_s": result.vz_m_s,
        }
    )
    return data


def _save_json_settings(settings: dict[str, Any]) -> str:
    return json.dumps(settings, indent=2)


def _load_settings_from_json(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object.")
    return payload


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
    st.title(APP_TITLE)
    st.caption("Ballistics calculator built on the repo's existing solver and physics modules.")

    with st.sidebar:
        st.header("Configuration")
        uploaded_json = st.file_uploader("Load settings (JSON)", type=["json"])

        if uploaded_json is not None:
            if st.button("Apply JSON settings"):
                try:
                    data = _load_settings_from_json(uploaded_json.read().decode("utf-8"))
                    for key, value in data.items():
                        st.session_state[key] = value
                    st.success("Settings loaded.")
                except Exception as exc:  # pragma: no cover
                    st.error(f"Could not load settings: {exc}")

        units = st.selectbox("Units", ("Metric", "Imperial"), key="units")

        st.header("Location")
        latitude = st.number_input(
            "Latitude (°)",
            min_value=-90.0,
            max_value=90.0,
            value=st.session_state.get("latitude", 0.0),
            step=0.01,
            key="latitude",
        )
        longitude = st.number_input(
            "Longitude (°)",
            min_value=-180.0,
            max_value=180.0,
            value=st.session_state.get("longitude", 0.0),
            step=0.01,
            key="longitude",
        )

        if st.button("Get Current Weather"):
            try:
                weather = _fetch_weather(float(latitude), float(longitude))
                st.session_state["temperature"] = float(weather["temperature_c"])
                st.session_state["pressure"] = float(weather["pressure_hpa"])
                st.session_state["humidity"] = float(weather["humidity_percent"])
                st.session_state["wind_speed"] = float(weather["wind_speed_m_s"])
                st.session_state["wind_direction"] = float(weather["wind_direction_deg"])
                st.session_state["weather_timestamp"] = str(weather["timestamp"])
                st.session_state.pop("weather_error", None)
            except Exception:
                st.session_state["weather_error"] = True

        if st.session_state.get("weather_error"):
            st.error("Could not fetch weather data")
        if st.session_state.get("weather_timestamp"):
            st.caption(f"Updated: {st.session_state['weather_timestamp']} from Open-Meteo")

        st.header("Environment")
        if units == "Metric":
            default_pressure = st.session_state.get("pressure", 1013.25)
            default_temperature = st.session_state.get("temperature", 15.0)
            default_wind_speed = st.session_state.get("wind_speed", 0.0)
            default_wind_direction = st.session_state.get("wind_direction", 0.0)
            default_humidity = st.session_state.get("humidity", 50)
        else:
            default_pressure = st.session_state.get("pressure", 29.92)
            default_temperature = st.session_state.get("temperature", 59.0)
            default_wind_speed = st.session_state.get("wind_speed", 0.0)
            default_wind_direction = st.session_state.get("wind_direction", 0.0)
            default_humidity = st.session_state.get("humidity", 50)

        pressure = st.number_input(
            "Pressure",
            min_value=0.01,
            value=float(default_pressure),
            step=1.0 if units == "Metric" else 0.1,
            key="pressure_input",
        )
        temperature = st.number_input(
            "Temperature",
            value=float(default_temperature),
            step=1.0 if units == "Metric" else 0.5,
            key="temperature_input",
        )
        humidity = st.slider("Humidity (%)", 0, 100, int(default_humidity), key="humidity_input")
        wind_speed = st.number_input(
            "Wind speed",
            min_value=0.0,
            value=float(default_wind_speed),
            step=0.1,
            key="wind_speed_input",
        )
        wind_direction = st.number_input(
            "Wind direction FROM (°)",
            min_value=0.0,
            max_value=360.0,
            value=float(default_wind_direction),
            step=1.0,
            key="wind_direction_input",
        )

        st.header("Shot setup")
        mass = st.number_input(
            "Mass",
            min_value=0.001,
            value=11.34 if units == "Metric" else 168.0,
            step=0.1,
            key="mass_input",
        )
        diameter = st.number_input(
            "Diameter",
            min_value=0.001,
            value=7.82 if units == "Metric" else 0.308,
            step=0.01,
            key="diameter_input",
        )
        bc = st.number_input(
            "BC",
            min_value=0.001,
            value=0.462,
            step=0.001,
            key="bc_input",
        )
        zero_range = st.number_input(
            "Zero range",
            min_value=0.1,
            value=100.0 if units == "Metric" else 109.4,
            step=10.0 if units == "Metric" else 10.0,
            key="zero_range_input",
        )
        max_range = st.number_input(
            "Max range",
            min_value=10.0,
            value=1000.0 if units == "Metric" else 1093.6,
            step=50.0 if units == "Metric" else 50.0,
            key="max_range_input",
        )
        muzzle_velocity = st.number_input(
            "Muzzle velocity",
            min_value=1.0,
            value=800.0 if units == "Metric" else 2625.0,
            step=10.0 if units == "Metric" else 10.0,
            key="muzzle_velocity_input",
        )
        profile = st.selectbox("Drag table", options=SUPPORTED_PROFILES, index=1, key="profile_input")

    if drift_module is None:
        st.info("Optional module drift.py is not present in this repo; the core solver still runs.")
    if monte_carlo_module is None:
        st.info("Optional module monte_carlo.py is not present in this repo; core trajectory analysis still works.")

    if st.button("Calculate", type="primary"):
        try:
            # Convert user display values into SI
            mass_kg = _to_si(float(mass), units, "mass")
            diameter_m = _to_si(float(diameter), units, "diameter")
            zero_range_m = _to_si(float(zero_range), units, "length")
            max_range_m = _to_si(float(max_range), units, "length")
            muzzle_velocity_m_s = _to_si(float(muzzle_velocity), units, "speed")
            pressure_pa = _to_si(float(pressure), units, "pressure")
            temperature_c = _to_si(float(temperature), units, "temperature")
            relative_humidity = float(humidity) / 100.0
            wind_speed_m_s = _to_si(float(wind_speed), units, "wind_speed")
            wind_direction_deg = float(wind_direction)

            if mass_kg <= 0.0:
                raise ValueError("Mass must be positive.")
            if diameter_m <= 0.0:
                raise ValueError("Diameter must be positive.")
            if muzzle_velocity_m_s <= 0.0:
                raise ValueError("Muzzle velocity must be positive.")
            if zero_range_m <= 0.0 or max_range_m <= zero_range_m:
                raise ValueError("Zero range must be positive and less than max range.")
            if not 0.0 <= relative_humidity <= 1.0:
                raise ValueError("Humidity must be between 0 and 100%.")
            air_density(pressure_pa, temperature_c, relative_humidity)
            speed_of_sound(temperature_c, pressure_pa, relative_humidity)

            # Compute a launch elevation that reaches the selected zero range.
            zero_angle_deg = _solve_zero_angle(
                muzzle_velocity_m_s=muzzle_velocity_m_s,
                mass_kg=mass_kg,
                diameter_m=diameter_m,
                drag_profile=profile,
                zero_range_m=zero_range_m,
                max_range_m=max_range_m,
                pressure_pa=pressure_pa,
                temperature_c=temperature_c,
                relative_humidity=relative_humidity,
                wind_speed_m_s=wind_speed_m_s,
                wind_direction_deg=wind_direction_deg,
            )

            # Use the solved zero angle for the final trajectory.
            final_duration = max(5.0, max_range_m / max(muzzle_velocity_m_s, 1.0) * 1.5 + 5.0)
            final_result = solve_trajectory(
                mass_kg=mass_kg,
                diameter_m=diameter_m,
                drag_profile=profile,
                launch_speed_m_s=muzzle_velocity_m_s,
                launch_elevation_rad=math.radians(zero_angle_deg),
                pressure_pa=pressure_pa,
                temperature_c=temperature_c,
                relative_humidity=relative_humidity,
                time_end_s=final_duration,
                max_step_s=0.02,
                wind_speed_m_s=wind_speed_m_s,
                wind_direction_deg=wind_direction_deg,
            )

            table = _build_table(final_result)
            table = table[table["range_m"] <= max_range_m].copy()

            # Display output metrics.
            peak_height = float(np.max(table["height_m"])) if len(table) else 0.0
            final_range = float(table["range_m"].iloc[-1]) if len(table) else 0.0
            final_drift = float(table["drift_m"].iloc[-1]) if len(table) else 0.0

            metrics = st.columns(4)
            metrics[0].metric("Zero angle", f"{zero_angle_deg:.2f}°")
            metrics[1].metric("Peak height", f"{peak_height:.2f} m")
            metrics[2].metric("Final range", f"{final_range:.2f} m")
            metrics[3].metric("Final drift", f"{final_drift:.2f} m")

            # Plot drop vs range.
            fig, ax = plt.subplots()
            ax.plot(table["range_m"], table["height_m"], label="Height")
            ax.set_xlabel("Range (m)")
            ax.set_ylabel("Height (m)")
            ax.set_title("Drop vs range")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            # Show a data table.
            st.subheader("Trajectory data")
            st.dataframe(table, use_container_width=True, hide_index=True)

            # CSV export.
            st.download_button(
                "Download trajectory CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="trajectory.csv",
                mime="text/csv",
            )

            # Save current config as JSON.
            settings = {
                "units": units,
                "profile": profile,
                "mass": mass,
                "diameter": diameter,
                "bc": bc,
                "zero_range": zero_range,
                "max_range": max_range,
                "muzzle_velocity": muzzle_velocity,
                "pressure": pressure,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "wind_direction": wind_direction,
                "latitude": latitude,
                "longitude": longitude,
            }
            st.download_button(
                "Save settings (JSON)",
                data=_save_json_settings(settings),
                file_name="balistik_settings.json",
                mime="application/json",
            )

        except Exception as exc:  # pragma: no cover - broad safety net
            st.error(f"Calculation failed: {exc}")

    # Optional drift/Monte Carlo notices.
    if drift_module is not None:
        st.sidebar.caption("drift.py detected: drift logic is available.")
    if monte_carlo_module is not None:
        st.sidebar.caption("monte_carlo.py detected: Monte Carlo support is available.")


if __name__ == "__main__":
    main()
