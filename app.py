"""Streamlit app for Balistik_valk trajectory calculations and bullet selection."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
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
    import drift as drift_module
except ImportError:  # Optional until drift.py is added to the repository.
    drift_module = None

try:
    import monte_carlo as monte_carlo_module
except ImportError:  # Optional until monte_carlo.py is added to the repository.
    monte_carlo_module = None


APP_TITLE = "Balistik_valk Ballistics"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
BULLET_BUNDLE_URL = "https://krzysztofgw-pl.github.io/ballistics-bullets/v1/bullets.json"
BULLET_DATA_PATH = Path(__file__).resolve().parent / "data" / "bullets.json"
METERS_PER_FOOT = 0.3048
METERS_PER_YARD = 0.9144
METERS_PER_INCH = 0.0254
GRAINS_PER_KG = 1.0 / 0.00006479891
PA_PER_INHG = 3386.389

CONFIG_FIELDS = (
    "units", "profile", "muzzle_velocity", "mass", "diameter", "bc",
    "bullet_length", "twist", "zero_range", "max_range", "pressure",
    "temperature", "humidity", "wind_speed", "wind_direction", "latitude",
    "longitude", "bullet_id",
)


def _to_si(value: float, units: str, kind: str) -> float:
    """Convert displayed values to SI units."""
    if kind == "mass":
        return value / 1000.0 if units == "Metric" else value / GRAINS_PER_KG
    if kind == "diameter":
        return value / 1000.0 if units == "Metric" else value * METERS_PER_INCH
    if kind == "length":
        return value if units == "Metric" else value * METERS_PER_YARD
    if kind == "bullet_length":
        return value / 1000.0 if units == "Metric" else value * METERS_PER_INCH
    if kind == "speed":
        return value if units == "Metric" else value * METERS_PER_FOOT
    if kind == "wind_speed":
        return value if units == "Metric" else value * 0.44704
    if kind == "pressure":
        return value * 100.0 if units == "Metric" else value * PA_PER_INHG
    if kind == "temperature":
        return value if units == "Metric" else (value - 32.0) * (5.0 / 9.0)
    raise ValueError(f"Unsupported conversion kind: {kind}")


def _from_si(value: float, units: str, kind: str) -> float:
    """Convert SI values to selected display units."""
    if kind == "length":
        return value if units == "Metric" else value / METERS_PER_YARD
    if kind == "speed":
        return value if units == "Metric" else value / METERS_PER_FOOT
    raise ValueError(f"Unsupported conversion kind: {kind}")


def _bullet_caliber(diameter_in: float) -> str:
    """Format bullet diameter as a conventional caliber search/display term."""
    return f".{diameter_in:.3f}" if diameter_in < 1.0 else f"{diameter_in:g} in"


def _bullet_label(bullet: dict[str, Any]) -> str:
    """Format one bundle record as manufacturer, caliber, weight, and model."""
    return (
        f"{bullet.get('maker', 'Unknown')} — "
        f"{_bullet_caliber(float(bullet['diameter_in']))} — "
        f"{float(bullet['weight_gr']):g} gr ({bullet.get('name', 'Bullet')})"
    )


@st.cache_data(show_spinner=False)
def _load_bullets() -> list[dict[str, Any]]:
    """Load the local bullet bundle or fetch the published format-1 bundle.

    The checked-in data file records the upstream source and license. If its
    bullet list is empty, the published bundle is retrieved and cached by
    Streamlit for the session/runtime.
    """
    local_payload = json.loads(BULLET_DATA_PATH.read_text(encoding="utf-8"))
    bullets = local_payload.get("bullets", [])
    if not bullets:
        response = requests.get(BULLET_BUNDLE_URL, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if payload.get("format") != 1 or not isinstance(payload.get("bullets"), list):
            raise ValueError("Unsupported or invalid bullet database format.")
        bullets = payload["bullets"]
    valid = []
    for bullet in bullets:
        try:
            if bullet.get("id") and bullet.get("maker") and bullet.get("name"):
                if float(bullet["diameter_in"]) > 0 and float(bullet["weight_gr"]) > 0:
                    valid.append(bullet)
        except (KeyError, TypeError, ValueError):
            continue
    return valid


def _bullet_bc(bullet: dict[str, Any]) -> tuple[float | None, str | None]:
    """Return preferred G7 BC, falling back to G1."""
    coefficients = bullet.get("bc") or {}
    if coefficients.get("g7") is not None:
        return float(coefficients["g7"]), "G7"
    if coefficients.get("g1") is not None:
        return float(coefficients["g1"]), "G1"
    return None, None


def _fetch_weather(latitude: float, longitude: float) -> dict[str, float | str]:
    """Fetch current weather and normalize wind speed to metres per second."""
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError("Coordinates are outside valid latitude/longitude ranges.")
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
    return {
        "temperature_c": float(current["temperature_2m"]),
        "humidity_percent": float(current["relative_humidity_2m"]),
        "pressure_hpa": float(current["pressure_msl"]),
        "wind_speed_m_s": float(current["wind_speed_10m"]) / 3.6,
        "wind_direction_deg": float(current["wind_direction_10m"]),
        "timestamp": str(current["time"]),
    }


def _solve_zero_angle(
    *, muzzle_velocity_m_s: float, mass_kg: float, diameter_m: float,
    drag_profile: str, zero_range_m: float, max_range_m: float,
    pressure_pa: float, temperature_c: float, relative_humidity: float,
    wind_speed_m_s: float, wind_direction_deg: float,
) -> tuple[float, Any]:
    """Find a launch elevation that returns to the muzzle line at zero range."""
    duration = min(max(max_range_m / max(muzzle_velocity_m_s, 1.0) * 2.5 + 2.0, 3.0), 120.0)
    solver_parameters = inspect.signature(solve_trajectory).parameters

    def evaluate(angle_rad: float) -> tuple[float, Any]:
        arguments = {
            "mass_kg": mass_kg,
            "diameter_m": diameter_m,
            "drag_profile": drag_profile,
            "launch_speed_m_s": muzzle_velocity_m_s,
            "launch_elevation_rad": angle_rad,
            "pressure_pa": pressure_pa,
            "temperature_c": temperature_c,
            "relative_humidity": relative_humidity,
            "time_end_s": duration,
            "max_step_s": 0.02,
            "wind_speed_m_s": wind_speed_m_s,
            "wind_direction_deg": wind_direction_deg,
        }
        # Keep compatibility with older solver versions that predate wind args.
        arguments = {key: value for key, value in arguments.items() if key in solver_parameters}
        trajectory = solve_trajectory(**arguments)
        if float(trajectory.x_m[-1]) < zero_range_m:
            raise ValueError("Trajectory did not reach zero range; reduce range or increase muzzle velocity.")
        height = float(np.interp(zero_range_m, trajectory.x_m, trajectory.y_m))
        return height, trajectory

    low, high = math.radians(-5.0), math.radians(35.0)
    low_height, _ = evaluate(low)
    high_height, best = evaluate(high)
    if low_height * high_height > 0:
        raise ValueError("Could not find a zeroing angle for these inputs.")
    for _ in range(24):
        middle = (low + high) / 2.0
        middle_height, best = evaluate(middle)
        if abs(middle_height) < 0.002:
            return middle, best
        if low_height * middle_height <= 0:
            high = middle
        else:
            low, low_height = middle, middle_height
    angle = (low + high) / 2.0
    _, best = evaluate(angle)
    return angle, best


def _trajectory_frame(result: Any, zero_range_m: float, max_range_m: float) -> pd.DataFrame:
    """Build drop, drift, and velocity samples through the requested range."""
    mask = result.x_m <= max_range_m
    if not np.any(mask):
        raise ValueError("No trajectory samples were produced within max range.")
    zero_height = float(np.interp(zero_range_m, result.x_m, result.y_m))
    return pd.DataFrame({
        "time_s": result.time_s[mask],
        "range_m": result.x_m[mask],
        "drop_m": zero_height - result.y_m[mask],
        "drift_m": result.z_m[mask],
        "velocity_m_s": np.sqrt(
            result.vx_m_s[mask] ** 2 + result.vy_m_s[mask] ** 2 + result.vz_m_s[mask] ** 2
        ),
    })


def _load_json_settings(uploaded: Any) -> dict[str, Any]:
    """Parse user settings JSON and return only recognized fields."""
    data = json.loads(uploaded.getvalue().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Settings JSON must contain an object.")
    return {key: data[key] for key in CONFIG_FIELDS if key in data}


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
    st.title(APP_TITLE)
    st.caption("Trajectory estimates use the repository's point-mass solver and SI internals.")

    with st.sidebar:
        st.header("Settings")
        settings_file = st.file_uploader("Load settings (JSON)", type=["json"], key="settings_upload")
        if st.button("Apply JSON settings", disabled=settings_file is None, key="apply_settings"):
            try:
                for key, value in _load_json_settings(settings_file).items():
                    st.session_state[key] = value
                st.success("Settings loaded; review the input values.")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                st.error("Could not load JSON settings")

        units = st.selectbox("Units", ("Metric", "Imperial"), key="units")
        profile = st.selectbox("Drag profile", SUPPORTED_PROFILES, key="profile")
        st.header("Location")
        latitude = st.number_input("Latitude (°)", min_value=-90.0, max_value=90.0, value=0.0, step=0.01, key="latitude")
        longitude = st.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=0.0, step=0.01, key="longitude")
        if st.button("Get Current Weather", key="get_weather"):
            try:
                weather = _fetch_weather(float(latitude), float(longitude))
                for key in ("temperature_c", "humidity_percent", "pressure_hpa", "wind_speed_m_s", "wind_direction_deg"):
                    st.session_state[f"weather_{key}"] = weather[key]
                st.session_state["weather_timestamp"] = weather["timestamp"]
                st.session_state.pop("weather_error", None)
            except (requests.RequestException, KeyError, TypeError, ValueError, OverflowError):
                st.session_state["weather_error"] = True
        if st.session_state.get("weather_error"):
            st.error("Could not fetch weather data")
        if st.session_state.get("weather_timestamp") and not st.session_state.get("weather_error"):
            st.caption(f"Updated: {st.session_state['weather_timestamp']} from Open-Meteo")

        st.header("Environment")
        pressure_default = float(st.session_state.get("weather_pressure_hpa", 1013.25))
        temperature_default = float(st.session_state.get("weather_temperature_c", 15.0))
        if units == "Imperial":
            pressure_default = pressure_default * 100.0 / PA_PER_INHG
            temperature_default = temperature_default * 9.0 / 5.0 + 32.0
        pressure = st.number_input(f"Pressure ({'hPa' if units == 'Metric' else 'inHg'})", min_value=0.01, value=pressure_default, key=f"pressure_{units}")
        temperature = st.number_input(f"Temperature ({'°C' if units == 'Metric' else '°F'})", value=temperature_default, key=f"temperature_{units}")
        humidity = st.slider("Humidity (%)", 0, 100, int(st.session_state.get("weather_humidity_percent", 50)), key="humidity")
        weather_wind = float(st.session_state.get("weather_wind_speed_m_s", 0.0))
        if units == "Imperial":
            weather_wind /= 0.44704
        wind_speed = st.number_input(f"Wind speed ({'m/s' if units == 'Metric' else 'mph'})", min_value=0.0, value=weather_wind, key=f"wind_speed_{units}")
        wind_direction = st.number_input("Wind direction FROM (°)", min_value=0.0, max_value=360.0, value=float(st.session_state.get("weather_wind_direction_deg", 0.0)), key="wind_direction")

        st.header("Shot setup")
        try:
            bullets = _load_bullets()
            bullet_load_error = None
        except (OSError, requests.RequestException, json.JSONDecodeError, ValueError) as error:
            bullets = []
            bullet_load_error = str(error)

        query = st.text_input("Search bullets by manufacturer or caliber", key="bullet_search").strip().casefold()
        filtered_bullets = [
            bullet for bullet in bullets
            if not query or query in str(bullet.get("maker", "")).casefold()
            or query in _bullet_caliber(float(bullet["diameter_in"])).casefold()
        ]
        bullet_by_id = {str(bullet["id"]): bullet for bullet in filtered_bullets}
        options = ["__custom__", *bullet_by_id.keys()]
        if st.session_state.get("bullet_id") not in options:
            st.session_state["bullet_id"] = "__custom__"
        selected_id = st.selectbox(
            "Bullet", options, format_func=lambda value: "Custom" if value == "__custom__" else _bullet_label(bullet_by_id[value]), key="bullet_id",
        )
        selected_bullet = bullet_by_id.get(selected_id)
        if selected_bullet is not None and st.session_state.get("last_autofilled_bullet") != selected_id:
            mass_gr = float(selected_bullet["weight_gr"])
            diameter_in = float(selected_bullet["diameter_in"])
            length_in = selected_bullet.get("length_in")
            bc_value, bc_model = _bullet_bc(selected_bullet)
            st.session_state["mass_input"] = mass_gr * 0.00006479891 * 1000.0 if units == "Metric" else mass_gr
            st.session_state["diameter_input"] = diameter_in * 25.4 if units == "Metric" else diameter_in
            if length_in is not None:
                st.session_state["bullet_length_input"] = float(length_in) * 25.4 if units == "Metric" else float(length_in)
            if bc_value is not None:
                st.session_state["bc_input"] = bc_value
            st.session_state["last_autofilled_bullet"] = selected_id
        elif selected_id == "__custom__":
            st.session_state["last_autofilled_bullet"] = "__custom__"

        if selected_bullet is not None:
            chosen_bc, chosen_model = _bullet_bc(selected_bullet)
            length_text = f", length {float(selected_bullet['length_in']):.3f} in" if selected_bullet.get("length_in") is not None else ""
            bc_text = f", {chosen_model} BC {chosen_bc:g}" if chosen_bc is not None else ", no published BC"
            st.caption(f"{selected_bullet.get('maker')} {selected_bullet.get('name')} · {_bullet_caliber(float(selected_bullet['diameter_in']))} · {float(selected_bullet['weight_gr']):g} gr{length_text}{bc_text}")
        elif bullet_load_error:
            st.caption(f"Bullet database could not be loaded: {bullet_load_error}")
        if not bullets:
            st.warning("Bullet list unavailable; Custom entry remains available.")

        mass = st.number_input(f"Mass ({'g' if units == 'Metric' else 'gr'})", min_value=0.001, value=10.0 if units == "Metric" else 168.0, key="mass_input")
        diameter = st.number_input(f"Diameter ({'mm' if units == 'Metric' else 'in'})", min_value=0.001, value=7.82 if units == "Metric" else 0.308, key="diameter_input")
        bc = st.number_input("BC (reference; solver uses G1/G7 drag tables)", min_value=0.001, value=0.462, step=0.001, key="bc_input")
        bullet_length = st.number_input(f"Bullet length ({'mm' if units == 'Metric' else 'in'})", min_value=0.001, value=31.5 if units == "Metric" else 1.24, key="bullet_length_input")
        twist = st.number_input(f"Twist ({'mm/turn' if units == 'Metric' else 'in/turn'})", min_value=0.001, value=254.0 if units == "Metric" else 10.0, key="twist_input")
        muzzle_velocity = st.number_input(f"Muzzle velocity ({'m/s' if units == 'Metric' else 'ft/s'})", min_value=1.0, value=800.0 if units == "Metric" else 2625.0, key="muzzle_velocity")
        zero_range = st.number_input(f"Zero range ({'m' if units == 'Metric' else 'yd'})", min_value=1.0, value=100.0, key="zero_range")
        max_range = st.number_input(f"Maximum range ({'m' if units == 'Metric' else 'yd'})", min_value=10.0, value=1000.0 if units == "Metric" else 1094.0, key="max_range")

    st.caption("Bullet database: KGW Labs Ballistics, CC BY-SA 4.0. BC is shown for reference; the current solver uses the selected drag table.")
    if drift_module is None or monte_carlo_module is None:
        missing = [name for name, module in (("drift.py", drift_module), ("monte_carlo.py", monte_carlo_module)) if module is None]
        st.info("Optional modules not available: " + ", ".join(missing))

    if st.button("Calculate", type="primary", key="calculate"):
        try:
            mass_kg = _to_si(float(mass), units, "mass")
            diameter_m = _to_si(float(diameter), units, "diameter")
            bullet_length_m = _to_si(float(bullet_length), units, "bullet_length")
            twist_m = _to_si(float(twist), units, "bullet_length")
            velocity_m_s = _to_si(float(muzzle_velocity), units, "speed")
            zero_range_m = _to_si(float(zero_range), units, "length")
            max_range_m = _to_si(float(max_range), units, "length")
            pressure_pa = _to_si(float(pressure), units, "pressure")
            temperature_c = _to_si(float(temperature), units, "temperature")
            wind_m_s = _to_si(float(wind_speed), units, "wind_speed")
            humidity_fraction = float(humidity) / 100.0
            if mass_kg <= 0 or diameter_m <= 0 or bullet_length_m <= 0 or twist_m <= 0 or velocity_m_s <= 0:
                raise ValueError("Projectile dimensions, mass, twist, and muzzle velocity must be positive.")
            if zero_range_m <= 0 or max_range_m <= zero_range_m:
                raise ValueError("Maximum range must exceed the positive zero range.")
            air_density(pressure_pa, temperature_c, humidity_fraction)
            sound_speed = speed_of_sound(temperature_c, pressure_pa, humidity_fraction)
            angle, result = _solve_zero_angle(
                muzzle_velocity_m_s=velocity_m_s, mass_kg=mass_kg, diameter_m=diameter_m,
                drag_profile=profile, zero_range_m=zero_range_m, max_range_m=max_range_m,
                pressure_pa=pressure_pa, temperature_c=temperature_c,
                relative_humidity=humidity_fraction, wind_speed_m_s=wind_m_s,
                wind_direction_deg=float(wind_direction),
            )
            table = _trajectory_frame(result, zero_range_m, max_range_m)
            display = table.copy()
            if units == "Imperial":
                display["range_yd"] = display.pop("range_m") / METERS_PER_YARD
                display["drop_in"] = display.pop("drop_m") / METERS_PER_INCH
                display["drift_in"] = display.pop("drift_m") / METERS_PER_INCH
                display["velocity_ft_s"] = display.pop("velocity_m_s") / METERS_PER_FOOT
            else:
                display = display.rename(columns={"range_m": "range_m", "drop_m": "drop_m", "drift_m": "drift_m", "velocity_m_s": "velocity_m_s"})

            st.session_state["last_stability"] = stability_factor(
                mass_kg, diameter_m, bullet_length_m, twist_m, velocity_m_s,
                temperature_c, pressure_pa,
            )
            cols = st.columns(4)
            cols[0].metric("Launch elevation", f"{math.degrees(angle):.3f}°")
            cols[1].metric("Peak height", f"{_from_si(float(result.y_m.max()), units, 'length'):.2f} {'m' if units == 'Metric' else 'yd'}")
            drift_m = float(table["drift_m"].iloc[-1])
            cols[2].metric("Lateral drift", f"{drift_m if units == 'Metric' else drift_m / METERS_PER_INCH:.2f} {'m' if units == 'Metric' else 'in'}")
            cols[3].metric("Stability estimate Sg", f"{st.session_state['last_stability']:.2f}")

            figure, axis = plt.subplots()
            range_column = "range_m" if units == "Metric" else "range_yd"
            drop_column = "drop_m" if units == "Metric" else "drop_in"
            axis.plot(display[range_column], display[drop_column], label="Drop from zero line")
            axis.set_xlabel(f"Range ({'m' if units == 'Metric' else 'yd'})")
            axis.set_ylabel(f"Drop ({'m' if units == 'Metric' else 'in'})")
            axis.set_title("Trajectory drop vs range")
            axis.grid(True, alpha=0.3)
            axis.legend()
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.download_button("Download trajectory CSV", display.to_csv(index=False).encode("utf-8"), "trajectory.csv", "text/csv")
            settings = {key: st.session_state.get(key) for key in CONFIG_FIELDS}
            st.download_button("Save settings (JSON)", json.dumps(settings, indent=2), "balistik_settings.json", "application/json")
        except (ValueError, RuntimeError, requests.RequestException, OSError) as error:
            st.error(f"Calculation failed: {error}")
