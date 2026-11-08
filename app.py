"""Streamlit UI for the SI-based Balistik_valk trajectory solver."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import requests

APP_TITLE = "Balistik_valk"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
BULLET_DATA_PATH = Path(__file__).resolve().parent / "data" / "bullets.json"
GRAINS_PER_KG = 1.0 / 0.00006479891
METERS_PER_FOOT = 0.3048
METERS_PER_YARD = 0.9144
METERS_PER_INCH = 0.0254
PA_PER_INHG = 3386.389


def _to_si(value: float, units: str, kind: str) -> float:
    """Convert a displayed quantity to SI."""
    if kind == "mass":
        return value / 1000.0 if units == "Metric" else value / GRAINS_PER_KG
    if kind == "diameter":
        return value / 1000.0 if units == "Metric" else value * METERS_PER_INCH
    if kind == "bullet_length":
        return value / 1000.0 if units == "Metric" else value * METERS_PER_INCH
    if kind == "twist":
        return value / 1000.0 if units == "Metric" else value * METERS_PER_INCH
    if kind == "speed":
        return value if units == "Metric" else value * METERS_PER_FOOT
    if kind == "range":
        return value if units == "Metric" else value * METERS_PER_YARD
    if kind == "pressure":
        return value * 100.0 if units == "Metric" else value * PA_PER_INHG
    if kind == "temperature":
        return value if units == "Metric" else (value - 32.0) * 5.0 / 9.0
    if kind == "wind":
        return value if units == "Metric" else value * 0.44704
    raise ValueError(f"Unsupported conversion kind: {kind}")


def _from_si(value: float, units: str, kind: str) -> float:
    """Convert SI values to display units."""
    if kind == "distance":
        return value if units == "Metric" else value / METERS_PER_FOOT
    if kind == "range":
        return value if units == "Metric" else value / METERS_PER_YARD
    if kind == "speed":
        return value if units == "Metric" else value / METERS_PER_FOOT
    if kind == "drop":
        return value if units == "Metric" else value / METERS_PER_INCH
    if kind == "pressure":
        return value / 100.0 if units == "Metric" else value / PA_PER_INHG
    if kind == "temperature":
        return value if units == "Metric" else value * 9.0 / 5.0 + 32.0
    raise ValueError(f"Unsupported conversion kind: {kind}")


def _load_bullets() -> list[dict[str, Any]]:
    """Load and validate bullet records from the local data bundle."""
    payload = json.loads(BULLET_DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("bullets"), list):
        raise ValueError("Invalid bullet database format.")
    bullets = []
    for bullet in payload["bullets"]:
        try:
            if bullet.get("id") and bullet.get("maker") and bullet.get("name"):
                if float(bullet["diameter_in"]) > 0 and float(bullet["weight_gr"]) > 0:
                    bullets.append(bullet)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return bullets


def _bullet_bc(bullet: dict[str, Any]) -> tuple[float | None, str | None]:
    """Prefer the bullet's G7 BC, falling back to G1."""
    bc = bullet.get("bc") or {}
    if bc.get("g7") is not None:
        return float(bc["g7"]), "g7"
    if bc.get("g1") is not None:
        return float(bc["g1"]), "g1"
    return None, None


def _bullet_label(bullet: dict[str, Any]) -> str:
    """Create the requested manufacturer/caliber/weight/model label."""
    diameter = float(bullet["diameter_in"])
    return f"{bullet['maker']} — .{diameter:.3f} — {float(bullet['weight_gr']):g} gr ({bullet['name']})"


def _fetch_weather(latitude: float, longitude: float) -> dict[str, Any]:
    """Fetch Open-Meteo current conditions for the selected coordinates."""
    if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m",
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
        # Open-Meteo's default wind-speed unit is km/h.
        "wind_speed_m_s": float(current["wind_speed_10m"]) / 3.6,
        "wind_direction_deg": float(current["wind_direction_10m"]),
        "timestamp": str(current["time"]),
    }


def _solve(
    *, mass_kg: float, diameter_m: float, profile: str, velocity_m_s: float,
    elevation_deg: float, pressure_pa: float, temperature_c: float,
    humidity: float, wind_m_s: float, wind_direction_deg: float,
    duration_s: float, ballistic_coefficient: float | None,
) -> Any:
    """Call the solver with the current SI API and optional bullet BC."""
    from solver import solve_trajectory

    return solve_trajectory(
        mass_kg=mass_kg,
        diameter_m=diameter_m,
        drag_profile=profile,
        launch_speed_m_s=velocity_m_s,
        launch_elevation_rad=math.radians(elevation_deg),
        pressure_pa=pressure_pa,
        temperature_c=temperature_c,
        relative_humidity=humidity,
        time_end_s=duration_s,
        max_step_s=0.02,
        wind_speed_m_s=wind_m_s,
        wind_direction_deg=wind_direction_deg,
        ballistic_coefficient=ballistic_coefficient,
    )


def main() -> None:
    """Render the Streamlit ballistics application."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import streamlit as st

    from atmosphere import air_density, speed_of_sound
    from drag import SUPPORTED_PROFILES
    from stability import stability_factor

    st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
    st.title(APP_TITLE)
    units = st.sidebar.selectbox("Measurement system", ("Metric", "Imperial"), key="units")
    pressure_unit = "hPa" if units == "Metric" else "inHg"
    temperature_unit = "°C" if units == "Metric" else "°F"
    wind_unit = "m/s" if units == "Metric" else "mph"
    range_unit = "m" if units == "Metric" else "yd"
    mass_unit = "g" if units == "Metric" else "gr"
    diameter_unit = "mm" if units == "Metric" else "in"
    velocity_unit = "m/s" if units == "Metric" else "ft/s"

    with st.sidebar:
        st.header("Location")
        latitude = st.number_input("Latitude (°)", min_value=-90.0, max_value=90.0, value=0.0, step=0.01, key="latitude")
        longitude = st.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=0.0, step=0.01, key="longitude")

        st.header("Environment")
        if st.button("Get Current Weather", key="get_current_weather"):
            try:
                weather = _fetch_weather(float(latitude), float(longitude))
                temperature_ui = _from_si(weather["temperature_c"], units, "temperature")
                pressure_ui = _from_si(weather["pressure_hpa"] * 100.0, units, "pressure")
                wind_ui = weather["wind_speed_m_s"] if units == "Metric" else weather["wind_speed_m_s"] / 0.44704
                # Update the widgets' keyed values before their creation below.
                st.session_state[f"pressure_{units}"] = pressure_ui
                st.session_state[f"temperature_{units}"] = temperature_ui
                st.session_state["humidity_percent"] = int(round(weather["humidity_percent"]))
                st.session_state[f"wind_speed_{units}"] = wind_ui
                st.session_state["wind_direction_deg"] = weather["wind_direction_deg"]
                st.session_state["weather_timestamp"] = weather["timestamp"]
                st.session_state.pop("weather_error", None)
            except (requests.RequestException, KeyError, TypeError, ValueError, OverflowError) as error:
                st.session_state["weather_error"] = str(error)
                st.session_state.pop("weather_timestamp", None)

        if st.session_state.get("weather_error"):
            st.error(f"Could not fetch current weather: {st.session_state['weather_error']}")
        if st.session_state.get("weather_timestamp"):
            st.caption(f"Weather fetched: {st.session_state['weather_timestamp']} (Open-Meteo)")

        pressure_default = 1013.25 if units == "Metric" else 29.92
        temperature_default = 15.0 if units == "Metric" else 59.0
        pressure = st.number_input(f"Pressure ({pressure_unit})", min_value=0.01, value=pressure_default, key=f"pressure_{units}")
        temperature = st.number_input(f"Temperature ({temperature_unit})", value=temperature_default, key=f"temperature_{units}")
        humidity = st.slider("Relative humidity (%)", 0, 100, 50, key="humidity_percent")
        wind_speed = st.number_input(f"Wind speed ({wind_unit})", min_value=0.0, value=0.0, step=0.1, key=f"wind_speed_{units}")
        wind_direction = st.number_input("Wind direction FROM (°)", min_value=0.0, max_value=360.0, value=0.0, step=1.0, key="wind_direction_deg")

        st.header("Projectile")
        try:
            bullets = _load_bullets()
        except (OSError, json.JSONDecodeError, ValueError):
            bullets = []
        bullet_search = st.text_input("Search bullets by manufacturer or caliber", key="bullet_search").strip().casefold()
        visible_bullets = [
            bullet for bullet in bullets
            if not bullet_search
            or bullet_search in str(bullet.get("maker", "")).casefold()
            or bullet_search in f".{float(bullet['diameter_in']):.3f}".casefold()
        ]
        bullet_map = {str(bullet["id"]): bullet for bullet in visible_bullets}
        bullet_options = ["__custom__", *bullet_map.keys()]
        if st.session_state.get("bullet_id") not in bullet_options:
            st.session_state["bullet_id"] = "__custom__"
        bullet_id = st.selectbox(
            "Bullet",
            bullet_options,
            format_func=lambda key: "Custom" if key == "__custom__" else _bullet_label(bullet_map[key]),
            key="bullet_id",
        )
        bullet = bullet_map.get(bullet_id)
        bc_value = None
        bc_profile = None
        if bullet:
            bc_value, bc_profile = _bullet_bc(bullet)
            if st.session_state.get("autofilled_bullet") != bullet_id:
                weight_gr = float(bullet["weight_gr"])
                diameter_in = float(bullet["diameter_in"])
                length_in = bullet.get("length_in")
                st.session_state["mass_value"] = weight_gr * 0.00006479891 * 1000.0 if units == "Metric" else weight_gr
                st.session_state["diameter_value"] = diameter_in * 25.4 if units == "Metric" else diameter_in
                if length_in is not None:
                    st.session_state["bullet_length_value"] = float(length_in) * 25.4 if units == "Metric" else float(length_in)
                if bc_value is not None:
                    st.session_state["bc_value"] = bc_value
                st.session_state["autofilled_bullet"] = bullet_id
            st.caption(f"{bullet['maker']} · {_bullet_label(bullet)} · {bc_profile.upper() + ' BC ' + format(bc_value, '.3f') if bc_value is not None else 'No published BC'}")
        else:
            st.session_state["autofilled_bullet"] = "__custom__"

        mass = st.number_input(f"Mass ({mass_unit})", min_value=0.001, value=11.34 if units == "Metric" else 175.0, key="mass_value")
        diameter = st.number_input(f"Diameter ({diameter_unit})", min_value=0.001, value=7.82 if units == "Metric" else 0.308, key="diameter_value")
        bullet_length = st.number_input(f"Projectile length ({diameter_unit})", min_value=0.001, value=31.5 if units == "Metric" else 1.24, key="bullet_length_value")
        twist = st.number_input(f"Twist length ({diameter_unit}/turn)", min_value=0.001, value=254.0 if units == "Metric" else 10.0, key="twist_value")
        velocity = st.number_input(f"Muzzle velocity ({velocity_unit})", min_value=0.001, value=800.0 if units == "Metric" else 2625.0, key="velocity_value")
        bc = st.number_input("Ballistic coefficient", min_value=0.001, value=0.462, step=0.001, key="bc_value", help="For a selected database bullet, G7 is preferred; G1 is used when G7 is unavailable.")
        profile_choice = st.selectbox("Reference drag profile", SUPPORTED_PROFILES, index=1, key="drag_profile")
        profile = bc_profile if bullet and bc_profile else profile_choice

        st.header("Zeroing")
        zero_range = st.number_input(
            f"Zero Range ({range_unit})",
            min_value=25.0,
            max_value=1000.0,
            value=91.0 if units == "Metric" else 100.0,
            step=1.0,
            key=f"zero_range_{units}",
        )
        max_range = st.number_input(
            f"Max Range ({range_unit})",
            min_value=100.0,
            max_value=3000.0,
            value=914.0 if units == "Metric" else 1000.0,
            step=10.0,
            key=f"max_range_{units}",
        )
        launch_angle = st.slider("Launch elevation (deg)", 0.1, 80.0, 15.0, step=0.5)
        duration = st.slider("Integration duration (s)", 1.0, 60.0, 10.0, step=0.5)
        debug = st.checkbox("Show detailed errors and SI values", value=False)

    if st.button("Calculate", type="primary", key="calculate"):
        try:
            mass_kg = _to_si(float(mass), units, "mass")
            diameter_m = _to_si(float(diameter), units, "diameter")
            velocity_m_s = _to_si(float(velocity), units, "speed")
            pressure_pa = _to_si(float(pressure), units, "pressure")
            temperature_c = _to_si(float(temperature), units, "temperature")
            wind_m_s = _to_si(float(wind_speed), units, "wind")
            humidity_fraction = float(humidity) / 100.0
            zero_range_m = _to_si(float(zero_range), units, "range")
            max_range_m = _to_si(float(max_range), units, "range")
            if max_range_m <= zero_range_m:
                raise ValueError("Max Range must be greater than Zero Range.")
            air_density(pressure_pa, temperature_c, humidity_fraction)
            sound_speed = speed_of_sound(temperature_c, pressure_pa, humidity_fraction)
            result = _solve(
                mass_kg=mass_kg,
                diameter_m=diameter_m,
                profile=profile,
                velocity_m_s=velocity_m_s,
                elevation_deg=float(launch_angle),
                pressure_pa=pressure_pa,
                temperature_c=temperature_c,
                humidity=humidity_fraction,
                wind_m_s=wind_m_s,
                wind_direction_deg=float(wind_direction),
                duration_s=float(duration),
                ballistic_coefficient=float(bc) if bullet and bc_profile else None,
            )
            if result.x_m[-1] < zero_range_m:
                raise ValueError("Trajectory did not reach Zero Range; increase integration duration or muzzle velocity.")
            mask = result.x_m <= max_range_m
            if not np.any(mask):
                raise ValueError("No trajectory samples were produced within Max Range.")
            zero_height = float(np.interp(zero_range_m, result.x_m, result.y_m))
            output = pd.DataFrame({
                "time_s": result.time_s[mask],
                f"range_{'m' if units == 'Metric' else 'yd'}": result.x_m[mask] if units == "Metric" else result.x_m[mask] / METERS_PER_YARD,
                f"drop_{'m' if units == 'Metric' else 'in'}": zero_height - result.y_m[mask] if units == "Metric" else (zero_height - result.y_m[mask]) / METERS_PER_INCH,
                f"drift_{'m' if units == 'Metric' else 'in'}": result.z_m[mask] if units == "Metric" else result.z_m[mask] / METERS_PER_INCH,
                f"velocity_{'m_s' if units == 'Metric' else 'ft_s'}": np.sqrt(result.vx_m_s[mask] ** 2 + result.vy_m_s[mask] ** 2 + result.vz_m_s[mask] ** 2) if units == "Metric" else np.sqrt(result.vx_m_s[mask] ** 2 + result.vy_m_s[mask] ** 2 + result.vz_m_s[mask] ** 2) / METERS_PER_FOOT,
            })
            length_m = _to_si(float(bullet_length), units, "bullet_length")
            twist_m = _to_si(float(twist), units, "twist")
            sg = stability_factor(mass_kg, diameter_m, length_m, twist_m, velocity_m_s, temperature_c, pressure_pa)
            metric_cols = st.columns(4)
            metric_cols[0].metric("Peak height", f"{_from_si(float(result.y_m.max()), units, 'distance'):.2f} {'m' if units == 'Metric' else 'ft'}")
            drift_m = float(result.z_m[mask][-1])
            metric_cols[1].metric("Lateral drift", f"{drift_m if units == 'Metric' else drift_m / METERS_PER_INCH:.2f} {'m' if units == 'Metric' else 'in'}")
            metric_cols[2].metric("Speed of sound", f"{_from_si(sound_speed, units, 'speed'):.1f} {velocity_unit}")
            metric_cols[3].metric("Estimated Miller Sg", f"{sg:.3f}")
            range_col = f"range_{'m' if units == 'Metric' else 'yd'}"
            drop_col = f"drop_{'m' if units == 'Metric' else 'in'}"
            figure, axis = plt.subplots()
            axis.plot(output[range_col], output[drop_col], label="Drop relative to zero-range height")
            axis.set_xlabel(f"Range ({range_unit})")
            axis.set_ylabel(f"Drop ({'m' if units == 'Metric' else 'in'})")
            axis.grid(True, alpha=0.3)
            axis.legend()
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)
            st.dataframe(output, use_container_width=True, hide_index=True)
            st.download_button("Download trajectory CSV", output.to_csv(index=False).encode("utf-8"), "trajectory.csv", "text/csv")
        except (ValueError, RuntimeError, requests.RequestException, OSError) as error:
            st.error(f"Calculation failed: {error}")


if __name__ == "__main__":
    main()
