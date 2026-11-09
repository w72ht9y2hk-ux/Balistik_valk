"""Streamlit UI for the SI-based Balistik_valk trajectory solver."""

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

APP_TITLE = "Balistik_valk"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
BULLET_DATA_PATH = Path(__file__).resolve().parent / "data" / "bullets.json"
GRAINS_PER_KG = 1.0 / 0.00006479891
METERS_PER_FOOT = 0.3048
METERS_PER_YARD = 0.9144
METERS_PER_INCH = 0.0254
PA_PER_INHG = 3386.389


def _to_si(value: float, units: str, kind: str) -> float:
    """Convert displayed units to SI."""
    if kind == "mass":
        return value / 1000.0 if units == "Metric" else value / GRAINS_PER_KG
    if kind in ("diameter", "bullet_length", "twist"):
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
    records = []
    for bullet in payload["bullets"]:
        try:
            if bullet.get("id") and bullet.get("maker") and bullet.get("name"):
                if float(bullet["diameter_in"]) > 0 and float(bullet["weight_gr"]) > 0:
                    records.append(bullet)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return records


def _bullet_bc(bullet: dict[str, Any]) -> tuple[float | None, str | None]:
    """Prefer the bullet's G7 BC, falling back to G1."""
    coefficients = bullet.get("bc") or {}
    if coefficients.get("g7") is not None:
        return float(coefficients["g7"]), "g7"
    if coefficients.get("g1") is not None:
        return float(coefficients["g1"]), "g1"
    return None, None


def _bullet_label(bullet: dict[str, Any]) -> str:
    """Format the bullet picker label."""
    return (
        f"{bullet['maker']} — .{float(bullet['diameter_in']):.3f} — "
        f"{float(bullet['weight_gr']):g} gr ({bullet['name']})"
    )


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
        "wind_speed_m_s": float(current["wind_speed_10m"]) / 3.6,
        "wind_direction_deg": float(current["wind_direction_10m"]),
        "timestamp": str(current["time"]),
    }


def _supported_solver_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Filter optional inputs for backward compatibility with solver versions."""
    from solver import solve_trajectory

    accepted = inspect.signature(solve_trajectory).parameters
    return {key: value for key, value in arguments.items() if key in accepted}


def _solve(
    *, mass_kg: float, diameter_m: float, profile: str, velocity_m_s: float,
    elevation_deg: float, pressure_pa: float, temperature_c: float,
    humidity: float, wind_m_s: float, wind_direction_deg: float,
    duration_s: float, ballistic_coefficient: float | None, zero_range_m: float,
) -> Any:
    """Run the trajectory solver with normalized SI inputs."""
    from solver import solve_trajectory

    args = {
        "mass_kg": mass_kg,
        "diameter_m": diameter_m,
        "drag_profile": profile,
        "launch_speed_m_s": velocity_m_s,
        "launch_elevation_rad": math.radians(elevation_deg),
        "pressure_pa": pressure_pa,
        "temperature_c": temperature_c,
        "relative_humidity": humidity,
        "time_end_s": duration_s,
        "max_step_s": 0.02,
        "wind_speed_m_s": wind_m_s,
        "wind_direction_deg": wind_direction_deg,
        "ballistic_coefficient": ballistic_coefficient,
        "zero_range_m": zero_range_m,
    }
    return solve_trajectory(**_supported_solver_args(args))


def _display_monte_carlo(st: Any, params: dict[str, Any]) -> None:
    """Render the collapsed Monte Carlo expander and persisted results."""
    from monte_carlo import monte_carlo_trajectory

    with st.expander("Monte Carlo Simulation", expanded=False):
        c1, c2 = st.columns(2)
        samples = c1.number_input("Samples", min_value=1000, max_value=100000, value=10000, step=1000, key="mc_samples")
        mv_percent = c2.number_input("MV Variation (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="mc_mv_variation")
        c3, c4 = st.columns(2)
        wind_sigma = c3.number_input("Wind Variation (m/s)", min_value=0.0, max_value=5.0, value=0.5, step=0.1, key="mc_wind_variation")
        seed = c4.number_input("Seed", min_value=0, max_value=99999, value=42, step=1, key="mc_seed")
        run_simulation = st.button("Run Simulation", key="run_monte_carlo", disabled=params is None)
        if params is None:
            st.info("Calculate a trajectory first to set the Monte Carlo impact range and zero line.")
        if run_simulation and params is not None:
            try:
                with st.spinner(f"Running {int(samples):,} samples..."):
                    result = monte_carlo_trajectory(
                        params,
                        n_samples=int(samples),
                        mv_sigma=float(mv_percent) / 100.0,
                        wind_sigma=float(wind_sigma),
                        seed=int(seed),
                    )
                st.session_state["monte_carlo_result"] = result
                st.session_state.pop("monte_carlo_error", None)
            except (ValueError, RuntimeError, OSError) as error:
                st.session_state.pop("monte_carlo_result", None)
                st.session_state["monte_carlo_error"] = str(error)

        if st.session_state.get("monte_carlo_error"):
            st.error(f"Monte Carlo simulation failed: {st.session_state['monte_carlo_error']}")
        result = st.session_state.get("monte_carlo_result")
        if result is None:
            return

        mean_vertical = result["mean_impact"]["vertical_m"]
        mean_lateral = result["mean_impact"]["lateral_m"]
        wez_radius = result["wez_radius_m"]
        metrics = st.columns(4)
        metrics[0].metric("WEZ Radius (95%)", f"{wez_radius * 100.0:.1f} cm")
        metrics[1].metric("Mean Drop", f"{mean_vertical * 100.0:.1f} cm")
        metrics[2].metric("Mean Wind Drift", f"{mean_lateral * 100.0:.1f} cm")
        metrics[3].metric("TOF", f"{result['mean_tof_s']:.2f} s")

        lateral_cm = np.asarray(result["impact_points"]["lateral_m"]) * 100.0
        vertical_cm = np.asarray(result["impact_points"]["vertical_m"]) * 100.0
        mean_lateral_cm = mean_lateral * 100.0
        mean_vertical_cm = mean_vertical * 100.0
        radius_cm = wez_radius * 100.0
        theta = np.linspace(0.0, 2.0 * math.pi, 361)
        figure, axis = plt.subplots()
        axis.scatter(lateral_cm, vertical_cm, s=5, alpha=0.2, label="Simulated impacts")
        axis.plot(
            mean_lateral_cm + radius_cm * np.cos(theta),
            mean_vertical_cm + radius_cm * np.sin(theta),
            color="tab:red", label="95% confidence circle",
        )
        axis.scatter([mean_lateral_cm], [mean_vertical_cm], color="black", marker="x", label="Mean impact")
        axis.set_xlabel("Lateral impact (cm)")
        axis.set_ylabel("Vertical impact (cm)")
        axis.set_title("Monte Carlo Impact Distribution")
        axis.grid(True, alpha=0.3)
        axis.axis("equal")
        axis.legend()
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)

        percentiles = result["percentiles"]
        percentile_table = pd.DataFrame([
            {
                "Percentile": percentile,
                "Vertical (cm)": percentiles["vertical_m"][percentile] * 100.0,
                "Lateral (cm)": percentiles["lateral_m"][percentile] * 100.0,
            }
            for percentile in ("5", "50", "95")
        ])
        st.dataframe(percentile_table, use_container_width=True, hide_index=True)
        st.caption(f"Completed in {result['elapsed_s']:.1f} s")


def main() -> None:
    """Render the Streamlit ballistics application."""
    from atmosphere import air_density, speed_of_sound
    from drift import coriolis_deflection, spin_drift
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
                st.session_state[f"pressure_{units}"] = _from_si(weather["pressure_hpa"] * 100.0, units, "pressure")
                st.session_state[f"temperature_{units}"] = _from_si(weather["temperature_c"], units, "temperature")
                st.session_state["humidity_percent"] = int(round(weather["humidity_percent"]))
                st.session_state[f"wind_speed_{units}"] = weather["wind_speed_m_s"] if units == "Metric" else weather["wind_speed_m_s"] / 0.44704
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
        pressure = st.number_input(f"Pressure ({pressure_unit})", min_value=0.01, value=1013.25 if units == "Metric" else 29.92, key=f"pressure_{units}")
        temperature = st.number_input(f"Temperature ({temperature_unit})", value=15.0 if units == "Metric" else 59.0, key=f"temperature_{units}")
        humidity = st.slider("Relative humidity (%)", 0, 100, 50, key="humidity_percent")
        wind_speed = st.number_input(f"Wind speed ({wind_unit})", min_value=0.0, value=0.0, step=0.1, key=f"wind_speed_{units}")
        wind_direction = st.number_input("Wind direction FROM (°)", min_value=0.0, max_value=360.0, value=0.0, step=1.0, key="wind_direction_deg")

        st.header("Projectile")
        try:
            bullets = _load_bullets()
        except (OSError, json.JSONDecodeError, ValueError):
            bullets = []
        search = st.text_input("Search bullets by manufacturer or caliber", key="bullet_search").strip().casefold()
        visible_bullets = [b for b in bullets if not search or search in str(b.get("maker", "")).casefold() or search in f".{float(b['diameter_in']):.3f}"]
        bullet_map = {str(b["id"]): b for b in visible_bullets}
        bullet_options = ["__custom__", *bullet_map.keys()]
        if st.session_state.get("bullet_id") not in bullet_options:
            st.session_state["bullet_id"] = "__custom__"
        bullet_id = st.selectbox("Bullet", bullet_options, format_func=lambda key: "Custom" if key == "__custom__" else _bullet_label(bullet_map[key]), key="bullet_id")
        bullet = bullet_map.get(bullet_id)
        bc_value, bc_profile = _bullet_bc(bullet) if bullet else (None, None)
        if bullet and st.session_state.get("autofilled_bullet") != bullet_id:
            st.session_state["mass_value"] = float(bullet["weight_gr"]) * 0.00006479891 * 1000.0 if units == "Metric" else float(bullet["weight_gr"])
            st.session_state["diameter_value"] = float(bullet["diameter_in"]) * 25.4 if units == "Metric" else float(bullet["diameter_in"])
            if bullet.get("length_in") is not None:
                st.session_state["bullet_length_value"] = float(bullet["length_in"]) * 25.4 if units == "Metric" else float(bullet["length_in"])
            if bc_value is not None:
                st.session_state["bc_value"] = bc_value
            st.session_state["autofilled_bullet"] = bullet_id
        elif not bullet:
            st.session_state["autofilled_bullet"] = "__custom__"
        if bullet:
            st.caption(f"{bullet['maker']} · {_bullet_label(bullet)} · {bc_profile.upper() + ' BC ' + format(bc_value, '.3f') if bc_value is not None else 'No published BC'}")
        mass = st.number_input(f"Mass ({mass_unit})", min_value=0.001, value=11.34 if units == "Metric" else 175.0, key="mass_value")
        diameter = st.number_input(f"Diameter ({diameter_unit})", min_value=0.001, value=7.82 if units == "Metric" else 0.308, key="diameter_value")
        bullet_length = st.number_input(f"Projectile length ({diameter_unit})", min_value=0.001, value=31.5 if units == "Metric" else 1.24, key="bullet_length_value")
        twist = st.number_input(f"Twist length ({diameter_unit}/turn)", min_value=0.001, value=254.0 if units == "Metric" else 10.0, key="twist_value")
        velocity = st.number_input(f"Muzzle velocity ({velocity_unit})", min_value=0.001, value=800.0 if units == "Metric" else 2625.0, key="velocity_value")
        bc = st.number_input("Ballistic coefficient", min_value=0.001, value=0.462, step=0.001, key="bc_value")
        profile_choice = st.selectbox("Reference drag profile", SUPPORTED_PROFILES, index=1, key="drag_profile")
        profile = bc_profile if bullet and bc_profile else profile_choice

        st.header("Wind & Drift")
        spin_rate_rpm = st.number_input("Spin Rate (RPM)", min_value=500, max_value=5000, value=2500, step=100, key="spin_rate_rpm")
        azimuth_deg = st.number_input("Azimuth (°)", min_value=0, max_value=359, value=0, step=1, key="azimuth_deg")
        st.header("Zeroing")
        zero_range = st.number_input(f"Zero Range ({range_unit})", min_value=25.0, max_value=1000.0, value=91.0 if units == "Metric" else 100.0, step=1.0, key=f"zero_range_{units}")
        max_range = st.number_input(f"Max Range ({range_unit})", min_value=100.0, max_value=3000.0, value=914.0 if units == "Metric" else 1000.0, step=10.0, key=f"max_range_{units}")
        launch_angle = st.slider("Launch elevation (deg)", 0.1, 80.0, 15.0, step=0.5)
        duration = st.slider("Integration duration (s)", 1.0, 60.0, 10.0, step=0.5)

    mc_params = st.session_state.get("monte_carlo_params")
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
            solver_args = {
                "mass_kg": mass_kg,
                "diameter_m": diameter_m,
                "drag_profile": profile,
                "launch_speed_m_s": velocity_m_s,
                "launch_elevation_rad": math.radians(float(launch_angle)),
                "pressure_pa": pressure_pa,
                "temperature_c": temperature_c,
                "relative_humidity": humidity_fraction,
                "time_end_s": float(duration),
                "max_step_s": 0.02,
                "wind_speed_m_s": wind_m_s,
                "wind_direction_deg": float(wind_direction),
                "ballistic_coefficient": float(bc) if bullet and bc_profile else None,
                "zero_range_m": zero_range_m,
            }
            solver_args = _supported_solver_args(solver_args)
            result = _solve(
                mass_kg=mass_kg, diameter_m=diameter_m, profile=profile,
                velocity_m_s=velocity_m_s, elevation_deg=float(launch_angle),
                pressure_pa=pressure_pa, temperature_c=temperature_c,
                humidity=humidity_fraction, wind_m_s=wind_m_s,
                wind_direction_deg=float(wind_direction), duration_s=float(duration),
                ballistic_coefficient=float(bc) if bullet and bc_profile else None,
                zero_range_m=zero_range_m,
            )
            if result.x_m[-1] < zero_range_m:
                raise ValueError("Trajectory did not reach Zero Range; increase integration duration or muzzle velocity.")
            mask = result.x_m <= max_range_m
            if not np.any(mask):
                raise ValueError("No trajectory samples were produced within Max Range.")
            ranges_m = result.x_m[mask]
            times_s = result.time_s[mask]
            heights_m = result.y_m[mask]
            lateral_m = result.z_m[mask]
            zero_height = float(np.interp(zero_range_m, result.x_m, result.y_m))
            speeds_m_s = np.sqrt(result.vx_m_s[mask]**2 + result.vy_m_s[mask]**2 + result.vz_m_s[mask]**2)
            drop_m = zero_height - heights_m
            output = pd.DataFrame({
                "time_s": times_s,
                f"range_{'m' if units == 'Metric' else 'yd'}": ranges_m if units == "Metric" else ranges_m / METERS_PER_YARD,
                f"drop_{'m' if units == 'Metric' else 'in'}": drop_m if units == "Metric" else drop_m / METERS_PER_INCH,
                f"drift_{'m' if units == 'Metric' else 'in'}": lateral_m if units == "Metric" else lateral_m / METERS_PER_INCH,
                f"velocity_{'m_s' if units == 'Metric' else 'ft_s'}": speeds_m_s if units == "Metric" else speeds_m_s / METERS_PER_FOOT,
            })
            length_m = _to_si(float(bullet_length), units, "bullet_length")
            twist_m = _to_si(float(twist), units, "twist")
            sg = stability_factor(mass_kg, diameter_m, length_m, twist_m, velocity_m_s, temperature_c, pressure_pa)
            st.session_state["monte_carlo_params"] = {
                **solver_args,
                "impact_range_m": max_range_m,
                "zero_height_m": zero_height,
            }
            st.session_state.pop("monte_carlo_result", None)
            st.session_state.pop("monte_carlo_error", None)

            st.subheader("Trajectory")
            figure, axis = plt.subplots()
            range_plot = ranges_m if units == "Metric" else ranges_m / METERS_PER_YARD
            drop_plot = drop_m if units == "Metric" else drop_m / METERS_PER_INCH
            axis.plot(range_plot, drop_plot, label="Drop relative to zero line")
            axis.set_xlabel(f"Range ({range_unit})")
            axis.set_ylabel(f"Drop ({'m' if units == 'Metric' else 'in'})")
            axis.grid(True, alpha=0.3)
            axis.legend()
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)
            st.dataframe(output, use_container_width=True, hide_index=True)
            st.download_button("Download trajectory CSV", output.to_csv(index=False).encode("utf-8"), "trajectory.csv", "text/csv")

            st.subheader("Wind & Drift")
            drift_table = pd.DataFrame({"Range (m)": ranges_m, "Lateral wind drift (cm)": lateral_m * 100.0})
            st.dataframe(drift_table, use_container_width=True, hide_index=True)
            drift_figure, drift_axis = plt.subplots()
            drift_axis.plot(ranges_m, lateral_m * 100.0, label="Solver z-axis wind drift")
            drift_axis.set_title("Lateral Drift (cm) vs Range (m)")
            drift_axis.set_xlabel("Range (m)")
            drift_axis.set_ylabel("Lateral drift (cm)")
            drift_axis.grid(True, alpha=0.3)
            drift_axis.legend()
            st.pyplot(drift_figure, clear_figure=True)
            plt.close(drift_figure)
            total_drift_cm = float(lateral_m[-1] * 100.0)
            direction = "right" if total_drift_cm >= 0 else "left"
            st.write(f"Total wind drift at {ranges_m[-1]:.0f} m: {abs(total_drift_cm):.1f} cm to the {direction}")
            tof_s = float(times_s[-1])
            final_velocity_m_s = float(speeds_m_s[-1])
            spin_cm = spin_drift(tof_s, float(spin_rate_rpm) / 60.0, velocity_m_s, final_velocity_m_s)
            coriolis_ew_m, coriolis_ns_m = coriolis_deflection(tof_s, final_velocity_m_s, float(latitude), float(azimuth_deg))
            st.write(f"Spin drift: {spin_cm:+.1f} cm | Coriolis EW: {coriolis_ew_m * 100.0:+.1f} cm | Coriolis NS: {coriolis_ns_m * 100.0:+.1f} cm")
        except (ValueError, RuntimeError, requests.RequestException, OSError) as error:
            st.error(f"Calculation failed: {error}")

    _display_monte_carlo(st, st.session_state.get("monte_carlo_params"))


if __name__ == "__main__":
    main()
