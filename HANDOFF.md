# Project handoff

## Current state

This handoff reflects the repository files inspected on 2026-11-08. Code exists for the core modules below, but **self-tests and the pytest suite have not been run/verified in this session**. Do not interpret existing `assert` blocks as proof that the current checkout passes.

- **`atmosphere.py`** — CIPM-style moist-air density and humid-air sound speed; has a `__main__` smoke test. Not run here.
- **`drag.py`** — reads G1/G7 Mach/Cd CSVs; generates a 200-point `CubicSpline` fine table when missing, prefers it, and caches parsed tables; has a `__main__` smoke test. The generated `data/g1_fine.csv` and `data/g7_fine.csv` are not currently checked in and will be created on first lookup if the environment permits file writes. Not run here.
- **`solver.py`** — RK45 point-mass integration, optional BC scaling, and adaptive downrange output samples (10 m through zero range, 25 m through 2× zero range, 50 m beyond). `interpolate_cd()` is called in the derivative used as the RK45 RHS, so Cd is evaluated at integration stages, not only output positions. Has a `__main__` self-test checking the first five samples and printing the point count. Not run here.
- **`stability.py`** — Miller-rule gyroscopic stability estimate; has a `__main__` smoke test. Not run here.
- **`app.py`** — Streamlit UI with units, bullet-picker UI, Open-Meteo weather fetch, zeroing/range inputs, drag profile, trajectory display, and CSV export. No standalone module self-test is defined. UI has not been launched or verified in this session.
- **`tests/test_basic.py`** — pytest regression tests are present, but currently import `_display_units` and `_si_inputs` from `app.py`; those names are absent from the current `app.py`. Expect test collection/import failure until the tests are aligned with the current app helpers or compatible helpers are restored.

## Still to do / not implemented

1. **Populate the bullet database.** `data/bullets.json` currently contains bundle metadata and an empty `bullets` array. The current app reads the local file only, so the bullet picker will offer Custom but no database bullets until the actual upstream bundle is downloaded and checked in.
2. **Verify the drag-table files.** Run the drag module to generate fine CSVs, confirm both have 200 data rows and valid finite Cd values, and decide whether to commit those generated files. Current loader falls back to original tables if generation raises `OSError` or `ValueError`.
3. **Resolve/apply the BC model carefully.** The solver applies `cd *= 1 / ballistic_coefficient`. This is a simple scaling, not a validated conversion from published G1/G7 ballistic coefficients to drag-table Cd. Validate the modeling assumption before treating BC-based results as physically authoritative.
4. **Update and run tests.** Fix stale app-helper imports in `tests/test_basic.py`; add assertions for the 200-point fine tables, fine-table preference/fallback, adaptive interval boundaries, and solver use of selected bullet BC/profile.
5. **Run the app and smoke tests locally.** Confirm Streamlit starts, bullet picker works after adding real data, weather updates all controls, and a trajectory calculation completes for both unit systems.
6. **Check app/solver zero-range integration.** The UI collects Zero Range and Max Range, but the current `_solve()` path does not pass Zero Range into the solver's adaptive `zero_range_m` parameter. The solver therefore uses its 100 m default for sampling unless adjusted.
7. Optional planned modules such as `drift.py` and `monte_carlo.py` are absent; the current app does not import them as active capabilities.

## Known issues / verification notes

- **No test execution is claimed here.** Run the commands below and record actual results before describing the modules as passing.
- `tests/test_basic.py` references missing names from `app.py`, so `python -m pytest -q` is likely to fail during collection as currently checked in.
- The bullet database file is an empty placeholder; this is the reason no database bullet options are available locally.
- Generated fine tables are runtime-created by `drag.py`; a read-only deployment filesystem or insufficient write permissions may prevent generation. In that case the loader uses the original table.
- `CubicSpline` plus linear interpolation on its 200 generated points densifies the curve but does not make the solver's Cd function a cubic spline between those points. If that distinction matters, interpolate directly with the spline at runtime or use a denser table.
- `solution.sol()` is time-parametrized. The solver maps requested ranges to times with `np.interp` over internal solver samples, assuming downrange x is monotonic. This is appropriate for the app's forward shots, but reverse/downrange-nonmonotone trajectories are rejected.
- The app currently sends `ballistic_coefficient` only for a selected database bullet. Custom BC input is not passed as a solver BC.
- Weather networking requires internet access and Open-Meteo availability; the app displays an error on request/data failures.

## Commands

From the repository root, with Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

Launch the app:

```bash
streamlit run app.py
```

Run automated tests:

```bash
python -m pytest -q
```

Run module smoke tests individually:

```bash
python atmosphere.py
python drag.py
python solver.py
python stability.py
```

`python drag.py` will try to generate the fine CSVs under `data/` before checking them. `python solver.py` runs the adaptive-sampling self-test and prints its output point count and initial samples.

## Module purposes and key signatures

- **`app.py`** — Streamlit front end, unit conversion, weather/bullet input, and trajectory visualization. Key functions: `_to_si(value: float, units: str, kind: str) -> float`, `_from_si(value: float, units: str, kind: str) -> float`, `_load_bullets() -> list[dict[str, Any]]`, `_bullet_bc(bullet: dict[str, Any]) -> tuple[float | None, str | None]`, `_fetch_weather(latitude: float, longitude: float) -> dict[str, Any]`, `_solve(*, mass_kg: float, diameter_m: float, profile: str, velocity_m_s: float, elevation_deg: float, pressure_pa: float, temperature_c: float, humidity: float, wind_m_s: float, wind_direction_deg: float, duration_s: float, ballistic_coefficient: float | None) -> Any`, `main() -> None`.
- **`atmosphere.py`** — moist-air properties. Key signatures: `air_density(pressure_pa: float, temperature_c: float, relative_humidity: float = 0.0) -> float`; `speed_of_sound(temperature_c: float, pressure_pa: float = STANDARD_PRESSURE_PA, relative_humidity: float = 0.0) -> float`.
- **`drag.py`** — source/fine drag-table loading and Cd lookup. Key signatures: `load_drag_table(profile: str, csv_path: str | Path | None = None) -> List[Tuple[float, float]]`; `interpolate_cd(mach: float, profile: str, csv_path: str | Path | None = None) -> float`.
- **`solver.py`** — RK45 trajectory integration and adaptive output sampling. Key signatures: `projectile_derivative(time_s: float, state: Sequence[float], mass_kg: float, diameter_m: float, drag_profile: str, pressure_pa: float, temperature_c: float, relative_humidity: float, azimuth_rad: float = 0.0, ballistic_coefficient: float | None = None) -> Sequence[float]`; `solve_trajectory(mass_kg: float, diameter_m: float, drag_profile: str, launch_speed_m_s: float, launch_elevation_rad: float, launch_azimuth_rad: float = 0.0, pressure_pa: float = 101325.0, temperature_c: float = 15.0, relative_humidity: float = 0.0, time_end_s: float = 10.0, initial_altitude_m: float = 0.0, max_step_s: float = 0.01, ballistic_coefficient: float | None = None, zero_range_m: float = DEFAULT_ZERO_RANGE_M, wind_speed_m_s: float = 0.0, wind_direction_deg: float = 0.0) -> TrajectoryResult`.
- **`stability.py`** — SI wrapper around the empirical Miller stability rule. Key signature: `stability_factor(bullet_mass_kg: float, bullet_diameter_m: float, bullet_length_m: float, twist_m_per_turn: float, velocity_m_s: float = STANDARD_VELOCITY_M_S, temperature_c: float = STANDARD_TEMPERATURE_C, pressure_pa: float = STANDARD_PRESSURE_PA) -> float`.
- **`tests/test_basic.py`** — pytest checks for atmosphere, drag data, solver, stability, and app unit helpers. Main command: `python -m pytest -q`.
