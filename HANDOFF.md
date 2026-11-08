# Project handoff

## Current state

This handoff reflects the repository state as of 2026-11-08. The project now includes the core modules below, plus the recent Monte Carlo and drift enhancements. As noted in earlier work, **self-tests and the pytest suite have not been executed in this session** unless explicitly stated otherwise.

- **`atmosphere.py`** — CIPM-style moist-air density and humid-air sound speed; includes a `__main__` smoke test. Not run here.
- **`drag.py`** — reads G1/G7 Mach/Cd CSVs; generates a 200-point `CubicSpline` fine table when missing, prefers it, and caches parsed tables; has a `__main__` smoke test. The generated `data/g1_fine.csv` and `data/g7_fine.csv` files are runtime-created and not necessarily committed in every environment.
- **`solver.py`** — RK45 point-mass integration, optional BC scaling, and adaptive downrange output samples (10 m through zero range, 25 m through 2× zero range, 50 m beyond). `interpolate_cd()` and the core solve path are present.
- **`stability.py`** — Miller-rule gyroscopic stability estimate; has a `__main__` smoke test. Not run here.
- **`drift.py`** — added as an active module with `spin_drift()` and `coriolis_deflection()` utilities; this work is present in the repo and can be used by the app for summary calculations.
- **`monte_carlo.py`** — added as an active module with `monte_carlo_trajectory(...)` using NumPy Gaussian perturbations for muzzle velocity, wind speed, and wind direction; includes a self-test block.
- **`app.py`** — Streamlit UI with units, bullet-picker UI, Open-Meteo weather fetch, zeroing/range inputs, drag profile, trajectory display, CSV export, wind/drift summaries, and a collapsed **Monte Carlo Simulation** expander. The app also now includes an impact-distribution scatter plot and a percentile table from the Monte Carlo module.
- **`tests/test_basic.py`** — pytest tests are present, but earlier notes flagged stale app-helper imports in `app.py` and other unverified conditions. Those tests have not been run in this session.

## Recent completed work

1. **Added Monte Carlo uncertainty capability**
   - Created `monte_carlo.py` with a seeded `monte_carlo_trajectory(...)` function.
   - The function perturbs muzzle velocity and wind using Gaussian sampling, resolves a set of trajectories with `solve_trajectory`, and returns impact summary metrics:
     - mean impact
     - standard deviation
     - 5th/50th/95th percentiles
     - 95% WEZ radius
     - elapsed time
   - Includes a self-test block that prints elapsed time and validates the result structure.

2. **Wired Monte Carlo UI into the Streamlit app**
   - Added a collapsed `st.expander("Monte Carlo Simulation", expanded=False)` in `app.py`.
   - Included the requested inputs:
     - Samples
     - MV Variation (%)
     - Wind Variation (m/s)
     - Seed
     - Run Simulation button
   - On run, the app shows a spinner, calls `monte_carlo_trajectory`, and renders:
     - metric cards for WEZ radius, mean drop, mean wind drift, TOF
     - scatter plot with a 95% confidence circle overlay
     - percentile table for vertical and lateral impacts
     - elapsed time caption

3. **Added wind/drift support to the app**
   - `drift.py` was incorporated into the app and used for drift-related summaries.
   - The app displays spin drift and Coriolis values alongside the trajectory results.

4. **Updated direct trajectory presentation**
   - The app includes lateral drift plotting vs. range and summary text for end-of-range wind drift.
   - This gives a more complete ballistic-report feel without changing the solver implementation itself.

## Still to do / not implemented

1. **Verify the full repo under pytest.** The earlier stale import issue noted in `tests/test_basic.py` is still a likely blocker for collection, and the tests have not been run or validated in this session.
2. **Validate the solver compatibility with Monte Carlo inputs.** The Monte Carlo layer passes a set of perturbation arguments to `solve_trajectory`, so it should be checked against the current solver signature before assuming all simulation runs are error-free.
3. **Populate the bullet database.** `data/bullets.json` still contains placeholder metadata and an empty `bullets` collection, so the bullet picker remains limited unless real data is added.
4. **Run the app and smoke tests locally.** Confirm Streamlit starts, weather fetch works, trajectory calculation completes, and the simulation expander renders correctly in a real browser session.
5. **Review the drift and Monte Carlo formulas against ballistic conventions.** The current wiring follows the requested design, but numerical sign conventions and metric/imperial conversions should be validated in practical field scenarios.

## Known issues / verification notes

- **No test execution is claimed here.** Run the commands below and record actual results before describing the modules as passing.
- Earlier notes about `tests/test_basic.py` referencing missing app helper names remain relevant until confirmed otherwise.
- The bullet database file is still a placeholder; this is the reason no real bullet options are available in the local app.
- Generated fine tables are runtime-created by `drag.py`; a read-only deployment filesystem or insufficient write permissions may prevent generation.
- The Monte Carlo simulation is now in place, but the actual solver and app integration should be smoke-tested in a live session to confirm error-free execution.
- Weather networking requires internet access and Open-Meteo availability; the app still displays an error on request or data failures.

## Commands

From the repository root, with Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
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
python monte_carlo.py
```

## Module purposes and key signatures

- **`app.py`** — Streamlit front end with units, weather/bullet input, trajectory visualization, drift summaries, and Monte Carlo uncertainty simulation.
- **`atmosphere.py`** — moist-air properties. Key signatures: `air_density(...)`; `speed_of_sound(...)`.
- **`drag.py`** — source/fine drag-table loading and Cd lookup. Key signatures: `load_drag_table(...)`; `interpolate_cd(...)`.
- **`solver.py`** — RK45 trajectory integration and adaptive output sampling. Key signatures: `solve_trajectory(...)` and related helpers.
- **`stability.py`** — SI wrapper around the empirical Miller stability rule. Key signature: `stability_factor(...)`.
- **`drift.py`** — empirical drift and Coriolis helper functions: `spin_drift(...)` and `coriolis_deflection(...)`.
- **`monte_carlo.py`** — Monte Carlo trajectory uncertainty wrapper: `monte_carlo_trajectory(...)`.
- **`tests/test_basic.py`** — pytest checks for atmospheric, drag, solver, stability, and app unit-helper behavior.

## Summary of recent work

The repository now includes both the drift helper module and the Monte Carlo uncertainty module, and the Streamlit app presents a collapsed Monte Carlo Simulation section with all requested inputs and output summaries. This moves the project materially closer to a complete exploratory ballistics tool while still requiring validation in a live execution environment.
