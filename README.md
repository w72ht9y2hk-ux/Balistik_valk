# Balistik_valk
Ballistic Calc Project
Python 3.11+ external ballistics engine. No external ballistic libraries.
Stack
numpy, scipy, matplotlib, streamlit, pandas, pytest
Conventions
SI units internally (m, kg, s, Pa, °C, rad). Convert at UI layer only.
Type hints on all function signatures.
Each module has an `if __name__ == "__main__"` self-test block.
Docstrings: one-line summary + Args/Returns.
Constants at top of file, not inline.
No external ballistic libraries (no py-ballisticcalc, no ballistics-engine).
Architecture
atmosphere.py — CIPM-2007 air density, Arden Buck, speed of sound
drag.py — load G1/G7 CSV tables, interpolate Cd(Mach)
solver.py — RK45 trajectory integration (scipy solve_ivp)
drift.py — Litz spin drift + Coriolis/Eötvös
stability.py — Miller twist rule
monte_carlo.py — WEZ via N-sample perturbation
app.py — Streamlit UI (all user-facing unit conversions happen here)
data/ — CSV drag tables
tests/ — pytest
Rules
Do not create files I haven't asked for.
Do not add dependencies beyond the stack listed above.
When I say "write module X", produce the full file with self-test.
