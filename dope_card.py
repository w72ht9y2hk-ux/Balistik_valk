"""DOPE card PDF generator for ballistic drop tables and field reference."""

from __future__ import annotations

import datetime
import io
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import qrcode

INCH_TO_MM = 25.4


class DOPEPDF(FPDF):
    """Custom PDF class for landscape 4x6 inch shooter pocket card."""

    def __init__(self) -> None:
        super().__init__(orientation="L", unit="mm", format=(152.4, 101.6))
        self.set_auto_page_break(auto=False, margin=5)

    def header(self) -> None:
        pass

    def footer(self) -> None:
        self.set_y(-8)
        self.set_font("Helvetica", "I", 6)
        self.set_text_color(100, 100, 100)
        self.cell(0, 4, f"Ballistic Calc — Generated {datetime.date.today().isoformat()}", align="C")


def generate_dope_card(
    params: Mapping[str, Any],
    result: Any,
    drift_result: Mapping[str, Any] | None = None,
    stability: float = 1.5,
    filename: str = "dope_card.pdf",
    app_url: str = "https://balistik-valk.streamlit.app",
) -> bytes:
    """Generate a landscape 4x6 inch shooter pocket card PDF."""
    pdf = DOPEPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 20, 20)
    
    bullet_name = params.get("bullet_name", ".308 Win — 168gr Match")
    zero_range = params.get("zero_range_m", 100.0)
    mv = params.get("launch_speed_m_s", 800.0)
    bc = params.get("ballistic_coefficient", 0.462)
    twist = params.get("twist_m_per_turn", 0.254) * INCH_TO_MM / 0.0254

    pdf.cell(100, 5, f"DOPE CARD: {bullet_name}", ln=0, align="L")
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(42, 5, f"Date: {datetime.date.today().isoformat()}", ln=1, align="R")

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(60, 60, 60)
    sub_header = f"Zero: {zero_range:.0f}m | MV: {mv:.0f} m/s | BC: {bc:.3f} | Twist: 1:{twist:.1f}\""
    pdf.cell(142, 4, sub_header, ln=1, align="L")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 7)
    pdf.set_fill_color(30, 30, 46)
    pdf.set_text_color(255, 255, 255)

    headers = ["Range", "Drop(cm)", "Drop(MOA)", "TOF(s)", "Vel(m/s)", "W 5m/s", "W 10m/s", "W 15m/s"]
    widths = [16, 17, 18, 16, 18, 19, 19, 19]

    for h, w in zip(headers, widths):
        pdf.cell(w, 5, h, border=1, ln=0, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(20, 20, 20)
    if hasattr(result, "x_m"):
        ranges = result.x_m
        drops = getattr(result, "y_m", np.zeros_like(ranges))
        times = getattr(result, "time_s", np.zeros_like(ranges))
        vx = getattr(result, "vx_m_s", np.ones_like(ranges))
        vy = getattr(result, "vy_m_s", np.ones_like(ranges))
        vz = getattr(result, "vz_m_s", np.ones_like(ranges))
        vels = np.sqrt(vx**2 + vy**2 + vz**2)
        zero_h = float(np.interp(zero_range, ranges, drops))
        drop_cm = (zero_h - drops) * 100.0
    else:
        ranges = np.asarray(result.get("x_m", [0, 100, 200, 300, 400, 500]))
        drop_cm = np.asarray(result.get("drop_cm", [0, 5, 20, 50, 100, 180]))
        times = np.asarray(result.get("time_s", [0, 0.1, 0.2, 0.3, 0.4, 0.5]))
        vels = np.asarray(result.get("vel_m_s", [800, 750, 700, 650, 600, 550]))

    sampled_indices = []
    last_r = -50.0
    for idx, r in enumerate(ranges):
        if r >= 0 and (r - last_r >= 45.0 or idx == len(ranges) - 1):
            sampled_indices.append(idx)
            last_r = r

    for idx in sampled_indices[:12]:
        r = ranges[idx]
        d_cm = drop_cm[idx]
        moa = (d_cm / max(r, 1.0)) * 34.38 if r > 0 else 0.0
        tof = times[idx]
        v = vels[idx]
        w5 = (5.0 * tof ** 1.5) * 10.0
        w10 = w5 * 2.0
        w15 = w5 * 3.0

        pdf.cell(16, 4.2, f"{r:.0f}m", border=1, align="C")
        pdf.cell(17, 4.2, f"{d_cm:+.1f}", border=1, align="C")
        pdf.cell(18, 4.2, f"{moa:+.1f}", border=1, align="C")
        pdf.cell(16, 4.2, f"{tof:.2f}", border=1, align="C")
        pdf.cell(18, 4.2, f"{v:.0f}", border=1, align="C")
        pdf.cell(19, 4.2, f"{w5:+.0f}cm", border=1, align="C")
        pdf.cell(19, 4.2, f"{w10:+.0f}cm", border=1, align="C")
        pdf.cell(19, 4.2, f"{w15:+.0f}cm", border=1, align="C")
        pdf.ln()

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 7)
    stability_status = "STABLE" if stability >= 1.5 else ("MARGINAL" if stability >= 1.3 else "UNSTABLE")
    env_summary = f"Stability Sg = {stability:.2f} ({stability_status}) | T: {params.get('temperature_c', 15):.0f}°C | P: {params.get('pressure_pa', 101325)/100:.0f}hPa"
    pdf.cell(110, 4, env_summary, ln=0, align="L")

    qr = qrcode.QRCode(box_size=2, border=1)
    qr.add_data(app_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    qr_io = io.BytesIO()
    img.save(qr_io, format="PNG")
    qr_io.seek(0)

    pdf.image(qr_io, x=128, y=pdf.get_y() - 4, w=18, h=18)

    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")

    if filename:
        Path(filename).write_bytes(pdf_bytes)

    return pdf_bytes


if __name__ == "__main__":
    from solver import solve_trajectory
    res = solve_trajectory(
        mass_kg=0.010886,
        diameter_m=0.00782,
        drag_profile="g7",
        launch_speed_m_s=823.0,
        launch_elevation_rad=0.0,
        pressure_pa=101325.0,
        temperature_c=15.0,
        relative_humidity=0.0,
        time_end_s=5.0,
        max_step_s=0.02,
        zero_range_m=91.44,
    )
    test_params = {
        "bullet_name": ".308 Win — 168gr Match",
        "zero_range_m": 91.44,
        "launch_speed_m_s": 823.0,
        "ballistic_coefficient": 0.462,
        "twist_m_per_turn": 0.3048,
        "temperature_c": 15.0,
        "pressure_pa": 101325.0,
        "relative_humidity": 0.5,
    }
    pdf_out = generate_dope_card(test_params, res, stability=1.65, filename="test_dope_card.pdf")
    if len(pdf_out) > 1024:
        print("PASS")
    else:
        print("FAIL")

