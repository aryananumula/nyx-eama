import os
from pathlib import Path
from typing import Dict, Tuple, List, Any
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

# Loads .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

# Reference ranges for tennis strokes
# Keys expected from your features dict:
#  - predicted_stroke (string label)
#  - racket_velocity_mps, peak_power_W, rotation_range_deg,
#    stroke_duration_frames_60fps, peak_angular_velocity_rad_s, impact_timing_pct
REFERENCE_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Backhand (Two-handed)": {
        "racket_velocity_mps": (22, 32),
        "peak_power_W": (300, 600),
        "rotation_range_deg": (60, 100),
        "stroke_duration_frames_60fps": (85, 125),
        "peak_angular_velocity_rad_s": (4, 7),
        "impact_timing_pct": (75, 85),
    },
    "Backhand (One-handed)": {
        "racket_velocity_mps": (20, 30),
        "peak_power_W": (250, 550),
        "rotation_range_deg": (70, 110),
        "stroke_duration_frames_60fps": (90, 130),
        "peak_angular_velocity_rad_s": (5, 8),
        "impact_timing_pct": (70, 80),
    },
    "Backhand Slice": {
        "racket_velocity_mps": (15, 25),
        "peak_power_W": (150, 350),
        "rotation_range_deg": (40, 80),
        "stroke_duration_frames_60fps": (95, 140),
        "peak_angular_velocity_rad_s": (3, 6),
        "impact_timing_pct": (65, 75),
    },
    "Backhand Volley": {
        "racket_velocity_mps": (8, 15),
        "peak_power_W": (100, 250),
        "rotation_range_deg": (20, 50),
        "stroke_duration_frames_60fps": (30, 60),
        "peak_angular_velocity_rad_s": (2, 4),
        "impact_timing_pct": (50, 70),
    },
    "Forehand Flat": {
        "racket_velocity_mps": (25, 35),
        "peak_power_W": (400, 800),
        "rotation_range_deg": (80, 120),
        "stroke_duration_frames_60fps": (80, 120),
        "peak_angular_velocity_rad_s": (6, 10),
        "impact_timing_pct": (70, 80),
    },
    "Forehand Open Stance": {
        "racket_velocity_mps": (23, 33),
        "peak_power_W": (350, 750),
        "rotation_range_deg": (90, 130),
        "stroke_duration_frames_60fps": (75, 115),
        "peak_angular_velocity_rad_s": (7, 11),
        "impact_timing_pct": (72, 82),
    },
    "Forehand Slice": {
        "racket_velocity_mps": (18, 28),
        "peak_power_W": (200, 450),
        "rotation_range_deg": (50, 90),
        "stroke_duration_frames_60fps": (90, 135),
        "peak_angular_velocity_rad_s": (4, 7),
        "impact_timing_pct": (65, 75),
    },
    "Forehand Volley": {
        "racket_velocity_mps": (10, 18),
        "peak_power_W": (120, 300),
        "rotation_range_deg": (30, 60),
        "stroke_duration_frames_60fps": (35, 65),
        "peak_angular_velocity_rad_s": (3, 5),
        "impact_timing_pct": (55, 75),
    },
    "Service Flat": {
        "racket_velocity_mps": (45, 65),
        "peak_power_W": (1200, 2500),
        "rotation_range_deg": (100, 150),
        "stroke_duration_frames_60fps": (120, 180),
        "peak_angular_velocity_rad_s": (8, 15),
        "impact_timing_pct": (85, 95),
    },
    "Service Kick": {
        "racket_velocity_mps": (35, 50),
        "peak_power_W": (800, 1800),
        "rotation_range_deg": (80, 130),
        "stroke_duration_frames_60fps": (130, 190),
        "peak_angular_velocity_rad_s": (10, 18),
        "impact_timing_pct": (80, 90),
    },
    "Service Slice": {
        "racket_velocity_mps": (40, 55),
        "peak_power_W": (900, 2000),
        "rotation_range_deg": (90, 140),
        "stroke_duration_frames_60fps": (125, 185),
        "peak_angular_velocity_rad_s": (6, 12),
        "impact_timing_pct": (82, 92),
    },
    "Smash": {
        "racket_velocity_mps": (30, 45),
        "peak_power_W": (600, 1200),
        "rotation_range_deg": (60, 110),
        "stroke_duration_frames_60fps": (50, 90),
        "peak_angular_velocity_rad_s": (8, 14),
        "impact_timing_pct": (75, 85),
    },
}


def _fmt_num(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def compare_to_reference(features: Dict[str, Any], stroke_type: str) -> List[str]:
    """
    Compare numeric features to the optimal ranges for a given stroke type.
    Returns a list of human-readable findings (low/high/in-range).
    """
    findings: List[str] = []
    ref = REFERENCE_RANGES.get(stroke_type)
    if not ref:
        return [f"No reference ranges found for stroke '{stroke_type}'."]

    # Label for the coach
    labels = {
        "racket_velocity_mps": "Racket velocity (m/s)",
        "peak_power_W": "Peak power (W)",
        "rotation_range_deg": "Rotation range (°)",
        "stroke_duration_frames_60fps": "Stroke duration (frames @60fps)",
        "peak_angular_velocity_rad_s": "Peak angular velocity (rad/s)",
        "impact_timing_pct": "Impact timing (%)",
    }

    for key, (lo, hi) in ref.items():
        if key not in features:
            findings.append(f"{labels[key]}: missing in features.")
            continue

        val = features[key]
        # Only compare if numeric
        try:
            v = float(val)
        except Exception:
            findings.append(f"{labels[key]}: non-numeric value '{val}'.")
            continue

        if v < lo:
            diff = (lo - v) / (hi - lo + 1e-9) * 100.0
            findings.append(
                f"{labels[key]} LOW: {_fmt_num(v)} vs optimal {lo}-{hi} (≈{diff:.0f}% below range)."
            )
        elif v > hi:
            diff = (v - hi) / (hi - lo + 1e-9) * 100.0
            findings.append(
                f"{labels[key]} HIGH: {_fmt_num(v)} vs optimal {lo}-{hi} (≈{diff:.0f}% above range)."
            )
        else:
            findings.append(f"{labels[key]} OK: {_fmt_num(v)} within {lo}-{hi}.")

    return findings

def safe_get(d, keys, default=None):
    """
    Safely retrieve a nested value from a dict.
    If any key is missing or value is None/empty, return default.
    """
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    # Handle numpy arrays / lists truth check
    if d is None:
        return default
    if isinstance(d, (list, np.ndarray)) and len(d) == 0:
        return default
    return d

def build_context_summary(features: Dict[str, Any]) -> str:
     # format raw features to match reference keys
    def peak_angular_velocity(joint_array):
        angles = np.array(joint_array)
        vel = np.diff(np.radians(angles))
        return np.max(np.abs(vel))

    # return None if feature can't be found
    condensed = {
        "racket_velocity_mps": (
            max(safe_get(features, ["racket_dynamics", "speed"], []))
            if safe_get(features, ["racket_dynamics", "speed"]) is not None
            else None
        ),
        "peak_power_W": safe_get(features, ["power_generation", "peak_power"]),
        "rotation_range_deg": (
            (max(safe_get(features, ["joint_angles", "hip_rotation"], [])) -
            min(safe_get(features, ["joint_angles", "hip_rotation"], [])))
            if safe_get(features, ["joint_angles", "hip_rotation"]) is not None
            else None
        ),
        "stroke_duration_frames_60fps": safe_get(features, ["timing_features", "stroke_duration"]),
        "peak_angular_velocity_rad_s": (
            max(
                peak_angular_velocity(joint_vals)
                for joint_vals in safe_get(features, ["joint_angles"], {}).values()
                if joint_vals is not None
            )
            if safe_get(features, ["joint_angles"]) is not None
            else None
        ),
        "impact_timing_pct": (
            100 * safe_get(features, ["racket_dynamics", "impact_frame"]) /
            safe_get(features, ["timing_features", "stroke_duration"])
            if safe_get(features, ["racket_dynamics", "impact_frame"]) is not None
            and safe_get(features, ["timing_features", "stroke_duration"]) not in (None, 0)
            else None
        ),
        "classification": safe_get(features, ["classification", "type"]),
    }


    stroke = (
        condensed.get("predicted_stroke")
        or condensed.get("classification")
        or "UNKNOWN"
    )
    cmp_lines = compare_to_reference(condensed, stroke)
    lines = [
        f"Predicted stroke: {stroke}",
        "Optimal-range comparison:",
        *[" - " + line for line in cmp_lines],
        "",
        "Raw features (subset):",
        f" - racket_velocity_mps: {condensed.get('racket_velocity_mps')}",
        f" - peak_power_W: {condensed.get('peak_power_W')}",
        f" - rotation_range_deg: {condensed.get('rotation_range_deg')}",
        f" - stroke_duration_frames_60fps: {condensed.get('stroke_duration_frames_60fps')}",
        f" - peak_angular_velocity_rad_s: {condensed.get('peak_angular_velocity_rad_s')}",
        f" - impact_timing_pct: {condensed.get('impact_timing_pct')}",
    ]
    return "\n".join(lines)


if not API_KEY:

    def generate_feedback(_features):
        return "[ERROR] GEMINI_API_KEY not set. Add it to .env or your environment."

else:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_feedback(features: Dict[str, Any]) -> str:
        """
        Generate grounded coaching feedback that references optimal ranges
        for the predicted stroke and highlights deviations.
        """
        context = build_context_summary(features)

        prompt = (
            "You are a professional tennis biomechanics coach. "
            "First line: 'Overall Score: X/10' (0 = very poor, 10 = perfect). "
            "Second, provide a list of metrics that are outside the optimal range, indicating whether each is LOW or HIGH and a descriptor of how much without stating specific numbers. "
            "Third, provide a concise diagnosis (2–3 sentences) "
            "Finally, provide exactly 3 actionable corrections phrased in concise, neutral, human-understandable coaching advice, focusing on how to improve performance rather than quoting exact numbers or ranges. "
            "Base your judgments on the optimal-range comparison; do not invent numbers."
            "\n\n"
            f"{context}"
        )

        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=650,
                ),
            )
            return resp.text.strip()
        except Exception as e:
            return f"[ERROR] Could not generate feedback: {e}"
