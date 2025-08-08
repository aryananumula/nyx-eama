import os
import json
import numbers
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from models.feature_extraction import perform_causal_analysis

# Load .env from repo root (adjust if yours lives elsewhere)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Toggle mock: true if env says so OR when running pytest
MOCK_LLM = os.getenv("MOCK_LLM", "False").lower() == "true" or "PYTEST_CURRENT_TEST" in os.environ

# Only import OpenAI client if we might actually use it
client = None
if not MOCK_LLM and os.getenv("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        client = None  # fall back to mock if import/creation fails



# JSON-safe summarization utils

def summarize_value(values):
    """
    Convert arrays/lists into mean/max/min and make JSON-safe.
    """
    # Plain numbers / numpy scalars
    if isinstance(values, (np.floating, np.integer, numbers.Number)):
        return float(values)

    # Array-like → stats
    if isinstance(values, (list, tuple, np.ndarray)):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size:
            return {
                "mean": round(float(np.mean(arr)), 2),
                "max": round(float(np.max(arr)), 2),
                "min": round(float(np.min(arr)), 2),
            }
        return {"mean": 0, "max": 0, "min": 0}

    # Anything JSON can already serialize
    try:
        json.dumps(values)
        return values
    except TypeError:
        return str(values)  # e.g., slice → "slice(10, 20, None)"


def summarize_features(features):
    """
    Recursively summarize biomechanical features into compact, JSON-safe stats.
    """
    out = {}
    for k, v in features.items():
        if isinstance(v, dict):
            out[k] = summarize_features(v)
        else:
            out[k] = summarize_value(v)
    return out



# Prompt compaction & causal filtering

def _safe_num(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _get_stat(d, key, stat="mean", default=None):
    """Pull {mean,max,min} from summarized dicts cleanly."""
    if not isinstance(d, dict):
        return default
    val = d.get(key)
    if isinstance(val, dict):
        return _safe_num(val.get(stat), default)
    return _safe_num(val, default)


def _select_key_signals(summarized):
    """
    Build a compact, unit-aware context from the summarized features.
    Focus on the essentials in consistent units/labels.
    """
    ja = summarized.get("joint_angles", {}) or {}
    rot = summarized.get("body_rotation", {}) or {}
    rd  = summarized.get("racket_dynamics", {}) or {}
    tf  = summarized.get("timing_features", {}) or {}
    pg  = summarized.get("power_generation", {}) or {}
    kc  = summarized.get("kinetic_chain", {}) or {}

    shoulder_deg = _get_stat(ja, "shoulder_flexion", "mean")
    elbow_deg    = _get_stat(ja, "elbow_flexion", "mean")
    wrist_ext    = _get_stat(ja, "wrist_extension", "mean")
    hip_rot      = _get_stat(ja, "hip_rotation", "mean")
    knee_flex    = _get_stat(ja, "knee_flexion", "mean")

    # Some datasets name torso rotation "trunk_rotation"
    torso_deg = _get_stat(rot, "torso_rotation_deg", "mean") or _get_stat(rot, "trunk_rotation", "mean")
    hip_deg   = _get_stat(rot, "hip_rotation_deg", "mean")

    tip_speed_mean = _get_stat(rd, "tip_speed", "mean") or _get_stat(rd, "speed", "mean")
    tip_speed_max  = _get_stat(rd, "tip_speed", "max")  or _get_stat(rd, "speed", "max")
    max_vel        = _safe_num(rd.get("max_velocity"))

    peak_power     = _safe_num(pg.get("peak_power")) or _safe_num(pg.get("estimated_power_watts"))

    # Timing/activation (handle -99 sentinels gracefully)
    spine_timing   = kc.get("spine_timing")
    racket_timing  = kc.get("racket_tip_timing")

    def _valid_timing(x):
        try:
            return (x is not None) and (float(x) != -99)
        except Exception:
            return False

    return {
        "angles_deg": {
            "shoulder_flexion_mean": shoulder_deg,
            "elbow_flexion_mean": elbow_deg,
            "wrist_extension_mean": wrist_ext,
            "hip_rotation_mean": hip_rot,
            "knee_flexion_mean": knee_flex,
        },
        "rotation_deg": {
            "torso_rotation_mean": torso_deg,
            "hip_rotation_mean": hip_deg,
        },
        "racket": {
            "tip_speed_mean_mps": tip_speed_mean,
            "tip_speed_max_mps": tip_speed_max,
            "max_velocity_mps": max_vel,
        },
        "power": {
            "peak_power": peak_power,
        },
        "timing": {
            "spine_timing_valid": _valid_timing(spine_timing),
            "racket_timing_valid": _valid_timing(racket_timing),
        }
    }


def _filter_causal_findings(causal_results, min_abs=0.3, top_k=5):
    """
    Keep only the most meaningful causal relations (e.g., |corr| ≥ min_abs).
    Works with perform_causal_analysis() output.
    """
    if not isinstance(causal_results, dict):
        return {}
    kept = {}

    t2p = (causal_results.get("timing_to_power") or {}) if isinstance(causal_results.get("timing_to_power"), dict) else {}
    ac  = (causal_results.get("angle_correlations") or {}) if isinstance(causal_results.get("angle_correlations"), dict) else {}

    t2p_kept = {k: float(v) for k, v in t2p.items() if isinstance(v, (int, float)) and abs(v) >= min_abs}
    ac_kept  = {k: float(v) for k, v in ac.items()  if isinstance(v, (int, float)) and abs(v) >= min_abs}

    def _top(d):
        return dict(sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k])

    if t2p_kept:
        kept["timing_to_power"] = _top(t2p_kept)
    if ac_kept:
        kept["angle_correlations"] = _top(ac_kept)
    return kept


def _uncertainties_from_features(summarized):
    """
    Surface known gaps (e.g., sentinel -99 timings, missing joints) so the LLM avoids guessing.
    """
    notes = []
    kc = summarized.get("kinetic_chain", {}) or {}
    for k in ("spine_timing", "racket_tip_timing"):
        v = kc.get(k)
        try:
            if v is not None and float(v) == -99:
                notes.append(f"{k} unavailable (sentinel -99)")
        except Exception:
            pass
    return notes



# Prompt construction

def format_input_for_llm(features, causal_results=None):
    """
    Build a compact, priority-ordered prompt using biomech + filtered causal info.
    """
    summarized = summarize_features(features)
    key_signals = _select_key_signals(summarized)
    causal_filtered = _filter_causal_findings(causal_results or {}, min_abs=0.3, top_k=5)
    uncertainties = _uncertainties_from_features(summarized)

    prompt_payload = {
        "stroke_type": features.get("stroke_type", "unknown"),
        "key_signals": key_signals,
        "causal_findings": causal_filtered,
        "uncertainties": uncertainties,
        "constraints": {
            "units": {
                "angles": "degrees",
                "rotation": "degrees",
                "speed": "m/s"
            },
            "style": {
                "audience": "high-school player",
                "max_sentences": 3,
                "avoid": ["jargon", "unmeasured claims", "hallucinations"]
            }
        }
    }

    return (
        "You are a tennis biomechanics coach. Using ONLY the provided JSON, "
        "explain the main issue and give 2–3 specific corrections. "
        "If something is missing/uncertain, mention it briefly and avoid guessing.\n\n"
        f"{json.dumps(prompt_payload, indent=2)}"
    )


# Feedback generation

def generate_feedback(features):
    """
    Runs causal analysis; returns mock, coach-like text during tests/dev,
    otherwise calls GPT-4o if a key/client is available.
    """
    # Always compute causal so both paths can use it
    causal_results = perform_causal_analysis(features)
    summarized = summarize_features(features)

    # MOCK PATH: no credits used
    if MOCK_LLM or client is None or not os.getenv("OPENAI_API_KEY"):
        st = (features.get("stroke_type") or "stroke").lower()

        # Pull optional signals from summarized features (key ones only)
        key = _select_key_signals(summarized)
        angles = key.get("angles_deg", {}) or {}
        rotation = key.get("rotation_deg", {}) or {}
        racket = key.get("racket", {}) or {}
        timing = key.get("timing", {}) or {}

        torso_mean = angles.get("shoulder_flexion_mean")  # keep legacy angle mention for tests if needed
        elbow_mean = angles.get("elbow_flexion_mean")
        torso_rot  = rotation.get("torso_rotation_mean")
        tip_max    = racket.get("tip_speed_max_mps") or racket.get("max_velocity_mps")
        arm_to_racket = (summarized.get("kinetic_chain", {}) or {}).get("arm_to_racket")

        issues, fixes = [], []

        # Realistic sentence
        if (torso_rot or 0) < 40:
            issues.append("limited torso rotation")
            fixes.append("rotate your hips and shoulders together through contact")

        if (tip_max or 0) < 7:
            issues.append("low racket head speed")
            fixes.append("accelerate the forearm earlier and extend through the ball")

        if (elbow_mean or 0) > 80:
            issues.append("excess elbow flexion")
            fixes.append("keep a slightly straighter hitting arm at impact")

        if isinstance(arm_to_racket, (int, float)) and arm_to_racket > 0.06:
            issues.append("late wrist release")
            fixes.append("initiate wrist release slightly earlier in the execution phase")

        if not issues:
            issues.append("good sequencing overall")
            fixes.append("keep a relaxed wrist and complete the follow-through")

        # Include a few flat keys if present (helps tests/logs)
        extras = []
        for k in ("accuracy", "power", "spin"):
            if k in features:
                extras.append(f"{k}={features[k]}")

        extra_txt = (" [" + ", ".join(extras) + "]") if extras else ""
        return (
            f"{st.capitalize()} feedback: "
            f"{' and '.join(issues)} detected. "
            f"Try to " + "; ".join(fixes) + f". (mock){extra_txt}"
        )

    # REAL PATH (uses credits)
    prompt = format_input_for_llm(features, causal_results)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional tennis biomechanics coach."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] Could not generate feedback: {e}"

# import os
# import json
# from pathlib import Path
# from dotenv import load_dotenv
# from openai import OpenAI
# import numpy as np
# from models.feature_extraction import perform_causal_analysis

# # Load .env 
# load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def summarize_value(values):
#     """
#     Convert arrays/lists into simple mean/max/min stats.
#     """
#     if hasattr(values, "__len__") and not isinstance(values, str):
#         arr = np.array(values, dtype=float).flatten()
#         if arr.size > 0:
#             return {
#                 "mean": round(float(np.mean(arr)), 2),
#                 "max": round(float(np.max(arr)), 2),
#                 "min": round(float(np.min(arr)), 2)
#             }
#         return {"mean": 0, "max": 0, "min": 0}
#     return values

# def summarize_features(features):
#     """
#     Recursively summarize biomechanical features into compact stats.
#     """
#     summary = {}
#     for k, v in features.items():
#         if isinstance(v, dict):
#             summary[k] = summarize_features(v)
#         else:
#             summary[k] = summarize_value(v)
#     return summary

# def format_input_for_llm(features, causal_results=None):
#     """
#     Format biomechanical + causal features into a readable LLM prompt.
#     """
#     summarized = summarize_features(features)

#     prompt = (
#         "You are a tennis biomechanics coach. "
#         "Analyze this stroke based on the biomechanical summary and explain what went wrong and how to improve.\n\n"
#         f"STROKE TYPE: {features.get('stroke_type', 'unknown')}\n\n"
#         "BIOMECHANICAL SUMMARY (aggregated stats in JSON):\n"
#         f"{json.dumps(summarized, indent=2)}\n"
#     )

#     if causal_results:
#         prompt += (
#             "\nCAUSAL ANALYSIS RESULTS (JSON):\n"
#             f"{json.dumps(causal_results, indent=2)}\n"
#         )

#     prompt += (
#         "\nGuidelines:\n"
#         "- Identify biomechanical strengths and weaknesses.\n"
#         "- Use causal reasoning to explain WHY the issue occurs.\n"
#         "- Suggest 2–3 specific, actionable corrections.\n"
#         "- Keep language clear for a high-school tennis player.\n"
#         "- Only reference provided data (no made-up facts).\n"
#     )

#     return prompt.strip()

# def generate_feedback(features):
#     """
#     Generate LLM feedback from biomechanical features.
#     Runs causal analysis automatically and summarizes data for clarity.
#     """
#     if not os.getenv("OPENAI_API_KEY"):
#         return "[FAKE FEEDBACK] No API key found. Returning placeholder feedback."

#     # Auto-run causal analysis
#     causal_results = perform_causal_analysis(features)

#     prompt = format_input_for_llm(features, causal_results)

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a professional tennis biomechanics coach."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.4,
#             max_tokens=300
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"[ERROR] Could not generate feedback: {e}"


# --------------------------------------------------------------------------

# def generate_feedback(stroke_data):
#     """
#     Generate biomechanically grounded feedback for the given stroke data using a large language model (LLM).

#     Parameters:
#     stroke_data (dict): A dictionary containing relevant features of the stroke for analysis.

#     Returns:
#     str: Feedback generated by the LLM.
#     """
#     # Initialize the LLM pipeline
#     llm = pipeline("text-generation", model="gpt-3.5-turbo")  # Replace with the appropriate model

#     # Format the input for the LLM
#     input_text = format_input_for_llm(stroke_data)

#     # Generate feedback
#     feedback = llm(input_text, max_length=150)[0]['generated_text']

#     return feedback


# def format_input_for_llm(stroke_data):
#     """
#     Format the stroke data into a string suitable for LLM input.

#     Parameters:
#     stroke_data (dict): A dictionary containing relevant features of the stroke for analysis.

#     Returns:
#     str: Formatted input string for the LLM.
#     """
#     formatted_input = "Analyze the following stroke data:\n"
#     for key, value in stroke_data.items():
#         formatted_input += f"{key}: {value}\n"

#     return formatted_input.strip()
