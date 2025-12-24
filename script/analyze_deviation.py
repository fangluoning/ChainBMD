"""
Compare a sample with an expert benchmark and output movement deficits plus training advice.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import requests

CHAIN_ORDER = [
    ("lower_chain", ["left_foot", "right_foot"]),
    ("leg_chain", ["left_leg_emg"]),
    ("core_chain", ["core_spine"]),
    ("upper_chain", ["upper_arm_emg"]),
    ("forearm_chain", ["lower_arm_emg"]),
    ("hand_chain", ["right_hand_euler"]),
]

CHAIN_DESCRIPTIONS = {
    "lower_chain": "Lower-body support and push-off",
    "leg_chain": "Thigh drive",
    "core_chain": "Core rotation and force transfer",
    "upper_chain": "Upper-arm loading",
    "forearm_chain": "Forearm whipping action",
    "hand_chain": "Terminal wrist acceleration",
}

RECOMMENDATION_RULES = {
    "lower_chain": "Insufficient plantar force. Emphasize landing stability, explosive push-off, and center-of-mass control.",
    "leg_chain": "Insufficient lower-limb drive. Train thigh strength and hip-knee coordination.",
    "core_chain": "Weak core kinetic chain. Reinforce trunk rotation and core stability.",
    "upper_chain": "Insufficient upper-arm loading. Focus on racket preparation and shoulder external rotation.",
    "forearm_chain": "Insufficient force transfer through the forearm. Train whipping mechanics and forearm strength.",
    "hand_chain": "Insufficient terminal wrist acceleration. Train impact-phase wrist explosiveness and control.",
}

NODE_CHAIN_MAP = {
    "left_foot": "lower_chain",
    "right_foot": "lower_chain",
    "left_leg_emg": "leg_chain",
    "core_spine": "core_chain",
    "upper_arm_emg": "upper_chain",
    "lower_arm_emg": "forearm_chain",
    "right_hand_euler": "hand_chain",
}

NODE_DESCRIPTIONS = {
    "left_foot": "Left-foot support",
    "right_foot": "Right-foot support",
    "left_leg_emg": "Left thigh musculature",
    "core_spine": "Trunk/core",
    "upper_arm_emg": "Upper-arm musculature",
    "lower_arm_emg": "Forearm musculature",
    "right_hand_euler": "Distal wrist",
}

COMPONENT_DESCRIPTIONS = {
    "left_total_force": "Left-foot total pressure",
    "left_forefoot": "Left forefoot",
    "left_heel": "Left heel",
    "right_total_force": "Right-foot total pressure",
    "right_forefoot": "Right forefoot",
    "right_heel": "Right heel",
    "hip_roll": "Hip roll",
    "hip_pitch": "Hip flexion/extension",
    "hip_yaw": "Hip rotation",
    "spine_roll": "Spinal lateral flexion",
    "spine_pitch": "Spinal flexion/extension",
    "spine_yaw": "Spinal rotation",
    "shoulder_roll": "Shoulder roll",
    "shoulder_pitch": "Shoulder elevation/depression",
    "shoulder_yaw": "Shoulder rotation",
    "hand_roll": "Wrist roll",
    "hand_pitch": "Wrist pitch",
    "hand_yaw": "Wrist yaw",
}

SKILL_LEVEL_NOTES = {
    0: "Beginner: New to badminton; control and continuity are limited.",
    1: "Intermediate: Has hitting experience; overall coordination still has room to improve.",
    2: "Advanced: Mature technique and rhythm; can consistently produce high-quality strokes.",
}


def severity_text(score: float) -> str:
    if score >= 2.0:
        return "major deviation"
    if score >= 1.5:
        return "large deviation"
    if score >= 1.0:
        return "minor deviation"
    return "close to benchmark"


def timing_text(delta: float) -> str:
    gap = abs(delta)
    if gap >= 12:
        level = "significant"
    elif gap >= 6:
        level = "moderate"
    elif gap >= 3:
        level = "slight"
    else:
        level = None
    if not level:
        return ""
    return f"{level} {'lag' if delta > 0 else 'lead'}"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_phase_metrics(curve: np.ndarray):
    peak_idx = int(curve.argmax())
    peak_phase = peak_idx / max(len(curve) - 1, 1) * 100
    cumulative = np.cumsum(curve)
    total = cumulative[-1] + 1e-6
    reach50 = np.argmax(cumulative >= 0.5 * total)
    rise_phase = reach50 / max(len(curve) - 1, 1) * 100
    return peak_phase, rise_phase


def detect_deviation(record, benchmark, threshold=1.0, phase_threshold=8.0):
    """threshold indicates how many standard deviations trigger an alert."""
    chain_reports = []
    chain_scores = {}
    node_stats = benchmark["node_stats"]
    for node, stats in node_stats.items():
        mean = np.array(stats["mean"])
        std = np.array(stats["std"]) + 1e-6
        series = np.array(record["node_time_series"][node])
        z = (series - mean) / std
        score = np.abs(z).mean()
        chain_scores[node] = score
    for chain_id, nodes in CHAIN_ORDER:
        scores = [chain_scores.get(n, 0.0) for n in nodes]
        if not scores:
            continue
        avg_score = float(np.mean(scores))
        if avg_score > threshold:
            chain_reports.append(
                {
                    "chain": chain_id,
                    "description": CHAIN_DESCRIPTIONS.get(chain_id, chain_id),
                    "avg_z_score": avg_score,
                    "nodes": {node: chain_scores.get(node, 0.0) for node in nodes},
                    "suggestion": RECOMMENDATION_RULES.get(
                        chain_id, "Please review synchronized video for a more precise diagnosis."
                    ),
                }
            )
    chain_reports.sort(key=lambda x: x["avg_z_score"], reverse=True)

    component_alerts = []
    comp_stats = benchmark.get("component_stats", {})
    sample_components = record.get("node_components", {})
    for node, comps in comp_stats.items():
        if node not in sample_components:
            continue
        for comp_name, stats in comps.items():
            series = np.array(sample_components[node].get(comp_name))
            if series.size == 0:
                continue
            mean = np.array(stats["mean"])
            std = np.array(stats["std"]) + 1e-6
            score = float(np.abs((series - mean) / std).mean())
            if score > threshold:
                component_alerts.append(
                    {
                        "node": node,
                        "component": comp_name,
                        "chain": NODE_CHAIN_MAP.get(node),
                        "z_score": score,
                        "description": COMPONENT_DESCRIPTIONS.get(comp_name, comp_name),
                        "suggestion": RECOMMENDATION_RULES.get(
                            NODE_CHAIN_MAP.get(node, ""),
                            "Please review synchronized video for a more precise diagnosis.",
                        ),
                    }
                )

    timing_alerts = []
    sample_series = record["node_time_series"]
    for node, stats in node_stats.items():
        curve = np.array(sample_series.get(node))
        if curve.size == 0:
            continue
        peak_phase, rise_phase = compute_phase_metrics(curve)
        baseline_peak = stats.get("peak_phase", peak_phase)
        delta_peak = peak_phase - baseline_peak
        if abs(delta_peak) > phase_threshold:
            timing_alerts.append(
                {
                    "node": node,
                    "issue": "peak_delay" if delta_peak > 0 else "peak_early",
                    "delta": delta_peak,
                    "description": (
                        f"{NODE_DESCRIPTIONS.get(node, node)} peak "
                        f"{'lag' if delta_peak > 0 else 'lead'} {abs(delta_peak):.1f}%"
                    ),
                }
            )
    return {
        "chain_reports": chain_reports,
        "component_alerts": component_alerts,
        "timing_alerts": timing_alerts,
    }


def generate_llm_advice(report, model="deepseek-r1:7b"):
    chain_lines = []
    for item in report.get("chain_reports", []):
        node_terms = []
        for node, score in item["nodes"].items():
            desc = NODE_DESCRIPTIONS.get(node, node)
            sev = severity_text(score)
            if sev == "close to benchmark":
                continue
            node_terms.append(f"{desc}{sev}")
        term_text = "; ".join(node_terms) if node_terms else "maintain smooth force transfer"
        chain_lines.append(f"- {item['description']}: {term_text}")
    component_lines = []
    for alert in report.get("component_alerts", []):
        sev = severity_text(alert["z_score"])
        if sev == "close to benchmark":
            continue
        component_lines.append(
            f"- {COMPONENT_DESCRIPTIONS.get(alert['component'], alert['component'])}: {sev}"
        )
    timing_lines = []
    for alert in report.get("timing_alerts", []):
        phr = timing_text(alert["delta"])
        if not phr:
            continue
        timing_lines.append(f"- {NODE_DESCRIPTIONS.get(alert['node'], alert['node'])}: {phr}")
    level_defs = "\n".join(f"{v}" for v in SKILL_LEVEL_NOTES.values())
    true_level = report["sample_info"]["true_label"]
    pred_level = report["sample_info"]["pred_label"]
    prompt = (
        "Role: Expert badminton coach and biomechanics analyst.\n"
        "Goal: Produce a professional, coach-like assessment with a sports-science perspective.\n"
        "Knowledge constraints:\n"
        f"{level_defs}\n"
        "Use only the labels 'Beginner', 'Intermediate', or 'Advanced' (do not use other wording).\n"
        "Style requirements:\n"
        "- English only; authoritative and concise; no hedging or apologies.\n"
        "- Use coach-grade terminology: kinetic chain, sequencing, proximal-to-distal transfer, "
        "load/drive/bracing, timing, stability, ground reaction.\n"
        "- Be specific and actionable. Use concrete cues like 'bigger hip turn', 'later wrist snap', "
        "'stronger push-off', 'earlier trunk rotation', 'stiffer front-leg brace'.\n"
        "- Link each issue to impact (e.g., loss of power, control, or rhythm) and a corrective focus.\n"
        "Output format (English only):\n"
        "- Write three short paragraphs with no labels or headings.\n"
        "- Paragraph 1 (2-4 sentences): State the athlete's current skill level and summarize overall kinetic chain "
        "quality and rhythm.\n"
        "- Paragraph 2 (3-5 sentences): Review the chain in order: foot -> leg -> core -> upper arm -> forearm -> wrist. "
        "Mention the most important issues or strengths, staying in that order.\n"
        "- Paragraph 3 (exactly 3 sentences): Provide exactly 3 training recommendations. Each sentence must include "
        "a drill and a performance goal.\n"
        "Constraints: Avoid exact numbers or percentages. Use qualitative terms such as major/minor/near-benchmark.\n"
        "If data are insufficient, state that explicitly and avoid guessing.\n"
        "Few-shot examples (tone and specificity only):\n"
        "Bad (too vague):\n"
        "The athlete should improve coordination and timing. Work on stability and power.\n"
        "Good (specific, coach-like):\n"
        "The core turn is late, so the arm is forced to overwork; open the hips earlier and keep the front leg "
        "stiffer at plant to anchor rotation. Wrist snap is early, so delay the snap to the last third of the swing "
        "to keep shuttle contact cleaner. Emphasize a stronger push-off through the forefoot to drive a fuller "
        "hip-shoulder separation.\n"
        "Data:\n"
        f"- Skill levels: true {true_level}, model predicted {pred_level}.\n"
        f"- Chain deviations:\n{chr(10).join(chain_lines) or 'No clear anomalies'}\n"
        f"- Component alerts:\n{chr(10).join(component_lines) or 'No clear anomalies'}\n"
        f"- Timing rhythm:\n{chr(10).join(timing_lines) or 'Rhythm is stable'}\n"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
        },
    }
    try:
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.RequestException as exc:
        return f"LLM request failed: {exc}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare against a benchmark and generate deficits plus training advice."
    )
    parser.add_argument("--record", default="outputs/figures/sample_0.json", help="Single-sample JSON")
    parser.add_argument("--benchmark", default="outputs/figures/benchmark_skill2.json")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--phase_threshold", type=float, default=8.0)
    parser.add_argument(
        "--use_llm", action="store_true", help="Call local Ollama to generate natural-language advice"
    )
    parser.add_argument("--llm_model", default="deepseek-r1:7b", help="Ollama model name")
    args = parser.parse_args()

    record = load_json(Path(args.record))
    if isinstance(record, list):
        record = record[0]
    benchmark = load_json(Path(args.benchmark))
    deviation_report = detect_deviation(
        record, benchmark, threshold=args.threshold, phase_threshold=args.phase_threshold
    )
    report = {
        "sample_info": {
            "true_label": record["true_label"],
            "pred_label": record["pred_label"],
        },
        **deviation_report,
    }
    if args.use_llm:
        llm_text = generate_llm_advice(report, model=args.llm_model)
        print(llm_text)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
