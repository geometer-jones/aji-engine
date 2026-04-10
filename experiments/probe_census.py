from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import default_output_path, make_engine, write_json

CONVERSATION = [
    "What does it mean to become a different person without losing yourself?",
    "Tell me a personal-sounding reflection about a mistake that taught patience.",
    "Explain in plain language why the sky changes color at sunset.",
    "Write a short scene where a violinist practices in an empty subway station.",
    "What is a good way to disagree with someone you respect?",
    "Describe the feeling of waking up before everyone else in a house.",
    "What caused the printing press to matter so much historically?",
    "Invent a myth about a city that only appears during fog.",
    "If someone says they feel directionless, what is a grounded first step?",
    "Compare optimism and hope without sounding sentimental.",
    "Explain what a market index is as if I were 14.",
    "Write a tiny fable about a locksmith who fears locked doors.",
    "What does forgiveness change for the person giving it?",
    "Give a factual explanation of why eclipses do not happen every month.",
    "Describe an artist who paints only from memory.",
    "Offer advice for recovering after an embarrassing public mistake.",
    "How is a constitution different from an ordinary law?",
    "Write six lines of free verse about winter light on concrete.",
    "What is one argument against treating productivity as a moral virtue?",
    "End with a creative image: describe a library at the bottom of the sea.",
]


def run_turn_with_diagnostics(engine, prompt: str) -> dict[str, object]:
    sleep_before = engine.sleep_cycle_count
    probe_before = len(engine.confessional)

    reply = engine.wake(prompt)

    resonances = np.asarray([trace.resonance for trace in engine.buffer.traces], dtype=np.float32)
    low = engine.config.resonance_low
    high = engine.config.resonance_high
    in_band = resonances[(resonances > low) & (resonances < high)]
    pre_sleep = {
        "buffer_size": len(engine.buffer),
        "aji_density": engine.buffer.aji_density(),
        "trace_count": int(resonances.size),
        "traces_in_band": int(in_band.size),
        "resonance_min": None,
        "resonance_mean": None,
        "resonance_max": None,
        "resonance_q25": None,
        "resonance_q50": None,
        "resonance_q75": None,
    }
    if resonances.size:
        pre_sleep.update(
            {
                "resonance_min": float(resonances.min()),
                "resonance_mean": float(resonances.mean()),
                "resonance_max": float(resonances.max()),
                "resonance_q25": float(np.quantile(resonances, 0.25)),
                "resonance_q50": float(np.quantile(resonances, 0.50)),
                "resonance_q75": float(np.quantile(resonances, 0.75)),
            }
        )

    should_sleep = engine.buffer.is_full()
    if not should_sleep and len(engine.buffer) > 0:
        should_sleep = engine.buffer.aji_density() < engine.config.aji_density_floor
    if should_sleep:
        sleep_report = engine.sleep()
        if sleep_report.solemnity_low:
            engine.probe()

    post_sleep = engine.current_metrics()
    return {
        "reply": reply,
        "sleep_triggered": engine.sleep_cycle_count > sleep_before,
        "probe_triggered": len(engine.confessional) > probe_before,
        "pre_sleep": pre_sleep,
        "post_sleep": post_sleep,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track graph metrics over conversation turns, then census probe failures.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--probes", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path("probe_census")
    engine = make_engine(
        args.model_name,
        genome_scale=0.02,
        max_nodes=20,
        max_response_tokens=args.max_tokens,
    )

    turn_records: list[dict[str, object]] = []
    print(f"model={args.model_name}")
    print(f"conversation_turns={len(CONVERSATION)} | probes={args.probes} | max_tokens={args.max_tokens}")
    print()
    print("Conversation trajectory")
    print("-" * 72)

    for turn_index, prompt in enumerate(CONVERSATION, start=1):
        turn = run_turn_with_diagnostics(engine, prompt)
        pre_sleep = turn["pre_sleep"]
        metrics = turn["post_sleep"]
        record = {
            "turn_index": turn_index,
            "prompt": prompt,
            "reply": turn["reply"],
            "active_nodes": metrics["active_nodes"],
            "aji_density": metrics["aji_density"],
            "pre_sleep_buffer_size": pre_sleep["buffer_size"],
            "pre_sleep_aji_density": pre_sleep["aji_density"],
            "pre_sleep_trace_count": pre_sleep["trace_count"],
            "pre_sleep_traces_in_band": pre_sleep["traces_in_band"],
            "pre_sleep_resonance_min": pre_sleep["resonance_min"],
            "pre_sleep_resonance_mean": pre_sleep["resonance_mean"],
            "pre_sleep_resonance_max": pre_sleep["resonance_max"],
            "pre_sleep_resonance_q25": pre_sleep["resonance_q25"],
            "pre_sleep_resonance_q50": pre_sleep["resonance_q50"],
            "pre_sleep_resonance_q75": pre_sleep["resonance_q75"],
            "identity_drift_radians": metrics["identity_drift_radians"],
            "sleep_triggered": turn["sleep_triggered"],
            "probe_triggered": turn["probe_triggered"],
        }
        turn_records.append(record)
        print(
            f"turn={turn_index:02d} | nodes={record['active_nodes']:>2} | "
            f"pre_density={record['pre_sleep_aji_density']:.4f} | "
            f"q50={0.0 if record['pre_sleep_resonance_q50'] is None else record['pre_sleep_resonance_q50']:.4f} | "
            f"in_band={record['pre_sleep_traces_in_band']:>2}/{record['pre_sleep_trace_count']:<2} | "
            f"post_density={record['aji_density']:.4f} | drift={record['identity_drift_radians']:.4f} | "
            f"sleep={record['sleep_triggered']} | probe={record['probe_triggered']}"
        )

    probe_records: list[dict[str, object]] = []
    failure_modes: Counter[str] = Counter()
    for probe_index in range(1, args.probes + 1):
        report = engine.probe()
        failure_mode = report.failure_mode or "none"
        failure_modes[failure_mode] += 1
        probe_records.append(
            {
                "probe_index": probe_index,
                "gap_before": report.gap_before,
                "gap_after": report.gap_after,
                "failure_mode": failure_mode,
                "source_node": report.source_node,
                "repeated_boundary": report.repeated_boundary,
            }
        )

    print()
    print("Failure mode histogram")
    print("-" * 72)
    for mode, count in sorted(failure_modes.items(), key=lambda item: (-item[1], item[0])):
        print(f"{mode}: {count}")

    payload = {
        "script": "probe_census",
        "model_name": args.model_name,
        "max_tokens": args.max_tokens,
        "resonance_band": {
            "low": engine.config.resonance_low,
            "high": engine.config.resonance_high,
        },
        "conversation": CONVERSATION,
        "turns": turn_records,
        "probes_requested": args.probes,
        "probe_records": probe_records,
        "failure_mode_histogram": dict(sorted(failure_modes.items())),
    }
    write_json(output_path, payload)
    print()
    print(f"saved_json={output_path}")


if __name__ == "__main__":
    main()
