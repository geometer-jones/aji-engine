from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import default_output_path, make_engine, set_genome_scale, short_text, write_json

PROMPTS = [
    "Explain why habits are easier to keep than goals in three sentences.",
    "Write a haiku about rust on a garden gate.",
    "Give me two practical ways to calm down before a difficult conversation.",
    "Argue for and against remote work in one short paragraph each.",
    "What is the difference between accuracy and precision? Answer plainly.",
    "Tell a one-paragraph story about a lighthouse keeper hiding a secret.",
    "List three signs that a startup idea is too broad.",
    "Summarize the trolley problem in under 60 words.",
    "Draft a polite message declining a meeting without sounding cold.",
    "Invent a motto for a city built inside a crater.",
]
WARMUP_PROMPTS = [
    "What does it mean to really understand something?",
    "Describe a sound that reminds you of home.",
    "Explain why silence can be uncomfortable.",
]
SCALES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure how often KV bias changes generated text.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path("gen_shift")
    engine = make_engine(
        args.model_name,
        genome_scale=SCALES[0],
        max_response_tokens=args.max_tokens,
    )

    # Warmup: run a few turns to populate the self-graph so serialize() is non-zero.
    print("Warmup turns to populate self-graph...")
    for warmup_prompt in WARMUP_PROMPTS:
        engine.respond(warmup_prompt)
    print(f"  active_nodes={engine.self_graph.num_active}")
    print()

    records: list[dict[str, object]] = []
    scale_summaries: list[dict[str, object]] = []

    print(f"model={args.model_name}")
    print(f"prompts={len(PROMPTS)} | max_tokens={args.max_tokens}")
    print()
    print("Generation shift summary")
    print("-" * 72)

    for scale in SCALES:
        set_genome_scale(engine.bridge, scale)
        engine.config.genome_scale = scale
        differing: list[dict[str, object]] = []
        for prompt_index, prompt in enumerate(PROMPTS, start=1):
            comparison = engine.evaluate_prompt(prompt, max_tokens=args.max_tokens)
            record = {
                "scale": scale,
                "prompt_index": prompt_index,
                "prompt": prompt,
                "biased": comparison["biased"],
                "unbiased": comparison["unbiased"],
                "different": bool(comparison["different"]),
            }
            records.append(record)
            if record["different"]:
                differing.append(record)

        fraction = len(differing) / len(PROMPTS)
        scale_summary = {
            "scale": scale,
            "different_count": len(differing),
            "prompt_count": len(PROMPTS),
            "fraction_different": fraction,
            "representative_differences": differing[:2],
        }
        scale_summaries.append(scale_summary)
        print(f"scale={scale:>4.2f} | different={len(differing):>2}/{len(PROMPTS)} | fraction={fraction:0.2f}")

    print()
    print("Representative differing pairs")
    print("-" * 72)
    any_difference = False
    for summary in scale_summaries:
        reps = summary["representative_differences"]
        if not reps:
            continue
        any_difference = True
        print(f"scale={summary['scale']:0.2f}")
        for item in reps:
            print(f"  prompt {item['prompt_index']}: {short_text(str(item['prompt']), 90)}")
            print(f"    biased:   {short_text(str(item['biased']))}")
            print(f"    unbiased: {short_text(str(item['unbiased']))}")
    if not any_difference:
        print("No differing outputs observed at the tested scales.")

    payload = {
        "script": "gen_shift",
        "model_name": args.model_name,
        "max_tokens": args.max_tokens,
        "scales": SCALES,
        "prompts": PROMPTS,
        "summary": scale_summaries,
        "records": records,
    }
    write_json(output_path, payload)
    print()
    print(f"saved_json={output_path}")


if __name__ == "__main__":
    main()
