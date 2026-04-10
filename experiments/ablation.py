from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import default_output_path, make_bridge, make_engine, short_text, similarity_ratio, write_json

PROMPTS = [
    "Describe a person who keeps every promise except the ones made to themselves.",
    "Give a concise explanation of opportunity cost with one everyday example.",
    "Write four lines of dialogue between a cartographer and a storm.",
    "What makes an apology sound sincere instead of strategic?",
    "Plan a one-hour study session for someone who feels scattered.",
    "Summarize the argument for free will and against it in simple language.",
    "Invent a product slogan for a notebook that can survive a flood.",
    "Answer this as a careful teacher: why do we round numbers at all?",
]

CONDITIONS = [
    {"key": "full_engine", "label": "full engine (scale=0.02, graph on)", "genome_scale": 0.02, "max_nodes": 20},
    {"key": "bias_no_graph", "label": "bias without graph (scale=0.02, graph empty)", "genome_scale": 0.02, "max_nodes": 0},
    {"key": "graph_high_scale", "label": "full engine (scale=0.05, graph on)", "genome_scale": 0.05, "max_nodes": 20},
    {"key": "baseline", "label": "baseline (scale=0.0, graph empty)", "genome_scale": 0.0, "max_nodes": 0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate graph and KV bias contributions across fixed prompts.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def run_condition(model_name: str, condition: dict[str, object], bridge, max_tokens: int) -> dict[str, object]:
    engine = make_engine(
        model_name,
        bridge=bridge,
        genome_scale=float(condition["genome_scale"]),
        max_nodes=int(condition["max_nodes"]),
        max_response_tokens=max_tokens,
    )

    outputs: list[dict[str, object]] = []
    for prompt_index, prompt in enumerate(PROMPTS, start=1):
        turn = engine.respond(prompt)
        outputs.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "reply": turn.reply,
                "metrics": turn.metrics,
            }
        )

    return {
        "key": condition["key"],
        "label": condition["label"],
        "genome_scale": condition["genome_scale"],
        "max_nodes": condition["max_nodes"],
        "outputs": outputs,
    }


def compare_conditions(results: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {result["key"]: result for result in results}
    comparisons: list[dict[str, object]] = []
    for left_key, right_key in itertools.combinations(by_key, 2):
        left_outputs = by_key[left_key]["outputs"]
        right_outputs = by_key[right_key]["outputs"]
        prompt_pairs: list[dict[str, object]] = []
        for left_item, right_item in zip(left_outputs, right_outputs):
            left_reply = str(left_item["reply"])
            right_reply = str(right_item["reply"])
            prompt_pairs.append(
                {
                    "prompt_index": left_item["prompt_index"],
                    "identical": left_reply == right_reply,
                    "similarity": similarity_ratio(left_reply, right_reply),
                    "left_reply": left_reply,
                    "right_reply": right_reply,
                }
            )

        identical_indices = [item["prompt_index"] for item in prompt_pairs if item["identical"]]
        different_indices = [item["prompt_index"] for item in prompt_pairs if not item["identical"]]
        average_similarity = sum(item["similarity"] for item in prompt_pairs) / len(prompt_pairs)
        comparisons.append(
            {
                "left": left_key,
                "right": right_key,
                "identical_indices": identical_indices,
                "different_indices": different_indices,
                "identical_count": len(identical_indices),
                "different_count": len(different_indices),
                "average_similarity": average_similarity,
                "sample_differences": [item for item in prompt_pairs if not item["identical"]][:2],
            }
        )
    return comparisons


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path("ablation")
    bridge = make_bridge(args.model_name)

    print(f"model={args.model_name}")
    print(f"prompts={len(PROMPTS)} | max_tokens={args.max_tokens}")
    print()
    print("Running conditions")
    print("-" * 72)

    results: list[dict[str, object]] = []
    for condition in CONDITIONS:
        print(f"{condition['key']}: {condition['label']}")
        results.append(run_condition(args.model_name, condition, bridge, args.max_tokens))

    comparisons = compare_conditions(results)

    print()
    print("Pairwise similarity")
    print("-" * 72)
    for comparison in comparisons:
        print(
            f"{comparison['left']} vs {comparison['right']} | "
            f"identical={comparison['identical_count']}/{len(PROMPTS)} | "
            f"avg_similarity={comparison['average_similarity']:.3f}"
        )
        print(
            f"  identical_prompts={comparison['identical_indices']} | "
            f"different_prompts={comparison['different_indices']}"
        )
        for sample in comparison["sample_differences"]:
            prompt = PROMPTS[int(sample["prompt_index"]) - 1]
            print(f"  prompt {sample['prompt_index']}: {short_text(prompt, 90)}")
            print(f"    left:  {short_text(str(sample['left_reply']))}")
            print(f"    right: {short_text(str(sample['right_reply']))}")

    payload = {
        "script": "ablation",
        "model_name": args.model_name,
        "max_tokens": args.max_tokens,
        "prompts": PROMPTS,
        "conditions": results,
        "comparisons": comparisons,
    }
    write_json(output_path, payload)
    print()
    print(f"saved_json={output_path}")


if __name__ == "__main__":
    main()
