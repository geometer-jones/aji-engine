from __future__ import annotations

import argparse

from .aji_engine import AjiEngine, AjiEngineConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aji-Engine with a complex self-graph over Qwen2.5-1.5B.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-response-tokens", type=int, default=128)
    parser.add_argument("--injection-layers", type=int, default=3)
    parser.add_argument("--genome-rank", type=int, default=32)
    parser.add_argument("--genome-scale", type=float, default=0.02)
    parser.add_argument("--max-nodes", type=int, default=20)
    parser.add_argument("--buffer-capacity", type=int, default=64)
    parser.add_argument("--state-path", default="aji_state.pkl")
    parser.add_argument("--metrics-path", default="aji_metrics.jsonl")
    parser.add_argument("--prompt", default=None, help="Run one turn and exit.")
    return parser.parse_args()


def _make_engine(args: argparse.Namespace) -> AjiEngine:
    engine = AjiEngine(
        AjiEngineConfig(
            model_name=args.model_name,
            max_response_tokens=args.max_response_tokens,
            injection_layers=args.injection_layers,
            genome_rank=args.genome_rank,
            genome_scale=args.genome_scale,
            max_nodes=args.max_nodes,
            buffer_capacity=args.buffer_capacity,
            state_path=args.state_path,
            metrics_path=args.metrics_path,
        )
    )
    engine.load_state()
    return engine


def main() -> None:
    args = _parse_args()
    engine = _make_engine(args)
    print(
        f"chat mode active | loaded_state={engine.loaded_from_state} "
        f"| active_nodes={engine.self_graph.num_active} "
        f"| sleep_cycles={engine.sleep_cycle_count}"
    )
    if engine.last_state_load_error:
        print(f"state load skipped: {engine.last_state_load_error}")
    print("commands: /exit, /state, /sleep, /probe, /eval <prompt>")

    if args.prompt:
        turn = engine.respond(args.prompt)
        print(turn.reply)
        print(turn.metrics)
        return

    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            engine.save_state()
            print()
            break

        if not user_input:
            continue
        if user_input in {"/exit", "exit", "quit"}:
            engine.save_state()
            break
        if user_input == "/state":
            print(engine.current_metrics())
            continue
        if user_input == "/sleep":
            print(engine.force_sleep())
            continue
        if user_input == "/probe":
            print(engine.force_probe())
            continue
        if user_input.startswith("/eval "):
            prompt = user_input[len("/eval ") :].strip()
            print(engine.evaluate_prompt(prompt))
            continue

        turn = engine.respond(user_input)
        print(f"aji> {turn.reply}")
        print(turn.metrics)


if __name__ == "__main__":
    main()
