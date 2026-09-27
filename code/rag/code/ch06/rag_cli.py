"""Chapter 6 - rag_cli.py: ask the handbook from your terminal.

Run:  uv run python code/ch06/rag_cli.py "How long is parental leave?"
      uv run python code/ch06/rag_cli.py --interactive
      uv run python code/ch06/rag_cli.py --golden 5        # answer the first 5 golden questions
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate import answer  # noqa: E402
from retrieve import get_store  # noqa: E402

from ragbook import load_golden  # noqa: E402


def show(result: dict) -> None:
    print(f"\nA: {result['answer']}")
    print("   sources: " + ", ".join(f"[{s['n']}] {s['source']}" for s in result["sources"]))
    print(f"   {result['usage']['input_tokens']} in / {result['usage']['output_tokens']} out tokens, ${result['cost_usd']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--golden", type=int, metavar="N", help="answer the first N golden questions")
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    store = get_store()          # one connection, reused for every question
    total = 0.0

    if args.golden:
        for q in load_golden()[: args.golden]:
            print(f"\nQ: {q['question']}")
            r = answer(q["question"], k=args.k, store=store)
            show(r)
            print(f"   reference: {q['answer']}")
            total += r["cost_usd"]
        print(f"\ntotal cost ${total:.5f}")
    elif args.interactive:
        print("Ask the handbook. Empty line to quit.")
        while (q := input("\nQ: ").strip()):
            show(answer(q, k=args.k, store=store))
    else:
        show(answer(args.question or "How long is parental leave for a primary caregiver?", k=args.k, store=store))
