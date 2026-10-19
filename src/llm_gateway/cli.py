"""CLI: decision mode, execution mode, health check, FinOps summary."""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .core import CompletionRequest, Gateway
from .finops import CostLedger


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-gateway",
        description="Multi-provider LLM gateway with purpose routing, fallback chains, and FinOps accounting.")
    parser.add_argument("--config", default=None, help="Path to gateway.yaml")
    parser.add_argument("--purpose", default="default", help="Routing purpose/route name")
    parser.add_argument("--prompt", default=None, help="Prompt (execution mode when given)")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ledger", default=None, help="Cost ledger path (default: ./ledger.jsonl)")
    parser.add_argument("--raw", action="store_true", help="Print only response content")
    parser.add_argument("--decision", action="store_true", help="Print the routing decision, don't execute")
    parser.add_argument("--health", action="store_true", help="Probe every configured backend")
    parser.add_argument("--finops", action="store_true", help="Print cost summary from the ledger")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ledger = CostLedger(path=args.ledger) if (args.prompt or args.finops) else None
    gw = Gateway(cfg, on_attempt=ledger.record_attempt if ledger else None)

    if args.finops:
        print(json.dumps({"by_model": gw and ledger.summary(),
                          "by_purpose": ledger.summary_by_purpose()}, indent=2))
        return 0

    if args.health:
        print(json.dumps(gw.health_check(), indent=2))
        return 0

    if args.decision:
        chain = gw.config.chain_for(args.purpose)
        print(json.dumps({"purpose": args.purpose,
                          "chain": [f"{e.backend}:{e.model}" for e in chain]}, indent=2))
        return 0

    if not args.prompt:
        parser.print_help()
        return 0

    try:
        result = gw.complete(CompletionRequest(
            prompt=args.prompt, purpose=args.purpose, system=args.system,
            max_tokens=args.max_tokens, temperature=args.temperature))
    except Exception as exc:
        print(f"gateway error: {exc}", file=sys.stderr)
        return 1

    if args.raw:
        print(result.content)
    else:
        print(f"[{result.purpose} -> {result.backend}:{result.model}] "
              f"tokens={result.tokens_total} cost=${result.cost_usd} "
              f"latency={result.elapsed_seconds}s fallback={result.fallback_used}")
        print(result.content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
