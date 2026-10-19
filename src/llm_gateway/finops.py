"""FinOps cost ledger — the first-class output of the gateway.

Every attempt (success or failure) is appended to a JSONL ledger with
computed cost. This is what turns "we use LLMs" into "we know what LLMs
cost us per purpose, per model, per day" — LLM FinOps.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .core import AttemptRecord, CompletionResult


class CostLedger:
    """Append-only JSONL cost ledger with in-memory aggregates.

    Thread-safe: multiple agents/processes can route through one ledger.
    Cost is computed from backend pricing tables when known; entries with
    unknown pricing carry cost_usd=null — never a fake zero.
    """

    def __init__(self, path: Optional[str] = None):
        path = path or os.environ.get("LLM_GATEWAY_LEDGER") or "ledger.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._totals = defaultdict(lambda: {"requests": 0, "tokens_in": 0,
                                            "tokens_out": 0, "cost_usd": 0.0})

    def record_attempt(self, record: AttemptRecord,
                       result: Optional[CompletionResult] = None) -> None:
        entry = {
            "ts": time.time(),
            "backend": record.backend,
            "model": record.model,
            "success": record.success,
            "error_type": record.error_type or None,
        }
        if result is not None:
            entry.update({
                "purpose": result.purpose,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "tokens_total": result.tokens_total,
                "cost_usd": result.cost_usd,  # None stays None — unknown, not free
                "elapsed_seconds": result.elapsed_seconds,
                "fallback_used": result.fallback_used,
                "attempts_failed": result.attempts_failed,
            })
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if result is not None:
                key = (result.backend, result.model)
                self._totals[key]["requests"] += 1
                self._totals[key]["tokens_in"] += result.tokens_in
                self._totals[key]["tokens_out"] += result.tokens_out
                if result.cost_usd is not None:
                    self._totals[key]["cost_usd"] += result.cost_usd

    def summary(self, since_ts: Optional[float] = None) -> list[dict]:
        """Aggregate cost by backend+model, optionally since a timestamp.
        Reads from disk so multiple processes stay consistent."""
        agg: dict = defaultdict(lambda: {"requests": 0, "tokens_in": 0,
                                         "tokens_out": 0, "cost_usd": 0.0,
                                         "unknown_cost_requests": 0})
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn write — skip, never crash accounting
                if not e.get("success"):
                    continue
                if since_ts and e.get("ts", 0) < since_ts:
                    continue
                key = (e.get("backend", "?"), e.get("model", "?"))
                a = agg[key]
                a["requests"] += 1
                a["tokens_in"] += e.get("tokens_in", 0)
                a["tokens_out"] += e.get("tokens_out", 0)
                cost = e.get("cost_usd")
                if cost is None:
                    a["unknown_cost_requests"] += 1
                else:
                    a["cost_usd"] += cost
        return [
            {"backend": b, "model": m, **stats}
            for (b, m), stats in sorted(agg.items(), key=lambda kv: -kv[1]["cost_usd"])
        ]

    def summary_by_purpose(self) -> list[dict]:
        """Aggregate cost by declared purpose — the FinOps question that
        actually matters: which workload burns the budget?"""
        agg: dict = defaultdict(lambda: {"requests": 0, "tokens_in": 0,
                                         "tokens_out": 0, "cost_usd": 0.0})
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not e.get("success") or "purpose" not in e:
                    continue
                key = e.get("purpose", "default")
                a = agg[key]
                a["requests"] += 1
                a["tokens_in"] += e.get("tokens_in", 0)
                a["tokens_out"] += e.get("tokens_out", 0)
                if e.get("cost_usd") is not None:
                    a["cost_usd"] += e["cost_usd"]
        return [
            {"purpose": p, **stats}
            for p, stats in sorted(agg.items(), key=lambda kv: -kv[1]["cost_usd"])
        ]

