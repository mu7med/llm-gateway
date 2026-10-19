# llm-gateway

**Multi-provider LLM gateway with purpose-based routing, fallback chains, circuit breaking, privacy gating, and first-class cost accounting (LLM FinOps).**

Built from patterns running 24/7 in production on a personal multi-agent fleet
(Termux + Windows + WSL agents routing across 10+ model endpoints daily).

## Why

Most agent failures in production are not model failures — they are *routing*
failures: a provider has an outage, a quota lapses, a model gets deprecated,
or a sensitive prompt lands on a provider that trains on your data. This
gateway treats those as first-class engineering problems:

- **Purpose-based chains** — every request declares an intent (`code_complex`,
  `summarization`, `high_volume_batch`…); a declarative table maps intent →
  ordered `backend:model` chain. One YAML change re-routes a whole fleet.
- **Per-attempt fallback** — a provider failure burns one chain entry, never
  the request. Full attempt trail is returned and logged.
- **Circuit breaker** — a backend that fails N times in a row is skipped for a
  cooldown window instead of poisoning every request's latency.
- **Privacy gating** — sensitive purposes (`security`, `production`,
  `code_complex`…) automatically drop providers whose terms permit training on
  prompts. Policy, not a suggestion.
- **Cost ledger (FinOps)** — every attempt is appended to a JSONL ledger with
  computed cost. Unknown pricing is recorded as `null` — never a fake zero.
  Aggregations: by model, by purpose ("which workload burns the budget?").

## Architecture

```
                        ┌────────────────────────────────────────┐
   request (purpose) ──▶│              Gateway                   │
                        │  1. resolve chain for purpose          │
                        │  2. privacy gate (sensitive purposes)  │
                        │  3. broken-model gate                  │
                        │  4. circuit-breaker gate               │
                        │  5. walk chain, fallback per attempt   │
                        └───────┬───────────────────────┬────────┘
                                │                       │
                     every attempt                      success
                                ▼                       ▼
                        ┌───────────────┐      ┌──────────────────┐
                        │  CostLedger   │      │ CompletionResult │
                        │ (JSONL, cost) │      │ content+cost+trail│
                        └───────┬───────┘      └──────────────────┘
                                ▼
                      finops summary (by model / by purpose)
```

## Install

```bash
pip install -e .            # library + CLI (deps: requests, pyyaml)
pip install -e ".[server]"  # + FastAPI OpenAI-compatible server
pip install -e ".[dev]"     # + pytest
```

## Quickstart

```bash
export PROVIDER_MAIN_API_KEY=sk-...
cp config/gateway.example.yaml config/gateway.yaml   # adapt backends/routes

# Decision mode: what would run, without executing?
llm-gateway --purpose code_complex --decision

# Execute with fallback + cost accounting
llm-gateway --purpose summarization --prompt "Summarize: ..." --system "Be terse"

# Health probes on all backends
llm-gateway --health

# FinOps: cost by model and by purpose
llm-gateway --finops
```

### Library use

```python
from llm_gateway import Gateway, CompletionRequest, CostLedger, load_config

cfg = load_config("config/gateway.yaml")
ledger = CostLedger(path="ledger.jsonl")
gw = Gateway(cfg, on_attempt=ledger.record_attempt)

result = gw.complete(CompletionRequest(
    prompt="Explain circuit breakers", purpose="reasoning"))

print(result.model, result.cost_usd, result.fallback_used)
```

### OpenAI-compatible server

```bash
uvicorn llm_gateway.server:app --port 8080
# Point any OpenAI SDK / agent framework at it:
#   base_url = "http://localhost:8080/v1"
curl -s localhost:8080/finops/summary | python3 -m json.tool
```

## Configuration

See `config/gateway.example.yaml` — fully commented. Key sections:

- `backends` — OpenAI-compatible endpoints, key names via `api_key_env`
  (keys come from the environment, never the config file), per-model pricing
  for cost computation.
- `routes` — purpose → ordered `backend:model` chain.
- `sensitive_purposes` + `sensitive_excluded_backends` — the privacy gate.
- `disabled_models` — known-broken models, skipped at resolution time.

## Testing

```bash
python -m pytest tests/ -v   # 23 offline tests: chains, gates, breaker, fallback, ledger
```

## Design notes

- **Requests-only transport.** No vendor SDKs — the gateway must run in cron
  containers and minimal environments. This mirrors how the production router
  it was extracted from survives on edge hardware.
- **Cost `null` ≠ cost `0`.** Unknown pricing is surfaced, not hidden — silent
  freebies are how LLM budgets leak.
- **Hooks never break routing.** `on_attempt` callbacks are wrapped; a failing
  metrics hook degrades observability, never availability.

## Roadmap

- [x] Core routing engine + gates + breaker + ledger (v0.1)
- [x] OpenAI-compatible FastAPI server
- [ ] Prometheus metrics endpoint (see companion project **agentops-observatory**)
- [ ] Complexity-aware escalation (classify → cheap-first routing)
- [ ] Response caching + token budget caps per purpose

## License

MIT © Mohammed Abdullah
