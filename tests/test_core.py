"""Unit tests for the gateway core: chains, gates, breaker, fallback, ledger.

All tests are offline — a FakeBackend simulates provider behavior, which also
demonstrates the Backend extension contract for custom backends.
"""

import json
import threading

import pytest

from llm_gateway.core import (
    AllBackendsFailedError,
    Backend,
    BackendError,
    ChainEntry,
    CircuitBreaker,
    CompletionRequest,
    CompletionResult,
    Gateway,
    GatewayConfig,
)
from llm_gateway.finops import CostLedger


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

class FakeBackend(Backend):
    """Scripted fake provider: raises for models in `fail_for`."""

    name = "fake"

    def __init__(self, fail_for=(), error_type="http_503"):
        self.fail_for = set(fail_for)
        self.error_type = error_type
        self.calls = []

    def complete(self, request, model):
        self.calls.append(model)
        if model in self.fail_for:
            raise BackendError(self.name, model, "simulated outage",
                               error_type=self.error_type)
        return CompletionResult(
            content=f"ok-from-{model}", backend=self.name, model=model,
            purpose=request.purpose, tokens_in=10, tokens_out=5,
            tokens_total=15, cost_usd=0.0001, elapsed_seconds=0.01,
        )


def make_config(fail_for=(), routes=None):
    cfg = GatewayConfig(
        backends={"fake": FakeBackend(fail_for=fail_for),
                  "spare": FakeBackend(fail_for=fail_for)},
        routes=routes or {
            "default": ["fake:primary", "spare:backup"],
            "code_complex": ["fake:big", "spare:backup"],
            "summarization": ["fake:primary", "spare:backup"],
        },
        sensitive_purposes={"code_complex"},
        sensitive_excluded_backends={"spare"},
        disabled_models={"dead-model"},
    )
    return cfg, cfg.backends["fake"]


# --------------------------------------------------------------------------- #
# Chain resolution & gates
# --------------------------------------------------------------------------- #

class TestChains:
    def test_unknown_purpose_falls_back_to_default(self):
        cfg, _ = make_config()
        gw = Gateway(cfg)
        chain = gw.config.chain_for("does_not_exist")
        assert [e.model for e in chain] == ["primary", "backup"]

    def test_sensitive_purpose_drops_excluded_backend(self):
        cfg, _ = make_config()
        chain = Gateway(cfg).config.chain_for("code_complex")
        assert all(e.backend != "spare" for e in chain), \
            "privacy gate must remove sensitive-excluded backends"

    def test_disabled_models_are_skipped(self):
        cfg, _ = make_config(routes={"default": ["fake:primary", "fake:dead-model", "spare:backup"]})
        chain = Gateway(cfg).config.chain_for("default")
        assert [e.model for e in chain] == ["primary", "backup"]


class TestChainEntry:
    def test_parse_shorthand(self):
        e = ChainEntry.parse("openrouter:meta-llama/llama-3")
        assert e.backend == "openrouter" and e.model == "meta-llama/llama-3"

    @pytest.mark.parametrize("bad", ["no-colon", "", ":model", "be:"])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            ChainEntry.parse(bad)


# --------------------------------------------------------------------------- #
# Routing / fallback behavior
# --------------------------------------------------------------------------- #

class TestRouting:
    def test_first_healthy_backend_wins(self):
        cfg, _ = make_config()
        gw = Gateway(cfg)
        result = gw.complete(CompletionRequest(prompt="hi", purpose="default"))
        assert result.model == "primary" and result.fallback_used is False

    def test_fallback_on_failure(self):
        cfg, fb = make_config(fail_for={"primary"})
        gw = Gateway(cfg)
        result = gw.complete(CompletionRequest(prompt="hi", purpose="default"))
        assert result.model == "backup"
        assert result.fallback_used is True
        assert result.attempts_failed == 1

    def test_all_fail_raises_with_trail(self):
        cfg, _ = make_config(fail_for={"primary", "backup"})
        gw = Gateway(cfg)
        with pytest.raises(AllBackendsFailedError) as excinfo:
            gw.complete(CompletionRequest(prompt="hi"))
        assert "ALL BACKENDS FAILED" in str(excinfo.value)
        assert len(excinfo.value.attempts) == 2

    def test_attempt_hook_receives_every_attempt(self):
        cfg, _ = make_config(fail_for={"primary"})
        events = []
        gw = Gateway(cfg, on_attempt=lambda rec, res: events.append((rec.backend, rec.success)))
        gw.complete(CompletionRequest(prompt="hi"))
        assert ("fake", False) in events and ("spare", True) in events

    def test_fallback_across_backends_preserves_purpose(self):
        cfg, _ = make_config(fail_for={"primary"})
        result = Gateway(cfg).complete(CompletionRequest(prompt="x", purpose="summarization"))
        assert result.purpose == "summarization"


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #

class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        br = CircuitBreaker(threshold=2, cooldown=60)
        br.record_failure("a"); br.record_failure("a")
        assert br.is_open("a") is True

    def test_resets_on_success(self):
        br = CircuitBreaker(threshold=2, cooldown=60)
        br.record_failure("a")
        br.record_success("a")
        br.record_failure("a")
        assert br.is_open("a") is False

    def test_cooldown_half_open(self):
        br = CircuitBreaker(threshold=1, cooldown=0.01)
        br.record_failure("a")
        assert br.is_open("a") is True
        threading.Event().wait(0.02)
        assert br.is_open("a") is False  # half-open after cooldown

    def test_gateway_skips_open_circuit(self):
        cfg, _ = make_config()
        gw = Gateway(cfg)
        gw.breaker.record_failure("fake"); gw.breaker.record_failure("fake")
        gw.breaker.record_failure("fake")
        result = gw.complete(CompletionRequest(prompt="hi"))
        assert result.model == "backup"

    def test_status_shape(self):
        br = CircuitBreaker()
        br.record_failure("x")
        st = br.status()["x"]
        assert st["consecutive_failures"] == 1 and st["open"] is False


# --------------------------------------------------------------------------- #
# FinOps ledger
# --------------------------------------------------------------------------- #

class TestCostLedger:
    def test_records_success_and_failure(self, tmp_path):
        ledger = CostLedger(path=str(tmp_path / "ledger.jsonl"))
        cfg, _ = make_config()
        gw = Gateway(cfg, on_attempt=ledger.record_attempt)
        gw.complete(CompletionRequest(prompt="hi"))
        lines = [json.loads(l) for l in (tmp_path / "ledger.jsonl").read_text().splitlines()]
        assert any(e["success"] for e in lines)
        assert lines[0]["cost_usd"] == 0.0001

    def test_unknown_pricing_stays_null_not_zero(self, tmp_path):
        ledger = CostLedger(path=str(tmp_path / "l.jsonl"))
        class NoPricing(FakeBackend):
            def complete(self, request, model):
                r = super().complete(request, model)
                r.cost_usd = None  # pricing unknown
                return r
        cfg = GatewayConfig(
            backends={"np": NoPricing()},
            routes={"default": ["np:m1"]},
        )
        Gateway(cfg, on_attempt=ledger.record_attempt).complete(CompletionRequest(prompt="hi"))
        entry = json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])
        assert entry["cost_usd"] is None, "unknown pricing must be null, never fake zero"

    def test_summary_aggregates_by_backend_model(self, tmp_path):
        ledger = CostLedger(path=str(tmp_path / "l.jsonl"))
        cfg, _ = make_config(fail_for={"primary"})
        Gateway(cfg, on_attempt=ledger.record_attempt).complete(CompletionRequest(prompt="hi"))
        rows = ledger.summary()
        backup_row = next(r for r in rows if r["model"] == "backup")
        assert backup_row["requests"] == 1
        assert backup_row["cost_usd"] == 0.0001

    def test_summary_by_purpose(self, tmp_path):
        ledger = CostLedger(path=str(tmp_path / "l.jsonl"))
        cfg, _ = make_config()
        gw = Gateway(cfg, on_attempt=ledger.record_attempt)
        gw.complete(CompletionRequest(prompt="hi", purpose="summarization"))
        rows = ledger.summary_by_purpose()
        assert rows[0]["purpose"] == "summarization"
        assert rows[0]["requests"] == 1

    def test_torn_write_is_skipped(self, tmp_path):
        ledger = CostLedger(path=str(tmp_path / "l.jsonl"))
        (tmp_path / "l.jsonl").write_text('{"success": true, "backend": "b", "model": "m"}\nGARBAGE\n')
        rows = ledger.summary()
        assert len(rows) == 1
