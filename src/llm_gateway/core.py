"""Core gateway engine: backends, routing, fallback chains, circuit breaking.

Design principles (proven in production on a personal multi-agent fleet):
- Requests-only transport: no vendor SDKs, safe in minimal/cron environments.
- Purpose-based chains: every request declares an intent; routing tables map
  intent -> ordered provider/model chain. One table change re-routes a fleet.
- Per-attempt fallback: a failure burns one entry, never the request.
- Privacy gating: sensitive purposes drop providers whose terms allow training
  on prompts. This is a guarantee, not a tuning knob.
- Cost ledger: every attempt (success or failure) is recorded with tokens,
  latency, and computed cost. FinOps is a first-class output, not an afterthought.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class CompletionRequest:
    """A single completion request with its declared purpose (route)."""
    prompt: str
    purpose: str = "default"
    system: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    metadata: dict = field(default_factory=dict)


@dataclass
class CompletionResult:
    """Outcome of a successful completion through the gateway."""
    content: str
    backend: str
    model: str
    purpose: str
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_total: int = 0
    cost_usd: Optional[float] = None  # None = pricing unknown, never fake 0
    elapsed_seconds: float = 0.0
    attempts_failed: int = 0
    fallback_used: bool = False


@dataclass
class AttemptRecord:
    """One attempt against one backend entry."""
    backend: str
    model: str
    success: bool
    elapsed_seconds: float = 0.0
    error: str = ""
    error_type: str = ""


class BackendError(Exception):
    """Raised by a backend when a completion attempt fails."""

    def __init__(self, backend: str, model: str, message: str,
                 error_type: str = "unknown", retryable: bool = True):
        super().__init__(message)
        self.backend = backend
        self.model = model
        self.error_type = error_type
        self.retryable = retryable


class AllBackendsFailedError(Exception):
    """Every entry in the chain failed. Carries the full attempt trail."""

    def __init__(self, purpose: str, attempts: list[AttemptRecord]):
        lines = [f"  {a.backend}:{a.model} -> {a.error_type}: {a.error}" for a in attempts]
        super().__init__(f"ALL BACKENDS FAILED for purpose '{purpose}':\n" + "\n".join(lines))
        self.purpose = purpose
        self.attempts = attempts


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class Backend(ABC):
    """A single LLM provider endpoint."""

    name: str = "abstract"

    @abstractmethod
    def complete(self, request: CompletionRequest, model: str) -> CompletionResult:
        """Execute a completion. Raises BackendError on failure."""

    def health_check(self) -> dict:
        """Cheap liveness probe. Backends may override."""
        return {"backend": self.name, "ok": True, "detail": "no probe implemented"}


class OpenAICompatBackend(Backend):
    """OpenAI-compatible chat-completions backend (works with the majority of
    providers/aggregators). Key is read from the environment — never hard-coded."""

    def __init__(self, name: str, base_url: str, api_key_env: str,
                 timeout: int = 120, models: Optional[dict[str, dict]] = None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.models = models or {}  # model_id -> {"input_cost_per_1k": x, "output_cost_per_1k": y}

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise BackendError(self.name, "", f"env var {self.api_key_env} not set",
                               error_type="no_key", retryable=False)
        return key

    def complete(self, request: CompletionRequest, model: str) -> CompletionResult:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        body = {"model": model, "messages": messages, "max_tokens": request.max_tokens}
        if request.temperature is not None:
            body["temperature"] = request.temperature

        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key()}",
                         "Content-Type": "application/json"},
                json=body, timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise BackendError(self.name, model, str(exc)[:200],
                               error_type="network") from exc

        elapsed = time.time() - start
        if resp.status_code != 200:
            err_type, err_msg = _parse_error(resp.text)
            raise BackendError(self.name, model, err_msg,
                               error_type=err_type or f"http_{resp.status_code}",
                               retryable=resp.status_code in (429, 500, 502, 503, 504))

        data = resp.json()
        content = ((data.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
        usage = data.get("usage") or {}
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        pricing = self.models.get(model, {})
        cost = None
        if pricing:
            cost = round(
                tokens_in / 1000 * pricing.get("input_cost_per_1k", 0)
                + tokens_out / 1000 * pricing.get("output_cost_per_1k", 0), 6,
            )

        return CompletionResult(
            content=content, backend=self.name, model=model,
            purpose=request.purpose, tokens_in=tokens_in, tokens_out=tokens_out,
            tokens_total=usage.get("total_tokens", tokens_in + tokens_out),
            cost_usd=cost, elapsed_seconds=round(elapsed, 2),
        )

    def health_check(self) -> dict:
        """Probe with a minimal request against the first configured model."""
        model = next(iter(self.models), None)
        if not model:
            return {"backend": self.name, "ok": False, "detail": "no models configured"}
        try:
            result = self.complete(
                CompletionRequest(prompt="Reply with exactly: OK",
                                  max_tokens=8, purpose="health"),
                model,
            )
            return {"backend": self.name, "ok": bool(result.content),
                    "model": model, "latency_seconds": result.elapsed_seconds}
        except BackendError as exc:
            return {"backend": self.name, "ok": False, "error": exc.error_type,
                    "detail": str(exc)[:160]}


def _parse_error(body: str) -> tuple[str, str]:
    """Extract (error_type, message) from an OpenAI-compatible error body."""
    err_type, err_msg = "unknown", (body or "")[:160]
    try:
        import json
        data = json.loads(body)
        err = data.get("error", {})
        if isinstance(err, dict):
            err_type = str(err.get("type") or err.get("code") or "unknown")
            err_msg = (err.get("message") or "")[:160]
    except Exception:
        pass
    return err_type, err_msg


# --------------------------------------------------------------------------- #
# Routing configuration
# --------------------------------------------------------------------------- #

@dataclass
class ChainEntry:
    """One attemptable entry in a routing chain."""
    backend: str
    model: str

    @classmethod
    def parse(cls, raw: str) -> "ChainEntry":
        """Parse 'backend:model' shorthand. Both parts must be non-empty."""
        if ":" not in raw:
            raise ValueError(f"chain entry must be 'backend:model', got: {raw!r}")
        backend, model = raw.split(":", 1)
        backend, model = backend.strip(), model.strip()
        if not backend or not model:
            raise ValueError(f"chain entry has empty backend or model: {raw!r}")
        return cls(backend=backend, model=model)


@dataclass
class GatewayConfig:
    """Declarative gateway configuration (typically loaded from YAML)."""
    backends: dict[str, Backend] = field(default_factory=dict)
    routes: dict[str, list[str]] = field(default_factory=dict)  # purpose -> ["be:model", ...]
    sensitive_purposes: set[str] = field(default_factory=set)
    sensitive_excluded_backends: set[str] = field(default_factory=set)
    disabled_models: set[str] = field(default_factory=set)  # known-broken, always skip
    default_purpose: str = "default"

    def chain_for(self, purpose: str) -> list[ChainEntry]:
        """Resolve the filtered attempt chain for a purpose.

        Filtering order (each step is a policy, applied every time):
        1. unknown purpose -> default route
        2. privacy gate: sensitive purposes drop excluded backends
        3. broken-model gate: disabled models are skipped
        """
        purpose = purpose if purpose in self.routes else self.default_purpose
        entries = [ChainEntry.parse(raw) for raw in self.routes.get(purpose, [])]
        if purpose in self.sensitive_purposes:
            entries = [e for e in entries if e.backend not in self.sensitive_excluded_backends]
        entries = [e for e in entries if e.model not in self.disabled_models]
        return entries


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #

class CircuitBreaker:
    """Per-backend consecutive-failure breaker with cooldown.

    A backend that fails `threshold` times in a row is skipped for `cooldown`
    seconds — the chain reroutes to the next entry instead of burning latency
    on a dead endpoint. Half-open probing happens naturally: after cooldown,
    the backend becomes attemptable again and the breaker resets on success.
    """

    def __init__(self, threshold: int = 3, cooldown: float = 300.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self._consecutive_failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def is_open(self, backend: str) -> bool:
        opened = self._opened_at.get(backend)
        if opened is None:
            return False
        if time.time() - opened >= self.cooldown:
            # cooldown elapsed -> half-open: allow one attempt through
            del self._opened_at[backend]
            self._consecutive_failures[backend] = self.threshold - 1
            return False
        return True

    def record_success(self, backend: str) -> None:
        self._consecutive_failures[backend] = 0
        self._opened_at.pop(backend, None)

    def record_failure(self, backend: str) -> None:
        count = self._consecutive_failures.get(backend, 0) + 1
        self._consecutive_failures[backend] = count
        if count >= self.threshold and backend not in self._opened_at:
            self._opened_at[backend] = time.time()

    def status(self) -> dict:
        return {
            b: {"consecutive_failures": n,
                "open": self.is_open(b),
                "cooldown_remaining": round(self.cooldown - (time.time() - self._opened_at[b]), 1)
                if b in self._opened_at else 0.0}
            for b, n in self._consecutive_failures.items()
        }


# --------------------------------------------------------------------------- #
# Gateway
# --------------------------------------------------------------------------- #

class Gateway:
    """The routing engine. Walk a purpose's chain with per-attempt fallback,
    enforce gates, break circuits, and emit a complete attempt trail."""

    def __init__(self, config: GatewayConfig,
                 breaker: Optional[CircuitBreaker] = None,
                 on_attempt=None):
        self.config = config
        self.breaker = breaker or CircuitBreaker()
        self.on_attempt = on_attempt  # callback(AttemptRecord) for ledger/hooks

    def complete(self, request: CompletionRequest) -> CompletionResult:
        chain = self.config.chain_for(request.purpose)
        if not chain:
            raise AllBackendsFailedError(request.purpose, [])

        attempts: list[AttemptRecord] = []
        for entry in chain:
            backend = self.config.backends.get(entry.backend)
            if backend is None:
                attempts.append(AttemptRecord(entry.backend, entry.model, False,
                                              error="backend not configured",
                                              error_type="config"))
                self._emit(attempts[-1])
                continue
            if self.breaker.is_open(entry.backend):
                attempts.append(AttemptRecord(entry.backend, entry.model, False,
                                              error="circuit open — skipped",
                                              error_type="circuit_open"))
                self._emit(attempts[-1])
                continue

            start = time.time()
            try:
                result = backend.complete(request, entry.model)
            except BackendError as exc:
                record = AttemptRecord(entry.backend, entry.model, False,
                                       elapsed_seconds=round(time.time() - start, 2),
                                       error=str(exc)[:200], error_type=exc.error_type)
                attempts.append(record)
                self.breaker.record_failure(entry.backend)
                self._emit(record)
                continue

            self.breaker.record_success(entry.backend)
            result.attempts_failed = len([a for a in attempts if not a.success])
            result.fallback_used = result.attempts_failed > 0
            success_record = AttemptRecord(entry.backend, entry.model, True,
                                           elapsed_seconds=result.elapsed_seconds)
            attempts.append(success_record)
            self._emit(success_record, result=result)
            return result

        raise AllBackendsFailedError(request.purpose, attempts)

    def health_check(self) -> dict:
        return {name: backend.health_check()
                for name, backend in self.config.backends.items()}

    def _emit(self, record: AttemptRecord, result: Optional[CompletionResult] = None):
        if self.on_attempt:
            try:
                self.on_attempt(record, result)
            except Exception:
                pass  # hooks must never break routing
