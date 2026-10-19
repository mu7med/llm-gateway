"""YAML configuration loading for the gateway.

Config shape (config/gateway.yaml):

    backends:
      - name: provider-a
        type: openai_compat
        base_url: https://api.provider-a.com/v1
        api_key_env: PROVIDER_A_API_KEY
        models:
          fast-small: {input_cost_per_1k: 0.0001, output_cost_per_1k: 0.0002}
    routes:
      code_complex: ["provider-a:big-model", "provider-b:backup"]
    sensitive_purposes: [code_complex, security]
    sensitive_excluded_backends: [data-training-provider]
    disabled_models: [broken-model-1]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from .core import Backend, GatewayConfig, OpenAICompatBackend


def load_config(path: Optional[str] = None) -> GatewayConfig:
    """Load a GatewayConfig from YAML. `path` defaults to
    $LLM_GATEWAY_CONFIG or ./config/gateway.yaml."""
    path = path or os.environ.get("LLM_GATEWAY_CONFIG") or "config/gateway.yaml"
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"gateway config not found: {file}")
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}

    backends: dict[str, Backend] = {}
    for spec in raw.get("backends", []):
        btype = spec.get("type", "openai_compat")
        if btype == "openai_compat":
            backends[spec["name"]] = OpenAICompatBackend(
                name=spec["name"],
                base_url=spec["base_url"],
                api_key_env=spec.get("api_key_env", f"{spec['name'].upper()}_API_KEY"),
                timeout=int(spec.get("timeout", 120)),
                models=spec.get("models") or {},
            )
        else:
            raise ValueError(f"unknown backend type: {btype!r}")

    return GatewayConfig(
        backends=backends,
        routes={p: list(entries) for p, entries in raw.get("routes", {}).items()},
        sensitive_purposes=set(raw.get("sensitive_purposes", [])),
        sensitive_excluded_backends=set(raw.get("sensitive_excluded_backends", [])),
        disabled_models=set(raw.get("disabled_models", [])),
        default_purpose=raw.get("default_purpose", "default"),
    )
