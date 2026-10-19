"""llm_gateway — multi-provider LLM gateway with FinOps accounting.

Core value: route every completion request through a purpose-based fallback
chain, record every attempt in a cost ledger, and enforce privacy gating for
sensitive workloads — the operational discipline that keeps agent fleets alive.
"""

from .core import Backend, Gateway, CompletionResult
from .config import GatewayConfig, load_config
from .finops import CostLedger

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "Gateway",
    "CompletionResult",
    "GatewayConfig",
    "load_config",
    "CostLedger",
    "__version__",
]
