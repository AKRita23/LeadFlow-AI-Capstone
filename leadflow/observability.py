"""Optional Arize Phoenix tracing for buyer/routing LLM calls and eval runs.

Off by default. Enable with LEADFLOW_PHOENIX=1 (or --phoenix on eval_buyer).
Requires the observability extras: arize-phoenix-otel, openinference-instrumentation-anthropic,
and arize-phoenix-client. Point at a local Phoenix (`phoenix serve`) or Arize-hosted Phoenix
via PHOENIX_COLLECTOR_ENDPOINT / PHOENIX_API_KEY.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TRACER_PROVIDER = None


def phoenix_enabled() -> bool:
    return os.environ.get("LEADFLOW_PHOENIX", "").strip().lower() in {"1", "true", "yes", "on"}


def setup_phoenix(project_name: str = "leadflow-ai", force: bool = False) -> Optional[Any]:
    """Register OpenInference → Phoenix. Idempotent. Returns the tracer provider or None."""
    global _TRACER_PROVIDER
    if _TRACER_PROVIDER is not None:
        return _TRACER_PROVIDER
    if not force and not phoenix_enabled():
        return None
    try:
        from phoenix.otel import register
    except ImportError as exc:
        raise SystemExit(
            "Phoenix packages missing. Install with:\n"
            "  .venv/bin/pip install -r requirements-observability.txt"
        ) from exc

    # Prefer OpenInference spans over Anthropic SDK's built-in OTEL (avoids double-counting).
    os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")

    _TRACER_PROVIDER = register(
        project_name=project_name,
        auto_instrument=True,
    )
    logger.info("Phoenix tracing on (project=%s)", project_name)
    return _TRACER_PROVIDER


def get_tracer(name: str = "leadflow"):
    """OpenTelemetry tracer after setup_phoenix(); a no-op tracer otherwise."""
    from opentelemetry import trace

    return trace.get_tracer(name)


def anthropic_client(**kwargs: Any) -> Any:
    """Anthropic client that skips the SDK's own OTEL when OpenInference is active."""
    import anthropic

    kwargs.setdefault("max_retries", 1)
    try:
        return anthropic.Anthropic(open_telemetry=False, **kwargs)
    except TypeError:
        return anthropic.Anthropic(**kwargs)
