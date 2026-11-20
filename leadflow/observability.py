"""Optional Arize Phoenix tracing for buyer/routing LLM calls and eval runs.

Off by default. Enable with LEADFLOW_PHOENIX=1 (or --phoenix on eval_buyer /
eval_routing / eval_capacity / eval_components). Requires the observability extras:
arize-phoenix-otel, openinference-instrumentation-anthropic, and
arize-phoenix-client. Point at a local Phoenix (`phoenix serve`) or Arize-hosted
Phoenix via PHOENIX_COLLECTOR_ENDPOINT / PHOENIX_API_KEY.

span() and set_output() group a lead's work into one trace: a span per lead, a child span per
stage, and the Claude calls inside them. Both are no-ops while tracing is off.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

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


@contextmanager
def span(name: str, kind: str = "CHAIN", input: Any = None, **attributes: Any) -> Iterator[Any]:
    """A span around a pipeline stage, nested under the current span. Yields None while tracing is off.

    kind is an OpenInference span kind (CHAIN, AGENT, TOOL...). Attributes with a None value are skipped.
    An exception inside the block marks the span as an error.
    """
    if _TRACER_PROVIDER is None:
        yield None
        return
    from opentelemetry.trace import Status, StatusCode

    with _TRACER_PROVIDER.get_tracer("leadflow").start_as_current_span(name) as current:
        current.set_attribute("openinference.span.kind", kind)
        if input is not None:
            current.set_attribute("input.value", _to_json(input))
            current.set_attribute("input.mime_type", "application/json")
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current
        current.set_status(Status(StatusCode.OK))


def set_output(current: Any, value: Any) -> None:
    """Record a stage's result on its span. Does nothing while tracing is off."""
    if current is None or value is None:
        return
    current.set_attribute("output.value", _to_json(value))
    current.set_attribute("output.mime_type", "application/json")


def _to_json(value: Any) -> str:
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    return json.dumps(value, default=str)
