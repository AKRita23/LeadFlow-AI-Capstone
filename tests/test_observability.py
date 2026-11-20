import json

import pytest

pytest.importorskip("opentelemetry.sdk", reason="needs requirements-observability.txt")

from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from leadflow import observability
from leadflow.api import app, get_buyer_agent, get_routing_agent, get_store
from leadflow.buyer_agent import BuyerContextAgent
from leadflow.data_loader import load_enrichment, load_leads
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.routing import RoutingAgent
from leadflow.store import DecisionStore

ROWS = load_enrichment()
FINPAY = next(lead for lead in load_leads()[0] if lead.submission_id == "sub_8f92a1c4")


@pytest.fixture
def exporter(monkeypatch):
    """Turn tracing on with an in-memory exporter instead of Phoenix."""
    spans = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    monkeypatch.setattr(observability, "_TRACER_PROVIDER", provider)
    return spans


def test_spans_are_no_ops_when_tracing_is_off(monkeypatch):
    monkeypatch.setattr(observability, "_TRACER_PROVIDER", None)
    with observability.span("lead", input={"a": 1}) as current:
        observability.set_output(current, {"b": 2})
    assert current is None


def test_each_lead_gets_one_trace_with_a_child_span_per_stage(exporter, tmp_path):
    app.dependency_overrides[get_store] = lambda: DecisionStore(tmp_path / "leadflow.db")
    app.dependency_overrides[get_buyer_agent] = lambda: BuyerContextAgent(MockTheirStack(ROWS), MockPostHog(ROWS))
    app.dependency_overrides[get_routing_agent] = lambda: RoutingAgent()
    try:
        response = TestClient(app).post("/leads", json=json.loads(FINPAY.model_dump_json()))
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 202

    spans = {s.name: s for s in exporter.get_finished_spans()}
    lead = spans["lead"]
    assert lead.attributes["lead.submission_id"] == "sub_8f92a1c4"
    assert json.loads(lead.attributes["output.value"])["assigned_rep_id"] == response.json()["assigned_rep_id"]
    assert lead.status.is_ok
    for stage in ("triage", "buyer_agent", "capacity_service", "routing_agent", "commit"):
        assert spans[stage].parent.span_id == lead.context.span_id, stage
        assert spans[stage].context.trace_id == lead.context.trace_id
    assert spans["buyer_agent"].attributes["openinference.span.kind"] == "AGENT"


def test_a_failing_stage_is_recorded_as_an_error(exporter):
    with pytest.raises(RuntimeError):
        with observability.span("routing_agent", "AGENT"):
            raise RuntimeError("Claude timed out")
    (failed,) = exporter.get_finished_spans()
    assert failed.status.status_code.name == "ERROR"
