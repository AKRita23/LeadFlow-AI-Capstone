"""Inbound submission API.

Run locally with:  .venv/bin/uvicorn leadflow.api:app --reload
Interactive docs:  http://127.0.0.1:8000/docs

Schema validation happens in the Lead model: a submission that breaks the
Inbound Form schema is rejected with HTTP 422 and a per-field error list.
A valid submission is triaged by the deterministic stages and recorded in
SQLite. The response carries the triage outcome as status. New prospects
also get a buyer assessment (Claude extracts the note signals when
ANTHROPIC_API_KEY is set, rules do otherwise), a capacity matrix for the
shortlisted reps, and a routing proposal (Claude when a key is set, with the
deterministic ranker as the fallback). Nothing is assigned yet: the proposal
is returned for the commit step.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple

from fastapi import Depends, FastAPI, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from leadflow.buyer_agent import BuyerContextAgent
from leadflow.capacity import CapacityService
from leadflow.contracts import BuyerAssessment, CapacityMatrix, RoutingOutcome
from leadflow.data_loader import TerritoryMap, load_accounts, load_enrichment, load_reps, load_territories
from leadflow.demo import demo_samples
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.extraction import claude_from_env
from leadflow.models import Account, Lead, Rep, TriageOutcome, TriageResult
from leadflow.pipeline import triage
from leadflow.routing import RoutingAgent, router_from_env
from leadflow.store import DecisionStore

app = FastAPI(title="LeadFlow AI", version="0.2.0")

DEMO_PAGE = Path(__file__).resolve().parent / "static" / "index.html"

ReferenceData = Tuple[Sequence[Account], Sequence[Rep], TerritoryMap]


@lru_cache(maxsize=1)
def get_reference_data() -> ReferenceData:
    return load_accounts(), load_reps(), load_territories()


@lru_cache(maxsize=1)
def get_store() -> DecisionStore:
    return DecisionStore()


@lru_cache(maxsize=1)
def get_buyer_agent() -> BuyerContextAgent:
    rows = load_enrichment()
    return BuyerContextAgent(MockTheirStack(rows), MockPostHog(rows), claude_from_env())


@lru_cache(maxsize=1)
def get_capacity_service() -> CapacityService:
    accounts, _, _ = get_reference_data()
    return CapacityService(accounts)


@lru_cache(maxsize=1)
def get_routing_agent() -> RoutingAgent:
    return RoutingAgent(router_from_env())


class SubmissionReceipt(BaseModel):
    submission_id: str
    status: TriageOutcome
    reason: str
    assigned_rep_id: Optional[str]
    lead: Lead
    triage: TriageResult
    buyer: Optional[BuyerAssessment] = None
    capacity: Optional[CapacityMatrix] = None
    routing: Optional[RoutingOutcome] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> str:
    """The demo page: send a sample lead and see each stage's result."""
    return DEMO_PAGE.read_text()


@app.get("/demo/samples")
def samples() -> list:
    """Workbook leads and scenario leads for the demo page's sample picker."""
    return demo_samples()


@app.post("/leads", status_code=status.HTTP_202_ACCEPTED, response_model=SubmissionReceipt)
def submit_lead(
    lead: Lead,
    reference: ReferenceData = Depends(get_reference_data),
    store: DecisionStore = Depends(get_store),
    buyer_agent: BuyerContextAgent = Depends(get_buyer_agent),
    capacity_service: CapacityService = Depends(get_capacity_service),
    routing_agent: RoutingAgent = Depends(get_routing_agent),
) -> SubmissionReceipt:
    """Accept an inbound form submission, triage it, and assess new prospects.

    Every schema-valid submission returns 202, whatever the outcome.
    """
    accounts, reps, territories = reference
    result = triage(lead, accounts, reps, territories, store)
    buyer = capacity = routing = None
    if result.outcome == "new_prospect":
        buyer = buyer_agent(lead)
        capacity = capacity_service(result.shortlist, reps, lead.email_domain)
        routing = routing_agent(lead, buyer, capacity, result.shortlist, result.related_rep_id)
    return SubmissionReceipt(
        submission_id=lead.submission_id,
        status=result.outcome,
        reason=result.reason,
        assigned_rep_id=result.assigned_rep_id,
        lead=lead,
        triage=result,
        buyer=buyer,
        capacity=capacity,
        routing=routing,
    )
