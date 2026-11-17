"""Inbound submission API.

Run locally with:  .venv/bin/uvicorn leadflow.api:app --reload
Interactive docs:  http://127.0.0.1:8000/docs

Schema validation happens in the Lead model: a submission that breaks the
Inbound Form schema is rejected with HTTP 422 and a per-field error list.
"""
from fastapi import FastAPI, status
from pydantic import BaseModel

from leadflow.models import Lead

app = FastAPI(title="LeadFlow AI", version="0.1.0")


class SubmissionReceipt(BaseModel):
    submission_id: str
    status: str
    lead: Lead


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/leads", status_code=status.HTTP_202_ACCEPTED, response_model=SubmissionReceipt)
def submit_lead(lead: Lead) -> SubmissionReceipt:
    """Accept an inbound form submission.

    The spam, duplicate and existing-account gates are added in later steps;
    for now a schema-valid submission is acknowledged as received.
    """
    return SubmissionReceipt(submission_id=lead.submission_id, status="received", lead=lead)
