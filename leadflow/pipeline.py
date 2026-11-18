"""Deterministic triage: every stage before the buyer agent and routing agent.

Order: screening, same-email duplicate, existing account, rep-pool filter.
A lead leaves at the first stage that decides it. Every lead is recorded in
the submissions table, and existing-account decisions are also saved to
routing_decisions.

A same-domain duplicate does not stop the lead. The earlier submission and its
rep, if any, are passed on as related_* for the routing step to weigh.
"""
from typing import Sequence

from leadflow.accounts import resolve_existing_account
from leadflow.data_loader import TerritoryMap
from leadflow.eligibility import filter_eligible_reps
from leadflow.models import Account, Lead, Rep, TriageResult
from leadflow.screening import find_duplicate, screen_lead
from leadflow.store import DecisionStore


def triage(
    lead: Lead,
    accounts: Sequence[Account],
    reps: Sequence[Rep],
    territories: TerritoryMap,
    store: DecisionStore,
) -> TriageResult:
    result = _decide(lead, accounts, reps, territories, store)
    store.record_submission(lead, result)
    return result


def _decide(
    lead: Lead,
    accounts: Sequence[Account],
    reps: Sequence[Rep],
    territories: TerritoryMap,
    store: DecisionStore,
) -> TriageResult:
    screen = screen_lead(lead)
    if screen.outcome != "pass":
        outcome = "dropped" if screen.outcome == "drop" else "self_serve"
        return TriageResult(lead.submission_id, outcome, screen.detail)

    duplicate = find_duplicate(lead, store.prior_submissions(lead))
    if duplicate.kind == "same_email":
        return TriageResult(
            lead.submission_id,
            "duplicate",
            f"Same email as {duplicate.prior_submission_id} within the last 30 days.",
            assigned_rep_id=duplicate.prior_rep_id,
            duplicate_of=duplicate.prior_submission_id,
        )

    decision = resolve_existing_account(lead, accounts, reps)
    if decision is not None:
        store.save(decision)
        return TriageResult(
            lead.submission_id, decision.outcome, decision.reason, decision.assigned_rep_id, ownership=decision
        )

    shortlist = filter_eligible_reps(lead, reps, territories)
    if shortlist.rep_ids:
        outcome = "new_prospect"
        reason = f"{len(shortlist.rep_ids)} eligible reps in {shortlist.territory}"
        reason += ", all outside working hours." if shortlist.after_hours else "."
    else:
        outcome = "no_eligible_reps"
        where = shortlist.territory or f"unmapped country {lead.country}"
        reason = f"No eligible rep in {where}."
    related = duplicate.kind == "same_domain"
    return TriageResult(
        lead.submission_id,
        outcome,
        reason,
        shortlist=shortlist,
        related_submission_id=duplicate.prior_submission_id if related else None,
        related_rep_id=duplicate.prior_rep_id if related else None,
    )
