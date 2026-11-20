"""Validate, reserve, and commit: turn a routing proposal into an assignment.

The workbook's rep counts are a fixed snapshot, so live load is the workbook
baseline plus what has been committed since: leads per rep for the lead's day
(UTC date of submitted_at) and intent points per rep for its ISO week. The rep
filter, the capacity service, and this step all read that live load, so each
assignment changes how the next lead is scored and routed.

commit_assignment runs in one locked SQLite transaction:

1. If this submission was already committed, return that assignment unchanged.
2. Order the candidates: the proposal's primary and backup if it still passes
   check_proposal, then the rest of the ranker's order.
3. Skip any candidate not on the shortlist, not on the roster, not Active, or
   at the daily cap under the live count.
4. The first free candidate gets the lead; the next free one is the backup.
   If none is free, the lead goes to the RevOps queue.
5. Save the assignment and set the submission's assigned rep.

The lock means two leads cannot both take a rep's last slot.
"""
import dataclasses
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from leadflow.contracts import Assignment, BuyerAssessment, RoutingOutcome, SkipReason, check_proposal
from leadflow.models import TIMESTAMP_FORMAT, EligibleReps, Lead, Rep
from leadflow.store import DecisionStore

Load = Dict[str, Tuple[int, int]]


def day_and_week(submitted_at: str) -> Tuple[str, str]:
    """The UTC date and ISO week of a submission, e.g. ("2026-11-16", "2026-W47")."""
    when = datetime.strptime(submitted_at, TIMESTAMP_FORMAT)
    year, week, _ = when.isocalendar()
    return when.date().isoformat(), f"{year}-W{week:02d}"


def live_reps(
    reps: Sequence[Rep],
    store: DecisionStore,
    submitted_at: str,
    exclude_submission_id: Optional[str] = None,
    load: Optional[Load] = None,
) -> List[Rep]:
    """The roster with committed assignments added to today's leads and the week's intent points.

    exclude_submission_id leaves out a submission's own assignment, so re-sending
    a committed lead does not count it against its own rep.
    """
    if load is None:
        day, week = day_and_week(submitted_at)
        load = store.rep_load(day, week, exclude_submission_id)
    live = []
    for rep in reps:
        today, points = load.get(rep.rep_id, (0, 0))
        if today or points:
            rep = dataclasses.replace(
                rep,
                leads_assigned_today=rep.leads_assigned_today + today,
                weekly_intent_points=rep.weekly_intent_points + points,
            )
        live.append(rep)
    return live


def commit_assignment(
    lead: Lead,
    buyer: BuyerAssessment,
    routing: RoutingOutcome,
    shortlist: EligibleReps,
    reps: Sequence[Rep],
    store: DecisionStore,
) -> Assignment:
    """Assign the lead to the first free candidate, or queue it for RevOps.

    reps is the workbook roster; live load is read inside the lock.
    """
    day, week = day_and_week(lead.submitted_at)
    with store.locked() as conn:
        existing = store.assignment(lead.submission_id, conn)
        if existing is not None:
            store.set_assigned_rep(conn, lead.submission_id, existing["rep_id"])
            return assignment_from_record(existing, already_committed=True)

        load = store.rep_load(day, week, conn=conn)
        live = {rep.rep_id: rep for rep in live_reps(reps, store, lead.submitted_at, load=load)}
        free: List[str] = []
        skipped: List[List[str]] = []
        for rep_id in _candidates(routing, shortlist):
            reason = _unavailable(rep_id, shortlist, live)
            if reason:
                skipped.append([rep_id, reason])
            else:
                free.append(rep_id)

        record = {
            "submission_id": lead.submission_id,
            "committed_at": datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT),
            "day": day,
            "week": week,
            "outcome": "assigned" if free else "revops_queue",
            "rep_id": free[0] if free else None,
            "backup_rep_id": free[1] if len(free) > 1 else None,
            "skipped": skipped,
            "intent_points": buyer.intent_score,
            "urgency_tier": buyer.urgency_tier,
            "method": routing.method,
            "rationale": routing.proposal.routing_rationale,
        }
        store.insert_assignment(conn, record)
    return assignment_from_record(record, already_committed=False)


def _candidates(routing: RoutingOutcome, shortlist: EligibleReps) -> List[str]:
    """Proposal first if it still passes the check, then the ranker's order, without repeats."""
    proposal = routing.proposal
    first = [] if check_proposal(proposal, shortlist) else [proposal.primary_rep_id, proposal.backup_rep_id]
    ordered = [rep_id for rep_id in first + list(routing.ranking) + list(shortlist.rep_ids) if rep_id]
    return list(dict.fromkeys(ordered))


def _unavailable(rep_id: str, shortlist: EligibleReps, live: Dict[str, Rep]) -> Optional[SkipReason]:
    if rep_id not in shortlist.rep_ids:
        return "not_on_shortlist"
    rep = live.get(rep_id)
    if rep is None:
        return "not_on_roster"
    if rep.working_status != "Active":
        return "inactive"
    if rep.leads_assigned_today >= rep.daily_lead_cap:
        return "daily_cap"
    return None


def assignment_from_record(record: dict, already_committed: bool = False) -> Assignment:
    return Assignment(
        submission_id=record["submission_id"],
        outcome=record["outcome"],
        rep_id=record["rep_id"],
        backup_rep_id=record["backup_rep_id"],
        skipped=record["skipped"],
        intent_points=record["intent_points"],
        committed_at=record["committed_at"],
        already_committed=already_committed,
    )
