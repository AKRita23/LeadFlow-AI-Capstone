"""Rep handoff, lead status, and the SLA watchdog.

Handoff: every committed lead gets a message in the mock Slack outbox (the
handoffs table). The assigned rep receives the routing brief, why they were
picked, and their time limit; a lead with no free rep goes to RevOps.
Existing-account owners receive the ownership brief.

Status: an assigned lead starts at New. The rep moves it to Working, then to
Converted or Disqualified. Every change is saved in status_events.

SLA watchdog: run_watchdog looks at assigned leads still at New.

| Urgency | Time limit | Action when the limit passes                      |
|---------|------------|---------------------------------------------------|
| High    | 10 minutes | Reassign to the backup rep, once; then RevOps      |
| Medium  | 60 minutes | Reported as overdue, no reassignment              |
| Low     | 4 hours    | Reported as overdue, no reassignment              |

A reassignment goes ahead only if the backup is still Active and under the
live daily cap, checked inside the same lock as the commit step. Otherwise,
or if the lead stalls a second time, it is escalated to RevOps. The clock
restarts at each reassignment. The Medium and Low limits are team assumptions;
the spec only sets the High-urgency limit (5 to 10 minutes).
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from leadflow.commit import assignment_from_record, live_reps
from leadflow.contracts import (
    Assignment,
    BuyerAssessment,
    HandoffMessage,
    LeadStatus,
    LeadTracking,
    RoutingOutcome,
    StatusEvent,
    WatchdogAction,
    WatchdogReport,
)
from leadflow.models import TIMESTAMP_FORMAT, Lead, OwnershipDecision, Rep
from leadflow.store import DecisionStore

REVOPS = "revops"

SLA_MINUTES: Dict[str, int] = {"High": 10, "Medium": 60, "Low": 240}

# Moves a rep (or the demo) may make. The watchdog sets New and Escalated itself.
TRANSITIONS: Dict[str, frozenset] = {
    "New": frozenset({"Working", "Disqualified"}),
    "Working": frozenset({"Converted", "Disqualified"}),
    "Converted": frozenset(),
    "Disqualified": frozenset(),
    "Escalated": frozenset({"Working", "Disqualified"}),
}


class StatusError(ValueError):
    """The status change is not allowed. not_found is True when the lead has no assignment."""

    def __init__(self, message: str, not_found: bool = False):
        super().__init__(message)
        self.not_found = not_found


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def send_assignment(
    store: DecisionStore, lead: Lead, buyer: BuyerAssessment, routing: RoutingOutcome, assignment: Assignment
) -> None:
    """Start the lead at New and message the assigned rep, or RevOps. Does nothing for a re-sent lead."""
    if assignment.already_committed:
        return
    at = _stamp(now_utc())
    who = f"{lead.first_name} {lead.last_name}, {lead.company} ({buyer.urgency_tier}, intent {buyer.intent_score})"
    brief = routing.proposal.rep_brief
    with store.locked() as conn:
        if assignment.outcome == "assigned":
            limit = SLA_MINUTES[buyer.urgency_tier]
            if buyer.urgency_tier == "High" and assignment.backup_rep_id:
                deadline = f"Mark it Working within {limit} minutes, or it moves to {assignment.backup_rep_id}."
            else:
                deadline = f"Mark it Working within {limit} minutes."
            body = "\n".join(
                [brief, "", f"Why you: {routing.proposal.routing_rationale}",
                 f"Backup: {assignment.backup_rep_id or 'none'}", deadline]
            )
            store.add_status(conn, lead.submission_id, at, "New", "system", assignment.rep_id, "Assigned at commit")
            store.add_handoff(conn, lead.submission_id, at, assignment.rep_id, "assignment", f"New lead: {who}", body)
        else:
            skipped = ", ".join(f"{rep} ({reason})" for rep, reason in assignment.skipped) or "none"
            body = f"{brief}\n\nNo shortlisted rep had a free slot. Skipped: {skipped}."
            store.add_handoff(conn, lead.submission_id, at, REVOPS, "revops_queue", f"No free rep: {who}", body)


def send_owner_handoff(store: DecisionStore, decision: OwnershipDecision) -> None:
    """Message the kept owner, or RevOps for an ownership exception. Sent once per submission."""
    at = _stamp(now_utc())
    with store.locked() as conn:
        if decision.outcome == "preserve_owner":
            store.add_handoff(
                conn, decision.submission_id, at, decision.assigned_rep_id, "existing_account",
                "Lead from your existing account", decision.brief,
            )
        else:
            store.add_handoff(
                conn, decision.submission_id, at, REVOPS, "ownership_exception",
                "Ownership exception to resolve", decision.brief,
            )


def update_status(store: DecisionStore, submission_id: str, status: LeadStatus, note: str = "") -> LeadTracking:
    """Apply a rep's status change, if the current status allows it."""
    with store.locked() as conn:
        record = store.assignment(submission_id, conn)
        if record is None:
            raise StatusError(f"{submission_id} has no assignment", not_found=True)
        history = store.status_history(submission_id, conn)
        current = history[-1]["status"] if history else None
        if current is None:
            raise StatusError(f"{submission_id} has no status to change")
        if status not in TRANSITIONS[current]:
            allowed = ", ".join(sorted(TRANSITIONS[current])) or "nothing"
            raise StatusError(f"{submission_id} is {current}; it can move to {allowed}, not {status}")
        store.add_status(conn, submission_id, _stamp(now_utc()), status, "rep", record["rep_id"], note)
    return tracking(store, submission_id)


def tracking(store: DecisionStore, submission_id: str) -> LeadTracking:
    record = store.assignment(submission_id)
    history = [StatusEvent(**_event_fields(e)) for e in store.status_history(submission_id)]
    messages = [HandoffMessage(**_message_fields(m)) for m in reversed(store.handoffs(submission_id))]
    return LeadTracking(
        submission_id=submission_id,
        assignment=assignment_from_record(record) if record else None,
        status=history[-1].status if history else None,
        status_history=history,
        messages=messages,
    )


def run_watchdog(store: DecisionStore, roster: Sequence[Rep], now: Optional[datetime] = None) -> WatchdogReport:
    """Reassign or escalate High-urgency leads stuck at New, and report overdue Medium and Low leads.

    now defaults to the current time; pass a later time to see what the
    watchdog would do after a delay.
    """
    now = now or now_utc()
    at = _stamp(now)
    actions: List[WatchdogAction] = []
    with store.locked() as conn:
        leads = store.assigned_leads(conn)
        for record in leads:
            sid, rep_id, urgency = record["submission_id"], record["rep_id"], record["urgency_tier"]
            history = store.status_history(sid, conn)
            if not history or history[-1]["status"] != "New":
                continue
            waited = int((now - _parse(history[-1]["changed_at"])).total_seconds() // 60)
            if waited < SLA_MINUTES[urgency]:
                continue
            if urgency != "High":
                actions.append(WatchdogAction(
                    submission_id=sid, action="overdue", urgency_tier=urgency, minutes_waiting=waited,
                    from_rep_id=rep_id, reason=f"Still New after {waited} minutes; limit is {SLA_MINUTES[urgency]}.",
                ))
                continue

            backup = record["backup_rep_id"]
            reassigned_before = any(e["actor"] == "watchdog" for e in history)
            blocked = _backup_problem(store, conn, roster, record) if backup and not reassigned_before else None
            brief = next((m["body"] for m in store.handoffs(sid, conn=conn) if m["kind"] == "assignment"), "")
            if backup and not reassigned_before and blocked is None:
                reason = f"{rep_id} did not start work within {SLA_MINUTES[urgency]} minutes."
                store.reassign(conn, sid, backup)
                store.add_status(conn, sid, at, "New", "watchdog", backup, f"Reassigned from {rep_id}. {reason}")
                store.add_handoff(conn, sid, at, backup, "reassigned_to", f"Reassigned to you: {sid}",
                                  f"{reason} The lead is yours now.\n\n{brief}")
                store.add_handoff(conn, sid, at, rep_id, "reassigned_away", f"Lead moved to {backup}: {sid}",
                                  f"{reason} It has been reassigned to {backup}.")
                actions.append(WatchdogAction(
                    submission_id=sid, action="reassigned", urgency_tier=urgency, minutes_waiting=waited,
                    from_rep_id=rep_id, to_rep_id=backup, reason=reason,
                ))
            else:
                if reassigned_before:
                    reason = f"Stalled again with {rep_id} after reassignment."
                elif not backup:
                    reason = f"{rep_id} did not start work, and there is no backup."
                else:
                    reason = f"{rep_id} did not start work, and backup {backup} is {blocked}."
                store.reassign(conn, sid, None)
                store.add_status(conn, sid, at, "Escalated", "watchdog", None, reason)
                store.add_handoff(conn, sid, at, REVOPS, "escalated", f"Escalated: {sid}", f"{reason}\n\n{brief}")
                store.add_handoff(conn, sid, at, rep_id, "reassigned_away", f"Lead escalated to RevOps: {sid}", reason)
                actions.append(WatchdogAction(
                    submission_id=sid, action="escalated", urgency_tier=urgency, minutes_waiting=waited,
                    from_rep_id=rep_id, reason=reason,
                ))
    return WatchdogReport(run_at=at, checked=len(leads), actions=actions)


def _backup_problem(store: DecisionStore, conn, roster: Sequence[Rep], record: dict) -> Optional[str]:
    """Why the backup cannot take the lead now, or None if it can. Uses live load, without this lead."""
    day, week = record["day"], record["week"]
    load = store.rep_load(day, week, exclude_submission_id=record["submission_id"], conn=conn)
    live = {rep.rep_id: rep for rep in live_reps(roster, store, f"{day}T00:00:00Z", load=load)}
    rep = live.get(record["backup_rep_id"])
    if rep is None:
        return "not on the roster"
    if rep.working_status != "Active":
        return "not active"
    if rep.leads_assigned_today >= rep.daily_lead_cap:
        return "at the daily cap"
    return None


def _stamp(when: datetime) -> str:
    return when.strftime(TIMESTAMP_FORMAT)


def _parse(stamp: str) -> datetime:
    return datetime.strptime(stamp, TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def _event_fields(row: dict) -> dict:
    return {k: row[k] for k in ("submission_id", "changed_at", "status", "actor", "rep_id", "note")}


def _message_fields(row: dict) -> dict:
    return {k: row[k] for k in ("submission_id", "sent_at", "recipient", "kind", "subject", "body")}
