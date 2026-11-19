"""Capacity service: score each shortlisted rep's workload and fit.

It sees the shortlist, the roster, and the prospect's domain, never buyer
intent. Each rep is kept as separate dimensions rather than one rank, so the
routing agent can trade them off (for example a Medium-bandwidth rep who wins
against the detected competitor over a High-bandwidth generalist).

stage_pressure comes from late-stage deals (Proposal + Negotiation):

| Late-stage opps | stage_pressure |
|-----------------|----------------|
| 0-3             | Low            |
| 4-5             | Medium         |
| 6 or more       | High           |

bandwidth_tier counts load flags. 0 flags is High, 1 is Medium, 2 or more is Low:

| Flag      | Raised when                                             |
|-----------|---------------------------------------------------------|
| daily_cap | 3 or fewer leads left under today's cap                 |
| backlog   | overdue tasks + unworked leads is 4 or more             |
| meetings  | 5 or more hours of meetings today                       |
| weekly    | 400 or more intent points already routed this week      |
| pipeline  | 16 or more open opportunities                           |

Stage pressure is not a bandwidth flag, so the two dimensions stay independent.
On the 50-rep roster the thresholds give 16 High, 20 Medium, and 14 Low
bandwidth reps. They are team assumptions, not values from the workbook.

The prospect's domain is checked against Closed Lost CRM rows, so a rep who
already lost a deal at this company is noted for the routing agent.
"""
import re
from typing import Dict, List, Sequence

from leadflow.contracts import COMPETITORS, CapacityMatrix, Competitor, Level, RepCapacity
from leadflow.models import Account, EligibleReps, Rep

LATE_STAGE_HIGH_MIN = 6
LATE_STAGE_MEDIUM_MIN = 4

CAP_HEADROOM_MAX = 3
BACKLOG_MIN = 4
MEETING_HOURS_MIN = 5.0
WEEKLY_POINTS_MIN = 400
OPEN_OPPS_MIN = 16

LOW_BANDWIDTH_FLAGS = 2


class CapacityService:
    """Builds a CapacityMatrix from the mock Salesforce and calendar data."""

    def __init__(self, accounts: Sequence[Account] = ()):
        self.accounts = accounts

    def __call__(self, shortlist: EligibleReps, reps: Sequence[Rep], domain: str) -> CapacityMatrix:
        reps_by_id = {rep.rep_id: rep for rep in reps}
        missing = [rep_id for rep_id in shortlist.rep_ids if rep_id not in reps_by_id]
        if missing:
            raise ValueError(f"Shortlisted reps not on the roster: {', '.join(missing)}")
        lost = self._closed_lost_by_owner(domain)
        return CapacityMatrix(
            submission_id=shortlist.submission_id,
            reps=[rep_capacity(reps_by_id[rep_id], lost.get(rep_id, [])) for rep_id in shortlist.rep_ids],
            tool_status={"salesforce": "mocked", "calendar": "mocked"},
        )

    def _closed_lost_by_owner(self, domain: str) -> Dict[str, List[str]]:
        lost: Dict[str, List[str]] = {}
        for account in self.accounts:
            if account.domain == domain and account.record_type == "Closed Lost" and account.owner_rep_id:
                lost.setdefault(account.owner_rep_id, []).append(account.account_name)
        return lost


def rep_capacity(rep: Rep, closed_lost: Sequence[str] = ()) -> RepCapacity:
    late = rep.proposal_opps + rep.negotiation_opps
    open_opps = rep.discovery_opps + rep.evaluation_opps + late
    remaining = max(0, rep.daily_lead_cap - rep.leads_assigned_today)
    backlog = rep.overdue_tasks + rep.unworked_leads

    flags = []
    if remaining <= CAP_HEADROOM_MAX:
        flags.append(f"{remaining} leads left today")
    if backlog >= BACKLOG_MIN:
        flags.append(
            f"backlog of {backlog} ({_count(rep.overdue_tasks, 'overdue task')}, "
            f"{_count(rep.unworked_leads, 'unworked lead')})"
        )
    if rep.meetings_today_hours >= MEETING_HOURS_MIN:
        flags.append(f"{rep.meetings_today_hours:g}h of meetings today")
    if rep.weekly_intent_points >= WEEKLY_POINTS_MIN:
        flags.append(f"{rep.weekly_intent_points} intent points routed this week")
    if open_opps >= OPEN_OPPS_MIN:
        flags.append(f"{open_opps} open opportunities")

    notes = [f"Load: {'; '.join(flags)}." if flags else "No load flags."]
    if closed_lost:
        notes.append(f"Previously lost a deal at {', '.join(closed_lost)}.")

    return RepCapacity(
        rep_id=rep.rep_id,
        bandwidth_tier=_bandwidth_tier(len(flags)),
        stage_pressure=_stage_pressure(late),
        open_opps=open_opps,
        late_stage_opps=late,
        open_pipeline_arr=rep.discovery_arr + rep.evaluation_arr + rep.proposal_arr + rep.negotiation_arr,
        overdue_tasks=rep.overdue_tasks,
        unworked_leads=rep.unworked_leads,
        weekly_intent_points=rep.weekly_intent_points,
        leads_remaining_today=remaining,
        meetings_today_hours=rep.meetings_today_hours,
        back_to_back_calls=rep.back_to_back_calls,
        vertical_expertise=list(rep.vertical_expertise),
        competitor_strengths=competitor_strengths(rep.competitor_win_rate_affinity),
        notes=" ".join(notes)[:300],
    )


def competitor_strengths(affinity: str) -> List[Competitor]:
    """Competitors named in a field like "High (Zendesk / Drift)", in the order written."""
    names = re.findall(r"[A-Za-z]+", affinity or "")
    return [name for name in names if name in COMPETITORS]


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _stage_pressure(late_stage_opps: int) -> Level:
    if late_stage_opps >= LATE_STAGE_HIGH_MIN:
        return "High"
    if late_stage_opps >= LATE_STAGE_MEDIUM_MIN:
        return "Medium"
    return "Low"


def _bandwidth_tier(flag_count: int) -> Level:
    if flag_count == 0:
        return "High"
    if flag_count < LOW_BANDWIDTH_FLAGS:
        return "Medium"
    return "Low"
