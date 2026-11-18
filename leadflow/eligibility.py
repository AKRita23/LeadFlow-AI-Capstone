"""Deterministic rep-pool filter for new prospects.

Territory and language come from data/territories.csv. A rep is eligible when
they cover the lead's territory and then pass every rule below, checked in
this order:

- segment: the rep's segment focus includes the lead's company-size segment
- language: the rep supports the territory row's language
- availability: the rep's working status is Active
- daily_cap: leads assigned today are below the rep's daily cap
- working_hours: the submission falls on a weekday, 09:00-18:00 in the rep's timezone

If every rep that passed the first four rules is outside working hours, they
are shortlisted anyway with after_hours set, so a lead arriving at night is
still routed. Capacity scoring happens after this filter, on the shortlist only.
"""
from datetime import datetime, time, timezone
from typing import List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from leadflow.data_loader import TerritoryMap
from leadflow.models import TIMESTAMP_FORMAT, EligibilityRule, EligibleReps, Lead, Rep, Segment, TerritoryRule

# The workbook rationales treat 16-100 as SMB and 51-250 as mid-market.
COMPANY_SIZE_SEGMENTS = {
    "1-15": "SMB",
    "16-100": "SMB",
    "51-250": "Mid-Market",
    "101-500": "Mid-Market",
    "251-1000": "Mid-Market",
    "501-1000": "Mid-Market",
    "1001+": "Enterprise",
}

WORKDAY_START = time(9, 0)
WORKDAY_END = time(18, 0)

MAX_SHORTLIST = 6


def territory_for(lead: Lead, territories: TerritoryMap) -> Optional[TerritoryRule]:
    return territories.get((lead.country, lead.state if lead.country == "US" else None))


def segment_for(company_size: str) -> Segment:
    return COMPANY_SIZE_SEGMENTS[company_size]


def in_working_hours(rep: Rep, submitted_at: str) -> bool:
    utc = datetime.strptime(submitted_at, TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    local = utc.astimezone(ZoneInfo(rep.timezone))
    return local.weekday() < 5 and WORKDAY_START <= local.time() < WORKDAY_END


def filter_eligible_reps(lead: Lead, reps: Sequence[Rep], territories: TerritoryMap) -> EligibleReps:
    """Shortlist up to six reps for a new prospect.

    When more than six pass, the reps with the most daily-cap headroom are
    kept, with rep_id breaking ties so the same lead always gets the same list.
    """
    rule = territory_for(lead, territories)
    segment = segment_for(lead.company_size)
    if rule is None:
        return EligibleReps(lead.submission_id, territory=None, segment=segment, language="en", rep_ids=(), excluded=())
    in_hours: List[Rep] = []
    off_hours: List[Rep] = []
    excluded: List[Tuple[str, EligibilityRule]] = []
    for rep in reps:
        if rep.territory != rule.territory:
            continue
        failed = _first_failed_rule(rep, segment, rule.language)
        if failed is not None:
            excluded.append((rep.rep_id, failed))
        elif in_working_hours(rep, lead.submitted_at):
            in_hours.append(rep)
        else:
            off_hours.append(rep)

    after_hours = not in_hours and bool(off_hours)
    eligible = off_hours if after_hours else in_hours
    if not after_hours:
        excluded.extend((rep.rep_id, "working_hours") for rep in off_hours)
    eligible.sort(key=lambda rep: (rep.leads_assigned_today - rep.daily_lead_cap, rep.rep_id))
    return EligibleReps(
        submission_id=lead.submission_id,
        territory=rule.territory,
        segment=segment,
        language=rule.language,
        rep_ids=tuple(rep.rep_id for rep in eligible[:MAX_SHORTLIST]),
        excluded=tuple(excluded),
        after_hours=after_hours,
    )


def _first_failed_rule(rep: Rep, segment: Segment, language: str) -> Optional[EligibilityRule]:
    if segment not in rep.segments:
        return "segment"
    if language not in rep.languages:
        return "language"
    if rep.working_status != "Active":
        return "availability"
    if rep.leads_assigned_today >= rep.daily_lead_cap:
        return "daily_cap"
    return None
