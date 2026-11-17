"""Rule-based intent score and urgency tier for the buyer agent.

The LLM extracts NoteSignals; this module turns them, the form, and PostHog
behavior into a score. Every point is listed in the breakdown, so the score
can be explained to a rep and replays identically.

Weights were tuned on the 15 workbook leads that have an expected score: all
15 land in their expected tier, with a mean absolute error of about 3 points.
Behavior carries most of the weight because that is what the workbook scores
track. The workbook has no Low leads, so MEDIUM_TIER_MIN is a team assumption.
"""
from typing import List, Optional, Tuple

from leadflow.contracts import BehaviorSignals, Level, NoteSignals, ScoreLine
from leadflow.models import Lead

HIGH_TIER_MIN = 80
MEDIUM_TIER_MIN = 40

BASE_POINTS = 44

ROLE_POINTS = {"CXO": 6, "VP": 4, "Director": 2, "Manager": 0, "Contributor": 0}

INQUIRY_POINTS = {
    "I want to buy licenses or upgrade": 3,
    "I need a compliance plan": 2,
    "I want to evaluate BeaconDesk for my organization": 1,
    "I want to understand which BeaconDesk plan is right for me": 0,
}

PRICING_POINTS, PRICING_VIEW_CAP = 3, 6
DOC_POINTS, DOC_VIEW_CAP = 1, 5
SESSION_POINTS, SESSION_CAP = 4, 5

# Used when PostHog has nothing for the lead, so a tool outage does not sink the score.
NEUTRAL_BEHAVIOR = BehaviorSignals(pricing_views_48h=2, doc_views_48h=2, sessions_7d=2)

DEADLINE_WINDOW_DAYS, DEADLINE_POINTS = 30, 3
SEATS_MIN, SEATS_POINTS = 10, 2
COMPETITOR_POINTS = 2
COMPLIANCE_POINTS = 7
READY_TO_BUY_POINTS = 2


def tier_for(score: int) -> Level:
    if score >= HIGH_TIER_MIN:
        return "High"
    if score >= MEDIUM_TIER_MIN:
        return "Medium"
    return "Low"


def score_buyer(
    lead: Lead, signals: NoteSignals, behavior: Optional[BehaviorSignals]
) -> Tuple[int, Level, List[ScoreLine]]:
    """Return intent_score, urgency_tier, and the lines that sum to the score."""
    lines = [ScoreLine(factor="base", points=BASE_POINTS, detail="Valid inbound lead")]

    def add(factor: str, points: int, detail: str) -> None:
        if points:
            lines.append(ScoreLine(factor=factor, points=points, detail=detail))

    add("role", ROLE_POINTS[lead.role], f"{lead.role} submitted the form")
    add("inquiry", INQUIRY_POINTS[lead.inquiry_reason], f'Inquiry: "{lead.inquiry_reason}"')

    observed = behavior is not None
    b = behavior if observed else NEUTRAL_BEHAVIOR
    suffix = "" if observed else " (PostHog unavailable, neutral default)"
    add(
        "pricing_views",
        PRICING_POINTS * min(b.pricing_views_48h, PRICING_VIEW_CAP),
        f"{b.pricing_views_48h} pricing views in 48h{suffix}",
    )
    add("doc_views", DOC_POINTS * min(b.doc_views_48h, DOC_VIEW_CAP), f"{b.doc_views_48h} doc views in 48h{suffix}")
    add("sessions", SESSION_POINTS * min(b.sessions_7d, SESSION_CAP), f"{b.sessions_7d} sessions in 7d{suffix}")

    if signals.deadline_days is not None and signals.deadline_days <= DEADLINE_WINDOW_DAYS:
        when = "immediately" if signals.deadline_days == 0 else f"in {signals.deadline_days} days"
        add("deadline", DEADLINE_POINTS, f"Needs to move {when}")
    if signals.seats is not None and signals.seats >= SEATS_MIN:
        add("seats", SEATS_POINTS, f"{signals.seats} seats needed")
    if signals.competitor_mentioned:
        add("competitor", COMPETITOR_POINTS, f"Names {signals.competitor_mentioned} in the note")
    if signals.compliance:
        add("compliance", COMPLIANCE_POINTS, f"Compliance needs: {', '.join(signals.compliance)}")
    if signals.buying_stage == "ready_to_buy":
        add("buying_stage", READY_TO_BUY_POINTS, "Says they are ready to buy")

    total = sum(line.points for line in lines)
    score = max(1, min(100, total))
    if score != total:
        lines.append(ScoreLine(factor="cap", points=score - total, detail="Score is capped at 1-100"))
    return score, tier_for(score), lines
