"""Formats exchanged between triage, the two agents, and the routing step.

Each component takes only the inputs listed on its protocol below:

- The buyer agent sees the lead and its own tool results, never rep data.
- The capacity service sees the shortlist, the roster, and the prospect's
  domain, never buyer intent.
- The routing agent sees both outputs plus the shortlist, and only proposes.
  check_proposal is run before anything is committed.

Agent outputs are Pydantic models, so an LLM response can be validated with
Model.model_validate_json(text), and Model.model_json_schema() can be passed
to a structured-output API.
"""
from typing import Dict, List, Literal, Optional, Protocol, Sequence, get_args

from pydantic import BaseModel, ConfigDict, Field

from leadflow.models import EligibleReps, Lead, Rep

Competitor = Literal["Intercom", "Zendesk", "Drift", "Crisp", "Tidio"]

COMPETITORS = get_args(Competitor)

Level = Literal["High", "Medium", "Low"]

ToolName = Literal["theirstack", "posthog", "salesforce", "calendar", "llm"]

BuyingStage = Literal["exploring", "evaluating", "ready_to_buy", "unknown"]

ExtractionMethod = Literal["llm", "rules"]

ToolStatus = Literal["ok", "mocked", "timeout", "error", "skipped"]

StepName = Literal[
    "screening",
    "duplicate_check",
    "account_match",
    "rep_filter",
    "buyer_agent",
    "capacity_service",
    "routing_agent",
    "commit",
]

StepStatus = Literal["ok", "timeout", "error", "fallback", "skipped"]


class Firmographics(BaseModel):
    """What TheirStack returned for the prospect's domain."""

    model_config = ConfigDict(frozen=True)

    domain: str
    industry: Optional[str] = None
    headcount: Optional[int] = Field(default=None, ge=0)
    tech_stack: List[str] = Field(default_factory=list)
    competitors_detected: List[Competitor] = Field(default_factory=list)


class BehaviorSignals(BaseModel):
    """First-party telemetry from PostHog."""

    model_config = ConfigDict(frozen=True)

    pricing_views_48h: int = Field(ge=0)
    doc_views_48h: int = Field(ge=0)
    sessions_7d: int = Field(ge=0)
    summary: Optional[str] = Field(default=None, max_length=500)


class NoteSignals(BaseModel):
    """What the free-text note says, extracted by the LLM (or the rules fallback).

    Only facts stated in the note belong here. Scoring happens afterwards, in
    leadflow.scoring, so the same signals always give the same score.
    """

    model_config = ConfigDict(frozen=True)

    deadline_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Days until a stated deadline (contract end, go-live). 0 for 'immediately' or 'now'. "
        "Null when no timing is stated.",
    )
    seats: Optional[int] = Field(
        default=None, ge=1, description="Number of seats, agents, or users the buyer says they need."
    )
    competitor_mentioned: Optional[Competitor] = Field(
        default=None, description="Competitor product the note names, if any."
    )
    compliance: List[str] = Field(
        default_factory=list,
        max_length=6,
        description="Compliance or security requirements named in the note, e.g. HIPAA, SOC 2, GDPR, "
        "data residency, encryption.",
    )
    buying_stage: BuyingStage = Field(
        default="unknown",
        description="exploring: learning or browsing. evaluating: comparing options or replacing a tool. "
        "ready_to_buy: says they are ready to purchase, upgrade, or review a contract.",
    )
    pain_points: List[str] = Field(
        default_factory=list, max_length=3, description="Up to 3 short problems the buyer wants solved."
    )
    summary: Optional[str] = Field(
        default=None, max_length=300, description="One sentence a sales rep can read in five seconds."
    )


class ScoreLine(BaseModel):
    """One rubric line that added points to intent_score."""

    model_config = ConfigDict(frozen=True)

    factor: str
    points: int
    detail: str


class BuyerAssessment(BaseModel):
    """Buyer agent output. intent_score and urgency_tier are scored against the workbook.

    The score is the sum of score_breakdown, computed by rules from signals,
    the form, and behavior. The LLM never sets the score directly.
    """

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "submission_id": "sub_8f92a1c4",
                    "intent_score": 95,
                    "urgency_tier": "High",
                    "detected_competitor": "Intercom",
                    "buying_signals": [
                        "Intercom contract expires in 2 weeks",
                        "40 seats needed before Q1",
                        "6 pricing views in 48h",
                    ],
                    "note_summary": "Replacing Intercom within 2 weeks; needs 40 seats live before Q1.",
                    "rationale": "Active Intercom displacement with a 2-week deadline, backed by heavy pricing "
                    "and doc views from a VP.",
                    "firmographics": {
                        "domain": "finpay.es",
                        "industry": "FinTech & Payments",
                        "headcount": 180,
                        "tech_stack": ["Stripe", "HubSpot", "AWS", "Intercom"],
                        "competitors_detected": ["Intercom"],
                    },
                    "behavior": {
                        "pricing_views_48h": 6,
                        "doc_views_48h": 4,
                        "sessions_7d": 5,
                        "summary": "Heavy pricing evaluation, reviewed Intercom migration guide 2 hours before submit.",
                    },
                    "tool_status": {"theirstack": "ok", "posthog": "mocked"},
                }
            ]
        },
    )

    submission_id: str
    intent_score: int = Field(ge=1, le=100)
    urgency_tier: Level
    detected_competitor: Optional[Competitor] = None
    buying_signals: List[str] = Field(default_factory=list, max_length=8)
    note_summary: Optional[str] = Field(default=None, max_length=500)
    rationale: str = Field(min_length=1, max_length=600)
    firmographics: Optional[Firmographics] = None
    behavior: Optional[BehaviorSignals] = None
    tool_status: Dict[ToolName, ToolStatus] = Field(default_factory=dict)
    signals: Optional[NoteSignals] = None
    extraction: Optional[ExtractionMethod] = None
    score_breakdown: List[ScoreLine] = Field(default_factory=list)


class RepCapacity(BaseModel):
    """One shortlisted rep, kept as separate dimensions rather than one rank.

    The routing agent weighs these against the buyer, for example preferring a
    Medium-bandwidth rep who is strong against the detected competitor.
    """

    model_config = ConfigDict(frozen=True)

    rep_id: str
    bandwidth_tier: Level
    stage_pressure: Level
    open_opps: int = Field(ge=0)
    late_stage_opps: int = Field(ge=0)
    open_pipeline_arr: float = Field(ge=0)
    overdue_tasks: int = Field(ge=0)
    unworked_leads: int = Field(ge=0)
    weekly_intent_points: int = Field(ge=0)
    leads_remaining_today: int = Field(ge=0)
    meetings_today_hours: float = Field(ge=0)
    back_to_back_calls: int = Field(ge=0)
    vertical_expertise: List[str] = Field(default_factory=list)
    competitor_strengths: List[Competitor] = Field(default_factory=list)
    notes: Optional[str] = Field(default=None, max_length=300)


class CapacityMatrix(BaseModel):
    """Capacity service output: one row per shortlisted rep."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "submission_id": "sub_8f92a1c4",
                    "reps": [
                        {
                            "rep_id": "rep_003",
                            "bandwidth_tier": "High",
                            "stage_pressure": "Medium",
                            "open_opps": 14,
                            "late_stage_opps": 4,
                            "open_pipeline_arr": 392995,
                            "overdue_tasks": 0,
                            "unworked_leads": 0,
                            "weekly_intent_points": 322,
                            "leads_remaining_today": 7,
                            "meetings_today_hours": 2.5,
                            "back_to_back_calls": 3,
                            "vertical_expertise": ["FinTech", "Travel & Hospitality"],
                            "competitor_strengths": ["Intercom", "Crisp"],
                        }
                    ],
                    "tool_status": {"salesforce": "mocked", "calendar": "mocked"},
                }
            ]
        },
    )

    submission_id: str
    reps: List[RepCapacity] = Field(min_length=1)
    tool_status: Dict[ToolName, ToolStatus] = Field(default_factory=dict)


class RoutingProposal(BaseModel):
    """Routing agent output. Nothing is assigned until check_proposal passes."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "submission_id": "sub_8f92a1c4",
                    "primary_rep_id": "rep_003",
                    "backup_rep_id": "rep_043",
                    "routing_rationale": "Urgent Intercom displacement in FinTech. rep_003 has High bandwidth, "
                    "FinTech expertise, and strong results against Intercom.",
                    "rep_brief": "Elena Rostova, VP at FinPay Technologies (180 people, FinTech). Intercom "
                    "contract ends in 2 weeks; needs 40 seats before Q1. 6 pricing views in 48h.",
                }
            ]
        },
    )

    submission_id: str
    primary_rep_id: str
    backup_rep_id: Optional[str] = None
    routing_rationale: str = Field(min_length=1, max_length=1000)
    rep_brief: str = Field(min_length=1, max_length=3000)


class StepTiming(BaseModel):
    """How long one stage took, for the latency, fallback-rate, and demo views."""

    model_config = ConfigDict(frozen=True)

    step: StepName
    duration_ms: float = Field(ge=0)
    status: StepStatus = "ok"
    detail: Optional[str] = Field(default=None, max_length=300)


class BuyerAgent(Protocol):
    def __call__(self, lead: Lead) -> BuyerAssessment: ...


class CapacityService(Protocol):
    def __call__(self, shortlist: EligibleReps, reps: Sequence[Rep], domain: str) -> CapacityMatrix: ...


class RoutingAgent(Protocol):
    def __call__(
        self, lead: Lead, buyer: BuyerAssessment, capacity: CapacityMatrix, shortlist: EligibleReps
    ) -> RoutingProposal: ...


def check_capacity(matrix: CapacityMatrix, shortlist: EligibleReps) -> List[str]:
    """Problems with a capacity matrix; empty when it covers exactly the shortlist."""
    problems = []
    if matrix.submission_id != shortlist.submission_id:
        problems.append(f"Matrix is for {matrix.submission_id}, not {shortlist.submission_id}.")
    rows = [rep.rep_id for rep in matrix.reps]
    if len(rows) != len(set(rows)):
        problems.append("Matrix lists a rep more than once.")
    missing = [rep_id for rep_id in shortlist.rep_ids if rep_id not in rows]
    extra = [rep_id for rep_id in rows if rep_id not in shortlist.rep_ids]
    if missing:
        problems.append(f"Matrix is missing shortlisted reps: {', '.join(missing)}.")
    if extra:
        problems.append(f"Matrix has reps not on the shortlist: {', '.join(extra)}.")
    return problems


def check_proposal(proposal: RoutingProposal, shortlist: EligibleReps) -> List[str]:
    """Problems with a routing proposal; empty when it can be committed.

    Both reps must be on the shortlist and differ. A backup is required
    whenever the shortlist has a second rep to offer.
    """
    problems = []
    if proposal.submission_id != shortlist.submission_id:
        problems.append(f"Proposal is for {proposal.submission_id}, not {shortlist.submission_id}.")
    if proposal.primary_rep_id not in shortlist.rep_ids:
        problems.append(f"Primary rep {proposal.primary_rep_id} is not on the shortlist.")
    if proposal.backup_rep_id is None:
        if len(shortlist.rep_ids) > 1:
            problems.append("A backup rep is required when the shortlist has more than one rep.")
    elif proposal.backup_rep_id == proposal.primary_rep_id:
        problems.append("Backup rep must differ from the primary rep.")
    elif proposal.backup_rep_id not in shortlist.rep_ids:
        problems.append(f"Backup rep {proposal.backup_rep_id} is not on the shortlist.")
    return problems
