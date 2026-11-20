"""Routing agent: pick a primary and backup rep from the shortlist.

Claude weighs the buyer assessment against the capacity matrix and returns a
RoutingProposal as structured output (output_config.format). The rep fields
are restricted to the shortlist in the JSON schema, and check_proposal runs on
the result. If
there is no API key, the call fails, or the proposal is rejected, the
deterministic ranker below makes the proposal instead, so a lead always gets
one. The agent only proposes; nothing is assigned here.

The ranker sorts the shortlist by, in order:

1. Urgency fit: for a High-urgency lead, reps with Low bandwidth go last.
2. Continuity: the rep of an earlier lead from the same company domain.
3. Competitor fit: the rep is strong against the buyer's competitor.
4. Vertical fit: the rep's expertise shares a word with the buyer's industry.
5. Bandwidth tier, High first.
6. Stage pressure, Low first.
7. Leads left today, most first.
8. rep_id, so ties always resolve the same way.
"""
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from pydantic import BaseModel, Field

from leadflow.contracts import (
    BuyerAssessment,
    CapacityMatrix,
    RepCapacity,
    RoutingOutcome,
    RoutingProposal,
    ToolStatus,
    check_proposal,
)
from leadflow.extraction import anthropic_key_available
from leadflow.models import EligibleReps, Lead

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-5-5"

LEVEL_ORDER = {"High": 0, "Medium": 1, "Low": 2}

SYSTEM_PROMPT = """You route inbound sales leads for BeaconDesk, an AI support and live-chat widget.

Pick a primary rep and a backup rep from the shortlist. Weigh the buyer against each rep:
- A High-urgency buyer needs a rep who can act today: avoid Low bandwidth unless nothing else fits.
- Prefer a rep who is strong against the buyer's competitor, even at Medium bandwidth, over a
  generalist with more bandwidth.
- Prefer vertical expertise that matches the buyer's industry.
- High stage pressure means many deals about to close, so less time for a new one.
- If a related rep is given, they already handled a lead from this company; prefer them unless
  they clearly cannot take it.
- The backup must be a different rep, and is required when the shortlist has more than one rep.

routing_rationale: two sentences on why the primary fits better than the others, citing the
capacity and buyer facts you used.
rep_brief: the handoff note the primary rep reads before calling. Plain text, no Markdown, under
120 words, in this layout:

<Name>, <role> at <company>
<industry> · <headcount> people · <company_size> · <country>
<urgency_tier> urgency · intent score <intent_score>

Why now
• <one bullet per deadline, seat count, or need in the note>
• <web activity, e.g. "6 pricing and 4 doc views in 48h, 5 sessions in 7d">
• <what they viewed on the site, if given>

Stack and competitor
• <their stack>
• <the competitor>

What they ask
• <the request or use case>

Opening angle
• <one first question>

Rules for the brief:
- Bullets start with "• ", hold one fact, and run 15 words at most. One blank line between sections.
- State each fact once, in the input's most specific wording: the input repeats some facts in other
  words (a deadline in the note and a "Needs to move in 14 days" signal; the role in the header and
  a "VP submitted the form" signal), so keep only one. Copy
  numbers and timeframes exactly, and keep separate facts separate.
- Leave out any bullet or section the input has nothing for.
- Only when after_hours is true, end with the line "Arrived outside working hours."
- Add no meetings, demos, dates, or next steps the input does not mention.

The input does not state anyone's gender, so do not guess it from a name. In both fields, never
write he, him, his, she, her, hers, himself, or herself for the buyer or a rep. Repeat the name,
use the role ("the VP"), or use they/their. Use only facts given in the input."""


class _ProposalDraft(BaseModel):
    """What Claude fills in; submission_id is added by code."""

    primary_rep_id: str
    backup_rep_id: Optional[str] = None
    routing_rationale: str = Field(min_length=1, max_length=1000)
    rep_brief: str = Field(min_length=1, max_length=3000)


class ClaudeRouter:
    def __init__(self, client: Any = None, model: Optional[str] = None, timeout: float = 10.0):
        if client is None:
            from leadflow.observability import anthropic_client

            client = anthropic_client(timeout=timeout)
        self.client = client
        self.model = model or os.environ.get("LEADFLOW_ROUTING_MODEL") or DEFAULT_MODEL

    def __call__(self, context: Dict[str, Any], shortlist: EligibleReps) -> RoutingProposal:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            # Haiku rejects output_config.effort; keep structured JSON only.
            output_config={
                "format": {"type": "json_schema", "schema": _output_schema(shortlist.rep_ids)},
            },
            messages=[{"role": "user", "content": json.dumps(context, indent=2)}],
        )
        if response.stop_reason != "end_turn":
            raise ValueError(f"Claude stopped with {response.stop_reason}")
        text = next(block.text for block in response.content if block.type == "text")
        draft = _ProposalDraft.model_validate_json(text)
        return RoutingProposal(submission_id=shortlist.submission_id, **draft.model_dump())


def router_from_env() -> Optional[ClaudeRouter]:
    """A ClaudeRouter when an API key is available, else None."""
    return ClaudeRouter() if anthropic_key_available() else None


class RoutingAgent:
    def __init__(self, router: Optional[ClaudeRouter] = None):
        self.router = router

    def __call__(
        self,
        lead: Lead,
        buyer: BuyerAssessment,
        capacity: CapacityMatrix,
        shortlist: EligibleReps,
        related_rep_id: Optional[str] = None,
    ) -> RoutingOutcome:
        ranked = rank_reps(buyer, capacity, related_rep_id)
        ranking = [row.rep_id for row in ranked]
        if self.router is None:
            return _ranker_outcome(lead, buyer, ranked, shortlist, related_rep_id, "skipped", [])

        context = routing_context(lead, buyer, capacity, shortlist, related_rep_id)
        try:
            proposal = self.router(context, shortlist)
        except Exception as exc:
            status: ToolStatus = "timeout" if "timeout" in type(exc).__name__.lower() else "error"
            logger.warning("Routing LLM failed for %s, using ranker: %s", lead.submission_id, exc)
            return _ranker_outcome(lead, buyer, ranked, shortlist, related_rep_id, status, [str(exc)[:300]])

        problems = check_proposal(proposal, shortlist)
        if problems:
            logger.warning("Routing proposal rejected for %s: %s", lead.submission_id, problems)
            return _ranker_outcome(lead, buyer, ranked, shortlist, related_rep_id, "error", problems)
        return RoutingOutcome(proposal=proposal, method="llm", llm_status="ok", ranking=ranking)


def rank_reps(
    buyer: BuyerAssessment, capacity: CapacityMatrix, related_rep_id: Optional[str] = None
) -> List[RepCapacity]:
    """The shortlist in the ranker's order, best first."""
    industry = _words(buyer.firmographics.industry if buyer.firmographics else None)

    def key(row: RepCapacity):
        return (
            buyer.urgency_tier == "High" and row.bandwidth_tier == "Low",
            row.rep_id != related_rep_id,
            not _beats_competitor(row, buyer),
            not _vertical_match(row, industry),
            LEVEL_ORDER[row.bandwidth_tier],
            -LEVEL_ORDER[row.stage_pressure],
            -row.leads_remaining_today,
            row.rep_id,
        )

    return sorted(capacity.reps, key=key)


def routing_context(
    lead: Lead,
    buyer: BuyerAssessment,
    capacity: CapacityMatrix,
    shortlist: EligibleReps,
    related_rep_id: Optional[str] = None,
) -> Dict[str, Any]:
    """The facts Claude sees. Only shortlisted reps, and no other rep data."""
    firm = buyer.firmographics
    return {
        "buyer": {
            "name": f"{lead.first_name} {lead.last_name}",
            "role": lead.role,
            "company": lead.company,
            "company_size": lead.company_size,
            "country": lead.country,
            "inquiry_reason": lead.inquiry_reason,
            "note": lead.free_text_note,
            "intent_score": buyer.intent_score,
            "urgency_tier": buyer.urgency_tier,
            "competitor": buyer.detected_competitor,
            "industry": firm.industry if firm else None,
            "headcount": firm.headcount if firm else None,
            "tech_stack": firm.tech_stack if firm else [],
            "buying_signals": buyer.buying_signals,
            "note_summary": buyer.note_summary,
        },
        "shortlist": list(shortlist.rep_ids),
        "after_hours": shortlist.after_hours,
        "related_rep_id": related_rep_id,
        "reps": [row.model_dump(mode="json") for row in capacity.reps],
    }


def _ranker_outcome(
    lead: Lead,
    buyer: BuyerAssessment,
    ranked: Sequence[RepCapacity],
    shortlist: EligibleReps,
    related_rep_id: Optional[str],
    llm_status: ToolStatus,
    problems: List[str],
) -> RoutingOutcome:
    primary = ranked[0]
    backup = ranked[1] if len(ranked) > 1 else None
    industry = _words(buyer.firmographics.industry if buyer.firmographics else None)
    rationale = f"{primary.rep_id} ranked first: {_reasons(primary, buyer, industry, related_rep_id)}."
    if backup:
        rationale += f" Backup {backup.rep_id}: {_reasons(backup, buyer, industry, related_rep_id)}."
    proposal = RoutingProposal(
        submission_id=shortlist.submission_id,
        primary_rep_id=primary.rep_id,
        backup_rep_id=backup.rep_id if backup else None,
        routing_rationale=rationale,
        rep_brief=_template_brief(lead, buyer, primary, shortlist.after_hours),
    )
    return RoutingOutcome(
        proposal=proposal,
        method="ranker",
        llm_status=llm_status,
        problems=problems,
        ranking=[row.rep_id for row in ranked],
    )


def _reasons(row: RepCapacity, buyer: BuyerAssessment, industry: Set[str], related_rep_id: Optional[str]) -> str:
    parts = []
    if row.rep_id == related_rep_id:
        parts.append("handled an earlier lead from this company")
    if _beats_competitor(row, buyer):
        parts.append(f"strong against {buyer.detected_competitor}")
    if _vertical_match(row, industry):
        parts.append(f"{', '.join(row.vertical_expertise)} expertise")
    parts.append(f"{row.bandwidth_tier} bandwidth, {row.stage_pressure} stage pressure")
    parts.append(f"{row.leads_remaining_today} leads left today")
    return "; ".join(parts)


def _template_brief(lead: Lead, buyer: BuyerAssessment, primary: RepCapacity, after_hours: bool) -> str:
    """Same layout as the brief Claude writes: a header, then short bulleted sections."""
    firm = buyer.firmographics
    details = [
        firm.industry if firm else None,
        f"{firm.headcount} people" if firm and firm.headcount else None,
        lead.country,
    ]
    sections = [
        f"{lead.first_name} {lead.last_name}, {lead.role} at {lead.company}\n"
        + " · ".join(d for d in details if d)
    ]
    why = [f"{buyer.urgency_tier} urgency, intent {buyer.intent_score}"] + list(buyer.buying_signals[:3])
    sections.append(_section("Why now", why))
    stack = []
    if firm and firm.tech_stack:
        stack.append(", ".join(firm.tech_stack))
    if buyer.detected_competitor:
        edge = " You are strong against them." if _beats_competitor(primary, buyer) else ""
        stack.append(f"Competitor: {buyer.detected_competitor}.{edge}")
    if stack:
        sections.append(_section("Stack and competitor", stack))
    ask = [f"Inquiry: {lead.inquiry_reason}"]
    if buyer.note_summary:
        ask.append(f"Note: {buyer.note_summary}")
    sections.append(_section("What they ask", ask))
    if after_hours:
        sections.append("Arrived outside working hours.")
    sections.append(f"Contact: {lead.work_email}, {lead.phone_number}")
    return "\n\n".join(sections)


def _section(title: str, bullets: Sequence[str]) -> str:
    return title + "\n" + "\n".join(f"• {b}" for b in bullets)


def _beats_competitor(row: RepCapacity, buyer: BuyerAssessment) -> bool:
    return buyer.detected_competitor is not None and buyer.detected_competitor in row.competitor_strengths


def _vertical_match(row: RepCapacity, industry: Set[str]) -> bool:
    return any(_words(vertical) & industry for vertical in row.vertical_expertise)


def _words(text: Optional[str]) -> Set[str]:
    return {word for word in re.split(r"[^a-z0-9-]+", (text or "").lower()) if word}


def _output_schema(rep_ids: Sequence[str]) -> Dict[str, Any]:
    """JSON schema for Claude's reply, with the rep fields restricted to the shortlist.

    Length limits are checked afterwards by _ProposalDraft.
    """
    reps = list(rep_ids)
    return {
        "type": "object",
        "properties": {
            "primary_rep_id": {"type": "string", "enum": reps},
            "backup_rep_id": {"anyOf": [{"type": "string", "enum": reps}, {"type": "null"}]},
            "routing_rationale": {"type": "string"},
            "rep_brief": {"type": "string"},
        },
        "required": ["primary_rep_id", "backup_rep_id", "routing_rationale", "rep_brief"],
        "additionalProperties": False,
    }
