"""Turn a lead's free-text note into NoteSignals.

ClaudeExtractor asks Claude for the signals through a forced tool call whose
input schema is NoteSignals, so the reply is always the right shape.
extract_with_rules is the deterministic fallback used when there is no API
key or the call fails; it also keeps tests offline.
"""
import os
import re
from typing import Any, List, Optional, Protocol

from leadflow.contracts import COMPETITORS, BuyingStage, NoteSignals
from leadflow.models import Lead

DEFAULT_MODEL = "claude-haiku-4-5"
TOOL_NAME = "record_note_signals"

SYSTEM_PROMPT = """You read inbound sales leads for BeaconDesk, an AI support and live-chat widget.
Competitors are Intercom, Zendesk, Drift, Crisp, and Tidio.

Record only what the note states. Do not guess or infer missing facts:
- deadline_days: days until a stated deadline ("2 weeks" = 14, "immediately" = 0). Words like
  "instant" or "real-time" describe the product, not timing. Null when no timing is stated.
- seats: seats, agents, or users the buyer says they need. Null if not stated.
- competitor_mentioned: a competitor the note names, else null.
- compliance: compliance or security requirements named in the note.
- buying_stage: ready_to_buy if they say they are ready to purchase, upgrade, or review a contract;
  evaluating if they compare options, replace or outgrow a tool, or state a concrete need;
  exploring if they are only looking or learning; unknown if the note gives no clue.
- pain_points: up to 3 short phrases.
- summary: one plain sentence for a sales rep."""


class NoteExtractor(Protocol):
    def __call__(self, lead: Lead) -> NoteSignals: ...


def _user_message(lead: Lead) -> str:
    return f"Inquiry reason: {lead.inquiry_reason}\nNote: {lead.free_text_note}"


class ClaudeExtractor:
    def __init__(self, client: Any = None, model: Optional[str] = None, timeout: float = 15.0):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout, max_retries=1)
        self.client = client
        self.model = model or os.environ.get("LEADFLOW_BUYER_MODEL") or DEFAULT_MODEL

    def __call__(self, lead: Lead) -> NoteSignals:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            temperature=0,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "name": TOOL_NAME,
                    "description": "Record the signals stated in the lead's note.",
                    "input_schema": NoteSignals.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": _user_message(lead)}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == TOOL_NAME:
                return NoteSignals.model_validate(block.input)
        raise ValueError("Claude did not return note signals")


def claude_from_env() -> Optional[ClaudeExtractor]:
    """A ClaudeExtractor when ANTHROPIC_API_KEY is set (in the environment or .env), else None."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return ClaudeExtractor()


_UNIT_DAYS = {"day": 1, "week": 7, "month": 30}
_DEADLINE_RE = re.compile(r"\b(\d+)\s*(day|week|month)s?\b", re.I)
_NOW_RE = re.compile(r"\b(immediately|asap|right away|right now|this week)\b", re.I)
_SEATS_RE = re.compile(r"\b(\d+)\s+(?:\w+\s+)?(?:seats|agents|licen[cs]es|users)\b", re.I)
_COMPLIANCE_RE = re.compile(
    r"\b(HIPAA|SOC ?2(?: Type I{1,2})?|GDPR(?: DPA)?|ISO ?27001|PCI(?:[- ]DSS)?|CCPA|data residency|encryption)\b",
    re.I,
)
_READY_RE = re.compile(r"\b(ready (?:to|for)|contract review|purchase|sign(?:ing)? (?:the|a) contract)\b", re.I)
_EVALUATING_RE = re.compile(
    r"\b(evaluat\w*|compar\w*|vs\.?|replac\w*|migrat\w*|switch\w*|outgrow\w*|alternative)\b", re.I
)
_EXPLORING_RE = re.compile(r"\b(looking|learn\w*|explor\w*|research\w*|curious)\b", re.I)

_STAGE_BY_INQUIRY = {
    "I want to buy licenses or upgrade": "ready_to_buy",
    "I need a compliance plan": "evaluating",
    "I want to evaluate BeaconDesk for my organization": "evaluating",
    "I want to understand which BeaconDesk plan is right for me": "exploring",
}


def _stage(note: str, inquiry_reason: str) -> BuyingStage:
    if _READY_RE.search(note):
        return "ready_to_buy"
    if _EVALUATING_RE.search(note):
        return "evaluating"
    if _EXPLORING_RE.search(note):
        return "exploring"
    return _STAGE_BY_INQUIRY[inquiry_reason]


def _compliance(note: str) -> List[str]:
    found: List[str] = []
    for match in _COMPLIANCE_RE.finditer(note):
        term = match.group(1)
        if term.lower() not in (f.lower() for f in found):
            found.append(term)
    return found[:6]


def extract_with_rules(lead: Lead) -> NoteSignals:
    note = (lead.free_text_note or "").strip()
    if not note:
        return NoteSignals()
    deadline = None
    if _NOW_RE.search(note):
        deadline = 0
    elif _DEADLINE_RE.search(note):
        amount, unit = _DEADLINE_RE.search(note).groups()
        deadline = int(amount) * _UNIT_DAYS[unit.lower()]
    seats = _SEATS_RE.search(note)
    mentioned = [(note.lower().find(c.lower()), c) for c in COMPETITORS if c.lower() in note.lower()]
    first_sentence = re.split(r"(?<=[.!?])\s", note, maxsplit=1)[0]
    return NoteSignals(
        deadline_days=deadline,
        seats=int(seats.group(1)) if seats else None,
        competitor_mentioned=min(mentioned)[1] if mentioned else None,
        compliance=_compliance(note),
        buying_stage=_stage(note, lead.inquiry_reason),
        summary=first_sentence[:300],
    )
