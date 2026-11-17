"""Buyer context agent: enrich the lead, extract note signals, score it.

The steps it takes depend on what it finds:
- Personal email, or a domain TheirStack does not know: look the company up by name.
- No note: skip the LLM.
- LLM missing or failing: fall back to rule extraction and record it in tool_status.
- Note names one competitor while TheirStack detects another: flag it for the rep.

It sees only the lead and its own tool results, never rep data.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from leadflow.contracts import (
    BehaviorSignals,
    BuyerAssessment,
    Competitor,
    ExtractionMethod,
    Firmographics,
    NoteSignals,
    ScoreLine,
    ToolName,
    ToolStatus,
)
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.extraction import NoteExtractor, extract_with_rules
from leadflow.models import Lead
from leadflow.screening import is_personal_domain
from leadflow.scoring import score_buyer


SIGNAL_ORDER = (
    "deadline", "seats", "competitor", "compliance", "buying_stage", "pricing_views", "doc_views", "sessions", "role"
)


@dataclass(frozen=True)
class Enrichment:
    firmographics: Optional[Firmographics]
    behavior: Optional[BehaviorSignals]
    tool_status: Dict[ToolName, ToolStatus]


@dataclass(frozen=True)
class Extraction:
    signals: NoteSignals
    method: ExtractionMethod
    llm_status: ToolStatus


class BuyerContextAgent:
    def __init__(
        self, theirstack: MockTheirStack, posthog: MockPostHog, extractor: Optional[NoteExtractor] = None
    ):
        self.theirstack = theirstack
        self.posthog = posthog
        self.extractor = extractor

    def __call__(self, lead: Lead) -> BuyerAssessment:
        enrichment = self.enrich(lead)
        return assess(lead, enrichment, self.extract(lead))

    def enrich(self, lead: Lead) -> Enrichment:
        if is_personal_domain(lead.email_domain):
            firmographics = self.theirstack.by_company(lead.company)
        else:
            firmographics = self.theirstack.by_domain(lead.email_domain) or self.theirstack.by_company(lead.company)
        behavior = self.posthog.behavior(lead.work_email)
        return Enrichment(firmographics, behavior, {"theirstack": "mocked", "posthog": "mocked"})

    def extract(self, lead: Lead) -> Extraction:
        if not (lead.free_text_note or "").strip() or self.extractor is None:
            return Extraction(extract_with_rules(lead), "rules", "skipped")
        try:
            return Extraction(self.extractor(lead), "llm", "ok")
        except Exception:
            return Extraction(extract_with_rules(lead), "rules", "error")


def assess(lead: Lead, enrichment: Enrichment, extraction: Extraction) -> BuyerAssessment:
    """Score the lead and assemble the BuyerAssessment."""
    firmographics, behavior, signals = enrichment.firmographics, enrichment.behavior, extraction.signals
    score, tier, breakdown = score_buyer(lead, signals, behavior)
    return BuyerAssessment(
        submission_id=lead.submission_id,
        intent_score=score,
        urgency_tier=tier,
        detected_competitor=signals.competitor_mentioned or _detected(firmographics),
        buying_signals=_buying_signals(signals, firmographics, behavior, breakdown),
        note_summary=signals.summary,
        rationale=_rationale(score, tier, breakdown, firmographics),
        firmographics=firmographics,
        behavior=behavior,
        tool_status={**enrichment.tool_status, "llm": extraction.llm_status},
        signals=signals,
        extraction=extraction.method,
        score_breakdown=breakdown,
    )


def _detected(firmographics: Optional[Firmographics]) -> Optional[Competitor]:
    if firmographics and firmographics.competitors_detected:
        return firmographics.competitors_detected[0]
    return None


def _buying_signals(
    signals: NoteSignals,
    firmographics: Optional[Firmographics],
    behavior: Optional[BehaviorSignals],
    breakdown: List[ScoreLine],
) -> List[str]:
    by_factor = {line.factor: line.detail for line in breakdown}
    items = [by_factor[factor] for factor in SIGNAL_ORDER if factor in by_factor]
    detected = _detected(firmographics)
    if detected and signals.competitor_mentioned is None:
        items.append(f"TheirStack detects {detected} on their site")
    elif detected and signals.competitor_mentioned and detected != signals.competitor_mentioned:
        items.append(f"Note names {signals.competitor_mentioned} but TheirStack detects {detected}")
    if firmographics is None:
        items.append("No TheirStack record; using form company size")
    if behavior is not None and behavior.summary:
        items.append(behavior.summary)
    return items[:8]


def _rationale(
    score: int, tier: str, breakdown: List[ScoreLine], firmographics: Optional[Firmographics]
) -> str:
    drivers = sorted((line for line in breakdown if line.factor not in ("base", "cap")), key=lambda l: -l.points)
    top = "; ".join(f"{line.detail} (+{line.points})" for line in drivers[:4])
    company = ""
    if firmographics and firmographics.industry:
        size = f", {firmographics.headcount} people" if firmographics.headcount else ""
        company = f" {firmographics.industry}{size}."
    text = f"{tier} intent, score {score}.{company} Main drivers: {top}."
    return text[:600]
