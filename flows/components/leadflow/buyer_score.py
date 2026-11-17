from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, Output
from lfx.schema.message import Message

from leadflow.buyer_agent import Enrichment, Extraction, assess
from leadflow.contracts import BehaviorSignals, Firmographics, NoteSignals
from leadflow.models import Lead


class BuyerScoreComponent(Component):
    display_name = "Buyer Score"
    description = "Rule-based intent score and urgency tier, with the points behind them."
    icon = "gauge"
    name = "LeadFlowBuyerScore"

    inputs = [
        DataInput(name="lead", display_name="Lead", required=True),
        DataInput(name="enrichment", display_name="Enrichment", required=True),
        DataInput(name="signals", display_name="Signals", required=True),
    ]

    outputs = [Output(display_name="Assessment", name="assessment", method="assessment_message")]

    def assessment_message(self) -> Message:
        e, s = self.enrichment.data, self.signals.data
        enrichment = Enrichment(
            firmographics=Firmographics.model_validate(e["firmographics"]) if e["firmographics"] else None,
            behavior=BehaviorSignals.model_validate(e["behavior"]) if e["behavior"] else None,
            tool_status=e["tool_status"],
        )
        extraction = Extraction(NoteSignals.model_validate(s["signals"]), s["method"], s["llm_status"])
        result = assess(Lead.model_validate(self.lead.data), enrichment, extraction)
        self.status = f"{result.urgency_tier} ({result.intent_score})"
        return Message(text=result.model_dump_json(indent=2))
