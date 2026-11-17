from functools import lru_cache

from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, Output
from lfx.schema.data import Data

from leadflow.buyer_agent import BuyerContextAgent
from leadflow.data_loader import load_enrichment
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.models import Lead


@lru_cache(maxsize=1)
def _agent() -> BuyerContextAgent:
    rows = load_enrichment()
    return BuyerContextAgent(MockTheirStack(rows), MockPostHog(rows))


class BuyerEnrichmentComponent(Component):
    display_name = "Buyer Enrichment"
    description = (
        "Look up firmographics (TheirStack, mocked) and site behavior (PostHog, mocked). "
        "Falls back to a company-name search for personal emails or unknown domains."
    )
    icon = "building-2"
    name = "LeadFlowBuyerEnrichment"

    inputs = [DataInput(name="lead", display_name="Lead", required=True)]

    outputs = [Output(display_name="Enrichment", name="enrichment", method="enrich")]

    def enrich(self) -> Data:
        result = _agent().enrich(Lead.model_validate(self.lead.data))
        firmographics = result.firmographics.model_dump(mode="json") if result.firmographics else None
        behavior = result.behavior.model_dump(mode="json") if result.behavior else None
        found = firmographics["industry"] if firmographics else "no TheirStack record"
        self.status = f"{found}; PostHog {'found' if behavior else 'no data'}"
        return Data(data={"firmographics": firmographics, "behavior": behavior, "tool_status": result.tool_status})
