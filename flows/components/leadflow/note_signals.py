from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, MessageTextInput, Output, SecretStrInput
from lfx.schema.data import Data

from leadflow.buyer_agent import BuyerContextAgent
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.extraction import DEFAULT_MODEL, ClaudeExtractor, claude_from_env
from leadflow.models import Lead


class NoteSignalsComponent(Component):
    display_name = "Note Signals (Claude)"
    description = (
        "Extract deadline, seats, competitor, compliance, and buying stage from the note with Claude. "
        "Uses rules when there is no note, no API key, or the call fails."
    )
    icon = "Anthropic"
    name = "LeadFlowNoteSignals"

    inputs = [
        DataInput(name="lead", display_name="Lead", required=True),
        SecretStrInput(
            name="api_key",
            display_name="Anthropic API Key",
            info="Leave empty to read ANTHROPIC_API_KEY from the environment or .env.",
            required=False,
        ),
        MessageTextInput(name="model", display_name="Model", value=DEFAULT_MODEL, advanced=True),
    ]

    outputs = [Output(display_name="Signals", name="signals", method="extract")]

    def extract(self) -> Data:
        if self.api_key:
            import anthropic

            extractor = ClaudeExtractor(client=anthropic.Anthropic(api_key=self.api_key, timeout=15.0), model=self.model)
        else:
            extractor = claude_from_env()
            if extractor is not None and self.model:
                extractor.model = self.model
        agent = BuyerContextAgent(MockTheirStack([]), MockPostHog([]), extractor)
        result = agent.extract(Lead.model_validate(self.lead.data))
        self.status = f"{result.method} ({result.llm_status})"
        return Data(
            data={
                "signals": result.signals.model_dump(mode="json"),
                "method": result.method,
                "llm_status": result.llm_status,
            }
        )
