import json

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput
from lfx.io import Output
from lfx.schema.data import Data
from lfx.schema.message import Message

from leadflow.models import Lead


class ParseLeadComponent(Component):
    display_name = "Parse Lead"
    description = "Validate an inbound form submission (JSON) against the Lead schema."
    icon = "user-check"
    name = "LeadFlowParseLead"

    inputs = [
        HandleInput(
            name="submission",
            display_name="Submission",
            info="Form submission as JSON text.",
            input_types=["Message", "Data"],
            required=True,
        ),
    ]

    outputs = [Output(display_name="Lead", name="lead", method="parse_lead")]

    def parse_lead(self) -> Data:
        value = self.submission
        if isinstance(value, Message):
            raw = json.loads(value.text)
        elif isinstance(value, Data):
            raw = value.data
        else:
            raw = json.loads(value)
        lead = Lead.model_validate(raw)
        self.status = f"{lead.first_name} {lead.last_name}, {lead.company}"
        return Data(data=lead.model_dump(mode="json"))
