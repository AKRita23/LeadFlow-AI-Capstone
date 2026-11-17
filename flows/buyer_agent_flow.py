"""Buyer agent as a Langflow graph.

    Chat Input -> Parse Lead -+-> Buyer Enrichment -+-> Buyer Score -> Chat Output
                              +-> Note Signals -----+

Run from the repo root (leadflow must be installed in the Langflow env, see README):
    .venv-langflow/bin/lfx run flows/buyer_agent_flow.py --input-value "$(cat flows/sample_lead.json)"

Regenerate flows/buyer_agent.json (for importing into the Langflow UI):
    .venv-langflow/bin/python flows/buyer_agent_flow.py
"""
import json
from pathlib import Path

from lfx.components.input_output import ChatInput, ChatOutput
from lfx.graph import Graph

from components.leadflow.buyer_enrichment import BuyerEnrichmentComponent
from components.leadflow.buyer_score import BuyerScoreComponent
from components.leadflow.note_signals import NoteSignalsComponent
from components.leadflow.parse_lead import ParseLeadComponent


def get_graph() -> Graph:
    chat_input = ChatInput(_id="submission")
    parse = ParseLeadComponent(_id="parse_lead").set(submission=chat_input.message_response)
    enrich = BuyerEnrichmentComponent(_id="buyer_enrichment").set(lead=parse.parse_lead)
    signals = NoteSignalsComponent(_id="note_signals").set(lead=parse.parse_lead)
    score = BuyerScoreComponent(_id="buyer_score").set(
        lead=parse.parse_lead, enrichment=enrich.enrich, signals=signals.extract
    )
    chat_output = ChatOutput(_id="assessment").set(input_value=score.assessment_message)
    return Graph(chat_input, chat_output)


# Stable so re-importing or `lfx push` updates the same flow instead of adding a copy.
FLOW_ID = "5b0c1f9e-4d1a-4b8e-9a51-1eadf10b0a01"

# Canvas positions for the Langflow UI, left to right.
POSITIONS = {
    "submission": (0, 200),
    "parse_lead": (400, 200),
    "buyer_enrichment": (800, 0),
    "note_signals": (800, 400),
    "buyer_score": (1200, 200),
    "assessment": (1600, 200),
}


def export(path: Path) -> None:
    flow = get_graph().dump(
        name="LeadFlow Buyer Agent",
        description="Enrich the lead, extract note signals with Claude, and score intent with rules.",
    )
    flow["id"] = FLOW_ID
    for node in flow["data"]["nodes"]:
        x, y = POSITIONS[node["id"]]
        node.update(type="genericNode", position={"x": x, "y": y})
        node["data"]["display_name"] = node["data"]["node"]["display_name"]
        node["data"]["description"] = node["data"]["node"]["description"]
    for edge in flow["data"]["edges"]:
        source, target = edge["data"]["sourceHandle"], edge["data"]["targetHandle"]
        edge.update(
            id=f"reactflow__edge-{edge['source']}{_handle(source, compact=True)}-{edge['target']}"
            f"{_handle(target, compact=True)}",
            sourceHandle=_handle(source),
            targetHandle=_handle(target),
            animated=False,
            className="",
        )
    path.write_text(json.dumps(flow, indent=2, sort_keys=True) + "\n")


def _handle(handle: dict, compact: bool = False) -> str:
    """The Langflow UI keys edges by the handle JSON with quotes replaced by 'œ'."""
    text = json.dumps(handle, separators=(",", ":") if compact else (", ", ": "))
    return text.replace('"', "œ")


if __name__ == "__main__":
    export(Path(__file__).with_name("buyer_agent.json"))
