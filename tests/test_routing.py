import json
from types import SimpleNamespace

import pytest

from leadflow.buyer_agent import BuyerContextAgent
from leadflow.capacity import CapacityService
from leadflow.contracts import CapacityMatrix, RoutingProposal, check_proposal
from leadflow.data_loader import load_accounts, load_enrichment, load_leads, load_reps, load_territories
from leadflow.eligibility import filter_eligible_reps
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.models import EligibleReps
from leadflow.routing import ClaudeRouter, RoutingAgent, rank_reps, routing_context

ROWS = load_enrichment()
REPS = load_reps()
BUYER_AGENT = BuyerContextAgent(MockTheirStack(ROWS), MockPostHog(ROWS))
CAPACITY = CapacityService(load_accounts())
LEADS = {lead.submission_id: lead for lead in load_leads()[0]}


def setup(submission_id="sub_8f92a1c4"):
    """Lead, buyer assessment, capacity matrix, and shortlist for a workbook lead (FinPay by default)."""
    lead = LEADS[submission_id]
    shortlist = filter_eligible_reps(lead, REPS, load_territories())
    return lead, BUYER_AGENT(lead), CAPACITY(shortlist, REPS, lead.email_domain), shortlist


# In FinPay's shortlist, rep_027, rep_035, and rep_011 are also strong against Intercom.
NO_OTHER_INTERCOM = {rep: {"competitor_strengths": []} for rep in ("rep_027", "rep_035", "rep_011")}


def with_rows(capacity, **changes_by_rep):
    rows = [row.model_copy(update=changes_by_rep.get(row.rep_id, {})) for row in capacity.reps]
    return capacity.model_copy(update={"reps": rows})


def proposal_json(primary, backup, rationale="Because.", brief="Brief."):
    return json.dumps(
        {"primary_rep_id": primary, "backup_rep_id": backup, "routing_rationale": rationale, "rep_brief": brief}
    )


class FakeRouter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, context, shortlist):
        self.calls.append(context)
        if isinstance(self.result, Exception):
            raise self.result
        return RoutingProposal(submission_id=shortlist.submission_id, **json.loads(self.result))


class APITimeoutError(Exception):
    pass


# Ranker


def test_finpay_ranks_the_intercom_specialist_with_bandwidth_first():
    _, buyer, capacity, _ = setup()
    assert [row.rep_id for row in rank_reps(buyer, capacity)][:2] == ["rep_003", "rep_027"]


def test_ranking_covers_exactly_the_shortlist():
    _, buyer, capacity, shortlist = setup()
    assert sorted(row.rep_id for row in rank_reps(buyer, capacity)) == sorted(shortlist.rep_ids)


def test_high_urgency_puts_low_bandwidth_last():
    _, buyer, capacity, _ = setup()
    assert buyer.urgency_tier == "High"
    capacity = with_rows(capacity, rep_003={"bandwidth_tier": "Low"})
    assert rank_reps(buyer, capacity)[-1].rep_id == "rep_003"


def test_medium_urgency_keeps_a_low_bandwidth_specialist():
    _, buyer, capacity, _ = setup()
    buyer = buyer.model_copy(update={"urgency_tier": "Medium"})
    capacity = with_rows(capacity, rep_003={"bandwidth_tier": "Low"}, **NO_OTHER_INTERCOM)
    assert rank_reps(buyer, capacity)[0].rep_id == "rep_003"


def test_related_rep_comes_first():
    _, buyer, capacity, _ = setup()
    assert rank_reps(buyer, capacity, related_rep_id="rep_011")[0].rep_id == "rep_011"


def test_competitor_fit_beats_bandwidth():
    _, buyer, capacity, _ = setup()
    capacity = with_rows(
        capacity,
        rep_003={"bandwidth_tier": "Medium"},
        rep_043={"bandwidth_tier": "High"},
        **NO_OTHER_INTERCOM,
    )
    assert rank_reps(buyer, capacity)[0].rep_id == "rep_003"


# Agent with no LLM, or a failing one


def test_without_a_router_the_ranker_proposes():
    lead, buyer, capacity, shortlist = setup()
    outcome = RoutingAgent()(lead, buyer, capacity, shortlist)
    assert outcome.method == "ranker"
    assert outcome.llm_status == "skipped"
    assert outcome.proposal.primary_rep_id == "rep_003"
    assert outcome.proposal.backup_rep_id == "rep_027"
    assert check_proposal(outcome.proposal, shortlist) == []
    assert "strong against Intercom" in outcome.proposal.routing_rationale
    assert "Competitor: Intercom. You are strong against them." in outcome.proposal.rep_brief


def test_single_rep_shortlist_has_no_backup():
    lead, buyer, capacity, shortlist = setup()
    only = EligibleReps(shortlist.submission_id, shortlist.territory, shortlist.segment, "es", ("rep_043",), ())
    capacity = CapacityMatrix(submission_id=shortlist.submission_id, reps=[r for r in capacity.reps if r.rep_id == "rep_043"])
    outcome = RoutingAgent()(lead, buyer, capacity, only)
    assert outcome.proposal.primary_rep_id == "rep_043"
    assert outcome.proposal.backup_rep_id is None
    assert check_proposal(outcome.proposal, only) == []


def test_valid_llm_proposal_is_used():
    lead, buyer, capacity, shortlist = setup()
    router = FakeRouter(proposal_json("rep_043", "rep_003"))
    outcome = RoutingAgent(router)(lead, buyer, capacity, shortlist)
    assert outcome.method == "llm"
    assert outcome.llm_status == "ok"
    assert outcome.proposal.primary_rep_id == "rep_043"
    assert outcome.ranking[0] == "rep_003"


@pytest.mark.parametrize(
    "primary, backup, problem",
    [
        ("rep_999", "rep_003", "Primary rep rep_999 is not on the shortlist."),
        ("rep_003", "rep_003", "Backup rep must differ from the primary rep."),
        ("rep_003", None, "A backup rep is required when the shortlist has more than one rep."),
    ],
)
def test_rejected_llm_proposal_falls_back_to_the_ranker(primary, backup, problem):
    lead, buyer, capacity, shortlist = setup()
    outcome = RoutingAgent(FakeRouter(proposal_json(primary, backup)))(lead, buyer, capacity, shortlist)
    assert outcome.method == "ranker"
    assert outcome.llm_status == "error"
    assert outcome.problems == [problem]
    assert check_proposal(outcome.proposal, shortlist) == []


@pytest.mark.parametrize("error, status", [(RuntimeError("boom"), "error"), (APITimeoutError("slow"), "timeout")])
def test_llm_failure_falls_back_to_the_ranker(error, status):
    lead, buyer, capacity, shortlist = setup()
    outcome = RoutingAgent(FakeRouter(error))(lead, buyer, capacity, shortlist)
    assert outcome.method == "ranker"
    assert outcome.llm_status == status
    assert outcome.proposal.primary_rep_id == "rep_003"


# What Claude sees and how it is called


def test_context_holds_only_shortlisted_reps():
    lead, buyer, capacity, shortlist = setup()
    context = routing_context(lead, buyer, capacity, shortlist, related_rep_id="rep_011")
    assert [row["rep_id"] for row in context["reps"]] == list(shortlist.rep_ids)
    assert context["shortlist"] == list(shortlist.rep_ids)
    assert context["related_rep_id"] == "rep_011"
    assert context["buyer"]["competitor"] == "Intercom"
    assert context["buyer"]["urgency_tier"] == "High"


class FakeClient:
    def __init__(self, text, stop_reason="end_turn"):
        self.kwargs = None
        self.response = SimpleNamespace(
            stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
        )
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_claude_router_requests_structured_output_limited_to_the_shortlist():
    lead, buyer, capacity, shortlist = setup()
    client = FakeClient(proposal_json("rep_003", "rep_027"))
    proposal = ClaudeRouter(client=client, model="test-model")(
        routing_context(lead, buyer, capacity, shortlist), shortlist
    )
    assert proposal.submission_id == shortlist.submission_id
    assert proposal.primary_rep_id == "rep_003"
    assert "temperature" not in client.kwargs
    schema = client.kwargs["output_config"]["format"]["schema"]
    assert schema["properties"]["primary_rep_id"]["enum"] == list(shortlist.rep_ids)
    assert schema["additionalProperties"] is False


def test_claude_router_rejects_a_truncated_reply():
    _, _, _, shortlist = setup()
    client = FakeClient('{"primary_rep_id": "rep_0', stop_reason="max_tokens")
    with pytest.raises(ValueError, match="max_tokens"):
        ClaudeRouter(client=client, model="test-model")({}, shortlist)


def test_every_workbook_lead_gets_a_valid_ranker_proposal():
    for submission_id, lead in LEADS.items():
        if submission_id.startswith("sub_ex"):
            continue
        lead, buyer, capacity, shortlist = setup(submission_id)
        outcome = RoutingAgent()(lead, buyer, capacity, shortlist)
        assert check_proposal(outcome.proposal, shortlist) == [], submission_id
