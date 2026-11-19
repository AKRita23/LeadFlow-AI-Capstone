import pytest
from fastapi.testclient import TestClient

from leadflow.api import app, get_buyer_agent, get_routing_agent, get_store
from leadflow.buyer_agent import BuyerContextAgent
from leadflow.data_loader import load_enrichment, load_leads
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.routing import RoutingAgent
from leadflow.store import DecisionStore

client = TestClient(app)
ROWS = load_enrichment()
RULES_AGENT = BuyerContextAgent(MockTheirStack(ROWS), MockPostHog(ROWS))


@pytest.fixture(autouse=True)
def store(tmp_path):
    test_store = DecisionStore(tmp_path / "leadflow.db")
    app.dependency_overrides[get_store] = lambda: test_store
    app.dependency_overrides[get_buyer_agent] = lambda: RULES_AGENT
    app.dependency_overrides[get_routing_agent] = lambda: RoutingAgent()
    yield test_store
    app.dependency_overrides.clear()

VALID = {
    "submission_id": "sub_test0001",
    "submitted_at": "2026-11-16T11:58:23Z",
    "first_name": "Elena",
    "last_name": "Rostova",
    "work_email": "Elena@FinPay.es",
    "role": "VP",
    "country": "es",
    "state": None,
    "company": "FinPay Technologies",
    "company_size": "101-500",
    "phone_number": "+34 915 550 199",
    "inquiry_reason": "I want to evaluate BeaconDesk for my organization",
    "free_text_note": "Our contract with Intercom expires in 2 weeks.",
}


def post(**overrides):
    payload = {**VALID, **overrides}
    payload = {k: v for k, v in payload.items() if v is not ...}
    return client.post("/leads", json=payload)


def error_fields(response):
    return {tuple(e["loc"][1:]) for e in response.json()["detail"]}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_valid_submission_is_triaged_and_normalised():
    response = post()
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "new_prospect"
    assert body["lead"]["work_email"] == "elena@finpay.es"
    assert body["lead"]["country"] == "ES"
    assert body["lead"]["phone_number"] == "+34915550199"


def test_auto_populated_fields_are_generated():
    response = post(submission_id=..., submitted_at=...)
    assert response.status_code == 202
    lead = response.json()["lead"]
    assert lead["submission_id"].startswith("sub_")
    assert lead["submitted_at"].endswith("Z")


def test_blank_note_becomes_null():
    assert post(free_text_note="   ").json()["lead"]["free_text_note"] is None


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"first_name": ""}, ("first_name",)),
        ({"last_name": ...}, ("last_name",)),
        ({"company": "  "}, ("company",)),
        ({"work_email": "not-an-email"}, ("work_email",)),
        ({"work_email": "a@b"}, ("work_email",)),
        ({"role": "Intern"}, ("role",)),
        ({"country": "Spain"}, ("country",)),
        ({"company_size": "10-20"}, ("company_size",)),
        ({"inquiry_reason": "Just browsing"}, ("inquiry_reason",)),
        ({"phone_number": "12345"}, ("phone_number",)),
        ({"submitted_at": "16/11/2026 11:58"}, ("submitted_at",)),
    ],
)
def test_invalid_fields_are_rejected(overrides, field):
    response = post(**overrides)
    assert response.status_code == 422
    assert field in error_fields(response)


@pytest.mark.parametrize(
    "overrides",
    [
        {"country": "US", "state": None},
        {"country": "US", "state": "ZZ"},
        {"country": "ES", "state": "CA"},
    ],
)
def test_state_must_match_country(overrides):
    response = post(**overrides)
    assert response.status_code == 422
    assert "state" in response.text


def test_us_lead_with_state_is_accepted():
    assert post(country="US", state="ca").json()["lead"]["state"] == "CA"


def test_every_dataset_lead_passes_schema_validation():
    leads, _ = load_leads()
    for lead in leads:
        response = client.post("/leads", json=lead.model_dump())
        assert response.status_code == 202, (lead.submission_id, response.text)


def test_dataset_leads_return_their_labeled_path():
    leads, expectations = load_leads()
    for lead, expected in zip(leads, expectations):
        status = client.post("/leads", json=lead.model_dump()).json()["status"]
        if expected.excluded:
            assert status in ("self_serve", "dropped"), lead.submission_id
        else:
            assert status == "new_prospect", lead.submission_id


def test_new_prospect_returns_its_shortlist():
    body = post().json()
    shortlist = body["triage"]["shortlist"]
    assert body["reason"] == "5 eligible reps in EMEA-Southern."
    assert body["assigned_rep_id"] is None
    assert shortlist["territory"] == "EMEA-Southern"
    assert set(shortlist["rep_ids"]) == {"rep_003", "rep_011", "rep_027", "rep_035", "rep_043"}
    assert shortlist["excluded"] == [["rep_019", "segment"]]
    assert shortlist["after_hours"] is False


def test_new_prospect_gets_a_buyer_assessment():
    buyer = post().json()["buyer"]
    assert buyer["submission_id"] == "sub_test0001"
    assert buyer["urgency_tier"] == "High"
    assert buyer["detected_competitor"] == "Intercom"
    assert buyer["extraction"] == "rules"
    assert sum(line["points"] for line in buyer["score_breakdown"]) == buyer["intent_score"]


def test_new_prospect_gets_a_capacity_matrix_for_its_shortlist():
    body = post().json()
    capacity = body["capacity"]
    assert capacity["submission_id"] == "sub_test0001"
    assert [row["rep_id"] for row in capacity["reps"]] == body["triage"]["shortlist"]["rep_ids"]
    assert capacity["tool_status"] == {"salesforce": "mocked", "calendar": "mocked"}


def test_new_prospect_gets_a_routing_proposal_from_its_shortlist():
    body = post().json()
    routing = body["routing"]
    shortlist = body["triage"]["shortlist"]["rep_ids"]
    assert routing["method"] == "ranker"
    assert routing["proposal"]["primary_rep_id"] in shortlist
    assert routing["proposal"]["backup_rep_id"] in shortlist
    assert body["assigned_rep_id"] is None


def test_other_outcomes_skip_the_agents():
    body = post(company_size="1-15").json()
    assert body["buyer"] is None
    assert body["capacity"] is None
    assert body["routing"] is None


@pytest.mark.parametrize(
    "overrides, status",
    [
        ({"work_email": "qa-bot@test-domain.org"}, "dropped"),
        ({"company_size": "1-15"}, "self_serve"),
        ({"work_email": "elena@atlasgroup.com"}, "ownership_exception"),
        ({"country": "ZZ"}, "no_eligible_reps"),
    ],
)
def test_other_outcomes(overrides, status):
    assert post(**overrides).json()["status"] == status


def test_existing_account_returns_owner_and_brief():
    body = post(work_email="elena@harborpay.com").json()
    assert body["status"] == "preserve_owner"
    assert body["assigned_rep_id"] == "rep_002"
    assert body["triage"]["shortlist"] is None
    assert body["triage"]["ownership"]["brief"].startswith("Owner kept: Mark Rivera (rep_002).")


def test_resubmission_is_a_duplicate(store):
    post(work_email="elena@harborpay.com")
    body = post(submission_id="sub_test0002", submitted_at="2026-11-17T11:58:23Z",
                work_email="elena@harborpay.com").json()
    assert body["status"] == "duplicate"
    assert body["triage"]["duplicate_of"] == "sub_test0001"
    assert body["assigned_rep_id"] == "rep_002"
    assert store.submission("sub_test0002")["outcome"] == "duplicate"
