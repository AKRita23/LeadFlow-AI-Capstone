import pytest
from fastapi.testclient import TestClient

from leadflow.api import app, get_buyer_agent, get_routing_agent, get_store
from leadflow.buyer_agent import BuyerContextAgent
from leadflow.data_loader import load_enrichment
from leadflow.demo import demo_samples
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.routing import RoutingAgent
from leadflow.store import DecisionStore

client = TestClient(app)
ROWS = load_enrichment()


@pytest.fixture(autouse=True)
def offline(tmp_path):
    store = DecisionStore(tmp_path / "leadflow.db")
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_buyer_agent] = lambda: BuyerContextAgent(MockTheirStack(ROWS), MockPostHog(ROWS))
    app.dependency_overrides[get_routing_agent] = lambda: RoutingAgent()
    yield
    app.dependency_overrides.clear()


def send(submission_id):
    lead = next(s["lead"] for s in demo_samples() if s["lead"]["submission_id"] == submission_id)
    return client.post("/leads", json=lead).json()


def test_page_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>LeadFlow AI</title>" in response.text
    assert "Send lead" in response.text


def test_eval_summary_for_demo():
    body = client.get("/demo/eval-summary").json()
    assert body["metrics"]["leads"] == 15
    assert body["metrics"]["tier_accuracy"] == 1.0
    assert body["metrics"]["critical_errors"] == 0
    assert body["buyer"]["tier_accuracy"] == 1.0
    assert body["routing"]["proposal_valid_rate"] == 1.0
    assert body["capacity"]["check_pass_rate"] == 1.0
    assert "buyer" in body["commands"] and "routing" in body["phoenix"]
    assert len(body["talking_points"]) >= 4


def test_samples_cover_the_workbook_and_scenarios():
    samples = client.get("/demo/samples").json()
    groups = [s["group"] for s in samples]
    assert groups.count("Workbook: new prospects") == 15
    assert groups.count("Workbook: excluded") == 5
    assert groups.count("Scenarios") == 6
    assert len({s["lead"]["submission_id"] for s in samples}) == len(samples)


def test_every_sample_passes_the_schema():
    for sample in demo_samples():
        assert client.post("/leads", json=sample["lead"]).status_code == 202, sample["label"]


@pytest.mark.parametrize(
    "submission_id, status",
    [
        ("sub_demo_cust", "preserve_owner"),
        ("sub_demo_twoown", "ownership_exception"),
        ("sub_demo_comp", "dropped"),
        ("sub_demo_pk", "no_eligible_reps"),
        ("sub_demo_lost", "new_prospect"),
    ],
)
def test_scenarios_reach_their_path(submission_id, status):
    assert send(submission_id)["status"] == status


def test_closed_lost_scenario_notes_the_earlier_loss():
    notes = {row["rep_id"]: row["notes"] for row in send("sub_demo_lost")["capacity"]["reps"]}
    assert "Previously lost a deal at FadedChat." in notes["rep_025"]


def test_duplicate_scenario_after_finpay():
    assert send("sub_8f92a1c4")["status"] == "new_prospect"
    body = send("sub_demo_dup")
    assert body["status"] == "duplicate"
    assert body["triage"]["duplicate_of"] == "sub_8f92a1c4"
