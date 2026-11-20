import dataclasses
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from leadflow.api import app, get_buyer_agent, get_routing_agent, get_store
from leadflow.buyer_agent import BuyerContextAgent
from leadflow.contracts import Assignment
from leadflow.data_loader import load_enrichment, load_reps
from leadflow.demo import demo_samples
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.handoff import now_utc, run_watchdog, send_assignment, tracking
from leadflow.routing import RoutingAgent
from leadflow.store import DecisionStore
from tests.test_routing import setup

client = TestClient(app)
ROWS = load_enrichment()
REPS = load_reps()
FINPAY = "sub_8f92a1c4"  # High urgency; ranker assigns rep_003 with backup rep_027
TRAVELPASS = "sub_5q99w3k2"  # Medium urgency


@pytest.fixture(autouse=True)
def store(tmp_path):
    test_store = DecisionStore(tmp_path / "leadflow.db")
    app.dependency_overrides[get_store] = lambda: test_store
    app.dependency_overrides[get_buyer_agent] = lambda: BuyerContextAgent(MockTheirStack(ROWS), MockPostHog(ROWS))
    app.dependency_overrides[get_routing_agent] = lambda: RoutingAgent()
    yield test_store
    app.dependency_overrides.clear()


def send(submission_id):
    lead = next(s["lead"] for s in demo_samples() if s["lead"]["submission_id"] == submission_id)
    return client.post("/leads", json=lead).json()


def watchdog(minutes_ahead):
    return client.post("/watchdog/run", params={"minutes_ahead": minutes_ahead}).json()


def set_status(submission_id, status):
    return client.post(f"/leads/{submission_id}/status", json={"status": status})


# Handoff


def test_assigned_rep_gets_the_brief_and_a_time_limit():
    body = send(FINPAY)
    track = body["tracking"]
    assert track["status"] == "New"
    [message] = track["messages"]
    assert message["recipient"] == "rep_003"
    assert message["kind"] == "assignment"
    assert message["subject"].startswith("New lead: Elena Rostova, FinPay Technologies (High, intent")
    assert body["routing"]["proposal"]["rep_brief"] in message["body"]
    assert "Mark it Working within 10 minutes, or it moves to rep_027." in message["body"]


def test_resending_does_not_send_again():
    send(FINPAY)
    assert len(send(FINPAY)["tracking"]["messages"]) == 1


def test_no_free_rep_messages_revops(store):
    lead, buyer, capacity, shortlist = setup()
    routing = RoutingAgent()(lead, buyer, capacity, shortlist)
    queued = Assignment(
        submission_id=lead.submission_id, outcome="revops_queue", skipped=[["rep_003", "daily_cap"]],
        intent_points=buyer.intent_score, committed_at="2026-11-16T12:00:00Z",
    )
    send_assignment(store, lead, buyer, routing, queued)
    [message] = tracking(store, lead.submission_id).messages
    assert message.recipient == "revops"
    assert "Skipped: rep_003 (daily_cap)." in message.body


def test_existing_account_owner_gets_the_ownership_brief():
    [message] = send("sub_demo_cust")["tracking"]["messages"]
    assert message["recipient"] == "rep_002"
    assert message["kind"] == "existing_account"
    assert message["body"].startswith("Owner kept: Mark Rivera (rep_002).")


def test_ownership_exception_goes_to_revops():
    [message] = send("sub_demo_twoown")["tracking"]["messages"]
    assert (message["recipient"], message["kind"]) == ("revops", "ownership_exception")


def test_outbox_lists_messages_newest_first():
    send(FINPAY)
    send("sub_demo_cust")
    assert [m["recipient"] for m in client.get("/outbox").json()] == ["rep_002", "rep_003"]


# Status


def test_rep_moves_the_lead_to_working_then_converted():
    send(FINPAY)
    assert set_status(FINPAY, "Working").json()["status"] == "Working"
    track = set_status(FINPAY, "Converted").json()
    assert [e["status"] for e in track["status_history"]] == ["New", "Working", "Converted"]
    assert track["status_history"][1]["actor"] == "rep"


@pytest.mark.parametrize("status", ["New", "Converted", "Escalated"])
def test_moves_that_skip_a_step_are_rejected(status):
    send(FINPAY)
    response = set_status(FINPAY, status)
    assert response.status_code == 409
    assert "can move to Disqualified, Working" in response.json()["detail"]


def test_status_for_an_unassigned_lead_is_404():
    assert set_status("sub_unknown", "Working").status_code == 404


# Watchdog


def test_watchdog_leaves_leads_inside_their_time_limit():
    send(FINPAY)
    report = watchdog(5)
    assert report["checked"] == 1
    assert report["actions"] == []


def test_watchdog_reassigns_a_stalled_high_urgency_lead_to_the_backup(store):
    send(FINPAY)
    [action] = watchdog(11)["actions"]
    assert (action["action"], action["from_rep_id"], action["to_rep_id"]) == ("reassigned", "rep_003", "rep_027")
    track = client.get(f"/leads/{FINPAY}/tracking").json()
    assert track["assignment"]["rep_id"] == "rep_027"
    assert track["status"] == "New"
    assert track["status_history"][-1]["actor"] == "watchdog"
    kinds = {(m["recipient"], m["kind"]) for m in track["messages"]}
    assert {("rep_027", "reassigned_to"), ("rep_003", "reassigned_away")} <= kinds
    moved = next(m for m in track["messages"] if m["kind"] == "reassigned_to")
    assert "Why you:" in moved["body"]
    assert store.submission(FINPAY)["assigned_rep_id"] == "rep_027"
    assert store.rep_load("2026-11-16", "2026-W47") == {"rep_027": (1, track["assignment"]["intent_points"])}


def test_a_second_stall_escalates_to_revops():
    send(FINPAY)
    watchdog(11)
    assert watchdog(15)["actions"] == []  # the clock restarted at reassignment
    [action] = watchdog(25)["actions"]
    assert action["action"] == "escalated"
    assert action["reason"] == "Stalled again with rep_027 after reassignment."
    track = client.get(f"/leads/{FINPAY}/tracking").json()
    assert track["status"] == "Escalated"
    assert track["assignment"]["outcome"] == "revops_queue"
    assert track["assignment"]["rep_id"] is None
    assert ("revops", "escalated") in {(m["recipient"], m["kind"]) for m in track["messages"]}


def test_working_leads_are_left_alone():
    send(FINPAY)
    set_status(FINPAY, "Working")
    assert watchdog(60)["actions"] == []


def test_backup_at_cap_escalates_instead(store):
    send(FINPAY)
    full = [dataclasses.replace(r, leads_assigned_today=8) if r.rep_id == "rep_027" else r for r in REPS]
    [action] = run_watchdog(store, full, now_utc() + timedelta(minutes=11)).actions
    assert action.action == "escalated"
    assert action.reason == "rep_003 did not start work, and backup rep_027 is at the daily cap."


def test_medium_urgency_is_only_reported_overdue():
    body = send(TRAVELPASS)
    assert body["buyer"]["urgency_tier"] == "Medium"
    rep = body["assigned_rep_id"]
    assert watchdog(30)["actions"] == []
    [action] = watchdog(61)["actions"]
    assert (action["action"], action["from_rep_id"], action["to_rep_id"]) == ("overdue", rep, None)
    assert client.get(f"/leads/{TRAVELPASS}/tracking").json()["assignment"]["rep_id"] == rep


def test_reset_clears_messages_and_status(store):
    send(FINPAY)
    client.post("/demo/reset")
    track = client.get(f"/leads/{FINPAY}/tracking").json()
    assert track["messages"] == [] and track["status_history"] == [] and track["assignment"] is None
