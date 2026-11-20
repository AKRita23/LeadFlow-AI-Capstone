import dataclasses
import threading

import pytest

from leadflow.capacity import CapacityService
from leadflow.commit import commit_assignment, day_and_week, live_reps
from leadflow.data_loader import load_accounts
from leadflow.models import EligibleReps, TriageResult
from leadflow.routing import RoutingAgent
from leadflow.store import DecisionStore
from tests.test_routing import REPS, setup

# FinPay: shortlist rep_003, rep_035, rep_027, rep_043, rep_011; the ranker puts rep_003 then rep_027 first.


@pytest.fixture
def store(tmp_path):
    return DecisionStore(tmp_path / "leadflow.db")


def finpay(submission_id="sub_8f92a1c4"):
    lead, buyer, capacity, shortlist = setup()
    lead = lead.model_copy(update={"submission_id": submission_id})
    shortlist = dataclasses.replace(shortlist, submission_id=submission_id)
    routing = RoutingAgent()(lead, buyer, capacity.model_copy(update={"submission_id": submission_id}), shortlist)
    return lead, buyer, routing, shortlist


def roster(**changes_by_rep):
    return [dataclasses.replace(rep, **changes_by_rep.get(rep.rep_id, {})) for rep in REPS]


def record(store, lead):
    store.record_submission(lead, TriageResult(lead.submission_id, "new_prospect", "test"))


def test_day_and_week():
    assert day_and_week("2026-11-16T11:58:23Z") == ("2026-11-16", "2026-W47")
    assert day_and_week("2027-01-01T09:00:00Z") == ("2027-01-01", "2026-W53")


def test_primary_gets_the_lead_and_backup_is_kept(store):
    lead, buyer, routing, shortlist = finpay()
    record(store, lead)
    assignment = commit_assignment(lead, buyer, routing, shortlist, REPS, store)
    assert assignment.outcome == "assigned"
    assert (assignment.rep_id, assignment.backup_rep_id) == ("rep_003", "rep_027")
    assert assignment.skipped == []
    assert assignment.intent_points == buyer.intent_score
    assert assignment.already_committed is False
    assert store.submission(lead.submission_id)["assigned_rep_id"] == "rep_003"
    saved = store.assignment(lead.submission_id)
    assert saved["method"] == "ranker"
    assert saved["urgency_tier"] == "High"


def test_committing_twice_returns_the_first_assignment(store):
    lead, buyer, routing, shortlist = finpay()
    first = commit_assignment(lead, buyer, routing, shortlist, REPS, store)
    again = commit_assignment(lead, buyer, routing, shortlist, roster(rep_003={"leads_assigned_today": 8}), store)
    assert again.already_committed is True
    assert again.rep_id == first.rep_id == "rep_003"
    assert store.rep_load("2026-11-16", "2026-W47") == {"rep_003": (1, buyer.intent_score)}


def test_primary_at_cap_hands_the_lead_to_the_backup(store):
    lead, buyer, routing, shortlist = finpay()
    assignment = commit_assignment(lead, buyer, routing, shortlist, roster(rep_003={"leads_assigned_today": 8}), store)
    assert assignment.rep_id == "rep_027"
    assert assignment.skipped == [["rep_003", "daily_cap"]]
    assert assignment.backup_rep_id not in (None, "rep_003", "rep_027")


def test_inactive_rep_is_skipped(store):
    lead, buyer, routing, shortlist = finpay()
    assignment = commit_assignment(lead, buyer, routing, shortlist, roster(rep_003={"working_status": "OOO"}), store)
    assert assignment.rep_id == "rep_027"
    assert assignment.skipped == [["rep_003", "inactive"]]


def test_committed_leads_count_toward_the_cap(store):
    # rep_003 starts with 1 of 8, so 7 more commits fill the day.
    for i in range(7):
        lead, buyer, routing, shortlist = finpay(f"sub_fill{i}")
        assert commit_assignment(lead, buyer, routing, shortlist, REPS, store).rep_id == "rep_003"
    lead, buyer, routing, shortlist = finpay("sub_next")
    assignment = commit_assignment(lead, buyer, routing, shortlist, REPS, store)
    assert assignment.rep_id == "rep_027"
    assert assignment.skipped == [["rep_003", "daily_cap"]]


def test_nobody_free_goes_to_revops(store):
    lead, buyer, routing, shortlist = finpay()
    full = {rep_id: {"leads_assigned_today": 8} for rep_id in shortlist.rep_ids}
    assignment = commit_assignment(lead, buyer, routing, shortlist, roster(**full), store)
    assert assignment.outcome == "revops_queue"
    assert assignment.rep_id is None and assignment.backup_rep_id is None
    assert sorted(rep for rep, _ in assignment.skipped) == sorted(shortlist.rep_ids)
    assert store.rep_load("2026-11-16", "2026-W47") == {}


def test_proposal_failing_the_check_falls_back_to_the_ranker_order(store):
    lead, buyer, routing, shortlist = finpay()
    bad = routing.proposal.model_copy(update={"primary_rep_id": "rep_999"})
    routing = routing.model_copy(update={"proposal": bad})
    assert commit_assignment(lead, buyer, routing, shortlist, REPS, store).rep_id == "rep_003"


def test_live_reps_add_committed_load(store):
    lead, buyer, routing, shortlist = finpay()
    commit_assignment(lead, buyer, routing, shortlist, REPS, store)
    live = {rep.rep_id: rep for rep in live_reps(REPS, store, lead.submitted_at)}
    base = {rep.rep_id: rep for rep in REPS}
    assert live["rep_003"].leads_assigned_today == base["rep_003"].leads_assigned_today + 1
    assert live["rep_003"].weekly_intent_points == base["rep_003"].weekly_intent_points + buyer.intent_score
    assert live["rep_027"] == base["rep_027"]
    own_excluded = {r.rep_id: r for r in live_reps(REPS, store, lead.submitted_at, lead.submission_id)}
    assert own_excluded["rep_003"] == base["rep_003"]


def test_other_days_do_not_count_toward_today(store):
    lead, buyer, routing, shortlist = finpay()
    commit_assignment(lead, buyer, routing, shortlist, REPS, store)
    next_day = {r.rep_id: r for r in live_reps(REPS, store, "2026-11-17T10:00:00Z")}
    assert next_day["rep_003"].leads_assigned_today == 1
    assert next_day["rep_003"].weekly_intent_points == dict((r.rep_id, r) for r in REPS)["rep_003"].weekly_intent_points + 98


def test_capacity_scores_see_live_load(store):
    lead, buyer, routing, shortlist = finpay()
    for i in range(5):
        lead_i, buyer_i, routing_i, shortlist_i = finpay(f"sub_fill{i}")
        commit_assignment(lead_i, buyer_i, routing_i, shortlist_i, REPS, store)
    row = CapacityService(load_accounts())(shortlist, live_reps(REPS, store, lead.submitted_at), lead.email_domain).reps[0]
    assert row.rep_id == "rep_003"
    assert row.leads_remaining_today == 2
    assert row.bandwidth_tier != "High"


def test_concurrent_commits_never_exceed_the_cap(store):
    # rep_003 and rep_027 have one slot left each; the other shortlisted reps are full.
    tight = roster(
        rep_003={"leads_assigned_today": 7},
        rep_027={"leads_assigned_today": 7},
        rep_035={"leads_assigned_today": 8},
        rep_043={"leads_assigned_today": 8},
        rep_011={"leads_assigned_today": 8},
    )
    leads = [finpay(f"sub_race{i}") for i in range(6)]
    results, errors = [], []
    barrier = threading.Barrier(len(leads))

    def run(args):
        try:
            barrier.wait()
            results.append(commit_assignment(*args, tight, store))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(args,)) for args in leads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assigned = sorted(a.rep_id for a in results if a.outcome == "assigned")
    assert assigned == ["rep_003", "rep_027"]
    assert sum(a.outcome == "revops_queue" for a in results) == 4
