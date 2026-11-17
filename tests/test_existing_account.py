import dataclasses

import pytest

from leadflow.accounts import match_domain, resolve_existing_account
from leadflow.data_loader import load_accounts, load_reps
from leadflow.models import Lead
from leadflow.store import DecisionStore


@pytest.fixture(scope="module")
def accounts():
    return load_accounts()


@pytest.fixture(scope="module")
def reps():
    return load_reps()


def make_lead(email, **overrides):
    fields = {
        "submission_id": "sub_acct0001",
        "submitted_at": "2026-11-16T15:00:00Z",
        "first_name": "Priya",
        "last_name": "Shah",
        "work_email": email,
        "role": "Director",
        "country": "US",
        "state": "NY",
        "company": "Acme",
        "company_size": "101-500",
        "phone_number": "+12125550100",
        "inquiry_reason": "I want to buy licenses or upgrade",
        "free_text_note": "Adding 15 seats for the support team.",
    }
    return Lead(**{**fields, **overrides})


def resolve(email, accounts, reps):
    lead = make_lead(email)
    return resolve_existing_account(lead, match_domain(lead.email_domain, accounts), accounts, reps)


def test_customer_keeps_its_owner(accounts, reps):
    decision = resolve("priya@harborpay.com", accounts, reps)
    assert decision.outcome == "preserve_owner"
    assert decision.assigned_rep_id == "rep_002"
    assert decision.account_ids == ("acc_harborpay",)
    assert decision.brief.startswith("Owner kept: Mark Rivera (rep_002).")
    assert "HarborPay (acc_harborpay), Customer" in decision.brief
    assert "Note: Adding 15 seats for the support team." in decision.brief


def test_open_opportunity_brief_shows_the_stage(accounts, reps):
    decision = resolve("priya@nimbusclinic.com", accounts, reps)
    assert decision.assigned_rep_id == "rep_001"
    assert "opportunity Patient chat rollout at Evaluation" in decision.brief


def test_two_owners_raise_an_exception(accounts, reps):
    decision = resolve("priya@atlasgroup.com", accounts, reps)
    assert decision.outcome == "ownership_exception"
    assert decision.assigned_rep_id is None
    assert decision.candidate_rep_ids == ("rep_009", "rep_038")
    assert decision.reason == "Active records on atlasgroup.com have 2 owners."
    assert "Jessica Taylor (rep_009), Gabriel Morel (rep_038)" in decision.brief


def test_missing_owner_raises_an_exception(accounts, reps):
    decision = resolve("priya@orphanstack.io", accounts, reps)
    assert decision.outcome == "ownership_exception"
    assert decision.candidate_rep_ids == ()
    assert decision.reason == "An active record on orphanstack.io has no owner."
    assert "Candidate owners: none." in decision.brief


def test_owner_missing_from_roster_raises_an_exception(accounts, reps):
    harbor = next(account for account in accounts if account.account_id == "acc_harborpay")
    stale = [dataclasses.replace(harbor, owner_rep_id="rep_999")]
    decision = resolve("priya@harborpay.com", stale, reps)
    assert decision.outcome == "ownership_exception"
    assert decision.reason == "Account owner rep_999 is not on the rep roster."
    assert "rep_999 (not on roster)" in decision.brief


def test_new_prospect_is_not_handled_here(accounts, reps):
    with pytest.raises(ValueError):
        resolve("priya@example.com", accounts, reps)


def test_decisions_are_persisted_per_run(tmp_path, accounts, reps):
    store = DecisionStore(tmp_path / "decisions.db")
    decision = resolve("priya@atlasgroup.com", accounts, reps)
    store.save(decision)
    store.save(decision)
    assert store.count("sub_acct0001") == 2
    saved = store.latest("sub_acct0001")
    assert saved["path"] == "existing_account"
    assert saved["outcome"] == "ownership_exception"
    assert saved["assigned_rep_id"] is None
    assert saved["candidate_rep_ids"] == ["rep_009", "rep_038"]
    assert saved["account_ids"] == ["acc_atlas_na", "acc_atlas_eu"]
    assert saved["brief"] == decision.brief
    assert saved["decided_at"].endswith("Z")


def test_unknown_submission_has_no_decision(tmp_path):
    assert DecisionStore(tmp_path / "decisions.db").latest("sub_missing") is None
