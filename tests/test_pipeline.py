import pytest

from leadflow.data_loader import load_accounts, load_leads, load_reps, load_territories
from leadflow.models import Lead
from leadflow.pipeline import triage
from leadflow.store import DecisionStore


@pytest.fixture(scope="module")
def data():
    return load_accounts(), load_reps(), load_territories()


@pytest.fixture
def store(tmp_path):
    return DecisionStore(tmp_path / "leadflow.db")


def run(lead, data, store):
    accounts, reps, territories = data
    return triage(lead, accounts, reps, territories, store)


def make_lead(**overrides):
    fields = {
        "submission_id": "sub_pipe0001",
        "submitted_at": "2026-11-16T15:00:00Z",
        "first_name": "Ana",
        "last_name": "Ruiz",
        "work_email": "ana@acme.es",
        "role": "VP",
        "country": "ES",
        "state": None,
        "company": "Acme",
        "company_size": "101-500",
        "phone_number": "+34915550100",
        "inquiry_reason": "I want to evaluate BeaconDesk for my organization",
    }
    return Lead(**{**fields, **overrides})


def test_workbook_leads_follow_their_labeled_path(data, store):
    leads, expectations = load_leads()
    for lead, expected in sorted(zip(leads, expectations), key=lambda pair: pair[0].submitted_at):
        result = run(lead, data, store)
        if expected.excluded:
            assert result.outcome in ("self_serve", "dropped"), lead.submission_id
        else:
            assert result.outcome == "new_prospect", lead.submission_id
            assert result.shortlist.rep_ids


def test_every_submission_is_recorded(data, store):
    run(make_lead(company_size="1-15"), data, store)
    saved = store.submission("sub_pipe0001")
    assert saved["outcome"] == "self_serve"
    assert saved["domain"] == "acme.es"


def test_existing_account_skips_the_rep_filter(data, store):
    result = run(make_lead(work_email="ana@harborpay.com", country="US", state="NY"), data, store)
    assert result.outcome == "preserve_owner"
    assert result.assigned_rep_id == "rep_002"
    assert result.shortlist is None
    assert store.latest("sub_pipe0001")["assigned_rep_id"] == "rep_002"
    assert store.submission("sub_pipe0001")["assigned_rep_id"] == "rep_002"


def test_ambiguous_account_is_an_exception(data, store):
    result = run(make_lead(work_email="ana@atlasgroup.com", country="US", state="NY"), data, store)
    assert result.outcome == "ownership_exception"
    assert result.assigned_rep_id is None
    assert result.ownership.candidate_rep_ids == ("rep_009", "rep_038")


def test_resubmission_from_the_same_email_is_a_duplicate(data, store):
    run(make_lead(work_email="ana@harborpay.com", country="US", state="NY"), data, store)
    again = make_lead(submission_id="sub_pipe0002", submitted_at="2026-11-17T15:00:00Z",
                      work_email="ana@harborpay.com", country="US", state="NY")
    result = run(again, data, store)
    assert result.outcome == "duplicate"
    assert result.duplicate_of == "sub_pipe0001"
    assert result.assigned_rep_id == "rep_002"
    assert store.count("sub_pipe0002") == 0


def test_same_domain_lead_continues_with_the_related_lead(data, store):
    run(make_lead(), data, store)
    colleague = make_lead(submission_id="sub_pipe0002", submitted_at="2026-11-16T16:00:00Z",
                          work_email="luis@acme.es")
    result = run(colleague, data, store)
    assert result.outcome == "new_prospect"
    assert result.related_submission_id == "sub_pipe0001"
    assert result.related_rep_id is None


def test_replaying_a_lead_gives_the_same_result(data, store):
    lead = make_lead()
    assert run(lead, data, store) == run(lead, data, store)


def test_after_hours_is_reported(data, store):
    result = run(make_lead(submitted_at="2026-11-16T21:00:00Z"), data, store)
    assert result.outcome == "new_prospect"
    assert result.shortlist.after_hours is True
    assert result.reason == "5 eligible reps in EMEA-Southern, all outside working hours."


def test_unmapped_country_has_no_eligible_reps(data, store):
    result = run(make_lead(country="ZZ"), data, store)
    assert result.outcome == "no_eligible_reps"
    assert result.reason == "No eligible rep in unmapped country ZZ."
