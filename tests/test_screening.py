import pytest

from leadflow.data_loader import load_leads
from leadflow.models import Lead, PriorSubmission
from leadflow.screening import find_duplicate, screen_lead


def make_lead(**overrides):
    fields = {
        "submission_id": "sub_scrn0001",
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


def prior(submission_id, email, submitted_at="2026-11-10T10:00:00Z", rep=None):
    return PriorSubmission(submission_id, submitted_at, email, email.split("@")[1], "new_prospect", rep)


def test_workbook_exclusions_are_caught_and_valid_leads_pass():
    leads, expectations = load_leads()
    for lead, expected in zip(leads, expectations):
        result = screen_lead(lead)
        assert (result.outcome != "pass") == expected.excluded, (lead.submission_id, result)


@pytest.mark.parametrize(
    "submission_id, outcome, reason",
    [
        ("sub_ex01z9m2", "self_serve", "micro_company"),
        ("sub_ex02k4x8", "self_serve", "micro_company"),
        ("sub_ex03p7v1", "self_serve", "recruiting_agency"),
        ("sub_ex04q2w5", "drop", "test_submission"),
        ("sub_ex05h8n9", "self_serve", "media"),
    ],
)
def test_workbook_exclusion_reasons(submission_id, outcome, reason):
    leads, _ = load_leads()
    lead = next(lead for lead in leads if lead.submission_id == submission_id)
    result = screen_lead(lead)
    assert (result.outcome, result.reason) == (outcome, reason)


@pytest.mark.parametrize(
    "overrides, outcome, reason",
    [
        ({"work_email": "noreply@acme.es"}, "drop", "test_submission"),
        ({"work_email": "ana@staging.acme.es"}, "drop", "test_submission"),
        ({"work_email": "ana@beacondesk.io"}, "drop", "internal"),
        ({"work_email": "ana@intercom.io"}, "drop", "competitor"),
        ({"work_email": "ana@eu.zendesk.com"}, "drop", "competitor"),
        ({"company": "Bright Staffing Partners"}, "self_serve", "recruiting_agency"),
        ({"company": "Daily Support News"}, "self_serve", "media"),
        ({"company_size": "1-15"}, "self_serve", "micro_company"),
    ],
)
def test_screen_rules(overrides, outcome, reason):
    result = screen_lead(make_lead(**overrides))
    assert (result.outcome, result.reason) == (outcome, reason)


@pytest.mark.parametrize("email", ["ana@contest.es", "ana@latest-tech.com", "ana@robotics.de"])
def test_noise_words_must_be_whole_tokens(email):
    assert screen_lead(make_lead(work_email=email)).outcome == "pass"


def test_notes_are_not_screened():
    note = "Student support ticketing spike during semester kickoff."
    assert screen_lead(make_lead(free_text_note=note)).outcome == "pass"


def test_same_email_is_a_duplicate():
    match = find_duplicate(make_lead(), [prior("sub_old", "ana@acme.es", rep="rep_003")])
    assert (match.kind, match.prior_submission_id, match.prior_rep_id) == ("same_email", "sub_old", "rep_003")


def test_same_domain_prefers_a_lead_with_a_rep():
    history = [
        prior("sub_a", "luis@acme.es", "2026-11-12T10:00:00Z", rep="rep_011"),
        prior("sub_b", "eva@acme.es", "2026-11-15T10:00:00Z"),
    ]
    match = find_duplicate(make_lead(), history)
    assert (match.kind, match.prior_submission_id, match.prior_rep_id) == ("same_domain", "sub_a", "rep_011")


def test_personal_domain_is_not_a_company_match():
    lead = make_lead(work_email="ana@gmail.com")
    assert find_duplicate(lead, [prior("sub_old", "luis@gmail.com")]).kind == "none"


def test_old_and_later_submissions_are_ignored():
    history = [
        prior("sub_old", "ana@acme.es", "2026-10-01T10:00:00Z"),
        prior("sub_later", "ana@acme.es", "2026-11-20T10:00:00Z"),
    ]
    assert find_duplicate(make_lead(), history).kind == "none"


def test_replay_of_the_same_submission_is_not_a_duplicate():
    assert find_duplicate(make_lead(), [prior("sub_scrn0001", "ana@acme.es")]).kind == "none"
