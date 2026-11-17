import re

import pytest

from leadflow.data_loader import load_leads, load_reps


@pytest.fixture(scope="module")
def leads_and_expectations():
    return load_leads()


@pytest.fixture(scope="module")
def reps():
    return load_reps()


def test_lead_counts(leads_and_expectations):
    leads, expectations = leads_and_expectations
    assert len(leads) == 20
    assert sum(e.excluded for e in expectations) == 5
    assert len({lead.submission_id for lead in leads}) == 20


def test_company_size_dates_are_restored(leads_and_expectations):
    leads, _ = leads_and_expectations
    for lead in leads:
        assert re.fullmatch(r"\d+-\d+|\d+\+", lead.company_size), lead.company_size
    excluded = [lead for lead in leads if lead.submission_id.startswith("sub_ex")]
    assert {lead.company_size for lead in excluded} == {"1-15"}


def test_phone_numbers_are_e164(leads_and_expectations):
    leads, _ = leads_and_expectations
    for lead in leads:
        assert re.fullmatch(r"\+\d{8,15}", lead.phone_number), lead.phone_number


def test_state_only_for_us(leads_and_expectations):
    leads, _ = leads_and_expectations
    for lead in leads:
        assert (lead.state is not None) == (lead.country == "US"), lead.submission_id


def test_missing_note_is_none(leads_and_expectations):
    leads, _ = leads_and_expectations
    nexus = next(lead for lead in leads if lead.submission_id == "sub_3c55d8e9")
    assert nexus.free_text_note is None


def test_email_domain(leads_and_expectations):
    leads, _ = leads_and_expectations
    assert leads[0].email_domain == "finpay.es"


def test_reps(reps):
    assert len(reps) == 50
    assert len({rep.rep_id for rep in reps}) == 50
    rep = next(r for r in reps if r.rep_id == "rep_003")
    assert rep.languages == ["en", "es", "pt"]
    assert rep.segments == ["Mid-Market", "Enterprise"]
    assert rep.territory == "EMEA-Southern"
    assert rep.daily_lead_cap == 8
