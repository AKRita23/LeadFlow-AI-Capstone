import dataclasses

import pytest

from leadflow.data_loader import load_leads, load_reps, load_territories
from leadflow.eligibility import filter_eligible_reps, in_working_hours, segment_for, territory_for
from leadflow.models import US_STATE_CODES, Lead


@pytest.fixture(scope="module")
def reps():
    return load_reps()


@pytest.fixture(scope="module")
def territories():
    return load_territories()


@pytest.fixture(scope="module")
def valid_leads():
    leads, expectations = load_leads()
    return [lead for lead, expected in zip(leads, expectations) if not expected.excluded]


def make_lead(**overrides):
    fields = {
        "submission_id": "sub_elig0001",
        "submitted_at": "2026-11-16T15:00:00Z",
        "first_name": "Ana",
        "last_name": "Ruiz",
        "work_email": "ana@example.es",
        "role": "VP",
        "country": "ES",
        "state": None,
        "company": "Example",
        "company_size": "101-500",
        "phone_number": "+34915550100",
        "inquiry_reason": "I want to evaluate BeaconDesk for my organization",
    }
    return Lead(**{**fields, **overrides})


def test_territory_file_covers_every_us_state_and_known_territory(territories, reps):
    us_states = {state for country, state in territories if country == "US"}
    assert us_states == US_STATE_CODES
    assert {rule.territory for rule in territories.values()} == {rep.territory for rep in reps}
    rep_languages = {language for rep in reps for language in rep.languages}
    assert {rule.language for rule in territories.values()} <= rep_languages


@pytest.mark.parametrize(
    "country, state, territory, language",
    [
        ("US", "CA", "NA-West", "en"),
        ("US", "WA", "NA-West", "en"),
        ("US", "NY", "NA-East", "en"),
        ("US", "TX", "NA-East", "en"),
        ("GB", None, "EMEA-UK/IE", "en"),
        ("FR", None, "EMEA-Central", "fr"),
        ("CH", None, "EMEA-Central", "en"),
        ("DE", None, "EMEA-Northern", "de"),
        ("AE", None, "EMEA-Southern", "en"),
        ("MX", None, "LATAM", "es"),
        ("BR", None, "LATAM", "pt"),
        ("JP", None, "APAC", "ja"),
    ],
)
def test_territory(territories, country, state, territory, language):
    rule = territory_for(make_lead(country=country, state=state), territories)
    assert (rule.territory, rule.language) == (territory, language)


def test_unmapped_country_has_no_territory(territories):
    assert territory_for(make_lead(country="ZZ"), territories) is None


def test_segment():
    assert segment_for("16-100") == "SMB"
    assert segment_for("51-250") == "Mid-Market"
    assert segment_for("1001+") == "Enterprise"


def test_workbook_leads_get_rule_passing_shortlists(valid_leads, reps, territories):
    reps_by_id = {rep.rep_id: rep for rep in reps}
    for lead in valid_leads:
        result = filter_eligible_reps(lead, reps, territories)
        assert 1 <= len(result.rep_ids) <= 6, lead.submission_id
        for rep_id in result.rep_ids:
            rep = reps_by_id[rep_id]
            assert rep.territory == result.territory
            assert result.segment in rep.segments
            assert result.language in rep.languages


def test_spanish_mid_market_lead(reps, territories):
    result = filter_eligible_reps(make_lead(), reps, territories)
    assert result.territory == "EMEA-Southern"
    assert set(result.rep_ids) == {"rep_003", "rep_011", "rep_027", "rep_035", "rep_043"}
    assert result.excluded == (("rep_019", "segment"),)
    assert result.after_hours is False


def test_unknown_country_has_no_reps(reps, territories):
    result = filter_eligible_reps(make_lead(country="ZZ"), reps, territories)
    assert result.rep_ids == ()
    assert result.excluded == ()


@pytest.mark.parametrize(
    "change, rule",
    [
        ({"working_status": "Out of Office"}, "availability"),
        ({"leads_assigned_today": 8}, "daily_cap"),
        ({"languages": ["en", "pt"]}, "language"),
        ({"segments": ["SMB"]}, "segment"),
        ({"timezone": "Asia/Tokyo"}, "working_hours"),
    ],
)
def test_each_rule_excludes_a_rep(reps, territories, change, rule):
    changed = [dataclasses.replace(rep, **change) if rep.rep_id == "rep_003" else rep for rep in reps]
    result = filter_eligible_reps(make_lead(), changed, territories)
    assert "rep_003" not in result.rep_ids
    assert ("rep_003", rule) in result.excluded


@pytest.mark.parametrize(
    "submitted_at, expected",
    [
        ("2026-11-16T08:00:00Z", True),
        ("2026-11-16T07:59:59Z", False),
        ("2026-11-16T16:59:59Z", True),
        ("2026-11-16T17:00:00Z", False),
        ("2026-11-14T12:00:00Z", False),
    ],
)
def test_working_hours_in_rep_timezone(reps, submitted_at, expected):
    madrid = next(rep for rep in reps if rep.rep_id == "rep_003")
    assert in_working_hours(madrid, submitted_at) is expected


def test_after_hours_lead_keeps_its_shortlist(reps, territories):
    result = filter_eligible_reps(make_lead(submitted_at="2026-11-16T21:00:00Z"), reps, territories)
    assert result.after_hours is True
    assert set(result.rep_ids) == {"rep_003", "rep_011", "rep_027", "rep_035", "rep_043"}
    assert ("rep_003", "working_hours") not in result.excluded


def test_shortlist_keeps_six_with_most_headroom(reps, territories):
    template = next(rep for rep in reps if rep.rep_id == "rep_003")
    pool = [
        dataclasses.replace(template, rep_id=f"rep_9{i:02d}", leads_assigned_today=assigned)
        for i, assigned in enumerate([5, 1, 4, 0, 2, 3, 6, 1])
    ]
    result = filter_eligible_reps(make_lead(), pool, territories)
    assert result.rep_ids == ("rep_903", "rep_901", "rep_907", "rep_904", "rep_905", "rep_902")


def test_same_lead_gets_the_same_shortlist(valid_leads, reps, territories):
    for lead in valid_leads:
        assert filter_eligible_reps(lead, reps, territories) == filter_eligible_reps(lead, reps, territories)
