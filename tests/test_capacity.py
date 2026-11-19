import dataclasses
from collections import Counter

import pytest

from leadflow.capacity import CapacityService, competitor_strengths, rep_capacity
from leadflow.contracts import CapacityMatrix, check_capacity
from leadflow.data_loader import load_accounts, load_reps
from leadflow.models import EligibleReps

REPS = load_reps()
REPS_BY_ID = {rep.rep_id: rep for rep in REPS}
SERVICE = CapacityService(load_accounts())


def shortlist(*rep_ids, submission_id="sub_cap0001"):
    return EligibleReps(
        submission_id=submission_id,
        territory="EMEA-Southern",
        segment="Mid-Market",
        language="es",
        rep_ids=rep_ids,
        excluded=(),
    )


def with_fields(rep_id, **changes):
    return dataclasses.replace(REPS_BY_ID[rep_id], **changes)


def test_matches_the_contract_example():
    """rep_003 is the worked example in CapacityMatrix's schema."""
    expected = CapacityMatrix.model_json_schema()["examples"][0]["reps"][0]
    row = SERVICE(shortlist("rep_003"), REPS, "finpay.es").reps[0]
    for field, value in expected.items():
        assert getattr(row, field) == value, field


def test_one_row_per_shortlisted_rep_in_order():
    sl = shortlist("rep_011", "rep_003", "rep_043")
    matrix = SERVICE(sl, REPS, "finpay.es")
    assert [row.rep_id for row in matrix.reps] == ["rep_011", "rep_003", "rep_043"]
    assert matrix.submission_id == "sub_cap0001"
    assert matrix.tool_status == {"salesforce": "mocked", "calendar": "mocked"}
    assert check_capacity(matrix, sl) == []


def test_rep_missing_from_roster_is_an_error():
    with pytest.raises(ValueError, match="rep_999"):
        SERVICE(shortlist("rep_003", "rep_999"), REPS, "finpay.es")


@pytest.mark.parametrize(
    "late, expected",
    [(0, "Low"), (3, "Low"), (4, "Medium"), (5, "Medium"), (6, "High"), (9, "High")],
)
def test_stage_pressure_from_late_stage_opps(late, expected):
    rep = with_fields("rep_003", proposal_opps=late, negotiation_opps=0)
    assert rep_capacity(rep).stage_pressure == expected


def test_no_flags_is_high_bandwidth():
    row = rep_capacity(REPS_BY_ID["rep_003"])
    assert row.bandwidth_tier == "High"
    assert row.notes == "No load flags."


@pytest.mark.parametrize(
    "changes, note",
    [
        ({"leads_assigned_today": 5}, "3 leads left today"),
        ({"overdue_tasks": 3, "unworked_leads": 1}, "backlog of 4 (3 overdue tasks, 1 unworked lead)"),
        ({"meetings_today_hours": 5.0}, "5h of meetings today"),
        ({"weekly_intent_points": 400}, "400 intent points routed this week"),
        ({"discovery_opps": 10}, "16 open opportunities"),
    ],
)
def test_each_flag_alone_gives_medium_bandwidth(changes, note):
    row = rep_capacity(with_fields("rep_003", **changes))
    assert row.bandwidth_tier == "Medium"
    assert note in row.notes


def test_values_just_below_thresholds_raise_no_flag():
    rep = with_fields(
        "rep_003",
        leads_assigned_today=4,
        overdue_tasks=3,
        unworked_leads=0,
        meetings_today_hours=4.9,
        weekly_intent_points=399,
        discovery_opps=7,
    )
    assert rep_capacity(rep).bandwidth_tier == "High"


def test_two_flags_give_low_bandwidth():
    rep = with_fields("rep_003", leads_assigned_today=6, meetings_today_hours=6.0)
    row = rep_capacity(rep)
    assert row.bandwidth_tier == "Low"
    assert row.notes == "Load: 2 leads left today; 6h of meetings today."


def test_stage_pressure_does_not_lower_bandwidth():
    rep = with_fields("rep_003", proposal_opps=5, negotiation_opps=3, discovery_opps=2)
    row = rep_capacity(rep)
    assert row.stage_pressure == "High"
    assert row.bandwidth_tier == "High"


def test_leads_remaining_never_negative():
    assert rep_capacity(with_fields("rep_003", leads_assigned_today=10)).leads_remaining_today == 0


def test_pipeline_totals():
    rep = REPS_BY_ID["rep_003"]
    row = rep_capacity(rep)
    assert row.open_opps == 14
    assert row.late_stage_opps == 4
    assert row.open_pipeline_arr == 392995


@pytest.mark.parametrize(
    "affinity, expected",
    [
        ("High (Zendesk / Drift)", ["Zendesk", "Drift"]),
        ("High (Tidio / Drift / Intercom)", ["Tidio", "Drift", "Intercom"]),
        ("", []),
        ("High (Freshdesk)", []),
    ],
)
def test_competitor_strengths(affinity, expected):
    assert competitor_strengths(affinity) == expected


def test_closed_lost_history_is_noted_for_the_owner_only():
    # acc_faded: FadedChat, fadedchat.io, Closed Lost, owned by rep_025.
    matrix = SERVICE(shortlist("rep_025", "rep_001"), REPS, "fadedchat.io")
    owner, other = matrix.reps
    assert "Previously lost a deal at FadedChat." in owner.notes
    assert "Previously lost" not in other.notes


def test_active_accounts_are_not_closed_lost_history():
    # harborpay.com is an active Customer owned by rep_002.
    row = SERVICE(shortlist("rep_002"), REPS, "harborpay.com").reps[0]
    assert "Previously lost" not in row.notes


def test_roster_tier_spread():
    """Guards the documented split; update the module docstring if thresholds change."""
    tiers = Counter(rep_capacity(rep).bandwidth_tier for rep in REPS)
    assert tiers == {"High": 16, "Medium": 20, "Low": 14}
