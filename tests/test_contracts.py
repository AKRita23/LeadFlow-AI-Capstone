import pytest
from pydantic import ValidationError

from leadflow.contracts import (
    BuyerAssessment,
    CapacityMatrix,
    RoutingProposal,
    StepTiming,
    check_capacity,
    check_proposal,
)
from leadflow.data_loader import load_leads, load_reps, load_territories
from leadflow.eligibility import filter_eligible_reps
from leadflow.models import EligibleReps


def example(model):
    return model.model_config["json_schema_extra"]["examples"][0]


def shortlist(*rep_ids, submission_id="sub_8f92a1c4"):
    return EligibleReps(submission_id, "EMEA-Southern", "Mid-Market", "es", tuple(rep_ids), ())


def proposal(primary="rep_003", backup="rep_043", submission_id="sub_8f92a1c4"):
    return RoutingProposal.model_validate(
        {**example(RoutingProposal), "submission_id": submission_id, "primary_rep_id": primary,
         "backup_rep_id": backup}
    )


@pytest.mark.parametrize("model", [BuyerAssessment, CapacityMatrix, RoutingProposal])
def test_examples_are_valid_and_round_trip(model):
    parsed = model.model_validate(example(model))
    assert model.model_validate_json(parsed.model_dump_json()) == parsed
    assert model.model_json_schema()["title"] == model.__name__


def test_example_matches_the_workbook():
    leads, expectations = load_leads()
    expected = next(e for e in expectations if e.submission_id == "sub_8f92a1c4")
    buyer = BuyerAssessment.model_validate(example(BuyerAssessment))
    assert buyer.intent_score == expected.expected_intent_score
    assert buyer.urgency_tier == expected.expected_urgency_tier


def test_example_reps_are_on_the_workbook_shortlist():
    leads, _ = load_leads()
    lead = next(lead for lead in leads if lead.submission_id == "sub_8f92a1c4")
    rep_ids = filter_eligible_reps(lead, load_reps(), load_territories()).rep_ids
    assert check_proposal(proposal(), shortlist(*rep_ids)) == []


@pytest.mark.parametrize(
    "change",
    [
        {"intent_score": 0},
        {"intent_score": 101},
        {"urgency_tier": "Med"},
        {"detected_competitor": "Freshdesk"},
        {"rationale": ""},
        {"buying_signals": [f"signal {i}" for i in range(9)]},
        {"tool_status": {"tavily": "ok"}},
    ],
)
def test_buyer_assessment_rejects_bad_output(change):
    with pytest.raises(ValidationError):
        BuyerAssessment.model_validate({**example(BuyerAssessment), **change})


def test_buyer_assessment_rejects_malformed_json():
    with pytest.raises(ValidationError):
        BuyerAssessment.model_validate_json('{"submission_id": "sub_1", "intent_score": "high"}')


def test_capacity_matrix_needs_a_rep():
    with pytest.raises(ValidationError):
        CapacityMatrix.model_validate({**example(CapacityMatrix), "reps": []})


def test_capacity_matrix_must_match_the_shortlist():
    matrix = CapacityMatrix.model_validate(example(CapacityMatrix))
    assert check_capacity(matrix, shortlist("rep_003")) == []
    assert check_capacity(matrix, shortlist("rep_003", "rep_011")) == [
        "Matrix is missing shortlisted reps: rep_011."
    ]
    assert check_capacity(matrix, shortlist("rep_011")) == [
        "Matrix is missing shortlisted reps: rep_011.",
        "Matrix has reps not on the shortlist: rep_003.",
    ]


@pytest.mark.parametrize(
    "candidate, reps, problems",
    [
        (proposal(), ("rep_003", "rep_043"), []),
        (proposal(backup=None), ("rep_003",), []),
        (proposal(primary="rep_999"), ("rep_003", "rep_043"), ["Primary rep rep_999 is not on the shortlist."]),
        (proposal(backup="rep_999"), ("rep_003", "rep_043"), ["Backup rep rep_999 is not on the shortlist."]),
        (proposal(backup="rep_003"), ("rep_003", "rep_043"), ["Backup rep must differ from the primary rep."]),
        (
            proposal(backup=None),
            ("rep_003", "rep_043"),
            ["A backup rep is required when the shortlist has more than one rep."],
        ),
        (
            proposal(submission_id="sub_other"),
            ("rep_003", "rep_043"),
            ["Proposal is for sub_other, not sub_8f92a1c4."],
        ),
    ],
)
def test_check_proposal(candidate, reps, problems):
    assert check_proposal(candidate, shortlist(*reps)) == problems


def test_step_timing():
    assert StepTiming(step="buyer_agent", duration_ms=4200, status="timeout").status == "timeout"
    with pytest.raises(ValidationError):
        StepTiming(step="buyer_agent", duration_ms=-1)
    with pytest.raises(ValidationError):
        StepTiming(step="tavily", duration_ms=1)
