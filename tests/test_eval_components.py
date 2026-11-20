import json
from types import SimpleNamespace

import pytest

from leadflow import eval_components as ev

ROWS = {row["submission_id"]: row for row in ev.dataset_rows()}
FINPAY = ROWS["sub_8f92a1c4"]

BRIEF = """Elena Rostova, VP at FinPay Technologies
Fintech · 320 people · 101-500 · ES
High urgency · intent score 98

Why now
• Intercom contract expires in 2 weeks
• Needs 40 seats live before Q1
• 6 pricing and 4 doc views in 48h, 5 sessions in 7d

Stack and competitor
• Intercom

What they ask
• Evaluate BeaconDesk for the support team

Opening angle
• What does the Intercom exit plan look like?"""


def rep(rep_id, bandwidth="High", left=5):
    return {"rep_id": rep_id, "bandwidth_tier": bandwidth, "leads_remaining_today": left}


def output(**overrides):
    base = {
        "triage": "new_prospect",
        "tier": "High",
        "extraction": "llm",
        "signals": {"deadline_days": 14, "seats": 40, "competitor_mentioned": "Intercom", "compliance": [],
                    "buying_stage": "evaluating", "pain_points": [], "summary": "Replacing Intercom."},
        "behavior": {"pricing_views_48h": 6, "doc_views_48h": 4, "sessions_7d": 5},
        "routing_method": "llm",
        "primary_rep_id": "rep_003",
        "backup_rep_id": "rep_027",
        "rep_brief": BRIEF,
        "routing_input": {
            "buyer": {"urgency_tier": "High", "competitor": "Intercom"},
            "reps": [rep("rep_003"), rep("rep_027")],
            "related_rep_id": None,
            "after_hours": False,
        },
        "counterfactual_primary": "rep_027",
    }
    return {**base, **overrides}


def test_dataset_has_the_15_labelled_leads_with_gold_signals():
    assert len(ROWS) == 15
    assert json.loads(FINPAY["gold_signals"])["seats"] == 40
    assert FINPAY["expected_tier"] == "High"


def test_a_clean_run_passes_every_code_check():
    for name, check in ev.evaluators(use_judge=False).items():
        assert check(output(), FINPAY).score == 1.0, name


def test_field_accuracy_lists_the_wrong_fields_and_ignores_compliance_case():
    signals = {**output()["signals"], "seats": 4, "deadline_days": 30}
    result = ev.field_accuracy(output(signals=signals), FINPAY)
    assert result.score == pytest.approx(3 / 5)
    assert "seats" in result.explanation and "deadline_days" in result.explanation
    row = {**FINPAY, "gold_signals": json.dumps({**json.loads(FINPAY["gold_signals"]), "compliance": ["SOC 2"]})}
    assert ev.field_accuracy(output(signals={**output()["signals"], "compliance": ["soc  2"]}), row).score == 1.0


def test_high_urgency_lead_on_a_low_bandwidth_rep_breaks_the_rules():
    context = {**output()["routing_input"], "reps": [rep("rep_003", "Low"), rep("rep_027")]}
    result = ev.rule_compliance(output(routing_input=context), FINPAY)
    assert result.score == 0.0 and "Low-bandwidth" in result.explanation


def test_skipping_an_available_related_rep_breaks_the_rules():
    context = {**output()["routing_input"], "related_rep_id": "rep_027"}
    assert ev.rule_compliance(output(routing_input=context), FINPAY).score == 0.0
    busy = {**context, "reps": [rep("rep_003"), rep("rep_027", left=0)]}
    assert ev.rule_compliance(output(routing_input=busy), FINPAY).score == 1.0


def test_ranker_fallback_is_not_scored_as_claude():
    result = ev.rule_compliance(output(routing_method="ranker"), FINPAY)
    assert result.score is None and result.label == "ranker fallback"


def test_counterfactual_fails_when_claude_keeps_the_disabled_rep():
    assert ev.counterfactual_sensitivity(output(counterfactual_primary="rep_003"), FINPAY).score == 0.0
    assert ev.counterfactual_sensitivity({k: v for k, v in output().items() if k != "counterfactual_primary"}, FINPAY).score is None


def test_brief_completeness_needs_exact_numbers():
    brief = BRIEF.replace("40 seats", "about forty seats").replace("6 pricing", "several pricing")
    result = ev.brief_completeness(output(rep_brief=brief), FINPAY)
    assert result.score == pytest.approx(3 / 5)
    assert "seats" in result.explanation and "pricing views" in result.explanation


def test_brief_format_flags_pronouns_long_bullets_and_markdown():
    brief = BRIEF + "\n• She said the team wants a very long rollout plan across every region and channel today"
    result = ev.brief_format(output(rep_brief=brief.replace("Why now", "**Why now**")), FINPAY)
    assert result.score == 0.0
    for problem in ("Markdown", "over 15 words", "she"):
        assert problem in result.explanation


def test_brief_format_wants_the_after_hours_line_only_when_due():
    context = {**output()["routing_input"], "after_hours": True}
    assert "missing" in ev.brief_format(output(routing_input=context), FINPAY).explanation
    late = BRIEF + "\n\nArrived outside working hours."
    assert ev.brief_format(output(routing_input=context, rep_brief=late), FINPAY).score == 1.0


class FakeMessages:
    def __init__(self, verdict, stop_reason="end_turn"):
        self.verdict, self.stop_reason, self.calls = verdict, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = SimpleNamespace(type="text", text=json.dumps(self.verdict))
        return SimpleNamespace(stop_reason=self.stop_reason, content=[text])


def test_judge_reports_unsupported_claims():
    messages = FakeMessages({"unsupported_claims": [{"claim": "40 seats in 2 weeks", "why": "merges two facts"}], "faithful": False})
    judge = ev.Judge(client=SimpleNamespace(messages=messages), model="claude-opus-5-5")
    result = ev.summary_faithfulness(judge)(output(), FINPAY)
    assert result.score == 0.0 and "merges two facts" in result.explanation
    sent = messages.calls[0]
    assert "Intercom expires in 2 weeks" in sent["messages"][0]["content"]
    assert sent["output_config"]["format"]["type"] == "json_schema"


def test_judge_skips_rules_extraction_and_survives_a_refusal():
    judge = ev.Judge(client=SimpleNamespace(messages=FakeMessages({}, stop_reason="refusal")), model="m")
    assert ev.summary_faithfulness(judge)(output(extraction="rules"), FINPAY).score is None
    assert ev.brief_faithfulness(judge)(output(), FINPAY).label == "judge stopped: refusal"
