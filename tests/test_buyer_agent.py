from types import SimpleNamespace

import pytest

from leadflow.buyer_agent import BuyerContextAgent
from leadflow.contracts import NoteSignals
from leadflow.data_loader import load_enrichment, load_leads
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.eval_buyer import load_gold, run, signal_agreement, summarize
from leadflow.extraction import DEFAULT_MODEL, TOOL_NAME, ClaudeExtractor, extract_with_rules
from leadflow.scoring import score_buyer, tier_for

LEADS, EXPECTATIONS = load_leads()
BY_ID = {lead.submission_id: lead for lead in LEADS}
ROWS = load_enrichment()
GOLD = load_gold()


def agent(extractor=None, posthog=None):
    return BuyerContextAgent(MockTheirStack(ROWS), posthog or MockPostHog(ROWS), extractor)


def finpay(**changes):
    return BY_ID["sub_8f92a1c4"].model_copy(update=changes)


def test_enrichment_loads_workbook_columns():
    stack = MockTheirStack(ROWS)
    found = stack.by_domain("FinPay.es")
    assert found.industry == "FinTech & Payments"
    assert found.headcount == 180
    assert found.competitors_detected == ["Intercom"]
    assert stack.by_domain("nexuscloud.ie").competitors_detected == []
    assert MockPostHog(ROWS).behavior("ELENA@finpay.es").pricing_views_48h == 6


def test_excluded_leads_have_no_enrichment():
    student = next(row for row in ROWS if row.company.startswith("Personal Portfolio"))
    assert student.firmographics is None and student.behavior is None


@pytest.mark.parametrize("submission_id", sorted(GOLD))
def test_rules_extraction_matches_gold(submission_id):
    got = extract_with_rules(BY_ID[submission_id])
    assert all(signal_agreement(got, GOLD[submission_id]).values()), got


def test_instant_is_not_a_deadline():
    assert extract_with_rules(BY_ID["sub_2h55j8b2"]).deadline_days is None


def test_score_breakdown_sums_to_score():
    score, tier, lines = score_buyer(finpay(), GOLD["sub_8f92a1c4"], MockPostHog(ROWS).behavior("elena@finpay.es"))
    assert sum(line.points for line in lines) == score
    assert tier == "High"
    assert {"deadline", "seats", "competitor"} <= {line.factor for line in lines}


def test_score_is_capped():
    signals = NoteSignals(deadline_days=0, seats=99, competitor_mentioned="Zendesk", compliance=["HIPAA"],
                          buying_stage="ready_to_buy")
    behavior = MockPostHog(ROWS).behavior("elena@finpay.es")
    score, _, lines = score_buyer(finpay(role="CXO"), signals, behavior)
    assert score == 100
    assert lines[-1].factor == "cap"
    assert sum(line.points for line in lines) == 100


@pytest.mark.parametrize("score, tier", [(100, "High"), (80, "High"), (79, "Medium"), (40, "Medium"), (39, "Low")])
def test_tier_thresholds(score, tier):
    assert tier_for(score) == tier


def test_workbook_tiers_with_gold_signals():
    cases = run(agent(), LEADS, EXPECTATIONS)
    metrics = summarize(cases, GOLD)
    assert metrics["leads"] == 15
    assert metrics["tier_accuracy"] == 1.0
    assert metrics["critical_errors"] == 0
    assert metrics["score_mae"] <= 4
    assert metrics["signal_buying_stage"] == 1.0


def test_personal_email_looks_up_company_by_name():
    result = agent()(finpay(work_email="elena.rostova@gmail.com"))
    assert result.firmographics.domain == "finpay.es"


def test_unknown_domain_falls_back_to_company_name():
    result = agent()(finpay(work_email="elena@finpay-group.com"))
    assert result.firmographics.domain == "finpay.es"


def test_unknown_company_has_no_firmographics():
    result = agent()(finpay(work_email="ana@unknown.example", company="Nobody Ltd"))
    assert result.firmographics is None
    assert "No TheirStack record; using form company size" in result.buying_signals


def test_missing_posthog_uses_neutral_behavior():
    result = agent(posthog=MockPostHog([]))(finpay())
    assert result.behavior is None
    assert any("neutral default" in line.detail for line in result.score_breakdown)


def test_blank_note_skips_llm():
    calls = []
    result = agent(extractor=calls.append)(finpay(free_text_note=None))
    assert calls == []
    assert result.tool_status["llm"] == "skipped"
    assert result.signals == NoteSignals()


def test_llm_signals_are_used():
    signals = NoteSignals(competitor_mentioned="Zendesk", buying_stage="ready_to_buy", summary="Wants to switch.")
    result = agent(extractor=lambda lead: signals)(finpay())
    assert result.extraction == "llm"
    assert result.tool_status["llm"] == "ok"
    assert result.note_summary == "Wants to switch."
    assert "Note names Zendesk but TheirStack detects Intercom" in result.buying_signals


def test_llm_failure_falls_back_to_rules():
    def broken(lead):
        raise TimeoutError

    result = agent(extractor=broken)(finpay())
    assert result.extraction == "rules"
    assert result.tool_status["llm"] == "error"
    assert result.signals.deadline_days == 14


def test_replay_is_consistent():
    cases = run(agent(), LEADS, EXPECTATIONS, repeat=3)
    assert summarize(cases)["replay_consistency"] == 1.0


class FakeMessages:
    def __init__(self, content):
        self.content = content
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(content=self.content)


def test_claude_extractor_forces_the_signals_tool():
    block = SimpleNamespace(type="tool_use", name=TOOL_NAME,
                            input={"deadline_days": 14, "seats": 40, "competitor_mentioned": "Intercom",
                                   "buying_stage": "evaluating"})
    messages = FakeMessages([block])
    signals = ClaudeExtractor(client=SimpleNamespace(messages=messages), model="test-model")(finpay())
    assert signals.seats == 40
    assert messages.request["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert messages.request["temperature"] == 0
    assert "Intercom expires in 2 weeks" in messages.request["messages"][0]["content"]


def test_blank_model_env_uses_default(monkeypatch):
    monkeypatch.setenv("LEADFLOW_BUYER_MODEL", "")
    extractor = ClaudeExtractor(client=SimpleNamespace(messages=FakeMessages([])))
    assert extractor.model == DEFAULT_MODEL


def test_claude_extractor_rejects_missing_tool_call():
    messages = FakeMessages([SimpleNamespace(type="text", text="no")])
    with pytest.raises(ValueError):
        ClaudeExtractor(client=SimpleNamespace(messages=messages), model="test-model")(finpay())
