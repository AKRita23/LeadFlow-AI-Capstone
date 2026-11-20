from leadflow.data_loader import load_leads
from leadflow.eval_failure_modes import load_cases, run_case
from leadflow.extraction import anthropic_key_available


def test_failure_mode_cases_pass_offline():
    workbook = {lead.submission_id: lead for lead in load_leads()[0]}
    cases = load_cases()
    assert len(cases) == 5
    # Offline: no LLM. After-hours brief line is skipped when method is ranker.
    for case in cases:
        report = run_case(case, workbook, use_llm=False)
        assert report["pass"], (case["id"], report["checks"])
