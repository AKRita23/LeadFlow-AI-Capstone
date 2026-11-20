from leadflow.capacity import CapacityService
from leadflow.data_loader import load_accounts, load_leads, load_reps, load_territories
from leadflow.eval_capacity import labeled_prospects
from leadflow.eval_routing import default_buyer, default_routing_agent, run, summarize


def test_routing_eval_ranker_valid_on_workbook():
    leads, expectations = load_leads()
    reps = load_reps()
    pairs = labeled_prospects(leads, expectations, reps, load_territories())
    assert len(pairs) == 15
    cases = run(
        default_routing_agent(use_llm=False),
        default_buyer(),
        CapacityService(load_accounts()),
        pairs,
        reps,
    )
    metrics = summarize(cases)
    assert metrics["proposal_valid_rate"] == 1.0
    assert metrics["fallback_rate"] == 1.0
    assert metrics["ranker_primary_match_rate"] == 1.0
    assert metrics["brief_nonempty_rate"] == 1.0
