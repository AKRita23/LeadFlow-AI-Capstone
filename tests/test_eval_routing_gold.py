from leadflow.capacity import CapacityService
from leadflow.data_loader import load_accounts, load_leads, load_reps, load_territories
from leadflow.eval_capacity import labeled_prospects
from leadflow.eval_routing import default_buyer, default_routing_agent, load_routing_gold, run, summarize


def test_routing_gold_covers_finpay_and_edges():
    gold = load_routing_gold()
    assert "sub_8f92a1c4" in gold
    assert gold["sub_8f92a1c4"]["expected_primary"] == "rep_003"
    assert len(gold) >= 8


def test_ranker_matches_routing_gold_primaries():
    gold = load_routing_gold()
    leads, expectations = load_leads()
    reps = load_reps()
    pairs = [
        (lead, shortlist)
        for lead, shortlist in labeled_prospects(leads, expectations, reps, load_territories())
        if lead.submission_id in gold
    ]
    cases = run(
        default_routing_agent(use_llm=False),
        default_buyer(),
        CapacityService(load_accounts()),
        pairs,
        reps,
    )
    metrics = summarize(cases, gold)
    assert metrics["gold_labeled_leads"] == len(gold)
    assert metrics["gold_primary_match_rate"] == 1.0
