from leadflow.eval_buyer import default_agent, load_gold, load_holdout_ids, run, summarize
from leadflow.data_loader import load_leads


def test_holdout_slice_has_three_labeled_leads():
    holdout = set(load_holdout_ids())
    assert len(holdout) == 3
    leads, expectations = load_leads()
    pairs = [(lead, exp) for lead, exp in zip(leads, expectations) if lead.submission_id in holdout]
    assert len(pairs) == 3
    cases = run(default_agent(False), [p[0] for p in pairs], [p[1] for p in pairs])
    metrics = summarize(cases, load_gold())
    assert metrics["leads"] == 3
    assert metrics["tier_accuracy"] == 1.0
