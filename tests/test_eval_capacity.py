from leadflow.capacity import CapacityService
from leadflow.data_loader import load_accounts, load_leads, load_reps, load_territories
from leadflow.eval_capacity import labeled_prospects, run, summarize


def test_capacity_eval_passes_contract_on_workbook():
    leads, expectations = load_leads()
    reps = load_reps()
    pairs = labeled_prospects(leads, expectations, reps, load_territories())
    assert len(pairs) == 15
    cases = run(CapacityService(load_accounts()), pairs, reps, repeat=2)
    metrics = summarize(cases)
    assert metrics["check_pass_rate"] == 1.0
    assert metrics["coverage_exact_rate"] == 1.0
    assert metrics["replay_consistency"] == 1.0
