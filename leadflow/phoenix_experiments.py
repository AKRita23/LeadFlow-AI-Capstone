"""Run Arize Phoenix Experiments on uploaded workbook datasets.

Datasets alone do not create Experiments in the UI. Call
``client.experiments.run_experiment`` so Phoenix shows task outputs and
evaluator scores under Datasets → Experiments.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union


def phoenix_client():
    try:
        from phoenix.client import Client
    except ImportError as exc:
        raise SystemExit(
            "Phoenix client missing. Install with:\n"
            "  .venv/bin/pip install -r requirements-observability.txt"
        ) from exc
    return Client()


def get_dataset(client: Any, name: str):
    matches = [d for d in client.datasets.list() if d.get("name") == name]
    if not matches:
        raise SystemExit(f"Phoenix dataset {name!r} not found. Run the eval with --phoenix first.")
    return client.datasets.get_dataset(dataset=matches[0]["id"])


def run_experiment(
    *,
    dataset_name: str,
    task: Callable[..., Any],
    evaluators: Union[Sequence[Callable[..., Any]], Mapping[str, Callable[..., Any]]],
    experiment_name: str,
    experiment_description: str = "",
    experiment_metadata: Optional[Mapping[str, Any]] = None,
    timeout: int = 120,
) -> Any:
    """Execute a Phoenix experiment and print its UI URL."""
    client = phoenix_client()
    dataset = get_dataset(client, dataset_name)
    result = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=evaluators,
        experiment_name=experiment_name,
        experiment_description=experiment_description or None,
        experiment_metadata=dict(experiment_metadata or {}),
        print_summary=True,
        timeout=timeout,
    )
    experiment_id = getattr(result, "id", None) or (result.get("id") if isinstance(result, dict) else None)
    try:
        if experiment_id:
            url = client.experiments.get_experiment_url(
                dataset_id=dataset.id, experiment_id=str(experiment_id)
            )
        else:
            url = client.experiments.get_dataset_experiments_url(dataset_id=dataset.id)
    except Exception:
        url = f"http://127.0.0.1:6006/datasets/{dataset.id}/experiments"
    print(f"Phoenix experiment: {experiment_name}")
    print(f"  Open: {url}")
    return result


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def buyer_experiment_task(agent: Any, leads_by_id: Mapping[str, Any]) -> Callable[..., Dict[str, Any]]:
    def task(input: Dict[str, Any]) -> Dict[str, Any]:
        submission_id = input["submission_id"]
        lead = leads_by_id[submission_id]
        assessment = agent(lead)
        signals = assessment.signals
        return {
            "submission_id": submission_id,
            "urgency_tier": assessment.urgency_tier,
            "intent_score": int(assessment.intent_score),
            "extraction": assessment.extraction,
            "deadline_days": None if signals is None else signals.deadline_days,
            "seats": None if signals is None else signals.seats,
            "competitor": None if signals is None else signals.competitor_mentioned,
            "compliance": [] if signals is None else list(signals.compliance or []),
            "buying_stage": None if signals is None else signals.buying_stage,
        }

    return task


def buyer_evaluators() -> Dict[str, Callable[..., Any]]:
    def tier_match(output: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        return _as_str(output.get("urgency_tier")) == _as_str(expected.get("expected_urgency_tier"))

    def score_error(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
        got = _as_float(output.get("intent_score"))
        exp = _as_float(expected.get("expected_intent_score"))
        if got is None or exp is None:
            return 999.0
        return abs(got - exp)

    def seats_match(output: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        got = _as_float(output.get("seats"))
        exp = _as_float(expected.get("gold_seats"))
        return got == exp

    def competitor_match(output: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        got = _as_str(output.get("competitor")) or None
        exp = _as_str(expected.get("gold_competitor")) or None
        return got == exp

    def buying_stage_match(output: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        return _as_str(output.get("buying_stage")) == _as_str(expected.get("gold_buying_stage"))

    def compliance_match(output: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        got = {c.strip() for c in (output.get("compliance") or []) if c}
        exp_raw = _as_str(expected.get("gold_compliance"))
        exp = {c.strip() for c in exp_raw.split(",") if c.strip()} if exp_raw else set()
        return got == exp

    return {
        "tier_match": tier_match,
        "score_error": score_error,
        "seats_match": seats_match,
        "competitor_match": competitor_match,
        "buying_stage_match": buying_stage_match,
        "compliance_match": compliance_match,
    }


def routing_experiment_task(
    routing_agent: Any,
    buyer_agent: Any,
    capacity_service: Any,
    leads_by_id: Mapping[str, Any],
    shortlists_by_id: Mapping[str, Any],
    reps: Sequence[Any],
) -> Callable[..., Dict[str, Any]]:
    def task(input: Dict[str, Any]) -> Dict[str, Any]:
        from leadflow.contracts import check_proposal

        submission_id = input["submission_id"]
        lead = leads_by_id[submission_id]
        shortlist = shortlists_by_id[submission_id]
        buyer = buyer_agent(lead)
        capacity = capacity_service(shortlist, reps, lead.email_domain)
        outcome = routing_agent(lead, buyer, capacity, shortlist, related_rep_id=None)
        problems = check_proposal(outcome.proposal, shortlist)
        ranker_top = outcome.ranking[0] if outcome.ranking else None
        return {
            "submission_id": submission_id,
            "method": outcome.method,
            "llm_status": outcome.llm_status,
            "primary_rep_id": outcome.proposal.primary_rep_id,
            "backup_rep_id": outcome.proposal.backup_rep_id,
            "ranker_primary": ranker_top,
            "proposal_valid": not problems,
            "ranker_match": outcome.proposal.primary_rep_id == ranker_top,
            "problems": problems,
            "brief_chars": len(outcome.proposal.rep_brief or ""),
        }

    return task


def routing_evaluators() -> Dict[str, Callable[..., Any]]:
    def proposal_valid(output: Dict[str, Any]) -> bool:
        return bool(output.get("proposal_valid"))

    def ranker_match(output: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        # Prefer live ranker from the task output; fall back to dataset baseline.
        baseline = output.get("ranker_primary") or expected.get("ranker_primary")
        return _as_str(output.get("primary_rep_id")) == _as_str(baseline)

    def brief_present(output: Dict[str, Any]) -> bool:
        return int(output.get("brief_chars") or 0) > 0

    return {
        "proposal_valid": proposal_valid,
        "ranker_match": ranker_match,
        "brief_present": brief_present,
    }


def capacity_experiment_task(
    service: Any,
    leads_by_id: Mapping[str, Any],
    shortlists_by_id: Mapping[str, Any],
    reps: Sequence[Any],
) -> Callable[..., Dict[str, Any]]:
    def task(input: Dict[str, Any]) -> Dict[str, Any]:
        from leadflow.contracts import check_capacity

        submission_id = input["submission_id"]
        lead = leads_by_id[submission_id]
        shortlist = shortlists_by_id[submission_id]
        matrix = service(shortlist, reps, lead.email_domain)
        problems = check_capacity(matrix, shortlist)
        got_ids = [row.rep_id for row in matrix.reps]
        return {
            "submission_id": submission_id,
            "rep_ids": got_ids,
            "check_pass": not problems,
            "coverage_exact": got_ids == list(shortlist.rep_ids),
            "n_reps": len(got_ids),
            "problems": problems,
        }

    return task


def capacity_evaluators() -> Dict[str, Callable[..., Any]]:
    def check_pass(output: Dict[str, Any]) -> bool:
        return bool(output.get("check_pass"))

    def coverage_exact(output: Dict[str, Any]) -> bool:
        return bool(output.get("coverage_exact"))

    return {
        "check_pass": check_pass,
        "coverage_exact": coverage_exact,
    }
