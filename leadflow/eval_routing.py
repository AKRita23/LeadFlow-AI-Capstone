"""Score the routing agent on workbook new-prospect shortlists.

    python -m leadflow.eval_routing              # ranker only (offline)
    python -m leadflow.eval_routing --llm        # Claude proposals (needs ANTHROPIC_API_KEY)
    python -m leadflow.eval_routing --llm --phoenix

The workbook has no labeled “correct rep.” Metrics are:
- proposal validity (check_proposal)
- whether Claude was used vs ranker fallback
- how often Claude’s primary matches the deterministic ranker’s first choice
  (consistency, not accuracy)
- brief / rationale present, latency

Buyer assessment uses rule extraction by default so this isolates routing.
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from leadflow.buyer_agent import BuyerContextAgent
from leadflow.capacity import CapacityService
from leadflow.contracts import BuyerAssessment, CapacityMatrix, RoutingOutcome, check_proposal
from leadflow.data_loader import load_accounts, load_enrichment, load_leads, load_reps, load_territories
from leadflow.eligibility import filter_eligible_reps
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.eval_capacity import labeled_prospects
from leadflow.models import EligibleReps, Lead, Rep
from leadflow.routing import RoutingAgent, router_from_env


@dataclass
class Case:
    lead: Lead
    shortlist: EligibleReps
    buyer: BuyerAssessment
    capacity: CapacityMatrix
    outcome: RoutingOutcome
    latency_ms: float
    span_id: Optional[str] = None


@dataclass
class TraceRow:
    span_id: str
    submission_id: str
    proposal_valid: bool
    method_llm: bool
    ranker_match: bool
    problems: List[str] = field(default_factory=list)


def default_buyer() -> BuyerContextAgent:
    rows = load_enrichment()
    return BuyerContextAgent(MockTheirStack(rows), MockPostHog(rows), None)


def default_routing_agent(use_llm: bool) -> RoutingAgent:
    if not use_llm:
        return RoutingAgent(router=None)
    router = router_from_env()
    if router is None:
        raise SystemExit("ANTHROPIC_API_KEY is not set (add it to .env).")
    return RoutingAgent(router=router)


def run(
    routing: RoutingAgent,
    buyer_agent: BuyerContextAgent,
    capacity_service: CapacityService,
    pairs: Sequence[tuple[Lead, EligibleReps]],
    reps: Sequence[Rep],
    tracer: Any = None,
) -> List[Case]:
    cases = []
    for lead, shortlist in pairs:
        span_id = None
        span_cm = tracer.start_as_current_span("routing.eval") if tracer is not None else None
        span = span_cm.__enter__() if span_cm is not None else None
        try:
            if span is not None:
                span.set_attribute("lead.submission_id", lead.submission_id)
                span.set_attribute("lead.company", lead.company)
                span.set_attribute("eval.shortlist_size", len(shortlist.rep_ids))
                ctx = span.get_span_context()
                span_id = format(ctx.span_id, "016x") if ctx.is_valid else None
            buyer = buyer_agent(lead)
            capacity = capacity_service(shortlist, reps, lead.email_domain)
            start = time.perf_counter()
            outcome = routing(lead, buyer, capacity, shortlist, related_rep_id=None)
            latency_ms = (time.perf_counter() - start) * 1000
            problems = check_proposal(outcome.proposal, shortlist)
            ranker_top = outcome.ranking[0] if outcome.ranking else None
            ranker_match = outcome.proposal.primary_rep_id == ranker_top
            if span is not None:
                span.set_attribute("eval.method", outcome.method)
                span.set_attribute("eval.llm_status", outcome.llm_status)
                span.set_attribute("eval.proposal_valid", not problems)
                span.set_attribute("eval.ranker_match", ranker_match)
                span.set_attribute("eval.primary_rep_id", outcome.proposal.primary_rep_id)
                if ranker_top:
                    span.set_attribute("eval.ranker_primary", ranker_top)
        finally:
            if span_cm is not None:
                span_cm.__exit__(None, None, None)
        cases.append(
            Case(
                lead=lead,
                shortlist=shortlist,
                buyer=buyer,
                capacity=capacity,
                outcome=outcome,
                latency_ms=latency_ms,
                span_id=span_id,
            )
        )
    return cases


def summarize(cases: Sequence[Case]) -> Dict[str, float]:
    n = len(cases) or 1
    valid = [not check_proposal(c.outcome.proposal, c.shortlist) for c in cases]
    llm = [c.outcome.method == "llm" for c in cases]
    llm_cases = [c for c in cases if c.outcome.method == "llm"]
    ranker_match = [
        c.outcome.proposal.primary_rep_id == (c.outcome.ranking[0] if c.outcome.ranking else None)
        for c in cases
    ]
    llm_match = [
        c.outcome.proposal.primary_rep_id == (c.outcome.ranking[0] if c.outcome.ranking else None)
        for c in llm_cases
    ]
    return {
        "leads": len(cases),
        "proposal_valid_rate": sum(valid) / n,
        "llm_ok_rate": sum(llm) / n,
        "fallback_rate": sum(c.outcome.method == "ranker" for c in cases) / n,
        "ranker_primary_match_rate": sum(ranker_match) / n,
        "llm_vs_ranker_match_rate": (sum(llm_match) / len(llm_match)) if llm_match else float("nan"),
        "brief_nonempty_rate": sum(bool(c.outcome.proposal.rep_brief.strip()) for c in cases) / n,
        "rationale_nonempty_rate": sum(bool(c.outcome.proposal.routing_rationale.strip()) for c in cases) / n,
        "latency_ms_p50": statistics.median(c.latency_ms for c in cases),
        "latency_ms_max": max(c.latency_ms for c in cases),
    }


def _trace_rows(cases: Sequence[Case]) -> List[TraceRow]:
    rows = []
    for c in cases:
        if not c.span_id:
            continue
        problems = check_proposal(c.outcome.proposal, c.shortlist)
        ranker_top = c.outcome.ranking[0] if c.outcome.ranking else None
        rows.append(
            TraceRow(
                span_id=c.span_id,
                submission_id=c.lead.submission_id,
                proposal_valid=not problems,
                method_llm=c.outcome.method == "llm",
                ranker_match=c.outcome.proposal.primary_rep_id == ranker_top,
                problems=problems,
            )
        )
    return rows


def upload_phoenix_dataset(
    cases: Sequence[Case],
    name: str = "leadflow-routing-workbook",
) -> None:
    try:
        import pandas as pd
        from phoenix.client import Client
    except ImportError as exc:
        raise SystemExit(
            "Phoenix client missing. Install with:\n"
            "  .venv/bin/pip install -r requirements-observability.txt"
        ) from exc

    rows = []
    for c in cases:
        ranker_primary = c.outcome.ranking[0] if c.outcome.ranking else ""
        ranker_backup = c.outcome.ranking[1] if len(c.outcome.ranking) > 1 else ""
        rows.append(
            {
                "submission_id": c.lead.submission_id,
                "company": c.lead.company,
                "urgency_tier": c.buyer.urgency_tier,
                "intent_score": int(c.buyer.intent_score),
                "shortlist_rep_ids": ",".join(c.shortlist.rep_ids),
                "ranker_primary": ranker_primary,
                "ranker_backup": ranker_backup,
                "note": "ranker is a consistency baseline, not a labeled correct rep",
            }
        )
    client = Client()
    existing = [d for d in client.datasets.list() if d.get("name") == name]
    if existing:
        print(f"Phoenix dataset already present: {name} ({existing[0].get('example_count')} examples)")
        return
    dataset = client.datasets.create_dataset(
        name=name,
        dataset_description=(
            "Workbook routing cases with the deterministic ranker as a consistency "
            "baseline. There is no hand-labeled correct rep in the Excel file."
        ),
        dataframe=pd.DataFrame(rows),
        input_keys=["submission_id", "company", "urgency_tier", "intent_score", "shortlist_rep_ids"],
        output_keys=["ranker_primary", "ranker_backup", "note"],
    )
    n = getattr(dataset, "example_count", None) or len(rows)
    print(f"Phoenix dataset created: {name} ({n} examples) — open Datasets in the UI")


def push_phoenix_annotations(rows: Sequence[TraceRow]) -> None:
    if not rows:
        print("Phoenix: no span ids to annotate (is the collector reachable?)")
        return
    try:
        import pandas as pd
        from phoenix.client import Client
    except ImportError as exc:
        raise SystemExit(
            "Phoenix client missing. Install with:\n"
            "  .venv/bin/pip install -r requirements-observability.txt"
        ) from exc

    client = Client()
    time.sleep(2.0)

    for annotation_name, getter in (
        ("proposal_valid", lambda r: r.proposal_valid),
        ("ranker_match", lambda r: r.ranker_match),
        ("method_llm", lambda r: r.method_llm),
    ):
        df = pd.DataFrame(
            [
                {
                    "span_id": r.span_id,
                    "label": "yes" if getter(r) else "no",
                    "score": 1.0 if getter(r) else 0.0,
                    "explanation": r.submission_id
                    if not r.problems
                    else f"{r.submission_id}: {'; '.join(r.problems)}",
                }
                for r in rows
            ]
        )
        client.spans.log_span_annotations_dataframe(
            dataframe=df, annotation_name=annotation_name, annotator_kind="CODE"
        )
    print(f"Phoenix: logged annotations for {len(rows)} leads (proposal_valid, ranker_match, method_llm)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="route with Claude instead of the ranker only")
    parser.add_argument(
        "--phoenix",
        action="store_true",
        help="trace each lead and push quality annotations to Arize Phoenix",
    )
    args = parser.parse_args()

    tracer = None
    if args.phoenix:
        from leadflow.observability import get_tracer, setup_phoenix

        setup_phoenix(project_name="leadflow-routing-eval", force=True)
        tracer = get_tracer("leadflow.eval_routing")

    leads, expectations = load_leads()
    reps = load_reps()
    pairs = labeled_prospects(leads, expectations, reps, load_territories())
    cases = run(
        default_routing_agent(args.llm),
        default_buyer(),
        CapacityService(load_accounts()),
        pairs,
        reps,
        tracer=tracer,
    )

    print(f"{'lead':14} {'company':24} {'method':7} {'primary':8} {'ranker':8} {'ok':3}")
    for c in cases:
        primary = c.outcome.proposal.primary_rep_id
        ranker = c.outcome.ranking[0] if c.outcome.ranking else "-"
        ok = "yes" if not check_proposal(c.outcome.proposal, c.shortlist) else "NO"
        mark = "" if primary == ranker else "  <-- vs ranker"
        print(
            f"{c.lead.submission_id:14} {c.lead.company[:24]:24} {c.outcome.method:7} "
            f"{primary:8} {ranker:8} {ok:3}{mark}"
        )
    print()
    metrics = summarize(cases)
    for name, value in metrics.items():
        if isinstance(value, float):
            if value != value:  # NaN
                print(f"{name:28} n/a")
            else:
                print(f"{name:28} {value:.3f}")
        else:
            print(f"{name:28} {value}")

    if args.phoenix:
        upload_phoenix_dataset(cases)
        push_phoenix_annotations(_trace_rows(cases))
        from leadflow.phoenix_experiments import (
            routing_evaluators,
            routing_experiment_task,
            run_experiment,
        )

        leads_by_id = {lead.submission_id: lead for lead, _ in pairs}
        shortlists_by_id = {lead.submission_id: shortlist for lead, shortlist in pairs}
        run_experiment(
            dataset_name="leadflow-routing-workbook",
            task=routing_experiment_task(
                default_routing_agent(args.llm),
                default_buyer(),
                CapacityService(load_accounts()),
                leads_by_id,
                shortlists_by_id,
                reps,
            ),
            evaluators=routing_evaluators(),
            experiment_name="routing-ranker" if not args.llm else "routing-llm",
            experiment_description=(
                "Routing on workbook shortlists: proposal validity and Claude↔ranker consistency "
                "(no labeled correct rep)."
            ),
            experiment_metadata={"router": "llm" if args.llm else "ranker"},
            timeout=600 if args.llm else 120,
        )
        print("Phoenix UI: http://127.0.0.1:6006")
        print("  Datasets → leadflow-routing-workbook → Experiments")
        print("  Projects → leadflow-routing-eval      (run traces + annotations)")


if __name__ == "__main__":
    main()
