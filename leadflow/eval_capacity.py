"""Score the capacity service on workbook shortlists.

    python -m leadflow.eval_capacity
    python -m leadflow.eval_capacity --repeat 3
    python -m leadflow.eval_capacity --phoenix

There is no hand-labeled “correct bandwidth” in the workbook. This eval checks
the contract: every shortlisted rep appears exactly once, check_capacity passes,
and the matrix is deterministic across repeats. Optional Phoenix upload stores
the shortlists as a dataset and attaches CODE annotations to the run.
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from leadflow.capacity import CapacityService
from leadflow.contracts import CapacityMatrix, check_capacity
from leadflow.data_loader import load_accounts, load_leads, load_reps, load_territories
from leadflow.eligibility import filter_eligible_reps
from leadflow.models import EligibleReps, Lead, LeadExpectation, Rep


@dataclass
class Case:
    lead: Lead
    shortlist: EligibleReps
    runs: List[CapacityMatrix]
    latencies_ms: List[float]
    span_id: Optional[str] = None


@dataclass
class TraceRow:
    span_id: str
    submission_id: str
    check_pass: bool
    coverage_exact: bool
    n_reps: int


def labeled_prospects(
    leads: Sequence[Lead],
    expectations: Sequence[LeadExpectation],
    reps: Sequence[Rep],
    territories: Any,
) -> List[tuple[Lead, EligibleReps]]:
    """Workbook leads with an expected tier and a non-empty shortlist."""
    out = []
    for lead, expected in zip(leads, expectations):
        if expected.expected_urgency_tier is None:
            continue
        shortlist = filter_eligible_reps(lead, reps, territories)
        if shortlist.rep_ids:
            out.append((lead, shortlist))
    return out


def run(
    service: CapacityService,
    pairs: Sequence[tuple[Lead, EligibleReps]],
    reps: Sequence[Rep],
    repeat: int = 1,
    tracer: Any = None,
) -> List[Case]:
    cases = []
    for lead, shortlist in pairs:
        runs, latencies = [], []
        span_id = None
        span_cm = tracer.start_as_current_span("capacity.eval") if tracer is not None else None
        span = span_cm.__enter__() if span_cm is not None else None
        try:
            if span is not None:
                span.set_attribute("lead.submission_id", lead.submission_id)
                span.set_attribute("lead.company", lead.company)
                span.set_attribute("eval.shortlist_size", len(shortlist.rep_ids))
                ctx = span.get_span_context()
                span_id = format(ctx.span_id, "016x") if ctx.is_valid else None
            for _ in range(repeat):
                start = time.perf_counter()
                matrix = service(shortlist, reps, lead.email_domain)
                latencies.append((time.perf_counter() - start) * 1000)
                runs.append(matrix)
            matrix = runs[0]
            problems = check_capacity(matrix, shortlist)
            coverage = [row.rep_id for row in matrix.reps] == list(shortlist.rep_ids)
            if span is not None:
                span.set_attribute("eval.check_pass", not problems)
                span.set_attribute("eval.coverage_exact", coverage)
                span.set_attribute("eval.n_reps", len(matrix.reps))
        finally:
            if span_cm is not None:
                span_cm.__exit__(None, None, None)
        cases.append(Case(lead, shortlist, runs, latencies, span_id))
    return cases


def summarize(cases: Sequence[Case]) -> Dict[str, float]:
    first = [c.runs[0] for c in cases]
    checks = [not check_capacity(m, c.shortlist) for m, c in zip(first, cases)]
    coverage = [
        [row.rep_id for row in m.reps] == list(c.shortlist.rep_ids) for m, c in zip(first, cases)
    ]
    tiers = [row.bandwidth_tier for m in first for row in m.reps]
    n_rows = len(tiers) or 1
    return {
        "leads": len(cases),
        "check_pass_rate": sum(checks) / len(checks),
        "coverage_exact_rate": sum(coverage) / len(coverage),
        "replay_consistency": sum(
            len({tuple((r.rep_id, r.bandwidth_tier, r.stage_pressure) for r in run.reps) for run in c.runs}) == 1
            for c in cases
        )
        / len(cases),
        "latency_ms_p50": statistics.median(ms for c in cases for ms in c.latencies_ms),
        "latency_ms_max": max(ms for c in cases for ms in c.latencies_ms),
        "bandwidth_high_share": sum(t == "High" for t in tiers) / n_rows,
        "bandwidth_medium_share": sum(t == "Medium" for t in tiers) / n_rows,
        "bandwidth_low_share": sum(t == "Low" for t in tiers) / n_rows,
        "mean_shortlist_size": statistics.mean(len(c.shortlist.rep_ids) for c in cases),
    }


def _trace_rows(cases: Sequence[Case]) -> List[TraceRow]:
    rows = []
    for c in cases:
        if not c.span_id:
            continue
        m = c.runs[0]
        rows.append(
            TraceRow(
                span_id=c.span_id,
                submission_id=c.lead.submission_id,
                check_pass=not check_capacity(m, c.shortlist),
                coverage_exact=[row.rep_id for row in m.reps] == list(c.shortlist.rep_ids),
                n_reps=len(m.reps),
            )
        )
    return rows


def upload_phoenix_dataset(
    pairs: Sequence[tuple[Lead, EligibleReps]],
    name: str = "leadflow-capacity-workbook",
) -> None:
    try:
        import pandas as pd
        from phoenix.client import Client
    except ImportError as exc:
        raise SystemExit(
            "Phoenix client missing. Install with:\n"
            "  .venv/bin/pip install -r requirements-observability.txt"
        ) from exc

    rows = [
        {
            "submission_id": lead.submission_id,
            "company": lead.company,
            "email_domain": lead.email_domain,
            "territory": shortlist.territory,
            "shortlist_rep_ids": ",".join(shortlist.rep_ids),
            "shortlist_size": len(shortlist.rep_ids),
            "expected_check": "pass",
            "expected_coverage": "exact_shortlist_order",
        }
        for lead, shortlist in pairs
    ]
    client = Client()
    existing = [d for d in client.datasets.list() if d.get("name") == name]
    if existing:
        print(f"Phoenix dataset already present: {name} ({existing[0].get('example_count')} examples)")
        return
    dataset = client.datasets.create_dataset(
        name=name,
        dataset_description=(
            "Workbook shortlists for capacity-service contract checks. "
            "No hand-labeled bandwidth tiers — expected output is check_capacity pass "
            "and exact shortlist coverage."
        ),
        dataframe=pd.DataFrame(rows),
        input_keys=["submission_id", "company", "email_domain", "territory", "shortlist_rep_ids"],
        output_keys=["expected_check", "expected_coverage", "shortlist_size"],
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
        ("check_pass", lambda r: r.check_pass),
        ("coverage_exact", lambda r: r.coverage_exact),
    ):
        df = pd.DataFrame(
            [
                {
                    "span_id": r.span_id,
                    "label": "pass" if getter(r) else "fail",
                    "score": 1.0 if getter(r) else 0.0,
                    "explanation": r.submission_id,
                }
                for r in rows
            ]
        )
        client.spans.log_span_annotations_dataframe(
            dataframe=df, annotation_name=annotation_name, annotator_kind="CODE"
        )

    size_df = pd.DataFrame(
        [
            {
                "span_id": r.span_id,
                "score": float(r.n_reps),
                "label": "n_reps",
                "explanation": r.submission_id,
            }
            for r in rows
        ]
    )
    client.spans.log_span_annotations_dataframe(
        dataframe=size_df, annotation_name="shortlist_size", annotator_kind="CODE"
    )
    print(f"Phoenix: logged annotations for {len(rows)} leads (check_pass, coverage_exact, shortlist_size)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=1, help="runs per lead, for replay consistency")
    parser.add_argument(
        "--phoenix",
        action="store_true",
        help="trace each shortlist and push quality annotations to Arize Phoenix",
    )
    args = parser.parse_args()

    tracer = None
    if args.phoenix:
        from leadflow.observability import get_tracer, setup_phoenix

        setup_phoenix(project_name="leadflow-capacity-eval", force=True)
        tracer = get_tracer("leadflow.eval_capacity")

    leads, expectations = load_leads()
    reps = load_reps()
    pairs = labeled_prospects(leads, expectations, reps, load_territories())
    service = CapacityService(load_accounts())
    cases = run(service, pairs, reps, args.repeat, tracer=tracer)

    print(f"{'lead':14} {'company':24} {'reps':>4} {'check':6} {'cover':6}")
    for c in cases:
        m = c.runs[0]
        ok = "pass" if not check_capacity(m, c.shortlist) else "FAIL"
        cover = "exact" if [r.rep_id for r in m.reps] == list(c.shortlist.rep_ids) else "MISS"
        print(f"{c.lead.submission_id:14} {c.lead.company[:24]:24} {len(m.reps):4} {ok:6} {cover:6}")
    print()
    metrics = summarize(cases)
    for name, value in metrics.items():
        print(f"{name:24} {value:.3f}" if isinstance(value, float) else f"{name:24} {value}")

    if args.phoenix:
        upload_phoenix_dataset(pairs)
        push_phoenix_annotations(_trace_rows(cases))
        from leadflow.phoenix_experiments import (
            capacity_evaluators,
            capacity_experiment_task,
            run_experiment,
        )

        leads_by_id = {lead.submission_id: lead for lead, _ in pairs}
        shortlists_by_id = {lead.submission_id: shortlist for lead, shortlist in pairs}
        run_experiment(
            dataset_name="leadflow-capacity-workbook",
            task=capacity_experiment_task(service, leads_by_id, shortlists_by_id, reps),
            evaluators=capacity_evaluators(),
            experiment_name="capacity-contract",
            experiment_description=(
                "Capacity service contract checks: check_capacity pass and exact shortlist coverage."
            ),
            experiment_metadata={"service": "CapacityService"},
            timeout=120,
        )
        print("Phoenix UI: http://127.0.0.1:6006")
        print("  Datasets → leadflow-capacity-workbook → Experiments")
        print("  Projects → leadflow-capacity-eval")


if __name__ == "__main__":
    main()
