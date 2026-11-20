"""Score the buyer agent against the workbook's expected intent scores and tiers.

    python -m leadflow.eval_buyer            # rules extraction, offline
    python -m leadflow.eval_buyer --llm      # Claude extraction (needs ANTHROPIC_API_KEY)
    python -m leadflow.eval_buyer --llm --repeat 3
    python -m leadflow.eval_buyer --llm --phoenix   # also push traces + quality annotations to Phoenix

Reports tier accuracy, High-tier recall and precision, critical errors
(High scored Low or Low scored High), score error, extraction agreement with
data/buyer_signals_gold.json, fallback rate, replay consistency, and latency.

With --phoenix, each lead is a traced span and CODE annotations for tier match,
score error, and signal agreement are logged to Arize Phoenix (local or hosted).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from leadflow.buyer_agent import BuyerContextAgent
from leadflow.contracts import BuyerAssessment, NoteSignals
from leadflow.data_loader import DATA_DIR, load_enrichment, load_leads
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.extraction import claude_from_env
from leadflow.models import Lead, LeadExpectation
from leadflow.scoring import DEADLINE_WINDOW_DAYS

GOLD_PATH = DATA_DIR / "buyer_signals_gold.json"
HOLDOUT_PATH = DATA_DIR / "buyer_holdout.json"

SIGNAL_FIELDS = ("deadline", "seats", "competitor", "compliance", "buying_stage")


def load_gold(path: Path = GOLD_PATH) -> Dict[str, NoteSignals]:
    raw = json.loads(path.read_text())
    return {key: NoteSignals.model_validate(value) for key, value in raw.items() if not key.startswith("_")}


def load_holdout_ids(path: Path = HOLDOUT_PATH) -> List[str]:
    raw = json.loads(path.read_text())
    return list(raw.get("submission_ids") or [])


def signal_agreement(got: NoteSignals, gold: NoteSignals) -> Dict[str, bool]:
    """Per-field match, compared at the granularity the scorer uses."""

    def urgent(s: NoteSignals) -> bool:
        return s.deadline_days is not None and s.deadline_days <= DEADLINE_WINDOW_DAYS

    return {
        "deadline": urgent(got) == urgent(gold),
        "seats": got.seats == gold.seats,
        "competitor": got.competitor_mentioned == gold.competitor_mentioned,
        "compliance": bool(got.compliance) == bool(gold.compliance),
        "buying_stage": got.buying_stage == gold.buying_stage,
    }


@dataclass
class Case:
    lead: Lead
    expected: LeadExpectation
    runs: List[BuyerAssessment]
    latencies_ms: List[float]
    span_id: Optional[str] = None


@dataclass
class TraceRow:
    span_id: str
    submission_id: str
    tier_match: bool
    score_error: float
    signals: Dict[str, bool] = field(default_factory=dict)


def run(
    agent: BuyerContextAgent,
    leads: Sequence[Lead],
    expectations: Sequence[LeadExpectation],
    repeat: int = 1,
    tracer: Any = None,
    gold: Optional[Dict[str, NoteSignals]] = None,
) -> List[Case]:
    cases = []
    for lead, expected in zip(leads, expectations):
        if expected.expected_urgency_tier is None:
            continue
        runs, latencies = [], []
        span_id = None
        span_cm = tracer.start_as_current_span("buyer.eval") if tracer is not None else None
        span = span_cm.__enter__() if span_cm is not None else None
        try:
            if span is not None:
                span.set_attribute("lead.submission_id", lead.submission_id)
                span.set_attribute("lead.company", lead.company)
                span.set_attribute("eval.expected_tier", expected.expected_urgency_tier)
                span.set_attribute("eval.expected_score", float(expected.expected_intent_score))
                ctx = span.get_span_context()
                span_id = format(ctx.span_id, "016x") if ctx.is_valid else None
            for _ in range(repeat):
                start = time.perf_counter()
                runs.append(agent(lead))
                latencies.append((time.perf_counter() - start) * 1000)
            a = runs[0]
            if span is not None:
                span.set_attribute("eval.got_tier", a.urgency_tier)
                span.set_attribute("eval.got_score", int(a.intent_score))
                span.set_attribute("eval.tier_match", a.urgency_tier == expected.expected_urgency_tier)
                span.set_attribute("eval.extraction", a.extraction)
                span.set_attribute("eval.score_error", abs(a.intent_score - expected.expected_intent_score))
                if gold and a.submission_id in gold:
                    for name, ok in signal_agreement(a.signals, gold[a.submission_id]).items():
                        span.set_attribute(f"eval.signal_{name}", ok)
        finally:
            if span_cm is not None:
                span_cm.__exit__(None, None, None)
        cases.append(Case(lead, expected, runs, latencies, span_id))
    return cases


def summarize(cases: Sequence[Case], gold: Optional[Dict[str, NoteSignals]] = None) -> Dict[str, float]:
    first = [c.runs[0] for c in cases]
    expected = [c.expected.expected_urgency_tier for c in cases]
    got = [a.urgency_tier for a in first]
    pairs = list(zip(expected, got))
    predicted_high = sum(g == "High" for g in got)
    actual_high = sum(e == "High" for e in expected)
    hits = sum(e == g == "High" for e, g in pairs)
    metrics = {
        "leads": len(cases),
        "tier_accuracy": sum(e == g for e, g in pairs) / len(pairs),
        "high_recall": hits / actual_high if actual_high else 1.0,
        "high_precision": hits / predicted_high if predicted_high else 1.0,
        "critical_errors": sum({e, g} == {"High", "Low"} for e, g in pairs),
        "score_mae": statistics.mean(abs(a.intent_score - c.expected.expected_intent_score) for a, c in zip(first, cases)),
        "fallback_rate": sum(a.tool_status.get("llm") == "error" for a in first) / len(first),
        "replay_consistency": sum(
            len({(r.intent_score, r.urgency_tier) for r in c.runs}) == 1 for c in cases
        ) / len(cases),
        "latency_ms_p50": statistics.median(ms for c in cases for ms in c.latencies_ms),
        "latency_ms_max": max(ms for c in cases for ms in c.latencies_ms),
    }
    if gold:
        checks = [signal_agreement(a.signals, gold[a.submission_id]) for a in first if a.submission_id in gold]
        for field_name in SIGNAL_FIELDS:
            metrics[f"signal_{field_name}"] = sum(check[field_name] for check in checks) / len(checks)
    return metrics


def default_agent(use_llm: bool) -> BuyerContextAgent:
    rows = load_enrichment()
    extractor = claude_from_env() if use_llm else None
    if use_llm and extractor is None:
        raise SystemExit("ANTHROPIC_API_KEY is not set (add it to .env).")
    return BuyerContextAgent(MockTheirStack(rows), MockPostHog(rows), extractor)


def _trace_rows(cases: Sequence[Case], gold: Optional[Dict[str, NoteSignals]]) -> List[TraceRow]:
    rows = []
    for c in cases:
        if not c.span_id:
            continue
        a = c.runs[0]
        signals = signal_agreement(a.signals, gold[a.submission_id]) if gold and a.submission_id in gold else {}
        rows.append(
            TraceRow(
                span_id=c.span_id,
                submission_id=a.submission_id,
                tier_match=a.urgency_tier == c.expected.expected_urgency_tier,
                score_error=abs(a.intent_score - c.expected.expected_intent_score),
                signals=signals,
            )
        )
    return rows


def upload_phoenix_dataset(
    leads: Sequence[Lead],
    expectations: Sequence[LeadExpectation],
    gold: Dict[str, NoteSignals],
    name: str = "leadflow-buyer-workbook",
) -> None:
    """Publish the workbook gold set under Phoenix → Datasets (not Projects)."""
    try:
        import pandas as pd
        from phoenix.client import Client
    except ImportError as exc:
        raise SystemExit(
            "Phoenix client missing. Install with:\n"
            "  .venv/bin/pip install -r requirements-observability.txt"
        ) from exc

    rows = []
    for lead, expected in zip(leads, expectations):
        if expected.expected_urgency_tier is None:
            continue
        g = gold.get(lead.submission_id)
        rows.append(
            {
                "submission_id": lead.submission_id,
                "company": lead.company,
                "work_email": lead.work_email,
                "role": lead.role,
                "inquiry_reason": lead.inquiry_reason,
                "free_text_note": lead.free_text_note or "",
                "expected_intent_score": float(expected.expected_intent_score),
                "expected_urgency_tier": expected.expected_urgency_tier,
                "gold_deadline_days": None if g is None else g.deadline_days,
                "gold_seats": None if g is None else g.seats,
                "gold_competitor": None if g is None else g.competitor_mentioned,
                "gold_compliance": "" if g is None else ",".join(g.compliance or []),
                "gold_buying_stage": None if g is None else g.buying_stage,
                "gold_summary": None if g is None else g.summary,
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
            "15 workbook leads with expected intent score/tier "
            "(data/LeadFlow AI mock dataset.xlsx) and gold note signals "
            "(data/buyer_signals_gold.json)."
        ),
        dataframe=pd.DataFrame(rows),
        input_keys=["submission_id", "company", "work_email", "role", "inquiry_reason", "free_text_note"],
        output_keys=[
            "expected_intent_score",
            "expected_urgency_tier",
            "gold_deadline_days",
            "gold_seats",
            "gold_competitor",
            "gold_compliance",
            "gold_buying_stage",
            "gold_summary",
        ],
    )
    n = getattr(dataset, "example_count", None) or len(rows)
    print(f"Phoenix dataset created: {name} ({n} examples) — open Datasets in the UI")


def push_phoenix_annotations(rows: Sequence[TraceRow]) -> None:
    """Log CODE annotations onto the eval spans in Phoenix."""
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
    # Flush can lag; give the collector a moment before attaching annotations.
    time.sleep(2.0)

    tier_df = pd.DataFrame(
        [
            {
                "span_id": r.span_id,
                "label": "match" if r.tier_match else "miss",
                "score": 1.0 if r.tier_match else 0.0,
                "explanation": r.submission_id,
            }
            for r in rows
        ]
    )
    client.spans.log_span_annotations_dataframe(
        dataframe=tier_df, annotation_name="tier_match", annotator_kind="CODE"
    )

    score_df = pd.DataFrame(
        [
            {
                "span_id": r.span_id,
                "score": float(r.score_error),
                "label": "mae",
                "explanation": r.submission_id,
            }
            for r in rows
        ]
    )
    client.spans.log_span_annotations_dataframe(
        dataframe=score_df, annotation_name="score_error", annotator_kind="CODE"
    )

    for field_name in SIGNAL_FIELDS:
        signal_rows = [r for r in rows if field_name in r.signals]
        if not signal_rows:
            continue
        df = pd.DataFrame(
            [
                {
                    "span_id": r.span_id,
                    "label": "match" if r.signals[field_name] else "miss",
                    "score": 1.0 if r.signals[field_name] else 0.0,
                    "explanation": r.submission_id,
                }
                for r in signal_rows
            ]
        )
        client.spans.log_span_annotations_dataframe(
            dataframe=df, annotation_name=f"signal_{field_name}", annotator_kind="CODE"
        )
    print(f"Phoenix: logged annotations for {len(rows)} leads (tier_match, score_error, signals)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="extract signals with Claude instead of rules")
    parser.add_argument("--repeat", type=int, default=1, help="runs per lead, for replay consistency")
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="only the 3 leads in data/buyer_holdout.json (slice for untuned talk-track)",
    )
    parser.add_argument(
        "--phoenix",
        action="store_true",
        help="trace each lead and push quality annotations to Arize Phoenix",
    )
    args = parser.parse_args()

    tracer = None
    if args.phoenix:
        from leadflow.observability import get_tracer, setup_phoenix

        setup_phoenix(project_name="leadflow-buyer-eval", force=True)
        tracer = get_tracer("leadflow.eval_buyer")

    leads, expectations = load_leads()
    if args.holdout:
        holdout = set(load_holdout_ids())
        pairs = [(lead, exp) for lead, exp in zip(leads, expectations) if lead.submission_id in holdout]
        leads = [p[0] for p in pairs]
        expectations = [p[1] for p in pairs]
        print(f"Holdout slice: {len(leads)} leads from {HOLDOUT_PATH.name}")
    gold = load_gold()
    cases = run(default_agent(args.llm), leads, expectations, args.repeat, tracer=tracer, gold=gold)

    print(f"{'lead':14} {'company':24} {'score':>5} {'exp':>4} {'tier':7} {'exp':7} via")
    for c in cases:
        a = c.runs[0]
        mark = "" if a.urgency_tier == c.expected.expected_urgency_tier else "  <-- tier miss"
        print(
            f"{a.submission_id:14} {c.lead.company[:24]:24} {a.intent_score:5} "
            f"{c.expected.expected_intent_score:4.0f} {a.urgency_tier:7} {c.expected.expected_urgency_tier:7} "
            f"{a.extraction}{mark}"
        )
    print()
    metrics = summarize(cases, gold)
    for name, value in metrics.items():
        print(f"{name:20} {value:.3f}" if isinstance(value, float) else f"{name:20} {value}")

    if args.phoenix:
        upload_phoenix_dataset(leads, expectations, gold)
        push_phoenix_annotations(_trace_rows(cases, gold))
        from leadflow.phoenix_experiments import buyer_evaluators, buyer_experiment_task, run_experiment

        leads_by_id = {lead.submission_id: lead for lead in leads}
        run_experiment(
            dataset_name="leadflow-buyer-workbook",
            task=buyer_experiment_task(default_agent(args.llm), leads_by_id),
            evaluators=buyer_evaluators(),
            experiment_name="buyer-rules" if not args.llm else "buyer-llm",
            experiment_description=(
                "Buyer agent on workbook leads: tier/score vs Excel expectations and "
                "note facts vs hand labels."
            ),
            experiment_metadata={"extraction": "llm" if args.llm else "rules"},
            timeout=300 if args.llm else 120,
        )
        print("Phoenix UI: http://127.0.0.1:6006")
        print("  Datasets → leadflow-buyer-workbook → Experiments")
        print("  Projects → leadflow-buyer-eval      (run traces + annotations)")


if __name__ == "__main__":
    main()
