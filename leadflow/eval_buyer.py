"""Score the buyer agent against the workbook's expected intent scores and tiers.

    python -m leadflow.eval_buyer            # rules extraction, offline
    python -m leadflow.eval_buyer --llm      # Claude extraction (needs ANTHROPIC_API_KEY)
    python -m leadflow.eval_buyer --llm --repeat 3

Reports tier accuracy, High-tier recall and precision, critical errors
(High scored Low or Low scored High), score error, extraction agreement with
data/buyer_signals_gold.json, fallback rate, replay consistency, and latency.
"""
import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from leadflow.buyer_agent import BuyerContextAgent
from leadflow.contracts import BuyerAssessment, NoteSignals
from leadflow.data_loader import DATA_DIR, load_enrichment, load_leads
from leadflow.enrichment import MockPostHog, MockTheirStack
from leadflow.extraction import claude_from_env
from leadflow.models import Lead, LeadExpectation
from leadflow.scoring import DEADLINE_WINDOW_DAYS

GOLD_PATH = DATA_DIR / "buyer_signals_gold.json"

SIGNAL_FIELDS = ("deadline", "seats", "competitor", "compliance", "buying_stage")


def load_gold(path: Path = GOLD_PATH) -> Dict[str, NoteSignals]:
    raw = json.loads(path.read_text())
    return {key: NoteSignals.model_validate(value) for key, value in raw.items() if not key.startswith("_")}


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


def run(agent: BuyerContextAgent, leads: Sequence[Lead], expectations: Sequence[LeadExpectation], repeat: int = 1):
    cases = []
    for lead, expected in zip(leads, expectations):
        if expected.expected_urgency_tier is None:
            continue
        runs, latencies = [], []
        for _ in range(repeat):
            start = time.perf_counter()
            runs.append(agent(lead))
            latencies.append((time.perf_counter() - start) * 1000)
        cases.append(Case(lead, expected, runs, latencies))
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
        for field in SIGNAL_FIELDS:
            metrics[f"signal_{field}"] = sum(check[field] for check in checks) / len(checks)
    return metrics


def default_agent(use_llm: bool) -> BuyerContextAgent:
    rows = load_enrichment()
    extractor = claude_from_env() if use_llm else None
    if use_llm and extractor is None:
        raise SystemExit("ANTHROPIC_API_KEY is not set (add it to .env).")
    return BuyerContextAgent(MockTheirStack(rows), MockPostHog(rows), extractor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="extract signals with Claude instead of rules")
    parser.add_argument("--repeat", type=int, default=1, help="runs per lead, for replay consistency")
    args = parser.parse_args()

    leads, expectations = load_leads()
    cases = run(default_agent(args.llm), leads, expectations, args.repeat)
    gold = load_gold()

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
    for name, value in summarize(cases, gold).items():
        print(f"{name:20} {value:.3f}" if isinstance(value, float) else f"{name:20} {value}")


if __name__ == "__main__":
    main()
