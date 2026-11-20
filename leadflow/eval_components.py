"""Component evals for the two Claude calls, on the 15 workbook leads that reach the buyer agent.

    python -m leadflow.eval_components               # run locally and print the results
    python -m leadflow.eval_components --phoenix     # also record the run as a Phoenix experiment
    python -m leadflow.eval_components --no-judge    # code evaluators only, no judge calls

Needs ANTHROPIC_API_KEY, and for --phoenix a running Phoenix (`phoenix serve`) with
requirements-observability.txt installed. Each lead runs the
real pipeline (triage, buyer agent, capacity, routing agent) without committing anything.

Evaluators, by component:
- Note extraction (buyer agent): field_accuracy against data/buyer_signals_gold.json (code),
  summary_faithfulness (LLM judge), and tier_accuracy against the workbook as a guard.
- Rep choice (routing agent): rule_compliance with the routing prompt's rules (code), and
  counterfactual_sensitivity: the primary rep is made unable to take the lead (Low bandwidth,
  High stage pressure, no lead left today, no competitor or vertical strength) and Claude
  must pick someone else (code, one extra routing call).
- Rep brief (routing agent): brief_completeness, every extracted signal with its exact number
  (code); brief_format, the layout and length rules (code); brief_faithfulness (LLM judge);
  brief_quality (LLM judge, 1–5 on tone / facts / usable-as-is).
"""
import argparse
import json
import os
import re
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from leadflow.contracts import CapacityMatrix, NoteSignals
from leadflow.data_loader import load_leads
from leadflow.eval_buyer import load_gold
from leadflow.extraction import anthropic_key_available
from leadflow.models import Lead
from leadflow.pipeline import triage
from leadflow.routing import ClaudeRouter, routing_context
from leadflow.store import DecisionStore

PHOENIX_DATASET = "leadflow-buyer-workbook"  # created by eval_buyer --phoenix, or here if missing
JUDGE_MODEL = "claude-opus-5-5"
SIGNAL_FIELDS = ("deadline_days", "seats", "competitor_mentioned", "compliance", "buying_stage")
GENDERED = re.compile(r"\b(he|him|his|she|her|hers|himself|herself)\b", re.I)
TIMEFRAME = re.compile(r"\b\d+\s*(?:day|week|month)s?\b|\b(?:immediately|asap|right away|right now|this week)\b", re.I)


@dataclass
class Score:
    """One evaluator's verdict. score is None when the check does not apply to this lead."""

    score: Optional[float]
    label: str
    explanation: str = ""


# ----- Dataset and task -------------------------------------------------------------------------


def dataset_rows() -> List[Dict[str, Any]]:
    """The 15 workbook leads with an expected tier, with their hand-labelled note signals."""
    gold = load_gold()
    leads, expectations = load_leads()
    return [
        {
            "submission_id": lead.submission_id,
            "company": lead.company,
            "lead": lead.model_dump_json(),
            "expected_tier": expected.expected_urgency_tier,
            "gold_signals": gold[lead.submission_id].model_dump_json(),
        }
        for lead, expected in zip(leads, expectations)
        if expected.expected_urgency_tier is not None
    ]


def run_lead(dataset_row: Dict[str, Any], counterfactual: bool = True) -> Dict[str, Any]:
    """Run one lead through the pipeline, up to the routing proposal, and return what the evaluators need."""
    from leadflow.api import get_buyer_agent, get_capacity_service, get_reference_data, get_routing_agent

    lead = Lead.model_validate_json(dataset_row["lead"])
    accounts, reps, territories = get_reference_data()
    store = DecisionStore(Path(tempfile.mkdtemp()) / "eval.db")  # fresh, so the lead is never a duplicate
    start = time.perf_counter()
    result = triage(lead, accounts, reps, territories, store)
    output: Dict[str, Any] = {"triage": result.outcome}
    if result.outcome != "new_prospect":
        return output
    buyer = get_buyer_agent()(lead)
    capacity = get_capacity_service()(result.shortlist, reps, lead.email_domain)
    routing = get_routing_agent()(lead, buyer, capacity, result.shortlist, result.related_rep_id)
    output.update(
        latency_s=round(time.perf_counter() - start, 2),
        tier=buyer.urgency_tier,
        intent_score=buyer.intent_score,
        extraction=buyer.extraction,
        signals=buyer.signals.model_dump(mode="json"),
        behavior=buyer.behavior.model_dump(mode="json") if buyer.behavior else None,
        routing_method=routing.method,
        primary_rep_id=routing.proposal.primary_rep_id,
        backup_rep_id=routing.proposal.backup_rep_id,
        rep_brief=routing.proposal.rep_brief,
        routing_input=routing_context(lead, buyer, capacity, result.shortlist, result.related_rep_id),
    )
    if counterfactual and routing.method == "llm" and len(capacity.reps) > 1:
        output["counterfactual_primary"] = _counterfactual_primary(
            lead, buyer, capacity, result.shortlist, result.related_rep_id, routing.proposal.primary_rep_id
        )
    return output


def _counterfactual_primary(lead, buyer, capacity: CapacityMatrix, shortlist, related_rep_id, primary_id) -> Optional[str]:
    """Claude's primary once the original primary clearly cannot take the lead. None if the call fails."""
    disabled = {
        "bandwidth_tier": "Low",
        "stage_pressure": "High",
        "leads_remaining_today": 0,
        "competitor_strengths": [],
        "vertical_expertise": [],
    }
    rows = [row.model_copy(update=disabled) if row.rep_id == primary_id else row for row in capacity.reps]
    changed = capacity.model_copy(update={"reps": rows})
    try:
        proposal = ClaudeRouter()(routing_context(lead, buyer, changed, shortlist, related_rep_id), shortlist)
    except Exception:
        return None
    return proposal.primary_rep_id


# ----- Code evaluators --------------------------------------------------------------------------


def tier_accuracy(output: Dict[str, Any], dataset_row: Dict[str, Any]) -> Score:
    expected, got = dataset_row["expected_tier"], output.get("tier")
    if got is None:
        return Score(None, "not routed", f"triage outcome {output.get('triage')}")
    return Score(float(got == expected), "match" if got == expected else "miss", f"expected {expected}, got {got}")


def field_accuracy(output: Dict[str, Any], dataset_row: Dict[str, Any]) -> Score:
    """Share of the 5 note fields equal to the hand labels. Compliance is compared as a set, ignoring case."""
    if "signals" not in output:
        return Score(None, "not routed")
    got = NoteSignals.model_validate(output["signals"])
    gold = NoteSignals.model_validate_json(dataset_row["gold_signals"])
    misses = []
    for field in SIGNAL_FIELDS:
        a, b = getattr(got, field), getattr(gold, field)
        if field == "compliance":
            a, b = _normalised(a), _normalised(b)
        if a != b:
            misses.append(f"{field}: got {getattr(got, field)!r}, expected {getattr(gold, field)!r}")
    score = 1 - len(misses) / len(SIGNAL_FIELDS)
    return Score(score, "all fields match" if not misses else f"{len(misses)} field(s) off", "; ".join(misses))


def _normalised(terms: List[str]) -> set:
    return {re.sub(r"\s+", " ", term).strip().lower() for term in terms}


def rule_compliance(output: Dict[str, Any], dataset_row: Dict[str, Any]) -> Score:
    """The routing prompt's hard rules, checked against the capacity rows Claude saw."""
    if output.get("routing_method") != "llm":
        return Score(None, "ranker fallback" if "routing_method" in output else "not routed")
    context = output["routing_input"]
    reps = {row["rep_id"]: row for row in context["reps"]}
    primary, backup = output["primary_rep_id"], output["backup_rep_id"]
    related = context.get("related_rep_id")
    violations = []
    if context["buyer"]["urgency_tier"] == "High" and reps[primary]["bandwidth_tier"] == "Low":
        others = [r for r, row in reps.items() if r != primary and row["bandwidth_tier"] != "Low"]
        if others:
            violations.append(f"High-urgency lead sent to Low-bandwidth {primary} while {', '.join(others)} fit")
    if related in reps and primary != related:
        row = reps[related]
        if row["bandwidth_tier"] != "Low" and row["leads_remaining_today"] > 0:
            violations.append(f"related rep {related} skipped although available")
    if len(reps) > 1 and (backup is None or backup == primary):
        violations.append("no distinct backup rep")
    return Score(float(not violations), "compliant" if not violations else "violation", "; ".join(violations))


def counterfactual_sensitivity(output: Dict[str, Any], dataset_row: Dict[str, Any]) -> Score:
    if "counterfactual_primary" not in output:
        return Score(None, "not run", "needs Claude routing and at least 2 shortlisted reps")
    before, after = output["primary_rep_id"], output["counterfactual_primary"]
    if after is None:
        return Score(None, "call failed")
    changed = after != before
    return Score(
        float(changed),
        "primary changed" if changed else "primary kept",
        f"{before} made unavailable, Claude then picked {after}",
    )


def brief_completeness(output: Dict[str, Any], dataset_row: Dict[str, Any]) -> Score:
    """Every extracted signal appears in the brief, with its number or timeframe copied exactly."""
    if "rep_brief" not in output:
        return Score(None, "not routed")
    brief = output["rep_brief"]
    lowered = brief.lower()
    signals = NoteSignals.model_validate(output["signals"])
    note = Lead.model_validate_json(dataset_row["lead"]).free_text_note or ""
    checks: Dict[str, bool] = {}
    if signals.deadline_days is not None:
        phrases = [m.group(0).lower() for m in TIMEFRAME.finditer(note)] + [f"{signals.deadline_days} days"]
        checks["deadline"] = any(p in lowered for p in phrases)
    if signals.seats is not None:
        checks["seats"] = re.search(rf"\b{signals.seats}\b", brief) is not None
    competitor = output["routing_input"]["buyer"]["competitor"]
    if competitor:
        checks["competitor"] = competitor.lower() in lowered
    behavior = output.get("behavior") or {}
    if behavior.get("pricing_views_48h"):
        checks["pricing views"] = re.search(rf"\b{behavior['pricing_views_48h']}\s+pricing", lowered) is not None
    if behavior.get("doc_views_48h"):
        checks["doc views"] = re.search(rf"\b{behavior['doc_views_48h']}\s+(?:doc|documentation)", lowered) is not None
    if not checks:
        return Score(1.0, "nothing to check")
    missing = [name for name, ok in checks.items() if not ok]
    score = 1 - len(missing) / len(checks)
    return Score(score, "complete" if not missing else f"{len(missing)} missing", "missing: " + ", ".join(missing) if missing else "")


def brief_format(output: Dict[str, Any], dataset_row: Dict[str, Any]) -> Score:
    """Plain text, under 120 words, short one-fact bullets, no gendered pronouns, after-hours line when due."""
    if output.get("routing_method") != "llm":
        return Score(None, "ranker fallback" if "routing_method" in output else "not routed")
    brief = output["rep_brief"]
    problems = []
    words = len(brief.split())
    if words > 120:
        problems.append(f"{words} words")
    if re.search(r"(^#|\*\*|^\* |^- )", brief, re.M):
        problems.append("Markdown")
    long_bullets = [b for b in brief.splitlines() if b.startswith("•") and len(b[1:].split()) > 15]
    if long_bullets:
        problems.append(f"{len(long_bullets)} bullet(s) over 15 words")
    pronouns = sorted({m.group(0).lower() for m in GENDERED.finditer(brief)})
    if pronouns:
        problems.append("gendered pronoun: " + ", ".join(pronouns))
    after_hours = output["routing_input"]["after_hours"]
    if after_hours != brief.rstrip().endswith("Arrived outside working hours."):
        problems.append("after-hours line missing" if after_hours else "after-hours line not due")
    return Score(float(not problems), "ok" if not problems else "format issue", "; ".join(problems))


# ----- LLM judge --------------------------------------------------------------------------------

JUDGE_SYSTEM = """You check whether a text written by another model is faithful to its source.

List every claim in the text that the source does not support:
- a fact, number, name, date, or timeframe that is not in the source or differs from it;
- two separate facts from the source merged into one claim the source does not make;
- a meeting, demo, date, or next step the source does not mention.
Rewording, shortening, and dropping facts are fine. Layout labels and an open question to ask the
buyer are not claims. Then set faithful to true only if you listed nothing."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "unsupported_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"}, "why": {"type": "string"}},
                "required": ["claim", "why"],
                "additionalProperties": False,
            },
        },
        "faithful": {"type": "boolean"},
    },
    "required": ["unsupported_claims", "faithful"],
    "additionalProperties": False,
}


class Judge:
    def __init__(self, client: Any = None, model: Optional[str] = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=120.0, max_retries=2)
        self.client = client
        self.model = model or os.environ.get("LEADFLOW_JUDGE_MODEL") or JUDGE_MODEL

    def __call__(self, source: str, text: str) -> Score:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=JUDGE_SYSTEM,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
            messages=[{"role": "user", "content": f"<source>\n{source}\n</source>\n\n<text>\n{text}\n</text>"}],
        )
        if response.stop_reason != "end_turn":
            return Score(None, f"judge stopped: {response.stop_reason}")
        verdict = json.loads(next(block.text for block in response.content if block.type == "text"))
        claims = verdict["unsupported_claims"]
        explanation = "; ".join(f"{c['claim']} ({c['why']})" for c in claims)
        return Score(float(verdict["faithful"]), "faithful" if verdict["faithful"] else "unfaithful", explanation)


def summary_faithfulness(judge: Judge) -> Callable[[Dict[str, Any], Dict[str, Any]], Score]:
    """The extraction summary against the extraction call's only inputs: inquiry reason and note."""

    def evaluate(output: Dict[str, Any], dataset_row: Dict[str, Any]) -> Score:
        if output.get("extraction") != "llm":
            return Score(None, "rules extraction" if "extraction" in output else "not routed")
        summary = output["signals"].get("summary")
        if not summary:
            return Score(None, "no summary")
        lead = Lead.model_validate_json(dataset_row["lead"])
        return judge(f"Inquiry reason: {lead.inquiry_reason}\nNote: {lead.free_text_note}", summary)

    return evaluate


def brief_faithfulness(judge: Judge) -> Callable[[Dict[str, Any], Dict[str, Any]], Score]:
    """The rep brief against the routing call's full input."""

    def evaluate(output: Dict[str, Any], dataset_row: Dict[str, Any]) -> Score:
        if output.get("routing_method") != "llm":
            return Score(None, "ranker fallback" if "routing_method" in output else "not routed")
        return judge(json.dumps(output["routing_input"], indent=2), output["rep_brief"])

    return evaluate


BRIEF_QUALITY_SYSTEM = """You rate a sales rep handoff brief on three rubrics. Score each 1–5.

Tone (1=robotic/unnatural, 3=ok but stiff, 5=warm professional conversational)
Facts (1=wrong numbers/names or invented, 3=mostly right with gaps, 5=all key note/form facts correct)
Usable as-is (1=needs full rewrite, 3=needs light edits, 5=rep could send/use almost unchanged)

Return JSON only. average is the mean of the three scores."""

BRIEF_QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "tone": {"type": "integer", "minimum": 1, "maximum": 5},
        "facts": {"type": "integer", "minimum": 1, "maximum": 5},
        "usable": {"type": "integer", "minimum": 1, "maximum": 5},
        "average": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["tone", "facts", "usable", "average", "notes"],
    "additionalProperties": False,
}


class BriefQualityJudge:
    def __init__(self, client: Any = None, model: Optional[str] = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=120.0, max_retries=2)
        self.client = client
        self.model = model or os.environ.get("LEADFLOW_JUDGE_MODEL") or JUDGE_MODEL

    def __call__(self, source: str, brief: str) -> Score:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=BRIEF_QUALITY_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": BRIEF_QUALITY_SCHEMA}},
            messages=[{"role": "user", "content": f"<source>\n{source}\n</source>\n\n<brief>\n{brief}\n</brief>"}],
        )
        if response.stop_reason != "end_turn":
            return Score(None, f"judge stopped: {response.stop_reason}")
        verdict = json.loads(next(block.text for block in response.content if block.type == "text"))
        avg = float(verdict["average"])
        explanation = (
            f"tone={verdict['tone']} facts={verdict['facts']} usable={verdict['usable']}; {verdict['notes']}"
        )
        # Normalize 1–5 → 0–1 so summarize means stay comparable; label keeps the 1–5 average.
        return Score(avg / 5.0, f"{avg:.1f}/5", explanation)


def brief_quality(judge: BriefQualityJudge) -> Callable[[Dict[str, Any], Dict[str, Any]], Score]:
    """Subjective 1–5 rubric: tone, facts from the note, usable as-is."""

    def evaluate(output: Dict[str, Any], dataset_row: Dict[str, Any]) -> Score:
        if output.get("routing_method") != "llm":
            return Score(None, "ranker fallback" if "routing_method" in output else "not routed")
        brief = output.get("rep_brief") or ""
        if not brief.strip():
            return Score(0.0, "empty brief", "no rep_brief")
        return judge(json.dumps(output["routing_input"], indent=2), brief)

    return evaluate


def evaluators(use_judge: bool = True) -> Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Score]]:
    found = {
        "tier_accuracy": tier_accuracy,
        "field_accuracy": field_accuracy,
        "rule_compliance": rule_compliance,
        "counterfactual_sensitivity": counterfactual_sensitivity,
        "brief_completeness": brief_completeness,
        "brief_format": brief_format,
    }
    if use_judge:
        judge = Judge()
        found["summary_faithfulness"] = summary_faithfulness(judge)
        found["brief_faithfulness"] = brief_faithfulness(judge)
        found["brief_quality"] = brief_quality(BriefQualityJudge())
    return found


# ----- Runners ----------------------------------------------------------------------------------


def run_local(rows, checks, counterfactual: bool, concurrency: int):
    """Run every lead, then every evaluator, in parallel. Returns (scores by lead, outputs by lead)."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outputs = list(pool.map(lambda row: run_lead(row, counterfactual), rows))
        jobs = {
            (row["submission_id"], name): pool.submit(_safe, check, output, row)
            for row, output in zip(rows, outputs)
            for name, check in checks.items()
        }
    results: Dict[str, Dict[str, Score]] = {row["submission_id"]: {} for row in rows}
    for (lead_id, name), job in jobs.items():
        results[lead_id][name] = job.result()
    for row, output in zip(rows, outputs):
        results[row["submission_id"]]["_latency"] = Score(output.get("latency_s"), output.get("routing_method", "-"))
    return results, {row["submission_id"]: output for row, output in zip(rows, outputs)}


def _safe(check, output, row) -> Score:
    try:
        return check(output, row)
    except Exception as exc:  # one broken check should not lose the whole run
        return Score(None, "evaluator error", f"{type(exc).__name__}: {exc}")


def summarize(results: Dict[str, Dict[str, Score]], names: List[str]) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for name in names:
        scores = [r[name].score for r in results.values() if r[name].score is not None]
        summary[name] = {"mean": statistics.mean(scores) if scores else None, "scored": len(scores)}
    return summary


def print_report(rows, results, names) -> None:
    for row in rows:
        scores = results[row["submission_id"]]
        failing = [f"{n}: {scores[n].label} ({scores[n].explanation})" for n in names if scores[n].score not in (None, 1.0)]
        print(f"{row['submission_id']} {row['company'][:24]:24} " + ("ok" if not failing else ""))
        for line in failing:
            print(f"    {line}")
    print()
    for name, stats in summarize(results, names).items():
        mean = "n/a" if stats["mean"] is None else f"{stats['mean']:.2f}"
        print(f"{name:28} {mean:>5}  on {stats['scored']} leads")
    latencies = [r["_latency"].score for r in results.values() if r["_latency"].score is not None]
    if latencies:
        print(f"{'latency_s_p50':28} {statistics.median(latencies):5.1f}")


def run_phoenix(rows, results: Dict[str, Dict[str, "Score"]], outputs: Dict[str, Dict[str, Any]], name: Optional[str]) -> None:
    """Record a finished run as an experiment on the Phoenix dataset leadflow-buyer-workbook.

    Scores are computed locally first, in parallel (Phoenix runs tasks one by one, which would take
    several minutes with the judge calls). The experiment's task and evaluators then hand Phoenix
    those outputs and scores, so the UI shows exactly what was computed.
    """
    try:
        from phoenix.client import Client
    except ImportError as exc:
        raise SystemExit("Phoenix client missing. Install with: .venv/bin/pip install -r requirements-observability.txt") from exc
    from leadflow.eval_buyer import upload_phoenix_dataset

    leads, expectations = load_leads()
    upload_phoenix_dataset(leads, expectations, load_gold())  # no-op when the dataset already exists
    client = Client()
    dataset = client.datasets.get_dataset(dataset=PHOENIX_DATASET)
    wanted = {row["submission_id"] for row in rows}

    def task(input):
        return outputs.get(input["submission_id"], {"skipped": "not in this run"})

    def as_phoenix(check_name):
        def evaluate(input):
            if input["submission_id"] not in wanted:
                return {"label": "not in this run"}
            result = results[input["submission_id"]][check_name]
            verdict = {"label": result.label, "explanation": result.explanation or result.label}
            if result.score is not None:
                verdict["score"] = result.score
            return verdict

        return evaluate

    names = [n for n in next(iter(results.values())) if not n.startswith("_")]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators={n: as_phoenix(n) for n in names},
        experiment_name=name or f"components {stamp}",
        experiment_description="Extraction, routing and rep brief evals from leadflow.eval_components",
        print_summary=False,
    )
    print(f"\nPhoenix: Datasets & Experiments → {PHOENIX_DATASET} → {name or f'components {stamp}'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phoenix", action="store_true", help="also record the run as a Phoenix experiment")
    parser.add_argument("--no-judge", action="store_true", help="skip the two LLM-judge evaluators")
    parser.add_argument("--no-counterfactual", action="store_true", help="skip the extra routing call per lead")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--name", help="experiment name in Phoenix")
    parser.add_argument("--limit", type=int, help="only the first N leads, for a quick check")
    args = parser.parse_args()

    if not anthropic_key_available():
        raise SystemExit("ANTHROPIC_API_KEY is not set (add it to .env).")
    rows = dataset_rows()[: args.limit]
    checks = evaluators(use_judge=not args.no_judge)
    results, outputs = run_local(rows, checks, not args.no_counterfactual, args.concurrency)
    print_report(rows, results, list(checks))
    if args.phoenix:
        run_phoenix(rows, results, outputs, args.name)


if __name__ == "__main__":
    main()
