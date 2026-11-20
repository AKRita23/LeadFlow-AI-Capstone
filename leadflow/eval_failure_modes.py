"""Run the five Demo Day failure-mode cases in data/failure_modes.json.

    python -m leadflow.eval_failure_modes

Offline checks where possible (triage, capacity force-low, related-rep ranker preference).
LLM brief after-hours line is checked only when Claude routing succeeds (needs API key).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from leadflow.contracts import check_proposal
from leadflow.data_loader import DATA_DIR, load_leads
from leadflow.extraction import anthropic_key_available
from leadflow.models import Lead
from leadflow.pipeline import triage
from leadflow.routing import RoutingAgent, rank_reps, router_from_env
from leadflow.store import DecisionStore

FAILURE_PATH = DATA_DIR / "failure_modes.json"


def load_cases(path: Path = FAILURE_PATH) -> List[Dict[str, Any]]:
    return list(json.loads(path.read_text())["cases"])


def _lead_for(case: Dict[str, Any], workbook: Dict[str, Lead]) -> Lead:
    if case.get("workbook_lead_id"):
        return workbook[case["workbook_lead_id"]]
    return Lead.model_validate(case["lead"])


def _with_all_low(capacity):
    rows = [row.model_copy(update={"bandwidth_tier": "Low"}) for row in capacity.reps]
    return capacity.model_copy(update={"reps": rows})


def run_case(case: Dict[str, Any], workbook: Dict[str, Lead], use_llm: bool) -> Dict[str, Any]:
    from leadflow.api import get_capacity_service, get_reference_data
    from leadflow.eval_buyer import default_agent

    lead = _lead_for(case, workbook)
    expect = case["expect"]
    accounts, roster, territories = get_reference_data()
    store = DecisionStore(Path(tempfile.mkdtemp()) / "fm.db")
    reps = roster
    result = triage(lead, accounts, reps, territories, store)
    report: Dict[str, Any] = {
        "id": case["id"],
        "title": case["title"],
        "outcome": result.outcome,
        "pass": True,
        "checks": [],
    }

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            report["pass"] = False

    if "outcome" in expect:
        check("outcome", result.outcome == expect["outcome"], f"got {result.outcome}")

    if result.outcome != "new_prospect" or result.shortlist is None:
        return report

    shortlist = result.shortlist
    if expect.get("after_hours") is not None:
        check("after_hours", bool(shortlist.after_hours) == bool(expect["after_hours"]))

    buyer = default_agent(use_llm=use_llm)(lead)
    capacity = get_capacity_service()(shortlist, reps, lead.email_domain)

    if expect.get("force_all_shortlist_low_bandwidth"):
        capacity = _with_all_low(capacity)

    related = case.get("related_rep_id")
    if expect.get("primary_is_related_when_eligible") and related:
        if related in shortlist.rep_ids:
            ranked = rank_reps(buyer, capacity, related)
            check("related_rep_preferred", ranked[0].rep_id == related, f"got {ranked[0].rep_id}")
        else:
            check("related_rep_on_shortlist", False, f"{related} not shortlisted")

    routing_agent = RoutingAgent(router_from_env() if use_llm else None)
    outcome = routing_agent(lead, buyer, capacity, shortlist, related_rep_id=related)
    problems = check_proposal(outcome.proposal, shortlist)
    if expect.get("routing_proposal_valid"):
        check("proposal_valid", not problems, "; ".join(problems))
    if expect.get("still_picks_a_shortlist_rep"):
        check("primary_on_shortlist", outcome.proposal.primary_rep_id in shortlist.rep_ids)

    if expect.get("buyer_extraction_ok_without_note"):
        check("ran_buyer", buyer.intent_score >= 1)

    if expect.get("brief_mentions_after_hours_when_llm") and outcome.method == "llm":
        check(
            "after_hours_brief_line",
            outcome.proposal.rep_brief.rstrip().endswith("Arrived outside working hours."),
        )

    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="use Claude for extract/route when API key is set")
    args = parser.parse_args()
    use_llm = bool(args.llm and anthropic_key_available())
    if args.llm and not use_llm:
        raise SystemExit("ANTHROPIC_API_KEY is not set (add it to .env).")
    if not use_llm:
        print("Offline mode (ranker/rules). Pass --llm to exercise Claude paths.")
    workbook = {lead.submission_id: lead for lead in load_leads()[0]}
    reports = [run_case(case, workbook, use_llm) for case in load_cases()]
    for report in reports:
        status = "PASS" if report["pass"] else "FAIL"
        print(f"{report['id']:28} {status:4}  {report['title']}")
        for item in report["checks"]:
            if not item["ok"]:
                print(f"  - {item['name']}: {item['detail']}")
    failed = sum(not r["pass"] for r in reports)
    print(f"\n{len(reports) - failed}/{len(reports)} cases passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
