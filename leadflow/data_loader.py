"""Load the LeadFlow mock dataset and clean known data-entry issues.

Fixes applied while loading:
- Company sizes such as "1-15" that Excel auto-converted to dates are restored.
- Empty cells become None instead of NaN.

Each lead is built through the Lead model, so the dataset gets the same
validation as API submissions (including phone normalisation to "+<digits>").
"""
import csv
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from leadflow.contracts import COMPETITORS, BehaviorSignals, Firmographics
from leadflow.enrichment import EnrichmentRow
from leadflow.models import Account, AccountRecordType, Lead, LeadExpectation, Rep, TerritoryRule

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "LeadFlow AI mock dataset.xlsx"
ACCOUNTS_PATH = DATA_DIR / "mock_accounts.csv"
TERRITORIES_PATH = DATA_DIR / "territories.csv"

TerritoryMap = Dict[Tuple[str, Optional[str]], TerritoryRule]

BUYER_SHEET = "Mock dataset - Buyer Context br"
REP_SHEET = "Mock dataset - CRM & Rep Calend"


def _clean(value: Any) -> Any:
    """Turn NaN/NaT into None and strip strings."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _company_size(value: Any) -> str:
    """Restore size ranges that Excel turned into dates (e.g. 2026-01-15 -> "1-15")."""
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        return f"{value.month}-{value.day}"
    return str(value).strip()


def _split_list(value: Optional[str], sep: str) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(sep) if part.strip()]


def load_leads(path: Path = DATASET_PATH) -> Tuple[List[Lead], List[LeadExpectation]]:
    """Return the form submissions and the dataset's expected outcome for each."""
    df = pd.read_excel(path, sheet_name=BUYER_SHEET, header=1)
    leads, expectations = [], []
    for _, row in df.iterrows():
        r = {k: _clean(v) for k, v in row.items()}
        if r["Submission ID"] is None:
            continue
        leads.append(
            Lead(
                submission_id=r["Submission ID"],
                submitted_at=r["Submitted At"],
                first_name=r["First Name"],
                last_name=r["Last Name"],
                work_email=r["Work Email"],
                role=r["Role"],
                country=r["Country"],
                state=r["State"],
                company=r["Company"],
                company_size=_company_size(r["Company Size"]),
                phone_number=r["Phone Number"],
                inquiry_reason=r["Inquiry Reason"],
                free_text_note=r["Free Text Note"],
            )
        )
        expectations.append(
            LeadExpectation(
                submission_id=r["Submission ID"],
                flow_path=r["Flow Path (via code)"],
                expected_intent_score=r["Expected Intent Score"],
                expected_urgency_tier=r["Expected Urgency Tier"],
                expected_rationale=r["Expected Rationale"],
            )
        )
    return leads, expectations


def load_enrichment(path: Path = DATASET_PATH) -> List[EnrichmentRow]:
    """Return the TheirStack and PostHog columns of the buyer sheet, one row per lead.

    Leads the workbook never enriched (the excluded self-serve and test rows)
    get None for both.
    """
    df = pd.read_excel(path, sheet_name=BUYER_SHEET, header=1)
    rows = []
    for _, row in df.iterrows():
        r = {k: _clean(v) for k, v in row.items()}
        if r["Submission ID"] is None:
            continue
        firmographics = None
        if r["Industry"] is not None:
            firmographics = Firmographics(
                domain=str(r["Domain"]).lower(),
                industry=r["Industry"],
                headcount=None if r["Headcount"] is None else int(r["Headcount"]),
                tech_stack=_split_list(r["Full Tech Stack"], ","),
                competitors_detected=[c for c in COMPETITORS if c in (r["Competitors Detected"] or "")],
            )
        behavior = None
        if r["Pricing Views (48h)"] is not None:
            behavior = BehaviorSignals(
                pricing_views_48h=int(r["Pricing Views (48h)"]),
                doc_views_48h=int(r["Doc Views (48h)"]),
                sessions_7d=int(r["Sessions (7d)"]),
                summary=r["Telemetry Summary"],
            )
        rows.append(EnrichmentRow(r["Company"], r["Work Email"], firmographics, behavior))
    return rows


def load_accounts(path: Path = ACCOUNTS_PATH) -> List[Account]:
    """Return the mock CRM accounts used by the existing-account gate.

    The workbook has no account or opportunity rows. This table is the
    stand-in. None of its domains appear on the 20 form submissions, so those
    leads keep the flow path written in the workbook.
    """
    accounts = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            accounts.append(
                Account(
                    account_id=row["account_id"],
                    scenario=row["scenario"],
                    account_name=row["account_name"],
                    domain=row["domain"].strip().lower(),
                    country=row["country"].strip().upper(),
                    segment=row["segment"],
                    record_type=_record_type(row["record_type"]),
                    owner_rep_id=row["owner_rep_id"].strip() or None,
                    opportunity_name=row["opportunity_name"].strip() or None,
                    opportunity_stage=row["opportunity_stage"].strip() or None,
                )
            )
    return accounts


def load_territories(path: Path = TERRITORIES_PATH) -> TerritoryMap:
    """Return territory and language by (country, state). state is None outside the US."""
    territories: TerritoryMap = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rule = TerritoryRule(
                country=row["country"].strip().upper(),
                state=row["state"].strip().upper() or None,
                territory=row["territory"].strip(),
                language=row["language"].strip().lower(),
            )
            key = (rule.country, rule.state)
            if key in territories:
                raise ValueError(f"duplicate territory row for {key}")
            territories[key] = rule
    return territories


def _record_type(value: str) -> AccountRecordType:
    if value not in ("Customer", "Open Opportunity", "Closed Lost"):
        raise ValueError(f"unknown account record type: {value}")
    return value


def load_reps(path: Path = DATASET_PATH) -> List[Rep]:
    """Return the sales rep roster with pipeline and calendar data."""
    df = pd.read_excel(path, sheet_name=REP_SHEET, header=1)
    reps = []
    for _, row in df.iterrows():
        r = {k: _clean(v) for k, v in row.items()}
        if r["Rep ID"] is None:
            continue
        reps.append(
            Rep(
                rep_id=r["Rep ID"],
                name=r["Rep Name"],
                email=r["Work Email"],
                territory=r["Territory Region"],
                languages=_split_list(r["Supported Languages"], ","),
                segments=_split_list(r["Segment Focus"], "/"),
                working_status=r["Working Status"],
                timezone=r["Timezone"],
                discovery_opps=int(r["Discovery Opps (Count)"]),
                discovery_arr=float(r["Discovery ARR ($)"]),
                evaluation_opps=int(r["Evaluation Opps (Count)"]),
                evaluation_arr=float(r["Evaluation ARR ($)"]),
                proposal_opps=int(r["Proposal Opps (Count)"]),
                proposal_arr=float(r["Proposal ARR ($)"]),
                negotiation_opps=int(r["Negotiation Opps (Count)"]),
                negotiation_arr=float(r["Negotiation ARR ($)"]),
                overdue_tasks=int(r["Overdue Tasks"]),
                unworked_leads=int(r["Unworked Leads"]),
                weekly_intent_points=int(r["Weekly Intent Points Assigned"]),
                vertical_expertise=_split_list(r["Top Vertical Expertise"], ","),
                competitor_win_rate_affinity=r["Competitor Win-Rate Affinity"],
                daily_lead_cap=int(r["Daily Lead Cap"]),
                leads_assigned_today=int(r["Leads Assigned Today"]),
                meetings_today_hours=float(r["Meetings Today (Hours)"]),
                back_to_back_calls=int(r["Back-to-Back Calls"]),
            )
        )
    return reps
