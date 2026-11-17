"""Load the LeadFlow mock dataset from Excel and clean known data-entry issues.

Fixes applied while loading:
- Company sizes such as "1-15" that Excel auto-converted to dates are restored.
- Empty cells become None instead of NaN.

Each lead is built through the Lead model, so the dataset gets the same
validation as API submissions (including phone normalisation to "+<digits>").
"""
import datetime as dt
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd

from leadflow.models import Lead, LeadExpectation, Rep

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "LeadFlow AI mock dataset.xlsx"

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
