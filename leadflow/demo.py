"""Sample leads for the demo page.

The 20 workbook submissions, plus scenario leads for the paths the workbook
never reaches: an existing customer, an ownership exception, a duplicate, a
competitor's address, an unmapped country, and a closed-lost account.
"""
from functools import lru_cache
from typing import Dict, List

from leadflow.data_loader import load_leads

_SCENARIO_BASE = {
    "first_name": "Priya",
    "last_name": "Shah",
    "role": "Director",
    "country": "US",
    "state": "NY",
    "company_size": "101-500",
    "phone_number": "+12125550100",
    "inquiry_reason": "I want to buy licenses or upgrade",
    "submitted_at": "2026-11-16T18:30:00Z",
    "free_text_note": "Adding 15 seats for the support team before the holidays.",
}


def _scenario(submission_id: str, label: str, **fields) -> Dict:
    return {"group": "Scenarios", "label": label, "lead": {**_SCENARIO_BASE, "submission_id": submission_id, **fields}}


@lru_cache(maxsize=1)
def demo_samples() -> List[Dict]:
    """Each sample is {"group", "label", "lead"}, where lead is a POST /leads body."""
    leads, expectations = load_leads()
    samples = []
    for lead, expected in zip(leads, expectations):
        group = "Workbook: excluded" if expected.excluded else "Workbook: new prospects"
        samples.append(
            {"group": group, "label": f"{lead.first_name} {lead.last_name}, {lead.company}", "lead": lead.model_dump()}
        )
    finpay = next(s["lead"] for s in samples if s["lead"]["submission_id"] == "sub_8f92a1c4")
    samples += [
        _scenario("sub_demo_cust", "Existing customer: HarborPay", work_email="priya@harborpay.com", company="HarborPay"),
        _scenario(
            "sub_demo_twoown", "Two account owners: Atlas Group", work_email="priya@atlasgroup.com", company="Atlas Group"
        ),
        _scenario(
            "sub_demo_lost",
            "Closed-lost account: FadedChat",
            work_email="priya@fadedchat.io",
            company="FadedChat",
            state="CA",
        ),
        _scenario(
            "sub_demo_comp", "Competitor address: Intercom", work_email="sam@intercom.com", company="Intercom", state="CA"
        ),
        _scenario(
            "sub_demo_pk",
            "Unmapped country: Pakistan",
            work_email="priya@karachiretail.pk",
            company="Karachi Retail",
            country="PK",
            state=None,
            phone_number="+922135550100",
        ),
        {
            "group": "Scenarios",
            "label": "Duplicate: FinPay again next day (send FinPay first)",
            "lead": {**finpay, "submission_id": "sub_demo_dup", "submitted_at": "2026-11-17T11:58:23Z"},
        },
    ]
    return samples
