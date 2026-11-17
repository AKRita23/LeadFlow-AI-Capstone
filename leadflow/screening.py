"""Pre-agent screening: noise, non-ICP leads, and duplicate submissions.

screen_lead checks the submission itself. The first rule that matches wins:

| Reason            | Outcome    | Signal                                                     |
|-------------------|------------|------------------------------------------------------------|
| test_submission   | drop       | test, qa, bot, staging or noreply as a word in the email   |
| internal          | drop       | a beacondesk.io address                                    |
| competitor        | drop       | an email domain belonging to a known competitor            |
| recruiting_agency | self_serve | recruiting or staffing words in the company name or domain |
| media             | self_serve | blog, media, news or similar in the company name or domain |
| micro_company     | self_serve | company size 1-15                                          |

Free-text notes are not screened. Words such as "student" appear in valid
notes (an EdTech buyer's student support spike), so notes are left to the
buyer agent.

find_duplicate compares the lead with leads already triaged in the last 30
days. A personal email domain such as gmail.com never counts as a company match.
"""
import re
from datetime import datetime, timedelta
from typing import Sequence

from leadflow.models import TIMESTAMP_FORMAT, DuplicateMatch, Lead, PriorSubmission, ScreenResult

NOISE_TOKENS = frozenset({"test", "testing", "qa", "bot", "staging", "noreply"})

INTERNAL_DOMAINS = frozenset({"beacondesk.io"})

COMPETITOR_DOMAINS = frozenset(
    {
        "intercom.com",
        "intercom.io",
        "zendesk.com",
        "drift.com",
        "salesloft.com",
        "crisp.chat",
        "tidio.com",
    }
)

RECRUITING_RE = re.compile(r"recruit|staffing|headhunt|executive search|talent agency")
MEDIA_RE = re.compile(r"\b(blog|blogger|media|magazine|news|press|podcast)\b")

PERSONAL_EMAIL_DOMAINS = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "live.com",
        "msn.com",
        "icloud.com",
        "me.com",
        "aol.com",
        "proton.me",
        "protonmail.com",
        "gmx.com",
        "gmx.de",
        "mail.com",
        "yandex.com",
        "qq.com",
        "163.com",
    }
)

DUPLICATE_WINDOW = timedelta(days=30)


def screen_lead(lead: Lead) -> ScreenResult:
    local, domain = lead.work_email.split("@", 1)
    if NOISE_TOKENS & _tokens(local + "." + domain):
        return ScreenResult("drop", "test_submission", f"Test or bot address: {lead.work_email}.")
    if _domain_in(domain, INTERNAL_DOMAINS):
        return ScreenResult("drop", "internal", f"Internal BeaconDesk address: {lead.work_email}.")
    if _domain_in(domain, COMPETITOR_DOMAINS):
        return ScreenResult("drop", "competitor", f"Competitor email domain: {domain}.")
    organisation = f"{lead.company} {' '.join(_tokens(domain))}".lower()
    if RECRUITING_RE.search(organisation):
        return ScreenResult("self_serve", "recruiting_agency", f"Recruiting or staffing company: {lead.company}.")
    if MEDIA_RE.search(organisation):
        return ScreenResult("self_serve", "media", f"Media or blog company: {lead.company}.")
    if lead.company_size == "1-15":
        return ScreenResult("self_serve", "micro_company", "Company size 1-15 is below the sales-assisted segment.")
    return ScreenResult("pass", None, "Passed screening.")


def is_personal_domain(domain: str) -> bool:
    return domain.lower() in PERSONAL_EMAIL_DOMAINS


def find_duplicate(lead: Lead, prior: Sequence[PriorSubmission]) -> DuplicateMatch:
    """Find the most recent earlier lead with the same email, or else the same company domain.

    For a domain match, a prior lead that was assigned a rep is preferred over
    a more recent one that was not.
    """
    submitted = _parse(lead.submitted_at)
    recent = [
        p
        for p in prior
        if p.submission_id != lead.submission_id and submitted - DUPLICATE_WINDOW <= _parse(p.submitted_at) <= submitted
    ]
    recent.sort(key=lambda p: p.submitted_at, reverse=True)

    same_email = next((p for p in recent if p.work_email == lead.work_email), None)
    if same_email is not None:
        return DuplicateMatch("same_email", same_email.submission_id, same_email.assigned_rep_id)

    if is_personal_domain(lead.email_domain):
        return DuplicateMatch("none")
    same_domain = [p for p in recent if p.domain == lead.email_domain]
    if not same_domain:
        return DuplicateMatch("none")
    chosen = next((p for p in same_domain if p.assigned_rep_id), same_domain[0])
    return DuplicateMatch("same_domain", chosen.submission_id, chosen.assigned_rep_id)


def _tokens(text: str) -> set:
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if token}


def _domain_in(domain: str, domains: frozenset) -> bool:
    return any(domain == known or domain.endswith("." + known) for known in domains)


def _parse(timestamp: str) -> datetime:
    return datetime.strptime(timestamp, TIMESTAMP_FORMAT)
