"""Existing-account gate over the mock CRM table.

A lead matches on its email domain. Only Customer and Open Opportunity rows
count. Closed-lost rows stay in the file so a finished deal is not mistaken
for a current account.

One distinct owner preserves that owner. More than one owner, or an active
row with no owner, is an ambiguous-ownership exception. The buyer agent and
the routing agent are not called on either of those outcomes.
"""
from typing import Dict, Iterable, List, Sequence

from leadflow.models import ACTIVE_ACCOUNT_TYPES, Account, AccountMatch, Lead, OwnershipDecision, Rep


def match_domain(domain: str, accounts: Sequence[Account]) -> AccountMatch:
    """Classify one email domain against the mock account rows."""
    key = domain.strip().lower()
    active = [account for account in accounts if account.domain == key and account.record_type in ACTIVE_ACCOUNT_TYPES]
    owner_rep_ids = _distinct_owners(active)
    account_ids = tuple(account.account_id for account in active)
    if not active:
        return AccountMatch(key, "new_prospect", None, (), ())
    if any(account.owner_rep_id is None for account in active) or len(owner_rep_ids) != 1:
        return AccountMatch(key, "ambiguous_ownership", None, owner_rep_ids, account_ids)
    return AccountMatch(key, "preserve_owner", owner_rep_ids[0], owner_rep_ids, account_ids)


def resolve_existing_account(
    lead: Lead, match: AccountMatch, accounts: Sequence[Account], reps: Sequence[Rep]
) -> OwnershipDecision:
    """Keep the account owner, or raise an ownership exception for RevOps.

    An owner that is not on the rep roster is also an exception, since the lead
    would otherwise be assigned to nobody.
    """
    if match.outcome == "new_prospect":
        raise ValueError(f"{lead.submission_id} has no existing account; use the rep filter instead")
    reps_by_id = {rep.rep_id: rep for rep in reps}
    matched = [account for account in accounts if account.account_id in match.account_ids]

    if match.outcome == "preserve_owner" and match.owner_rep_id in reps_by_id:
        owner = reps_by_id[match.owner_rep_id]
        reason = f"Existing account owned by {owner.rep_id}; owner preserved."
        headline = f"Owner kept: {_rep_label(owner.rep_id, reps_by_id)}."
        outcome, assigned = "preserve_owner", owner.rep_id
    else:
        problems = []
        if match.outcome == "preserve_owner":
            problems.append(f"Account owner {match.owner_rep_id} is not on the rep roster.")
        if len(match.owner_rep_ids) > 1:
            problems.append(f"Active records on {match.domain} have {len(match.owner_rep_ids)} owners.")
        if any(account.owner_rep_id is None for account in matched):
            problems.append(f"An active record on {match.domain} has no owner.")
        reason = " ".join(problems)
        owners = ", ".join(_rep_label(rep_id, reps_by_id) for rep_id in match.owner_rep_ids) or "none"
        headline = f"Ownership exception for RevOps. Candidate owners: {owners}."
        outcome, assigned = "ownership_exception", None

    return OwnershipDecision(
        submission_id=lead.submission_id,
        outcome=outcome,
        assigned_rep_id=assigned,
        candidate_rep_ids=match.owner_rep_ids,
        account_ids=match.account_ids,
        reason=reason,
        brief=_brief(lead, headline, reason, matched, reps_by_id),
    )


def _brief(lead: Lead, headline: str, reason: str, matched: Sequence[Account], reps_by_id: Dict[str, Rep]) -> str:
    lines = [headline, reason, "", "Accounts:"]
    for account in matched:
        owner = _rep_label(account.owner_rep_id, reps_by_id) if account.owner_rep_id else "no owner"
        line = f"- {account.account_name} ({account.account_id}), {account.record_type}, {account.segment}, owner {owner}"
        if account.opportunity_name:
            line += f"; opportunity {account.opportunity_name} at {account.opportunity_stage}"
        lines.append(line)
    lines += [
        "",
        f"Contact: {lead.first_name} {lead.last_name}, {lead.role}, {lead.company}",
        f"Email: {lead.work_email}  Phone: {lead.phone_number}",
        f"Inquiry: {lead.inquiry_reason}",
        f"Note: {lead.free_text_note or 'none'}",
        f"Submitted: {lead.submitted_at} ({lead.submission_id})",
    ]
    return "\n".join(lines)


def _rep_label(rep_id: str, reps_by_id: Dict[str, Rep]) -> str:
    rep = reps_by_id.get(rep_id)
    return f"{rep.name} ({rep_id})" if rep else f"{rep_id} (not on roster)"


def _distinct_owners(accounts: Iterable[Account]) -> tuple:
    seen: List[str] = []
    for account in accounts:
        owner = account.owner_rep_id
        if owner and owner not in seen:
            seen.append(owner)
    return tuple(seen)
