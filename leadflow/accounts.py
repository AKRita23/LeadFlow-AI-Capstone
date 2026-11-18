"""Existing-account gate over the mock CRM table.

A lead matches on its email domain. Only Customer and Open Opportunity rows
count. Closed-lost rows stay in the file so a finished deal is not mistaken
for a current account.

One distinct owner on the rep roster is preserved. More than one owner, an
active row with no owner, or an owner missing from the roster is an ownership
exception for RevOps. The buyer agent and the routing agent are not called on
either of those outcomes.
"""
from typing import Dict, Optional, Sequence

from leadflow.models import ACTIVE_ACCOUNT_TYPES, Account, Lead, OwnershipDecision, Rep


def resolve_existing_account(
    lead: Lead, accounts: Sequence[Account], reps: Sequence[Rep]
) -> Optional[OwnershipDecision]:
    """Keep the account owner, or raise an ownership exception for RevOps.

    Returns None when the lead's domain has no active account, i.e. a new prospect.
    """
    domain = lead.email_domain
    matched = [account for account in accounts if account.domain == domain and account.record_type in ACTIVE_ACCOUNT_TYPES]
    if not matched:
        return None
    reps_by_id = {rep.rep_id: rep for rep in reps}
    owners = tuple(dict.fromkeys(account.owner_rep_id for account in matched if account.owner_rep_id))
    unowned = any(account.owner_rep_id is None for account in matched)

    problems = []
    if len(owners) == 1 and not unowned and owners[0] not in reps_by_id:
        problems.append(f"Account owner {owners[0]} is not on the rep roster.")
    if len(owners) > 1:
        problems.append(f"Active records on {domain} have {len(owners)} owners.")
    if unowned:
        problems.append(f"An active record on {domain} has no owner.")

    if problems:
        outcome, assigned = "ownership_exception", None
        reason = " ".join(problems)
        candidates = ", ".join(_rep_label(rep_id, reps_by_id) for rep_id in owners) or "none"
        headline = f"Ownership exception for RevOps. Candidate owners: {candidates}."
    else:
        outcome, assigned = "preserve_owner", owners[0]
        reason = f"Existing account owned by {assigned}; owner preserved."
        headline = f"Owner kept: {_rep_label(assigned, reps_by_id)}."

    return OwnershipDecision(
        submission_id=lead.submission_id,
        outcome=outcome,
        assigned_rep_id=assigned,
        candidate_rep_ids=owners,
        account_ids=tuple(account.account_id for account in matched),
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
