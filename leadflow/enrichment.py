"""Mock TheirStack and PostHog lookups for the buyer agent.

Both read the enrichment columns of the workbook's buyer sheet. TheirStack is
indexed by domain and by company name, so a lead with a personal email can
still be looked up. PostHog is indexed by work email, the identity the form
sends to it.
"""
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from leadflow.contracts import BehaviorSignals, Firmographics


@dataclass(frozen=True)
class EnrichmentRow:
    company: str
    work_email: str
    firmographics: Optional[Firmographics]
    behavior: Optional[BehaviorSignals]


def company_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


class MockTheirStack:
    def __init__(self, rows: Iterable[EnrichmentRow]):
        self._by_domain: Dict[str, Firmographics] = {}
        self._by_company: Dict[str, Firmographics] = {}
        for row in rows:
            if row.firmographics is None:
                continue
            self._by_domain[row.firmographics.domain] = row.firmographics
            self._by_company[company_key(row.company)] = row.firmographics

    def by_domain(self, domain: str) -> Optional[Firmographics]:
        return self._by_domain.get(domain.lower())

    def by_company(self, company: str) -> Optional[Firmographics]:
        return self._by_company.get(company_key(company))


class MockPostHog:
    def __init__(self, rows: Iterable[EnrichmentRow]):
        self._by_email: Dict[str, BehaviorSignals] = {
            row.work_email.lower(): row.behavior for row in rows if row.behavior is not None
        }

    def behavior(self, work_email: str) -> Optional[BehaviorSignals]:
        return self._by_email.get(work_email.lower())
