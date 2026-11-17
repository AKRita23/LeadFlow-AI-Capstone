"""Typed records shared across the routing pipeline."""
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Role = Literal["CXO", "VP", "Director", "Manager", "Contributor"]

# The schema lists 1-15, 16-100, 101-500, 501-1000 and 1001+. The dataset also
# uses 51-250 and 251-1000 on valid leads, so those are accepted too.
CompanySize = Literal["1-15", "16-100", "51-250", "101-500", "251-1000", "501-1000", "1001+"]

InquiryReason = Literal[
    "I want to evaluate BeaconDesk for my organization",
    "I want to understand which BeaconDesk plan is right for me",
    "I want to buy licenses or upgrade",
    "I need a compliance plan",
]

US_STATE_CODES = frozenset(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
    "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY "
    "DC PR GU VI AS MP".split()
)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _new_submission_id() -> str:
    return f"sub_{uuid.uuid4().hex[:8]}"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


class Lead(BaseModel):
    """An inbound web form submission, validated against the Inbound Form schema.

    submission_id and submitted_at are auto-populated when the caller omits them.
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "first_name": "Elena",
                    "last_name": "Rostova",
                    "work_email": "elena@finpay.es",
                    "role": "VP",
                    "country": "ES",
                    "state": None,
                    "company": "FinPay Technologies",
                    "company_size": "101-500",
                    "phone_number": "+34915550199",
                    "inquiry_reason": "I want to evaluate BeaconDesk for my organization",
                    "free_text_note": "Our contract with Intercom expires in 2 weeks and we need 40 seats live before Q1.",
                }
            ]
        },
    )

    submission_id: str = Field(default_factory=_new_submission_id, min_length=1)
    submitted_at: str = Field(default_factory=_now_utc)
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)
    work_email: str
    role: Role
    country: str
    state: Optional[str] = None
    company: str = Field(min_length=1)
    company_size: CompanySize
    phone_number: str
    inquiry_reason: InquiryReason
    free_text_note: Optional[str] = Field(default=None, max_length=5000)

    @field_validator("submitted_at")
    @classmethod
    def _check_timestamp(cls, value: str) -> str:
        try:
            datetime.strptime(value, TIMESTAMP_FORMAT)
        except ValueError:
            raise ValueError("must be an ISO-8601 UTC timestamp like 2026-11-16T11:58:23Z")
        return value

    @field_validator("work_email")
    @classmethod
    def _check_email(cls, value: str) -> str:
        if not EMAIL_RE.match(value):
            raise ValueError("must be a valid email address like user@domain.com")
        return value.lower()

    @field_validator("country", "state")
    @classmethod
    def _upper(cls, value: Optional[str]) -> Optional[str]:
        return value.upper() if value else None

    @field_validator("country")
    @classmethod
    def _check_country(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Z]{2}", value or ""):
            raise ValueError("must be an ISO 3166-1 alpha-2 country code like US or ES")
        return value

    @field_validator("phone_number", mode="before")
    @classmethod
    def _normalise_phone(cls, value: Any) -> str:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        digits = re.sub(r"\D", "", str(value))
        if not 8 <= len(digits) <= 15:
            raise ValueError("must be an international phone number with 8-15 digits")
        return f"+{digits}"

    @field_validator("free_text_note")
    @classmethod
    def _blank_note_is_none(cls, value: Optional[str]) -> Optional[str]:
        return value or None

    @model_validator(mode="after")
    def _check_state_matches_country(self) -> "Lead":
        if self.country == "US":
            if self.state not in US_STATE_CODES:
                raise ValueError("state must be a USPS 2-letter code when country is US")
        elif self.state is not None:
            raise ValueError("state must be null when country is not US")
        return self

    @property
    def email_domain(self) -> str:
        return self.work_email.rsplit("@", 1)[-1]


@dataclass(frozen=True)
class LeadExpectation:
    """The dataset's expected outcome for a lead, used to evaluate the pipeline."""

    submission_id: str
    flow_path: str
    expected_intent_score: Optional[float]
    expected_urgency_tier: Optional[str]
    expected_rationale: str

    @property
    def excluded(self) -> bool:
        return "Excluded" in self.flow_path


@dataclass(frozen=True)
class Rep:
    """A sales rep from the CRM and calendar mock data."""

    rep_id: str
    name: str
    email: str
    territory: str
    languages: List[str]
    segments: List[str]
    working_status: str
    timezone: str
    discovery_opps: int
    discovery_arr: float
    evaluation_opps: int
    evaluation_arr: float
    proposal_opps: int
    proposal_arr: float
    negotiation_opps: int
    negotiation_arr: float
    overdue_tasks: int
    unworked_leads: int
    weekly_intent_points: int
    vertical_expertise: List[str]
    competitor_win_rate_affinity: str
    daily_lead_cap: int
    leads_assigned_today: int
    meetings_today_hours: float
    back_to_back_calls: int
