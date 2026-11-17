import pytest
from fastapi.testclient import TestClient

from leadflow.api import app
from leadflow.data_loader import load_leads

client = TestClient(app)

VALID = {
    "submission_id": "sub_test0001",
    "submitted_at": "2026-11-16T11:58:23Z",
    "first_name": "Elena",
    "last_name": "Rostova",
    "work_email": "Elena@FinPay.es",
    "role": "VP",
    "country": "es",
    "state": None,
    "company": "FinPay Technologies",
    "company_size": "101-500",
    "phone_number": "+34 915 550 199",
    "inquiry_reason": "I want to evaluate BeaconDesk for my organization",
    "free_text_note": "Our contract with Intercom expires in 2 weeks.",
}


def post(**overrides):
    payload = {**VALID, **overrides}
    payload = {k: v for k, v in payload.items() if v is not ...}
    return client.post("/leads", json=payload)


def error_fields(response):
    return {tuple(e["loc"][1:]) for e in response.json()["detail"]}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_valid_submission_is_received_and_normalised():
    response = post()
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "received"
    assert body["lead"]["work_email"] == "elena@finpay.es"
    assert body["lead"]["country"] == "ES"
    assert body["lead"]["phone_number"] == "+34915550199"


def test_auto_populated_fields_are_generated():
    response = post(submission_id=..., submitted_at=...)
    assert response.status_code == 202
    lead = response.json()["lead"]
    assert lead["submission_id"].startswith("sub_")
    assert lead["submitted_at"].endswith("Z")


def test_blank_note_becomes_null():
    assert post(free_text_note="   ").json()["lead"]["free_text_note"] is None


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"first_name": ""}, ("first_name",)),
        ({"last_name": ...}, ("last_name",)),
        ({"company": "  "}, ("company",)),
        ({"work_email": "not-an-email"}, ("work_email",)),
        ({"work_email": "a@b"}, ("work_email",)),
        ({"role": "Intern"}, ("role",)),
        ({"country": "Spain"}, ("country",)),
        ({"company_size": "10-20"}, ("company_size",)),
        ({"inquiry_reason": "Just browsing"}, ("inquiry_reason",)),
        ({"phone_number": "12345"}, ("phone_number",)),
        ({"submitted_at": "16/11/2026 11:58"}, ("submitted_at",)),
    ],
)
def test_invalid_fields_are_rejected(overrides, field):
    response = post(**overrides)
    assert response.status_code == 422
    assert field in error_fields(response)


@pytest.mark.parametrize(
    "overrides",
    [
        {"country": "US", "state": None},
        {"country": "US", "state": "ZZ"},
        {"country": "ES", "state": "CA"},
    ],
)
def test_state_must_match_country(overrides):
    response = post(**overrides)
    assert response.status_code == 422
    assert "state" in response.text


def test_us_lead_with_state_is_accepted():
    assert post(country="US", state="ca").json()["lead"]["state"] == "CA"


def test_every_dataset_lead_passes_schema_validation():
    leads, _ = load_leads()
    for lead in leads:
        response = client.post("/leads", json=lead.model_dump())
        assert response.status_code == 202, (lead.submission_id, response.text)
