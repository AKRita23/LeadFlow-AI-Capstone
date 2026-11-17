# LeadFlow-AI-Capstone
An autonomous multi-agent system to continuously evaluate live technographic and behavioral intent data, dynamically balance sales workloads, and instantly match high-intent prospects with the right sales reps.

## Status

Work so far covers the first flow stage, **inbound submission**, and the schema-check part of the second, **validate, deduplicate, ownership**.

| Flow stage | Status |
|---|---|
| Inbound submission (FastAPI) | Done |
| Required-field / schema validation | Done |
| Spam and noise filtering | Not started |
| Duplicate email/domain check | Not started |
| Existing-account detection | Not started |

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running the API

```bash
.venv/bin/uvicorn leadflow.api:app --reload
```

Interactive docs are at http://127.0.0.1:8000/docs. The `POST /leads` form there is pre-filled with a valid example lead.

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Returns `{"status": "ok"}` |
| `POST /leads` | Validates a form submission. Returns `202` with `status: "received"` and the normalised lead, or `422` with a per-field error list. |

Example:

```bash
curl -X POST http://127.0.0.1:8000/leads \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Elena",
    "last_name": "Rostova",
    "work_email": "elena@finpay.es",
    "role": "VP",
    "country": "ES",
    "state": null,
    "company": "FinPay Technologies",
    "company_size": "101-500",
    "phone_number": "+34915550199",
    "inquiry_reason": "I want to evaluate BeaconDesk for my organization",
    "free_text_note": "Our contract with Intercom expires in 2 weeks."
  }'
```

## Running the tests

```bash
.venv/bin/python -m pytest -q
```

## Project layout

```
data/LeadFlow AI mock dataset.xlsx   Mock form submissions, expected outcomes, and rep roster
leadflow/models.py                   Lead (Pydantic, enforces the form schema), LeadExpectation, Rep
leadflow/data_loader.py              Loads and cleans the Excel dataset
leadflow/api.py                      FastAPI app
tests/test_data_loader.py            Dataset loading and cleaning
tests/test_api.py                    Endpoint and validation rules
```

## Validation rules

Applied by the `Lead` model, so API submissions and dataset rows are checked the same way.

| Field | Rule |
|---|---|
| `submission_id` | Generated as `sub_<8 hex chars>` if omitted |
| `submitted_at` | `YYYY-MM-DDTHH:MM:SSZ`; set to the current UTC time if omitted |
| `first_name`, `last_name`, `company` | Non-empty after trimming whitespace |
| `work_email` | Must match `user@domain.tld`; lowercased |
| `role` | `CXO`, `VP`, `Director`, `Manager`, `Contributor` |
| `country` | 2-letter code; uppercased |
| `state` | Required USPS code when `country` is `US`; must be null otherwise |
| `company_size` | `1-15`, `16-100`, `51-250`, `101-500`, `251-1000`, `501-1000`, `1001+` |
| `phone_number` | 8–15 digits; normalised to `+<digits>` |
| `inquiry_reason` | One of the four form dropdown values |
| `free_text_note` | Optional, max 5,000 characters; blank becomes null |

## Dataset notes

The loader reads two sheets: `Mock dataset - Buyer Context br` (20 leads, 5 expected to be excluded) and `Mock dataset - CRM & Rep Calend` (50 reps). It corrects these issues:

- **Company size stored as a date.** Excel converted `1-15` to `2026-01-15` on all five excluded leads. The loader converts dates back to `month-day`.
- **Phone numbers stored as numbers.** They have no leading `+`. The `Lead` model normalises them.
- **Empty cells.** Converted to `None` rather than NaN.

Choices that depart from the form schema:

- `51-250` and `251-1000` are accepted as company sizes. The schema does not list them, but valid leads in the dataset use them.
- The email check is a format check only. Domains such as `.local` pass validation and are left for the spam filter to exclude.

Other things to know about the data:

- The dataset has no existing-account or open-opportunity data. Mock account data will be added for the existing-account check.
- Several rep names are shared with lead names, and two rep names appear twice under different IDs. Code identifies reps by `rep_id` only.
