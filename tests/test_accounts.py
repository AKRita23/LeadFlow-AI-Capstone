from leadflow.accounts import resolve_existing_account
from leadflow.data_loader import load_accounts, load_leads, load_reps
from leadflow.models import Lead

ACCOUNTS = load_accounts()
REPS = load_reps()


def match(email):
    lead = Lead(
        first_name="Priya",
        last_name="Shah",
        work_email=email,
        role="Director",
        country="US",
        state="NY",
        company="Acme",
        company_size="101-500",
        phone_number="+12125550100",
        inquiry_reason="I want to buy licenses or upgrade",
    )
    return resolve_existing_account(lead, ACCOUNTS, REPS)


def test_account_rows_cover_the_gate_cases():
    accounts = load_accounts()
    scenarios = {account.scenario for account in accounts}
    assert scenarios == {
        "clear_customer",
        "clear_open_opportunity",
        "same_owner_expansion",
        "ambiguous_two_accounts",
        "ambiguous_split_opportunity",
        "closed_lost_ignored",
        "missing_owner",
    }
    assert len({account.domain for account in accounts}) == 11
    assert len({account.account_id for account in accounts}) == len(accounts)


def test_owners_are_real_reps():
    rep_ids = {rep.rep_id for rep in load_reps()}
    for account in load_accounts():
        if account.owner_rep_id is not None:
            assert account.owner_rep_id in rep_ids, account.account_id


def test_workbook_leads_stay_new_prospects():
    """The 20 labeled submissions are not customers in this table."""
    leads, _ = load_leads()
    for lead in leads:
        assert resolve_existing_account(lead, ACCOUNTS, REPS) is None, lead.submission_id


def test_clear_customer_preserves_the_owner():
    decision = match("priya@HarborPay.com")
    assert decision.outcome == "preserve_owner"
    assert decision.assigned_rep_id == "rep_002"
    assert decision.account_ids == ("acc_harborpay",)


def test_open_opportunity_preserves_the_owner():
    decision = match("priya@nimbusclinic.com")
    assert decision.outcome == "preserve_owner"
    assert decision.assigned_rep_id == "rep_001"
    assert decision.account_ids == ("acc_nimbus",)


def test_two_rows_with_one_owner_stay_clear():
    decision = match("priya@lumenlogistics.de")
    assert decision.outcome == "preserve_owner"
    assert decision.assigned_rep_id == "rep_004"
    assert decision.account_ids == ("acc_lumen", "acc_lumen_exp")


def test_two_accounts_on_one_domain_are_ambiguous():
    decision = match("priya@atlasgroup.com")
    assert decision.outcome == "ownership_exception"
    assert decision.assigned_rep_id is None
    assert decision.candidate_rep_ids == ("rep_009", "rep_038")
    assert decision.account_ids == ("acc_atlas_na", "acc_atlas_eu")


def test_customer_and_opportunity_with_different_owners_are_ambiguous():
    decision = match("priya@brightretail.com")
    assert decision.outcome == "ownership_exception"
    assert decision.candidate_rep_ids == ("rep_042", "rep_026")


def test_missing_owner_is_ambiguous():
    decision = match("priya@orphanstack.io")
    assert decision.outcome == "ownership_exception"
    assert decision.candidate_rep_ids == ()
    assert decision.account_ids == ("acc_orphan",)


def test_closed_lost_domain_is_a_new_prospect():
    assert match("priya@fadedchat.io") is None


def test_unknown_domain_is_a_new_prospect():
    assert match("priya@example.com") is None
