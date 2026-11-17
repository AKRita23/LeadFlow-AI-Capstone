from leadflow.accounts import match_domain
from leadflow.data_loader import load_accounts, load_leads, load_reps


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
    accounts = load_accounts()
    leads, _ = load_leads()
    for lead in leads:
        match = match_domain(lead.email_domain, accounts)
        assert match.outcome == "new_prospect", lead.submission_id


def test_clear_customer_preserves_the_owner():
    match = match_domain("HarborPay.com", load_accounts())
    assert match.outcome == "preserve_owner"
    assert match.owner_rep_id == "rep_002"
    assert match.account_ids == ("acc_harborpay",)


def test_open_opportunity_preserves_the_owner():
    match = match_domain("nimbusclinic.com", load_accounts())
    assert match.outcome == "preserve_owner"
    assert match.owner_rep_id == "rep_001"
    assert match.account_ids == ("acc_nimbus",)


def test_two_rows_with_one_owner_stay_clear():
    match = match_domain("lumenlogistics.de", load_accounts())
    assert match.outcome == "preserve_owner"
    assert match.owner_rep_id == "rep_004"
    assert match.account_ids == ("acc_lumen", "acc_lumen_exp")


def test_two_accounts_on_one_domain_are_ambiguous():
    match = match_domain("atlasgroup.com", load_accounts())
    assert match.outcome == "ambiguous_ownership"
    assert match.owner_rep_id is None
    assert match.owner_rep_ids == ("rep_009", "rep_038")
    assert match.account_ids == ("acc_atlas_na", "acc_atlas_eu")


def test_customer_and_opportunity_with_different_owners_are_ambiguous():
    match = match_domain("brightretail.com", load_accounts())
    assert match.outcome == "ambiguous_ownership"
    assert match.owner_rep_ids == ("rep_042", "rep_026")


def test_missing_owner_is_ambiguous():
    match = match_domain("orphanstack.io", load_accounts())
    assert match.outcome == "ambiguous_ownership"
    assert match.owner_rep_ids == ()
    assert match.account_ids == ("acc_orphan",)


def test_closed_lost_domain_is_a_new_prospect():
    match = match_domain("fadedchat.io", load_accounts())
    assert match.outcome == "new_prospect"
    assert match.account_ids == ()


def test_unknown_domain_is_a_new_prospect():
    match = match_domain("example.com", load_accounts())
    assert match.outcome == "new_prospect"
