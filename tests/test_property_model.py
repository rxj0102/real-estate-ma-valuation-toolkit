"""Sanity tests for property_model.py: amortization math and portfolio roll-up."""

import pytest

from real_estate_ma_toolkit.property_model import (
    CorporateOverhead,
    GrowthAssumptions,
    PortfolioCompany,
    Property,
    amortization_schedule,
    balance_payment,
)


def _sample_property(property_id: str = "P01") -> Property:
    return Property(
        property_id=property_id,
        name="Test Gardens",
        submarket="Test Submarket",
        units=200,
        year_built=2015,
        in_place_monthly_rent_per_unit=1500.0,
        current_occupancy=0.92,
        other_income_per_unit_monthly=90.0,
        controllable_opex_ratio=0.37,
        property_tax_annual=350_000.0,
        insurance_annual=120_000.0,
        capex_reserve_per_unit_annual=300.0,
        going_in_cap_rate=0.05,
        debt_balance=20_000_000.0,
        interest_rate=0.05,
        amortization_years=30,
        io_period_years=0,
    )


def test_balance_payment_matches_standard_mortgage_formula():
    # $500,000 at 6%/yr (0.5%/mo) over 360 months -> standard payment ~$2,997.75
    payment = balance_payment(500_000.0, 0.06 / 12, 360)
    assert payment == pytest.approx(2997.75, abs=0.5)


def test_level_payment_fully_amortizes_balance_to_zero():
    principal = 1_000_000.0
    rate = 0.06
    years = 30
    schedule = amortization_schedule(principal, rate, years, io_years=0, projection_years=years)
    assert schedule.loc[years, "ending_balance"] == pytest.approx(0.0, abs=1.0)
    # debt service should be level (same payment every year, ignoring rounding)
    assert schedule["debt_service"].std() < 1.0


def test_interest_only_period_defers_principal_paydown():
    principal = 1_000_000.0
    rate = 0.06
    schedule = amortization_schedule(principal, rate, amortization_years=30, io_years=3, projection_years=5)
    # No principal paid down during the first 3 years
    assert schedule.loc[1, "principal_paydown"] == pytest.approx(0.0, abs=1e-6)
    assert schedule.loc[3, "ending_balance"] == pytest.approx(principal, abs=1e-6)
    # Principal paydown resumes in year 4
    assert schedule.loc[4, "principal_paydown"] > 0


def test_direct_cap_value_and_ltv():
    prop = _sample_property()
    value = prop.direct_cap_value()
    assert value == pytest.approx(prop.year1_noi() / prop.going_in_cap_rate)
    assert prop.loan_to_value() == pytest.approx(prop.debt_balance / value)


def test_portfolio_rollup_sums_two_identical_properties():
    p1 = _sample_property("P01")
    p2 = _sample_property("P02")
    overhead = CorporateOverhead(annual_ga=1_000_000.0, ga_growth_rate=0.025)
    company = PortfolioCompany(name="Test Co", properties=[p1, p2], overhead=overhead)

    assumptions = GrowthAssumptions()
    single = company.properties[0]
    from real_estate_ma_toolkit.property_model import project_property_cash_flows

    single_projection = project_property_cash_flows(single, assumptions, years=5)
    consolidated = company.project(assumptions, years=5)

    # Two identical properties -> consolidated NOI should be exactly 2x a single property's NOI
    assert consolidated["noi"].tolist() == pytest.approx((single_projection["noi"] * 2).tolist())
    assert company.total_units() == 400
    assert company.total_property_debt() == pytest.approx(p1.debt_balance + p2.debt_balance)
