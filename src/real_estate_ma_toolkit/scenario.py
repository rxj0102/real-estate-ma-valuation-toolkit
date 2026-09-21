"""The concrete deal scenario: Magnolia Multifamily Partners, LLC.

Single source of truth for the company, portfolio, and assumption set used
throughout the notebook and the output-generation script, so the two never
drift out of sync. All figures are illustrative -- see the project README
for the full disclaimer.

**Scenario**: CenterCap Group (fictional, illustrative advisory context) is
retained as sell-side M&A advisor to Magnolia Multifamily Partners, LLC, a
privately held multifamily operating company with a 12-property, ~3,000
unit garden-style/mid-rise portfolio concentrated in the Atlanta, GA metro
area. The engagement is exploring a full sale or majority recapitalization
of the company.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dcf_valuation import WACCInputs
from .property_model import CorporateOverhead, GrowthAssumptions, PortfolioCompany, Property

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

COMPANY_NAME = "Magnolia Multifamily Partners, LLC"
PROJECTION_YEARS = 5

GROWTH_ASSUMPTIONS = GrowthAssumptions(
    rent_growth_rate=0.030,
    other_income_growth_rate=0.025,
    stabilized_occupancy=0.93,
    occupancy_ramp_years=2,
    controllable_opex_growth_rate=0.030,
    property_tax_growth_rate=0.035,
    insurance_growth_rate=0.060,
    vacancy_frictional=0.0,
)

WACC_INPUTS = WACCInputs(
    risk_free_rate=0.043,
    equity_risk_premium=0.060,
    beta=1.0,
    cost_of_debt=0.0575,
    tax_rate=0.0,
    target_debt_weight=0.55,
)

TERMINAL_GROWTH_RATE = 0.025
EXIT_CAP_RATE_PREMIUM = 0.0025  # exit cap rate = going-in weighted avg cap rate + 25 bps


def load_portfolio_company() -> PortfolioCompany:
    """Build the :class:`PortfolioCompany` for Magnolia Multifamily Partners
    from ``data/portfolio.csv``.
    """
    df = pd.read_csv(DATA_DIR / "portfolio.csv")
    properties = [Property(**row) for row in df.to_dict(orient="records")]
    overhead = CorporateOverhead(
        annual_ga=3_500_000.0,
        ga_growth_rate=0.025,
        corporate_debt_balance=15_000_000.0,
        corporate_debt_rate=0.070,
        cash_and_equivalents=4_000_000.0,
    )
    return PortfolioCompany(name=COMPANY_NAME, properties=properties, overhead=overhead)


def default_exit_cap_rate(company: PortfolioCompany) -> float:
    """Exit cap rate: the portfolio's going-in weighted average cap rate
    plus a modest premium, reflecting the standard conservative convention
    of assuming some cap rate expansion (softer pricing) at exit versus
    today's entry pricing.
    """
    return company.weighted_avg_going_in_cap_rate() + EXIT_CAP_RATE_PREMIUM
