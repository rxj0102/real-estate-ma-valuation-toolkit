"""Asset-level and portfolio-level financial modeling for multifamily real estate.

This module builds the property-level engine that everything else in the
toolkit sits on top of. Real estate operating income is modeled the way an
owner/operator actually underwrites it -- gross potential rent less vacancy
plus ancillary income, less operating expenses (split into controllable
opex, property tax, and insurance, since each grows at a different rate) to
arrive at Net Operating Income (NOI). NOI, not EBITDA or net income, is the
central unit of real estate cash flow analysis.

Debt service is modeled per property using standard mortgage amortization
(with an optional interest-only period, common on agency multifamily
loans), so each asset also produces a levered cash flow and a debt service
coverage ratio (DSCR).

The portfolio roll-up aggregates every property's projection into a single
consolidated, company-level financial model, which is the input to the DCF
module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Property:
    """A single multifamily asset in the portfolio.

    Rent, other income, property tax, and insurance are all expressed as
    current (Year 0 / in-place) annualized dollar figures. Growth is applied
    by :class:`GrowthAssumptions` during projection, not stored here.

    Attributes:
        property_id: Short unique identifier (e.g. ``"P01"``).
        name: Property name as it would appear in an offering memorandum.
        submarket: Metro submarket, used for market-level color, not math.
        units: Unit count.
        year_built: Year the property was built or last substantially renovated.
        in_place_monthly_rent_per_unit: Current average in-place rent per unit, $/month.
        current_occupancy: Current physical occupancy, as a fraction (e.g. 0.92).
        other_income_per_unit_monthly: Ancillary income per unit, $/month
            (utility reimbursement (RUBS), parking, pet fees, admin fees).
        controllable_opex_ratio: Controllable operating expenses (payroll,
            repairs & maintenance, marketing, G&A at the property) as a
            fraction of Effective Gross Income (EGI).
        property_tax_annual: Current annual property tax bill, $.
        insurance_annual: Current annual property insurance premium, $.
        capex_reserve_per_unit_annual: Replacement reserve / recurring CapEx, $/unit/year.
        going_in_cap_rate: Market cap rate used to derive an as-is direct
            capitalization value for the asset (Year 1 NOI / cap rate).
        debt_balance: Outstanding mortgage principal balance, $.
        interest_rate: Annual note rate on the mortgage, as a fraction.
        amortization_years: Amortization period in years (30 is standard
            for agency multifamily debt).
        io_period_years: Interest-only period at the start of the loan term, years.
    """

    property_id: str
    name: str
    submarket: str
    units: int
    year_built: int
    in_place_monthly_rent_per_unit: float
    current_occupancy: float
    other_income_per_unit_monthly: float
    controllable_opex_ratio: float
    property_tax_annual: float
    insurance_annual: float
    capex_reserve_per_unit_annual: float
    going_in_cap_rate: float
    debt_balance: float
    interest_rate: float
    amortization_years: int = 30
    io_period_years: int = 0

    def gross_potential_rent(self) -> float:
        """Annualized gross potential rent (GPR) at 100% occupancy, $."""
        return self.in_place_monthly_rent_per_unit * self.units * 12

    def other_income(self) -> float:
        """Annualized ancillary/other income, $."""
        return self.other_income_per_unit_monthly * self.units * 12

    def year1_noi(self) -> float:
        """Approximate Year 1 (current in-place) NOI, $, used for the
        direct-capitalization as-is value below.
        """
        egi = self.gross_potential_rent() * self.current_occupancy + self.other_income()
        opex = egi * self.controllable_opex_ratio + self.property_tax_annual + self.insurance_annual
        return egi - opex

    def direct_cap_value(self) -> float:
        """As-is value implied by direct capitalization of Year 1 NOI.

        Value = NOI / Cap Rate. This is the standard real estate appraisal
        shortcut and is used here as a sanity check against the property's
        carrying value / debt basis, not as the company-level valuation
        methodology (see ``dcf_valuation.py``, ``comps_analysis.py``, and
        ``precedent_transactions.py`` for that).
        """
        return self.year1_noi() / self.going_in_cap_rate

    def loan_to_value(self) -> float:
        """Current LTV based on the direct-cap as-is value."""
        return self.debt_balance / self.direct_cap_value()


@dataclass
class GrowthAssumptions:
    """Portfolio-wide projection assumptions.

    A single assumption set is applied across all properties for
    consistency, mirroring how a company-level model is typically built
    even though individual submarkets could in principle be differentiated.

    Attributes:
        rent_growth_rate: Annual market/renewal rent growth, as a fraction.
        other_income_growth_rate: Annual growth in ancillary income.
        stabilized_occupancy: Portfolio-wide occupancy target that
            below-target properties ramp toward.
        occupancy_ramp_years: Number of years over which a property below
            ``stabilized_occupancy`` ramps linearly to it. Properties
            already at or above target hold flat.
        controllable_opex_growth_rate: Annual growth in controllable opex ratio's
            underlying dollar base (applied to the prior year's controllable
            opex dollars, not the ratio itself).
        property_tax_growth_rate: Annual growth in property tax, reflecting
            reassessment trends.
        insurance_growth_rate: Annual growth in insurance premiums. Set
            deliberately higher than other expense lines to reflect the
            well-documented run-up in Sun Belt multifamily property
            insurance costs.
        vacancy_frictional: Structural/frictional vacancy applied on top of
            the occupancy ramp (turnover downtime), as a fraction of GPR.
    """

    rent_growth_rate: float = 0.030
    other_income_growth_rate: float = 0.025
    stabilized_occupancy: float = 0.93
    occupancy_ramp_years: int = 2
    controllable_opex_growth_rate: float = 0.030
    property_tax_growth_rate: float = 0.035
    insurance_growth_rate: float = 0.060
    vacancy_frictional: float = 0.0


def _occupancy_path(current_occupancy: float, assumptions: GrowthAssumptions, years: int) -> np.ndarray:
    """Year 1..years occupancy path: linear ramp to stabilized occupancy,
    then flat. Properties already at/above target hold flat throughout.
    """
    target = assumptions.stabilized_occupancy
    ramp_years = max(assumptions.occupancy_ramp_years, 1)
    path = np.empty(years)
    for yr in range(1, years + 1):
        if current_occupancy >= target:
            path[yr - 1] = current_occupancy
        else:
            step = min(yr, ramp_years) / ramp_years
            path[yr - 1] = current_occupancy + step * (target - current_occupancy)
    return path


def project_property_cash_flows(prop: Property, assumptions: GrowthAssumptions, years: int = 5) -> pd.DataFrame:
    """Project a single property's revenue, NOI, debt service, and DSCR.

    Args:
        prop: The property to project.
        assumptions: Portfolio-wide growth assumptions.
        years: Projection horizon in years.

    Returns:
        A DataFrame indexed by projection year (1..years) with columns:
        ``gross_potential_rent``, ``vacancy_loss``, ``other_income``, ``egi``,
        ``controllable_opex``, ``property_tax``, ``insurance``, ``total_opex``,
        ``noi``, ``capex_reserve``, ``net_cash_flow_before_debt``,
        ``interest_expense``, ``principal_paydown``, ``debt_service``,
        ``levered_cash_flow``, ``dscr``.
    """
    occ_path = _occupancy_path(prop.current_occupancy, assumptions, years)

    rows = []
    gpr0 = prop.gross_potential_rent()
    other_income0 = prop.other_income()
    # Year 0 (current in-place) controllable opex, held as a dollar base so
    # it can be grown independently of the (growing) EGI base -- this
    # avoids silently re-deriving opex as a fixed ratio of a moving EGI
    # target, which would understate real-world expense stickiness.
    egi0 = gpr0 * prop.current_occupancy + other_income0
    controllable_opex0 = egi0 * prop.controllable_opex_ratio

    debt_schedule = amortization_schedule(
        prop.debt_balance, prop.interest_rate, prop.amortization_years, prop.io_period_years, years
    )

    for yr in range(1, years + 1):
        gpr_yr = gpr0 * (1 + assumptions.rent_growth_rate) ** yr
        other_income_yr = other_income0 * (1 + assumptions.other_income_growth_rate) ** yr
        occupancy_yr = occ_path[yr - 1] - assumptions.vacancy_frictional
        vacancy_loss = gpr_yr * (1 - occupancy_yr)
        egi = gpr_yr - vacancy_loss + other_income_yr

        controllable_opex_yr = controllable_opex0 * (1 + assumptions.controllable_opex_growth_rate) ** yr
        property_tax_yr = prop.property_tax_annual * (1 + assumptions.property_tax_growth_rate) ** yr
        insurance_yr = prop.insurance_annual * (1 + assumptions.insurance_growth_rate) ** yr
        total_opex = controllable_opex_yr + property_tax_yr + insurance_yr

        noi = egi - total_opex
        capex_reserve = prop.capex_reserve_per_unit_annual * prop.units
        net_cash_flow_before_debt = noi - capex_reserve

        interest_expense_yr = debt_schedule.loc[yr, "interest"]
        principal_paydown_yr = debt_schedule.loc[yr, "principal_paydown"]
        debt_service_yr = debt_schedule.loc[yr, "debt_service"]
        levered_cf = net_cash_flow_before_debt - debt_service_yr
        dscr = noi / debt_service_yr if debt_service_yr > 0 else np.nan

        rows.append(
            {
                "year": yr,
                "gross_potential_rent": gpr_yr,
                "vacancy_loss": vacancy_loss,
                "other_income": other_income_yr,
                "egi": egi,
                "controllable_opex": controllable_opex_yr,
                "property_tax": property_tax_yr,
                "insurance": insurance_yr,
                "total_opex": total_opex,
                "noi": noi,
                "capex_reserve": capex_reserve,
                "net_cash_flow_before_debt": net_cash_flow_before_debt,
                "interest_expense": interest_expense_yr,
                "principal_paydown": principal_paydown_yr,
                "debt_service": debt_service_yr,
                "levered_cash_flow": levered_cf,
                "dscr": dscr,
            }
        )

    return pd.DataFrame(rows).set_index("year")


def amortization_schedule(
    principal: float, annual_rate: float, amortization_years: int, io_years: int, projection_years: int
) -> pd.DataFrame:
    """Build a standard mortgage amortization schedule with an optional
    interest-only period.

    Args:
        principal: Original/current loan balance, $.
        annual_rate: Annual note rate, as a fraction.
        amortization_years: Full amortization period in years, once
            amortization begins.
        io_years: Number of initial years of interest-only payments.
        projection_years: Number of years of the schedule to return.

    Returns:
        A DataFrame indexed by year (1..projection_years) with columns
        ``beginning_balance``, ``interest``, ``principal_paydown``,
        ``debt_service``, ``ending_balance``.
    """
    monthly_rate = annual_rate / 12
    io_months = io_years * 12
    remaining_amort_months = amortization_years * 12 - io_months
    # The level payment is set once, based on the balance at the start of
    # amortization (which equals the original principal, since no principal
    # is paid down during the interest-only period).
    level_payment = balance_payment(principal, monthly_rate, remaining_amort_months)

    balance = principal
    rows = []
    month = 0
    for yr in range(1, projection_years + 1):
        beg_balance = balance
        interest_yr = 0.0
        principal_yr = 0.0
        for _ in range(12):
            month += 1
            interest_pmt = balance * monthly_rate
            if month <= io_months:
                principal_pmt = 0.0
            else:
                principal_pmt = min(level_payment - interest_pmt, balance)
            balance -= principal_pmt
            interest_yr += interest_pmt
            principal_yr += principal_pmt
        rows.append(
            {
                "year": yr,
                "beginning_balance": beg_balance,
                "interest": interest_yr,
                "principal_paydown": principal_yr,
                "debt_service": interest_yr + principal_yr,
                "ending_balance": balance,
            }
        )
    return pd.DataFrame(rows).set_index("year")


def balance_payment(balance: float, monthly_rate: float, n_months: int) -> float:
    """Level monthly payment that fully amortizes ``balance`` over
    ``n_months`` at ``monthly_rate``, i.e. the standard mortgage payment formula.
    """
    if monthly_rate == 0:
        return balance / n_months
    return balance * (monthly_rate * (1 + monthly_rate) ** n_months) / ((1 + monthly_rate) ** n_months - 1)


@dataclass
class CorporateOverhead:
    """Company-level (non-property) items that sit above the property roll-up.

    Attributes:
        annual_ga: Corporate general & administrative expense, $/year
            (executive team, deal/asset management staff, corporate
            insurance/legal in excess of property-level policies).
        ga_growth_rate: Annual growth rate applied to G&A.
        corporate_debt_balance: Non-property-level debt (e.g. a corporate
            revolver), $.
        corporate_debt_rate: Interest rate on corporate debt.
        cash_and_equivalents: Unrestricted cash on the balance sheet, $, used
            to compute net debt.
    """

    annual_ga: float
    ga_growth_rate: float = 0.025
    corporate_debt_balance: float = 0.0
    corporate_debt_rate: float = 0.0
    cash_and_equivalents: float = 0.0


@dataclass
class PortfolioCompany:
    """The full operating company: a set of properties plus corporate overhead."""

    name: str
    properties: list[Property]
    overhead: CorporateOverhead

    def total_units(self) -> int:
        return sum(p.units for p in self.properties)

    def total_property_debt(self) -> float:
        return sum(p.debt_balance for p in self.properties)

    def net_debt(self) -> float:
        """Consolidated net debt: property-level mortgages + corporate debt,
        less unrestricted cash.
        """
        return self.total_property_debt() + self.overhead.corporate_debt_balance - self.overhead.cash_and_equivalents

    def portfolio_direct_cap_value(self) -> float:
        """Sum of each property's as-is direct-capitalization value."""
        return sum(p.direct_cap_value() for p in self.properties)

    def weighted_avg_going_in_cap_rate(self) -> float:
        """Portfolio going-in cap rate, weighted by each property's as-is value."""
        total_value = self.portfolio_direct_cap_value()
        return sum(p.going_in_cap_rate * p.direct_cap_value() for p in self.properties) / total_value

    def project(self, assumptions: GrowthAssumptions, years: int = 5) -> pd.DataFrame:
        """Roll up every property's projection into a consolidated,
        company-level financial model.

        FFO and AFFO are computed here using the standard real estate
        approximation (rather than a full GAAP net-income-plus-depreciation
        build, which would require a full balance sheet/depreciation
        schedule this toolkit does not model): FFO = company NOI after
        corporate G&A, less cash interest expense; AFFO = FFO less
        recurring CapEx reserves.

        Returns:
            A DataFrame indexed by year (1..years) with the same line items
            as :func:`project_property_cash_flows`, summed across the
            portfolio, plus ``corporate_ga``, ``company_noi_after_ga``
            (the toolkit's EBITDA/EBITDAre proxy), ``corporate_debt_interest``,
            ``total_interest_expense``, ``company_ffo``, and ``company_affo``.
        """
        per_property = [project_property_cash_flows(p, assumptions, years) for p in self.properties]
        consolidated = sum(per_property)  # elementwise DataFrame sum, aligned on the year index

        ga = np.array([self.overhead.annual_ga * (1 + self.overhead.ga_growth_rate) ** yr for yr in range(1, years + 1)])
        consolidated["corporate_ga"] = ga
        consolidated["company_noi_after_ga"] = consolidated["noi"] - consolidated["corporate_ga"]

        # Corporate debt (e.g. a revolver) is treated as interest-only.
        corporate_debt_interest = self.overhead.corporate_debt_balance * self.overhead.corporate_debt_rate
        consolidated["corporate_debt_interest"] = corporate_debt_interest
        consolidated["total_interest_expense"] = consolidated["interest_expense"] + corporate_debt_interest
        consolidated["company_ffo"] = consolidated["company_noi_after_ga"] - consolidated["total_interest_expense"]
        consolidated["company_affo"] = consolidated["company_ffo"] - consolidated["capex_reserve"]
        return consolidated

    def property_summary(self) -> pd.DataFrame:
        """One-row-per-property summary table (Year 1 NOI, as-is value, LTV)."""
        rows = []
        for p in self.properties:
            rows.append(
                {
                    "property_id": p.property_id,
                    "name": p.name,
                    "submarket": p.submarket,
                    "units": p.units,
                    "year_built": p.year_built,
                    "year1_noi": p.year1_noi(),
                    "going_in_cap_rate": p.going_in_cap_rate,
                    "direct_cap_value": p.direct_cap_value(),
                    "debt_balance": p.debt_balance,
                    "loan_to_value": p.loan_to_value(),
                }
            )
        return pd.DataFrame(rows).set_index("property_id")
