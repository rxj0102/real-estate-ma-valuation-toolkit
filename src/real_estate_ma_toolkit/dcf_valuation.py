"""Discounted cash flow valuation of the operating company.

Real estate DCFs differ from a generic corporate DCF in two important ways
that this module makes explicit rather than hiding inside generic
"EBITDA" and "exit multiple" labels:

1. **Unlevered free cash flow starts from NOI, not EBIT.** There is no
   separate D&A add-back/subtract dance -- NOI already excludes debt
   service and corporate items, and real estate ownership vehicles
   (REITs and most operating partnerships) are typically structured to
   avoid entity-level income tax, so no tax shield/unlevered tax
   adjustment is layered onto property cash flow. The ``tax_rate`` in
   :class:`WACCInputs` exists only for the cost-of-debt tax shield in the
   WACC calculation, and defaults to 0 to reflect that structure -- it is
   kept as a parameter so the module can be reused for a taxable
   corporate structure if needed.
2. **Terminal value has a real-estate-native second method.** Alongside
   the standard Gordon Growth perpetuity, this module computes terminal
   value the way a real estate buyer actually underwrites an exit: the
   projected forward NOI in the year after the holding period, divided by
   an exit capitalization rate. This is the real estate analogue of an
   "exit multiple" method and is materially more defensible in a real
   estate deal context than applying a generic EV/EBITDA multiple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WACCInputs:
    """Inputs to a standard CAPM-based weighted average cost of capital.

    Attributes:
        risk_free_rate: Risk-free rate (e.g. 10-year Treasury yield), as a fraction.
        equity_risk_premium: Equity market risk premium, as a fraction.
        beta: Levered equity beta for the company (multifamily REIT betas
            are typically below 1.0, reflecting relatively stable cash flows).
        cost_of_debt: Pre-tax cost of debt (blended portfolio rate), as a fraction.
        tax_rate: Entity-level tax rate used only for the debt tax shield.
            Defaults to 0.0, reflecting a REIT/pass-through structure with
            no entity-level income tax.
        target_debt_weight: Target Debt / (Debt + Equity) in the capital
            structure used to value the company, as a fraction.
    """

    risk_free_rate: float
    equity_risk_premium: float
    beta: float
    cost_of_debt: float
    tax_rate: float = 0.0
    target_debt_weight: float = 0.55

    def cost_of_equity(self) -> float:
        """CAPM cost of equity: risk-free rate + beta x equity risk premium."""
        return self.risk_free_rate + self.beta * self.equity_risk_premium

    def after_tax_cost_of_debt(self) -> float:
        return self.cost_of_debt * (1 - self.tax_rate)

    def wacc(self) -> float:
        """Weighted average cost of capital."""
        d = self.target_debt_weight
        e = 1 - d
        return e * self.cost_of_equity() + d * self.after_tax_cost_of_debt()


def unlevered_free_cash_flow(consolidated_projection: pd.DataFrame) -> pd.Series:
    """Unlevered free cash flow to the firm, by year.

    Built as company NOI (after corporate G&A) less recurring CapEx
    reserves already embedded in the property-level projection. No interest
    expense (unlevered) and no tax adjustment (pass-through structure, see
    module docstring).

    Args:
        consolidated_projection: Output of
            ``PortfolioCompany.project()``, which must include
            ``company_noi_after_ga`` and ``capex_reserve`` columns.

    Returns:
        A Series of unlevered FCF indexed by year.
    """
    return consolidated_projection["company_noi_after_ga"] - consolidated_projection["capex_reserve"]


def terminal_value_gordon_growth(final_year_fcf: float, wacc: float, terminal_growth: float) -> float:
    """Gordon Growth terminal value as of the end of the final projection year.

    TV = FCF_(n+1) / (WACC - g) = FCF_n x (1 + g) / (WACC - g)
    """
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed the terminal growth rate for Gordon Growth to be defined.")
    return final_year_fcf * (1 + terminal_growth) / (wacc - terminal_growth)


def terminal_value_exit_cap_rate(terminal_year_forward_noi: float, exit_cap_rate: float) -> float:
    """Real-estate-native terminal value: forward NOI capitalized at an exit cap rate.

    Args:
        terminal_year_forward_noi: Projected NOI for the year immediately
            following the holding period (i.e. Year n+1 NOI, the NOI a
            buyer at exit would underwrite against).
        exit_cap_rate: Assumed capitalization rate at exit, as a fraction.

    Returns:
        Terminal enterprise value as of the end of the final projection year.
    """
    return terminal_year_forward_noi / exit_cap_rate


def discount_factors(wacc: float, years: int) -> np.ndarray:
    """Mid-year-convention-free discount factors for years 1..years at WACC."""
    return np.array([1 / (1 + wacc) ** yr for yr in range(1, years + 1)])


@dataclass
class DCFResult:
    """Container for the two parallel DCF outcomes (Gordon Growth and exit cap rate)."""

    wacc: float
    pv_of_explicit_fcf: float
    terminal_value_gordon: float
    pv_terminal_value_gordon: float
    enterprise_value_gordon: float
    equity_value_gordon: float
    terminal_value_exit: float
    pv_terminal_value_exit: float
    enterprise_value_exit: float
    equity_value_exit: float

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Gordon Growth": {
                    "PV of explicit-period FCF": self.pv_of_explicit_fcf,
                    "Terminal value (undiscounted)": self.terminal_value_gordon,
                    "PV of terminal value": self.pv_terminal_value_gordon,
                    "Enterprise value": self.enterprise_value_gordon,
                    "Equity value": self.equity_value_gordon,
                },
                "Exit Cap Rate": {
                    "PV of explicit-period FCF": self.pv_of_explicit_fcf,
                    "Terminal value (undiscounted)": self.terminal_value_exit,
                    "PV of terminal value": self.pv_terminal_value_exit,
                    "Enterprise value": self.enterprise_value_exit,
                    "Equity value": self.equity_value_exit,
                },
            }
        )


def run_dcf(
    unlevered_fcf: pd.Series,
    terminal_year_forward_noi: float,
    wacc: float,
    terminal_growth: float,
    exit_cap_rate: float,
    net_debt: float,
) -> DCFResult:
    """Run the full DCF: discount explicit-period FCF, add both terminal
    value methods, and back into enterprise and equity value.

    Args:
        unlevered_fcf: Unlevered FCF by year (years 1..n), e.g. from
            :func:`unlevered_free_cash_flow`.
        terminal_year_forward_noi: Year n+1 NOI, used for the exit-cap-rate
            terminal value.
        wacc: Discount rate.
        terminal_growth: Long-run FCF growth rate for the Gordon Growth
            terminal value. Must be less than ``wacc``.
        exit_cap_rate: Assumed cap rate at exit for the exit-cap-rate
            terminal value.
        net_debt: Consolidated net debt, subtracted from enterprise value
            to arrive at equity value.

    Returns:
        A :class:`DCFResult`.
    """
    years = len(unlevered_fcf)
    dfs = discount_factors(wacc, years)
    pv_fcf = float(np.sum(unlevered_fcf.values * dfs))
    final_year_fcf = float(unlevered_fcf.iloc[-1])
    terminal_discount_factor = dfs[-1]

    tv_gordon = terminal_value_gordon_growth(final_year_fcf, wacc, terminal_growth)
    pv_tv_gordon = tv_gordon * terminal_discount_factor
    ev_gordon = pv_fcf + pv_tv_gordon
    equity_gordon = ev_gordon - net_debt

    tv_exit = terminal_value_exit_cap_rate(terminal_year_forward_noi, exit_cap_rate)
    pv_tv_exit = tv_exit * terminal_discount_factor
    ev_exit = pv_fcf + pv_tv_exit
    equity_exit = ev_exit - net_debt

    return DCFResult(
        wacc=wacc,
        pv_of_explicit_fcf=pv_fcf,
        terminal_value_gordon=tv_gordon,
        pv_terminal_value_gordon=pv_tv_gordon,
        enterprise_value_gordon=ev_gordon,
        equity_value_gordon=equity_gordon,
        terminal_value_exit=tv_exit,
        pv_terminal_value_exit=pv_tv_exit,
        enterprise_value_exit=ev_exit,
        equity_value_exit=equity_exit,
    )


def sensitivity_table_gordon_growth(
    unlevered_fcf: pd.Series,
    net_debt: float,
    wacc_range: list[float],
    terminal_growth_range: list[float],
) -> pd.DataFrame:
    """Equity value sensitivity: WACC (rows) x terminal growth rate (columns), Gordon Growth method."""
    years = len(unlevered_fcf)
    final_year_fcf = float(unlevered_fcf.iloc[-1])

    data = {}
    for g in terminal_growth_range:
        col = []
        for w in wacc_range:
            dfs = discount_factors(w, years)
            pv_fcf = float(np.sum(unlevered_fcf.values * dfs))
            tv = terminal_value_gordon_growth(final_year_fcf, w, g)
            equity_value = pv_fcf + tv * dfs[-1] - net_debt
            col.append(equity_value)
        data[f"g = {g:.2%}"] = col
    return pd.DataFrame(data, index=[f"WACC = {w:.2%}" for w in wacc_range])


def sensitivity_table_exit_cap_rate(
    unlevered_fcf: pd.Series,
    terminal_year_forward_noi: float,
    net_debt: float,
    wacc_range: list[float],
    exit_cap_rate_range: list[float],
) -> pd.DataFrame:
    """Equity value sensitivity: WACC (rows) x exit cap rate (columns), exit cap rate method."""
    years = len(unlevered_fcf)

    data = {}
    for exit_cap in exit_cap_rate_range:
        col = []
        for w in wacc_range:
            dfs = discount_factors(w, years)
            pv_fcf = float(np.sum(unlevered_fcf.values * dfs))
            tv = terminal_value_exit_cap_rate(terminal_year_forward_noi, exit_cap)
            equity_value = pv_fcf + tv * dfs[-1] - net_debt
            col.append(equity_value)
        data[f"exit cap = {exit_cap:.2%}"] = col
    return pd.DataFrame(data, index=[f"WACC = {w:.2%}" for w in wacc_range])
