"""Precedent M&A transactions analysis.

Precedent transactions capture what buyers have actually paid, in a
change-of-control context, which typically embeds a control premium not
present in public trading multiples (comps). For multifamily real estate,
the three multiples that matter are:

- **Price per unit**: the most commonly quoted headline multiple in
  multifamily M&A, since it is directly comparable across deals of
  different size and is easy to sanity-check against replacement cost.
- **Cap rate paid**: NOI at closing / purchase price, the private-market
  equivalent of the comps' implied cap rate.
- **EV/EBITDA**: less universally quoted in multifamily deals than in
  general M&A, but relevant when the target is being acquired as an
  operating platform/company (as here) rather than a pure real estate
  asset purchase.

The transactions in ``data/precedent_transactions.csv`` are illustrative
composites reflecting realistic 2021-2024 Sun Belt multifamily deal
patterns (including the cap-rate expansion that occurred as interest rates
rose over that period). They are deliberately anonymized/generic rather
than naming real transactions, since asserting specific dollar figures
against real, named deals risks being factually wrong; the multiples
themselves are calibrated to be realistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PrecedentTransaction:
    """A single precedent M&A transaction.

    Attributes:
        deal_id: Short identifier.
        year: Year the transaction closed.
        target_profile: Description of the target (asset type, geography, scale).
        buyer_profile: Description of the buyer (type and strategy).
        units: Unit count of the target.
        price_per_unit: Gross purchase price per unit, $.
        cap_rate_paid: In-place NOI at closing / purchase price, as a fraction.
        ev_ebitda_multiple: Enterprise value / EBITDA multiple paid, if disclosed.
    """

    deal_id: str
    year: int
    target_profile: str
    buyer_profile: str
    units: int
    price_per_unit: float
    cap_rate_paid: float
    ev_ebitda_multiple: float


def load_precedent_transactions(path: str) -> list[PrecedentTransaction]:
    """Load precedent transactions from CSV (see ``data/precedent_transactions.csv``)."""
    df = pd.read_csv(path)
    return [PrecedentTransaction(**row) for row in df.to_dict(orient="records")]


def transactions_table(transactions: list[PrecedentTransaction]) -> pd.DataFrame:
    """Multiples table for the precedent transaction set, with a median/mean summary."""
    df = pd.DataFrame([t.__dict__ for t in transactions]).set_index("deal_id")
    numeric_cols = ["price_per_unit", "cap_rate_paid", "ev_ebitda_multiple"]
    summary = pd.DataFrame(
        {c: [df[c].median(), df[c].mean()] for c in numeric_cols},
        index=["Median", "Mean"],
    )
    return pd.concat([df[numeric_cols], summary])


@dataclass
class PrecedentTransactionsValuationResult:
    """Implied valuation range for the target from precedent transaction multiples."""

    implied_equity_value_price_per_unit_low: float
    implied_equity_value_price_per_unit_high: float
    implied_equity_value_cap_rate_low: float
    implied_equity_value_cap_rate_high: float
    implied_equity_value_ev_ebitda_low: float
    implied_equity_value_ev_ebitda_high: float

    def equity_value_range(self) -> tuple[float, float]:
        lows = [
            self.implied_equity_value_price_per_unit_low,
            self.implied_equity_value_cap_rate_low,
            self.implied_equity_value_ev_ebitda_low,
        ]
        highs = [
            self.implied_equity_value_price_per_unit_high,
            self.implied_equity_value_cap_rate_high,
            self.implied_equity_value_ev_ebitda_high,
        ]
        return min(lows), max(highs)


def value_target_from_precedents(
    transactions: list[PrecedentTransaction],
    target_units: int,
    target_forward_noi: float,
    target_ebitda: float,
    target_net_debt: float,
    lookback_years: int | None = 3,
    multiple_range: str = "quartile",
) -> PrecedentTransactionsValuationResult:
    """Apply precedent transaction multiples to the target.

    Args:
        transactions: The precedent transaction set.
        target_units: Target unit count.
        target_forward_noi: Target's forward NOI, $, used with cap rate paid.
        target_ebitda: Target EBITDA proxy, $, used with EV/EBITDA multiple.
        target_net_debt: Target net debt, $, subtracted from EV-based
            methods (price/unit and EV/EBITDA) to arrive at equity value.
            Cap-rate-paid values a debt-free real estate basis directly.
        lookback_years: If set, only include transactions from the most
            recent N years (more relevant to current market pricing).
            Recent multifamily cap rates have moved materially since 2021,
            so this matters.
        multiple_range: ``"quartile"`` or ``"min_max"``, as in
            :func:`comps_analysis.value_target_from_comps`.

    Returns:
        A :class:`PrecedentTransactionsValuationResult`.
    """
    if lookback_years is not None:
        most_recent_year = max(t.year for t in transactions)
        cutoff = most_recent_year - lookback_years + 1
        transactions = [t for t in transactions if t.year >= cutoff]

    price_per_unit = [t.price_per_unit for t in transactions]
    cap_rates = [t.cap_rate_paid for t in transactions]
    ev_ebitda = [t.ev_ebitda_multiple for t in transactions]

    def lo_hi(values: list[float]) -> tuple[float, float]:
        s = pd.Series(values)
        if multiple_range == "min_max":
            return float(s.min()), float(s.max())
        return float(s.quantile(0.25)), float(s.quantile(0.75))

    ppu_lo, ppu_hi = lo_hi(price_per_unit)
    cap_lo, cap_hi = lo_hi(cap_rates)
    ev_ebitda_lo, ev_ebitda_hi = lo_hi(ev_ebitda)

    gross_value_ppu_low = target_units * ppu_lo
    gross_value_ppu_high = target_units * ppu_hi

    return PrecedentTransactionsValuationResult(
        implied_equity_value_price_per_unit_low=gross_value_ppu_low - target_net_debt,
        implied_equity_value_price_per_unit_high=gross_value_ppu_high - target_net_debt,
        # Higher cap rate paid = lower price; low cap rate = high price.
        implied_equity_value_cap_rate_low=target_forward_noi / cap_hi - target_net_debt,
        implied_equity_value_cap_rate_high=target_forward_noi / cap_lo - target_net_debt,
        implied_equity_value_ev_ebitda_low=target_ebitda * ev_ebitda_lo - target_net_debt,
        implied_equity_value_ev_ebitda_high=target_ebitda * ev_ebitda_hi - target_net_debt,
    )
