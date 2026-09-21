"""Comparable public company ("comps") analysis using real estate-native multiples.

Real estate comps are not analyzed on P/E -- REITs report Funds From
Operations (FFO) and Adjusted Funds From Operations (AFFO) specifically
because GAAP net income is distorted by real estate depreciation, which
does not reflect actual economic value decline for well-maintained
property. The standard real estate public-comps multiples are:

- **P/FFO** and **P/AFFO**: the REIT-world equivalent of P/E, using FFO/AFFO
  per share instead of EPS.
- **EV/EBITDA** (or EV/EBITDAre in REIT-specific terminology): enterprise
  value relative to property-level cash earnings before interest, taxes,
  depreciation, and amortization.
- **Implied cap rate**: backed out of trading multiples as EBITDA / EV,
  which lets you compare where a REIT trades in cap-rate terms against
  where its physical real estate would trade in a private direct-market
  transaction.

The comp figures shipped in ``data/comps.csv`` are illustrative,
order-of-magnitude approximations for a set of real, publicly traded
multifamily REITs -- they are a plausible starting point for the
methodology, not live market data, and should be refreshed from public
filings before any real use.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Comp:
    """A single comparable public company.

    Attributes:
        ticker: Exchange ticker.
        name: Company name.
        share_price: Current share price, $.
        shares_outstanding_millions: Diluted shares outstanding, millions.
        ffo_per_share: Funds From Operations per share (trailing or NTM), $.
        affo_per_share: Adjusted Funds From Operations per share, $.
        net_debt_millions: Net debt, $ millions.
        ebitda_millions: EBITDA (EBITDAre), $ millions.
    """

    ticker: str
    name: str
    share_price: float
    shares_outstanding_millions: float
    ffo_per_share: float
    affo_per_share: float
    net_debt_millions: float
    ebitda_millions: float

    def market_cap_millions(self) -> float:
        return self.share_price * self.shares_outstanding_millions

    def enterprise_value_millions(self) -> float:
        return self.market_cap_millions() + self.net_debt_millions

    def p_ffo(self) -> float:
        return self.share_price / self.ffo_per_share

    def p_affo(self) -> float:
        return self.share_price / self.affo_per_share

    def ev_ebitda(self) -> float:
        return self.enterprise_value_millions() / self.ebitda_millions

    def implied_cap_rate(self) -> float:
        """Cap rate implied by the current trading multiple: EBITDA / EV.

        This treats EBITDA as a proxy for portfolio NOI, which is the
        standard (if approximate) simplification used to translate a public
        trading multiple into a private-market cap-rate equivalent.
        """
        return self.ebitda_millions / self.enterprise_value_millions()


def load_comps(path: str) -> list[Comp]:
    """Load comps from a CSV (see ``data/comps.csv`` for the expected schema)."""
    df = pd.read_csv(path)
    return [Comp(**row) for row in df.to_dict(orient="records")]


def comps_table(comps: list[Comp]) -> pd.DataFrame:
    """Multiples table for the comp set, one row per comp plus a median/mean summary row."""
    rows = []
    for c in comps:
        rows.append(
            {
                "ticker": c.ticker,
                "name": c.name,
                "market_cap_millions": c.market_cap_millions(),
                "enterprise_value_millions": c.enterprise_value_millions(),
                "p_ffo": c.p_ffo(),
                "p_affo": c.p_affo(),
                "ev_ebitda": c.ev_ebitda(),
                "implied_cap_rate": c.implied_cap_rate(),
            }
        )
    df = pd.DataFrame(rows).set_index("ticker")
    summary = pd.DataFrame(
        {
            "market_cap_millions": [df["market_cap_millions"].median(), df["market_cap_millions"].mean()],
            "enterprise_value_millions": [df["enterprise_value_millions"].median(), df["enterprise_value_millions"].mean()],
            "p_ffo": [df["p_ffo"].median(), df["p_ffo"].mean()],
            "p_affo": [df["p_affo"].median(), df["p_affo"].mean()],
            "ev_ebitda": [df["ev_ebitda"].median(), df["ev_ebitda"].mean()],
            "implied_cap_rate": [df["implied_cap_rate"].median(), df["implied_cap_rate"].mean()],
            "name": ["Median", "Mean"],
        },
        index=["Median", "Mean"],
    ).drop(columns="name")
    return pd.concat([df, summary])


@dataclass
class CompsValuationResult:
    """Implied valuation range for the target from comp multiples."""

    implied_equity_value_p_ffo_low: float
    implied_equity_value_p_ffo_high: float
    implied_equity_value_p_affo_low: float
    implied_equity_value_p_affo_high: float
    implied_equity_value_ev_ebitda_low: float
    implied_equity_value_ev_ebitda_high: float
    implied_value_cap_rate_low: float
    implied_value_cap_rate_high: float

    def equity_value_range(self) -> tuple[float, float]:
        """Overall low/high equity value range across the P/FFO, P/AFFO, and
        EV/EBITDA methods (the cap-rate-implied figure is asset-level and
        reported separately, since it does not net out corporate net debt).
        """
        lows = [
            self.implied_equity_value_p_ffo_low,
            self.implied_equity_value_p_affo_low,
            self.implied_equity_value_ev_ebitda_low,
        ]
        highs = [
            self.implied_equity_value_p_ffo_high,
            self.implied_equity_value_p_affo_high,
            self.implied_equity_value_ev_ebitda_high,
        ]
        return min(lows), max(highs)


def value_target_from_comps(
    comps: list[Comp],
    target_ffo: float,
    target_affo: float,
    target_ebitda: float,
    target_forward_noi: float,
    target_net_debt: float,
    multiple_range: str = "quartile",
) -> CompsValuationResult:
    """Apply the comp set's multiples to the target's own metrics.

    Args:
        comps: The comparable company set.
        target_ffo: Target company FFO for the relevant year, $.
        target_affo: Target company AFFO for the relevant year, $.
        target_ebitda: Target company EBITDA proxy (NOI after corporate
            G&A) for the relevant year, $.
        target_forward_noi: Target's forward (next-twelve-months) NOI, $,
            used with the comps' implied cap rate to derive an asset-level
            direct-cap value.
        target_net_debt: Target's consolidated net debt, $, subtracted from
            EV-based methods to arrive at equity value.
        multiple_range: ``"quartile"`` uses the 25th/75th percentile of the
            comp set for a tighter, more defensible range; ``"min_max"``
            uses the full observed range.

    Returns:
        A :class:`CompsValuationResult`.
    """
    p_ffo = [c.p_ffo() for c in comps]
    p_affo = [c.p_affo() for c in comps]
    ev_ebitda = [c.ev_ebitda() for c in comps]
    implied_cap_rates = [c.implied_cap_rate() for c in comps]

    def lo_hi(values: list[float]) -> tuple[float, float]:
        s = pd.Series(values)
        if multiple_range == "min_max":
            return float(s.min()), float(s.max())
        return float(s.quantile(0.25)), float(s.quantile(0.75))

    p_ffo_lo, p_ffo_hi = lo_hi(p_ffo)
    p_affo_lo, p_affo_hi = lo_hi(p_affo)
    ev_ebitda_lo, ev_ebitda_hi = lo_hi(ev_ebitda)
    cap_rate_lo, cap_rate_hi = lo_hi(implied_cap_rates)

    ev_low = target_ebitda * ev_ebitda_lo
    ev_high = target_ebitda * ev_ebitda_hi

    return CompsValuationResult(
        implied_equity_value_p_ffo_low=target_ffo * p_ffo_lo,
        implied_equity_value_p_ffo_high=target_ffo * p_ffo_hi,
        implied_equity_value_p_affo_low=target_affo * p_affo_lo,
        implied_equity_value_p_affo_high=target_affo * p_affo_hi,
        implied_equity_value_ev_ebitda_low=ev_low - target_net_debt,
        implied_equity_value_ev_ebitda_high=ev_high - target_net_debt,
        # Cap rate and multiple move inversely: the LOW cap rate (richest
        # pricing) produces the HIGH implied value.
        implied_value_cap_rate_low=target_forward_noi / cap_rate_hi,
        implied_value_cap_rate_high=target_forward_noi / cap_rate_lo,
    )
