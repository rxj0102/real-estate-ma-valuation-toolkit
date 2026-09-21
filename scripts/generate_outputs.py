#!/usr/bin/env python3
"""Runs the full valuation pipeline end-to-end and writes the client-ready
deliverables to ``output/``: the football field chart and the deal
memorandum. This is the same pipeline the walkthrough notebook narrates
step by step; run this script when you just want the final artifacts.

Usage:
    python scripts/generate_outputs.py
"""

from __future__ import annotations

import datetime as dt

from real_estate_ma_toolkit import buyer_screening, comps_analysis, precedent_transactions, scenario
from real_estate_ma_toolkit.dcf_valuation import (
    run_dcf,
    sensitivity_table_exit_cap_rate,
    sensitivity_table_gordon_growth,
    unlevered_free_cash_flow,
)
from real_estate_ma_toolkit.valuation_summary import (
    MethodologyRange,
    build_summary_table,
    overall_recommended_range,
    plot_football_field,
)


def main() -> None:
    scenario.OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Asset-level model -> consolidated company projection
    # ------------------------------------------------------------------
    company = scenario.load_portfolio_company()
    # Project one extra year so the exit-cap-rate terminal value can be
    # based on Year 6 (forward) NOI, per standard convention.
    projection = company.project(scenario.GROWTH_ASSUMPTIONS, years=scenario.PROJECTION_YEARS + 1)
    explicit_period = projection.loc[1 : scenario.PROJECTION_YEARS]

    net_debt = company.net_debt()
    weighted_cap_rate = company.weighted_avg_going_in_cap_rate()
    exit_cap_rate = scenario.default_exit_cap_rate(company)

    # ------------------------------------------------------------------
    # 2. DCF valuation
    # ------------------------------------------------------------------
    unlevered_fcf = unlevered_free_cash_flow(explicit_period)
    terminal_year_forward_noi = float(projection.loc[scenario.PROJECTION_YEARS + 1, "noi"])
    wacc = scenario.WACC_INPUTS.wacc()

    dcf_result = run_dcf(
        unlevered_fcf=unlevered_fcf,
        terminal_year_forward_noi=terminal_year_forward_noi,
        wacc=wacc,
        terminal_growth=scenario.TERMINAL_GROWTH_RATE,
        exit_cap_rate=exit_cap_rate,
        net_debt=net_debt,
    )

    gordon_sensitivity = sensitivity_table_gordon_growth(
        unlevered_fcf,
        net_debt,
        wacc_range=[wacc - 0.01, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.01],
        terminal_growth_range=[
            scenario.TERMINAL_GROWTH_RATE - 0.005,
            scenario.TERMINAL_GROWTH_RATE - 0.0025,
            scenario.TERMINAL_GROWTH_RATE,
            scenario.TERMINAL_GROWTH_RATE + 0.0025,
            scenario.TERMINAL_GROWTH_RATE + 0.005,
        ],
    )
    exit_sensitivity = sensitivity_table_exit_cap_rate(
        unlevered_fcf,
        terminal_year_forward_noi,
        net_debt,
        wacc_range=[wacc - 0.01, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.01],
        exit_cap_rate_range=[
            exit_cap_rate - 0.005,
            exit_cap_rate - 0.0025,
            exit_cap_rate,
            exit_cap_rate + 0.0025,
            exit_cap_rate + 0.005,
        ],
    )

    # DCF ranges for the football field: a tighter +/-50bp WACC band around
    # the base case (the same band used in the sensitivity tables), not the
    # full sensitivity table extremes.
    dcf_gordon_low = float(gordon_sensitivity.iloc[-1].min())
    dcf_gordon_high = float(gordon_sensitivity.iloc[0].max())
    dcf_exit_low = float(exit_sensitivity.iloc[-1].min())
    dcf_exit_high = float(exit_sensitivity.iloc[0].max())

    # ------------------------------------------------------------------
    # 3. Comparable companies analysis
    # ------------------------------------------------------------------
    comps = comps_analysis.load_comps(str(scenario.DATA_DIR / "comps.csv"))
    year1 = projection.loc[1]
    comps_result = comps_analysis.value_target_from_comps(
        comps,
        target_ffo=float(year1["company_ffo"]),
        target_affo=float(year1["company_affo"]),
        target_ebitda=float(year1["company_noi_after_ga"]),
        target_forward_noi=float(year1["noi"]),
        target_net_debt=net_debt,
    )
    comps_low, comps_high = comps_result.equity_value_range()

    # ------------------------------------------------------------------
    # 4. Precedent transactions analysis
    # ------------------------------------------------------------------
    precedents = precedent_transactions.load_precedent_transactions(str(scenario.DATA_DIR / "precedent_transactions.csv"))
    precedents_result = precedent_transactions.value_target_from_precedents(
        precedents,
        target_units=company.total_units(),
        target_forward_noi=float(year1["noi"]),
        target_ebitda=float(year1["company_noi_after_ga"]),
        target_net_debt=net_debt,
    )
    precedents_low, precedents_high = precedents_result.equity_value_range()

    # ------------------------------------------------------------------
    # 5. Football field
    # ------------------------------------------------------------------
    ranges = [
        MethodologyRange("Precedent Transactions", precedents_low, precedents_high),
        MethodologyRange("Comparable Companies", comps_low, comps_high),
        MethodologyRange("DCF - Exit Cap Rate", dcf_exit_low, dcf_exit_high),
        MethodologyRange("DCF - Gordon Growth", dcf_gordon_low, dcf_gordon_high),
    ]
    summary_table = build_summary_table(ranges)
    overall_low, overall_high, overall_mid = overall_recommended_range(ranges)

    chart_path = scenario.OUTPUT_DIR / "football_field.png"
    plot_football_field(ranges, scenario.COMPANY_NAME, str(chart_path))

    # ------------------------------------------------------------------
    # 6. Buyer / investor screening
    # ------------------------------------------------------------------
    buyers = buyer_screening.load_buyers(str(scenario.DATA_DIR / "buyers.csv"))
    scored_buyers = buyer_screening.score_buyers(buyers, implied_deal_size_millions=overall_mid / 1e6)
    top_5 = buyer_screening.top_targets(scored_buyers, n=5)

    # ------------------------------------------------------------------
    # 7. Deal memorandum
    # ------------------------------------------------------------------
    memo = render_deal_memo(
        company=company,
        projection=projection,
        net_debt=net_debt,
        weighted_cap_rate=weighted_cap_rate,
        exit_cap_rate=exit_cap_rate,
        wacc=wacc,
        summary_table=summary_table,
        overall_low=overall_low,
        overall_high=overall_high,
        overall_mid=overall_mid,
        top_5=top_5,
        chart_relative_path="football_field.png",
    )
    memo_path = scenario.OUTPUT_DIR / "deal_memorandum.md"
    memo_path.write_text(memo)

    print(f"Wrote {chart_path}")
    print(f"Wrote {memo_path}")
    print()
    print(f"Base-case DCF equity value -- Gordon Growth: ${dcf_result.equity_value_gordon/1e6:,.0f}M | "
          f"Exit Cap Rate: ${dcf_result.equity_value_exit/1e6:,.0f}M")
    print()
    print(summary_table)
    print()
    print(f"Recommended range: ${overall_low/1e6:,.0f}M - ${overall_high/1e6:,.0f}M (midpoint ${overall_mid/1e6:,.0f}M)")


def render_deal_memo(
    *,
    company,
    projection,
    net_debt: float,
    weighted_cap_rate: float,
    exit_cap_rate: float,
    wacc: float,
    summary_table,
    overall_low: float,
    overall_high: float,
    overall_mid: float,
    top_5,
    chart_relative_path: str,
) -> str:
    today = dt.date.today().isoformat()
    year1 = projection.loc[1]
    year5 = projection.loc[scenario.PROJECTION_YEARS]

    summary_rows = "\n".join(
        f"| {idx} | ${row['Low ($M)']:,.0f}M | ${row['High ($M)']:,.0f}M | ${row['Midpoint ($M)']:,.0f}M |"
        for idx, row in summary_table.iterrows()
    )

    buyer_rows = "\n".join(
        f"| {rank} | {row['name']} | {row['buyer_type']} | {row['weighted_score']:.1f} | {row['description']} |"
        for rank, row in top_5.iterrows()
    )

    return f"""# Project Magnolia -- Sell-Side Advisory Deal Memorandum

**Prepared for:** Board of Directors, {company.name}
**Prepared by:** CenterCap Group (illustrative engagement)
**Date:** {today}

> **Illustrative data notice.** This memorandum is a portfolio/demonstration
> artifact built on synthetic company, portfolio, comparable-company, and
> transaction data. It is not investment advice and should not be relied
> upon for any actual transaction decision. See the project README for full
> methodology and data notes.

---

## 1. Situation Overview

CenterCap Group has been retained as exclusive sell-side financial advisor
to {company.name} ("the Company") to explore a full sale or majority
recapitalization. The Company owns and operates {len(company.properties)}
garden-style and mid-rise multifamily communities totaling
{company.total_units():,} units, concentrated in the Atlanta, GA metro
area. This memorandum summarizes the Company's financial profile, the
valuation work performed to date, and our recommended buyer/investor
outreach strategy.

## 2. Company Overview

- **Portfolio:** {len(company.properties)} properties, {company.total_units():,} units, built {min(p.year_built for p in company.properties)}-{max(p.year_built for p in company.properties)}
- **Weighted average going-in cap rate:** {weighted_cap_rate:.2%}
- **Year 1 portfolio NOI:** ${year1['noi']:,.0f}
- **Year {scenario.PROJECTION_YEARS} portfolio NOI (projected):** ${year5['noi']:,.0f}
- **Consolidated net debt:** ${net_debt:,.0f}

Rent growth is assumed at {scenario.GROWTH_ASSUMPTIONS.rent_growth_rate:.1%} annually with occupancy
stabilizing toward {scenario.GROWTH_ASSUMPTIONS.stabilized_occupancy:.0%} across the portfolio. Operating
expense growth is differentiated by line item, notably a
{scenario.GROWTH_ASSUMPTIONS.insurance_growth_rate:.1%} annual growth assumption for property insurance,
reflecting the well-documented run-up in Sun Belt multifamily insurance
costs in recent years.

## 3. Valuation Summary

Four methodologies were used to triangulate an implied equity valuation
range, summarized in the football field below.

![Football Field]({chart_relative_path})

| Methodology | Low | High | Midpoint |
|---|---|---|---|
{summary_rows}

**Recommended valuation range: ${overall_low/1e6:,.0f}M - ${overall_high/1e6:,.0f}M
in implied equity value** (the band where all four methodologies overlap;
individual methodology ranges are wider, as shown above, with the DCF -
Gordon Growth method the most sensitive to its underlying assumptions).
Across all four methodologies, the average midpoint is approximately
${overall_mid/1e6:,.0f}M.

- **DCF (unlevered FCF discounted at a WACC of {wacc:.2%}):** run under both a
  Gordon Growth terminal value ({scenario.TERMINAL_GROWTH_RATE:.2%} long-term growth) and an
  exit-cap-rate terminal value ({exit_cap_rate:.2%} assumed exit cap rate), each
  sensitized across a range of discount rates and terminal assumptions.
- **Comparable public companies:** P/FFO, P/AFFO, and EV/EBITDA multiples
  from a set of publicly traded multifamily REITs, applied to the
  Company's projected Year 1 FFO, AFFO, and EBITDA.
- **Precedent transactions:** price-per-unit, cap-rate-paid, and EV/EBITDA
  multiples from a set of realistic recent multifamily M&A transactions,
  applied to the Company's unit count, forward NOI, and EBITDA.

## 4. Buyer / Investor Recommendation

Based on a structured screen of {len(top_5) and 'the'} strategic and
financial buyer universe -- scored on geographic and asset-class fit,
alignment with each buyer's stated investment thesis, and balance sheet
capacity relative to the implied deal size -- the following are our
top-ranked outreach targets:

| Rank | Buyer | Type | Score | Rationale |
|---|---|---|---|---|
{buyer_rows}

We recommend a targeted first-round outreach to the top five buyers above,
running strategic and financial buyers on a parallel process to maximize
competitive tension.

## 5. Next Steps

1. Finalize the confidential information memorandum (CIM) and data room.
2. Launch first-round outreach to the buyer list above.
3. Collect indications of interest and narrow to a second-round group.
4. Facilitate management presentations and property tours for finalists.
5. Solicit best-and-final offers and negotiate definitive documentation.
"""


if __name__ == "__main__":
    main()
