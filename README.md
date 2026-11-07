# Real Estate M&A Valuation Toolkit

[![CI](https://github.com/rxj0102/real-estate-ma-valuation-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/real-estate-ma-valuation-toolkit/actions/workflows/ci.yml)

A sell-side M&A advisory-style valuation toolkit for a multifamily real estate
operating company, built around the methodologies a real estate investment
banking analyst actually uses -- not a generic corporate DCF, and not
portfolio/fund-of-funds analysis.

**Scenario:** CenterCap Group (illustrative advisory context) is retained as
exclusive sell-side financial advisor to **Magnolia Multifamily Partners,
LLC**, a privately held operating company with a 12-property, ~3,060-unit
garden-style/mid-rise multifamily portfolio concentrated in the Atlanta, GA
metro area. The engagement explores a full sale or majority recapitalization.
See [`notebooks/01_valuation_walkthrough.ipynb`](notebooks/01_valuation_walkthrough.ipynb)
for the full analyst walkthrough and [`output/deal_memorandum.md`](output/deal_memorandum.md)
for the client-ready deliverable.

> **This is a portfolio/demonstration project.** All company, portfolio,
> comparable-company, and precedent-transaction data is illustrative and
> synthetic. Nothing here is investment advice, and none of it should be
> relied upon for any actual transaction, investment, or valuation decision.

---

## Why real estate valuation is different

A generic corporate DCF/comps model gets several conventions wrong if applied
naively to a real estate operating company. This toolkit is built around the
conventions that actually govern real estate valuation:

| Generic corporate finance | Real estate equivalent, used here |
|---|---|
| EBIT / EBITDA | **NOI (Net Operating Income)** -- revenue (gross potential rent less vacancy, plus ancillary income) less property-level operating expenses. NOI, not EBITDA, is the central real estate cash flow metric, and it excludes corporate G&A and debt service. |
| EPS | **FFO (Funds From Operations)** and **AFFO (Adjusted FFO)** -- FFO exists specifically because GAAP net income is distorted by real estate depreciation, which does not reflect a well-maintained property's actual economic value decline. This toolkit approximates FFO as NOI after corporate G&A less cash interest expense, and AFFO as FFO less recurring CapEx reserves (see `property_model.PortfolioCompany.project`). |
| P/E | **P/FFO** and **P/AFFO** -- the REIT-world equivalent of P/E, used in `comps_analysis.py`. |
| Generic EV/EBITDA exit multiple | **Cap rate** -- NOI / value (or, run in reverse as value = NOI / cap rate). Used both for as-is asset-level valuation (`property_model.Property.direct_cap_value`) and as the real-estate-native DCF terminal value method (`dcf_valuation.terminal_value_exit_cap_rate`), applied to forward NOI the way a buyer at exit would actually underwrite the deal. |
| A single blended opex line | Operating expenses are split into **controllable opex, property tax, and insurance**, each grown at its own rate -- notably, insurance is grown faster than other expense lines here, reflecting the real, well-documented run-up in Sun Belt multifamily property insurance costs. |
| Corporate tax shield on debt | Real estate ownership vehicles (REITs and most operating partnerships) are typically structured to avoid entity-level income tax by distributing income, so this toolkit's unlevered FCF has **no tax adjustment** by default (the `tax_rate` parameter exists only for the cost-of-debt tax shield inside the WACC calculation, and defaults to 0). |

## Methodology overview

1. **Asset-level financial model** (`property_model.py`) -- each of the 12
   properties is modeled individually: gross potential rent, vacancy,
   ancillary income, a three-line operating expense build (controllable
   opex / property tax / insurance), CapEx reserves, mortgage amortization
   (with an optional interest-only period), and DSCR. Properties roll up into
   a consolidated, company-level 5-year projection, plus corporate G&A and
   corporate-level debt.
2. **DCF valuation** (`dcf_valuation.py`) -- unlevered FCF (NOI after
   corporate G&A, less CapEx reserves) discounted at a CAPM-based WACC, with
   terminal value computed **two ways**: Gordon Growth and an exit-cap-rate
   method (forward NOI / assumed exit cap rate). Includes a WACC-vs-terminal-growth
   sensitivity table and a WACC-vs-exit-cap-rate sensitivity table. The core
   DCF math is validated against an independent, from-scratch hand
   calculation in `tests/test_dcf_valuation.py`.
3. **Comparable companies analysis** (`comps_analysis.py`) -- a set of real,
   publicly traded multifamily REITs, valued on P/FFO, P/AFFO, EV/EBITDA, and
   an implied cap rate backed out of the trading multiple, applied to the
   target's own projected FFO/AFFO/EBITDA/NOI.
4. **Precedent transactions analysis** (`precedent_transactions.py`) -- an
   illustrative set of realistic multifamily M&A transactions (price per
   unit, cap rate paid, EV/EBITDA), reflecting the cap-rate expansion that
   occurred industry-wide as interest rates rose from 2021 to 2024, applied
   to the target's unit count, forward NOI, and EBITDA proxy.
5. **Valuation summary / football field** (`valuation_summary.py`) --
   combines all four methodology ranges (DCF x2, comps, precedents) into the
   standard IB range-of-outcomes chart. The headline "recommended range" is
   the band where every methodology overlaps, not the full min-to-max span,
   since a single volatile method (typically Gordon Growth) can otherwise
   dominate the outer bookends.
6. **Buyer / investor screening** (`buyer_screening.py`) -- a structured,
   rubric-based framework covering both strategic (REITs, private operators)
   and financial (PE real estate funds, institutional capital) buyers,
   scored on geographic/submarket fit, asset-class fit, stated investment
   thesis alignment, and balance-sheet capacity relative to the deal's
   implied size, producing a ranked outreach list.

### A note on precedent transaction and comp data

The comp set (`data/comps.csv`) uses real, publicly traded multifamily
REIT tickers with illustrative/approximate multiples -- a plausible starting
point for the methodology, not live market data; refresh from public filings
before any real use. The precedent transaction set (`data/precedent_transactions.csv`)
is deliberately **anonymized/composite** (generic buyer/seller profiles
rather than named real deals) so the multiples can be realistic without
asserting specific dollar figures against real, named transactions. The
buyer universe (`data/buyers.csv`) is entirely fictional, since buyer
screening involves claims about acquisition intent and strategy that
shouldn't be attributed to real firms without their actual, current,
public statements.

## Project layout

```
src/real_estate_ma_toolkit/
    property_model.py         # Property, PortfolioCompany, NOI/debt-service projection
    dcf_valuation.py           # WACC, unlevered FCF, DCF, terminal value, sensitivity tables
    comps_analysis.py          # Public company comps (P/FFO, P/AFFO, EV/EBITDA)
    precedent_transactions.py  # M&A precedent transaction multiples
    valuation_summary.py       # Football field chart + recommended range
    buyer_screening.py         # Buyer/investor scoring and ranking
    scenario.py                 # The concrete Magnolia Multifamily scenario & assumptions
data/                          # Illustrative portfolio, comps, precedent transactions, buyers (CSV)
notebooks/01_valuation_walkthrough.ipynb   # End-to-end analyst walkthrough
scripts/generate_outputs.py     # Regenerates output/football_field.png and output/deal_memorandum.md
output/                         # Generated deliverables (football field chart, deal memorandum)
tests/                          # DCF hand-calculation validation + module unit tests
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt   # adds jupyter and pytest

# Run the test suite (includes the DCF hand-calculation validation)
pytest

# Regenerate the football field chart and deal memorandum
python scripts/generate_outputs.py

# Or walk through the full analysis interactively
jupyter notebook notebooks/01_valuation_walkthrough.ipynb
```

## Disclaimer

This project uses **illustrative, synthetic data throughout** -- the
company, its portfolio, the comparable companies' multiples, the precedent
transactions, and the buyer universe are all constructed to be realistic in
magnitude and convention for demonstration purposes. This is a methodology
showcase, not a real valuation. Nothing in this repository is investment
advice, and it should not be used as the basis for any actual investment,
acquisition, or valuation decision.

## License

MIT -- see [LICENSE](LICENSE).
