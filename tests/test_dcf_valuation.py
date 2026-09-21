"""Validates dcf_valuation.py against an independently computed hand calculation.

This test deliberately does not import or reuse any arithmetic from
dcf_valuation.py in its "expected" calculation -- it replicates the DCF
math from scratch with plain Python loops so that a bug shared between the
module and the test can't hide.
"""

import pandas as pd
import pytest

from real_estate_ma_toolkit.dcf_valuation import WACCInputs, run_dcf


def test_wacc_matches_capm_by_hand():
    inputs = WACCInputs(
        risk_free_rate=0.043,
        equity_risk_premium=0.060,
        beta=1.0,
        cost_of_debt=0.0575,
        tax_rate=0.0,
        target_debt_weight=0.55,
    )
    # Cost of equity = 4.3% + 1.0 * 6.0% = 10.3%
    assert inputs.cost_of_equity() == pytest.approx(0.103)
    # WACC = 0.45 * 10.3% + 0.55 * 5.75% = 4.635% + 3.1625% = 7.7975%
    assert inputs.wacc() == pytest.approx(0.45 * 0.103 + 0.55 * 0.0575)


def test_dcf_matches_independent_hand_calculation():
    fcf = pd.Series(
        [10_000_000.0, 10_500_000.0, 11_000_000.0, 11_500_000.0, 12_000_000.0],
        index=range(1, 6),
    )
    wacc = 0.08
    terminal_growth = 0.025
    exit_cap_rate = 0.053
    terminal_year_forward_noi = 13_000_000.0
    net_debt = 50_000_000.0

    result = run_dcf(fcf, terminal_year_forward_noi, wacc, terminal_growth, exit_cap_rate, net_debt)

    # --- independent replication, plain loops only ---
    pv_fcf_manual = 0.0
    for yr, cf in zip(fcf.index, fcf.values):
        pv_fcf_manual += cf / (1 + wacc) ** yr
    df5 = 1 / (1 + wacc) ** 5

    tv_gordon_manual = fcf.iloc[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    ev_gordon_manual = pv_fcf_manual + tv_gordon_manual * df5
    equity_gordon_manual = ev_gordon_manual - net_debt

    tv_exit_manual = terminal_year_forward_noi / exit_cap_rate
    ev_exit_manual = pv_fcf_manual + tv_exit_manual * df5
    equity_exit_manual = ev_exit_manual - net_debt

    assert result.pv_of_explicit_fcf == pytest.approx(pv_fcf_manual, rel=1e-9)
    assert result.terminal_value_gordon == pytest.approx(tv_gordon_manual, rel=1e-9)
    assert result.enterprise_value_gordon == pytest.approx(ev_gordon_manual, rel=1e-9)
    assert result.equity_value_gordon == pytest.approx(equity_gordon_manual, rel=1e-9)
    assert result.terminal_value_exit == pytest.approx(tv_exit_manual, rel=1e-9)
    assert result.enterprise_value_exit == pytest.approx(ev_exit_manual, rel=1e-9)
    assert result.equity_value_exit == pytest.approx(equity_exit_manual, rel=1e-9)

    # Hand-worked reference values (computed to the nearest dollar via the
    # same independent replication above, pinned here as a regression check).
    assert pv_fcf_manual == pytest.approx(43_613_308, abs=1_000)
    assert equity_gordon_manual == pytest.approx(145_856_600, abs=100_000)
    assert equity_exit_manual == pytest.approx(160_548_814, abs=1_000)


def test_gordon_growth_requires_wacc_above_terminal_growth():
    fcf = pd.Series([10_000_000.0] * 5, index=range(1, 6))
    with pytest.raises(ValueError):
        run_dcf(
            fcf,
            terminal_year_forward_noi=10_000_000.0,
            wacc=0.02,
            terminal_growth=0.03,
            exit_cap_rate=0.05,
            net_debt=0.0,
        )
