import pytest

from real_estate_ma_toolkit.comps_analysis import Comp, value_target_from_comps


def test_comp_multiples_match_manual_calculation():
    comp = Comp(
        ticker="TST",
        name="Test REIT",
        share_price=100.0,
        shares_outstanding_millions=50.0,
        ffo_per_share=5.0,
        affo_per_share=4.0,
        net_debt_millions=1000.0,
        ebitda_millions=400.0,
    )
    assert comp.market_cap_millions() == pytest.approx(5000.0)
    assert comp.enterprise_value_millions() == pytest.approx(6000.0)
    assert comp.p_ffo() == pytest.approx(20.0)
    assert comp.p_affo() == pytest.approx(25.0)
    assert comp.ev_ebitda() == pytest.approx(15.0)
    assert comp.implied_cap_rate() == pytest.approx(400.0 / 6000.0)


def test_value_target_from_comps_scales_with_target_metrics_and_returns_ordered_range():
    # Two comps with a known, narrow multiple spread so the math is checkable by hand.
    low_comp = Comp("A", "Comp A", share_price=100, shares_outstanding_millions=10,
                     ffo_per_share=10, affo_per_share=8, net_debt_millions=100, ebitda_millions=80)
    high_comp = Comp("B", "Comp B", share_price=150, shares_outstanding_millions=10,
                      ffo_per_share=10, affo_per_share=8, net_debt_millions=100, ebitda_millions=80)
    comps = [low_comp, high_comp]

    result = value_target_from_comps(
        comps,
        target_ffo=1_000_000.0,
        target_affo=800_000.0,
        target_ebitda=2_000_000.0,
        target_forward_noi=2_200_000.0,
        target_net_debt=5_000_000.0,
        multiple_range="min_max",
    )

    # p_ffo: A = 10.0, B = 15.0 -> target FFO of $1,000,000 implies $10M-$15M equity value
    assert result.implied_equity_value_p_ffo_low == pytest.approx(10_000_000.0)
    assert result.implied_equity_value_p_ffo_high == pytest.approx(15_000_000.0)

    lo, hi = result.equity_value_range()
    assert lo <= hi


def test_multiple_range_quartile_is_narrower_than_min_max():
    comps = [
        Comp(f"C{i}", f"Comp {i}", share_price=p, shares_outstanding_millions=10,
             ffo_per_share=10, affo_per_share=8, net_debt_millions=100, ebitda_millions=80)
        for i, p in enumerate([80, 100, 110, 120, 140, 200])
    ]
    result_quartile = value_target_from_comps(
        comps, target_ffo=1.0, target_affo=1.0, target_ebitda=1.0,
        target_forward_noi=1.0, target_net_debt=0.0, multiple_range="quartile",
    )
    result_minmax = value_target_from_comps(
        comps, target_ffo=1.0, target_affo=1.0, target_ebitda=1.0,
        target_forward_noi=1.0, target_net_debt=0.0, multiple_range="min_max",
    )
    quartile_spread = result_quartile.implied_equity_value_p_ffo_high - result_quartile.implied_equity_value_p_ffo_low
    minmax_spread = result_minmax.implied_equity_value_p_ffo_high - result_minmax.implied_equity_value_p_ffo_low
    assert quartile_spread < minmax_spread
