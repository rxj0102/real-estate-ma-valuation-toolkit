import pytest

from real_estate_ma_toolkit.precedent_transactions import PrecedentTransaction, value_target_from_precedents


def _txn(deal_id, year, ppu, cap_rate, ev_ebitda):
    return PrecedentTransaction(
        deal_id=deal_id,
        year=year,
        target_profile="Test portfolio",
        buyer_profile="Test buyer",
        units=1000,
        price_per_unit=ppu,
        cap_rate_paid=cap_rate,
        ev_ebitda_multiple=ev_ebitda,
    )


def test_lookback_years_filters_older_transactions():
    old = _txn("OLD", 2019, 300_000, 0.03, 30.0)
    recent = _txn("NEW", 2024, 150_000, 0.06, 15.0)
    transactions = [old, recent]

    result_full = value_target_from_precedents(
        transactions, target_units=1000, target_forward_noi=1.0, target_ebitda=1.0,
        target_net_debt=0.0, lookback_years=None, multiple_range="min_max",
    )
    result_recent_only = value_target_from_precedents(
        transactions, target_units=1000, target_forward_noi=1.0, target_ebitda=1.0,
        target_net_debt=0.0, lookback_years=2, multiple_range="min_max",
    )

    # With both transactions included, price/unit spans 150k-300k.
    assert result_full.implied_equity_value_price_per_unit_low == pytest.approx(1000 * 150_000)
    assert result_full.implied_equity_value_price_per_unit_high == pytest.approx(1000 * 300_000)

    # With only the recent transaction, low == high (a single data point).
    assert result_recent_only.implied_equity_value_price_per_unit_low == pytest.approx(1000 * 150_000)
    assert result_recent_only.implied_equity_value_price_per_unit_high == pytest.approx(1000 * 150_000)


def test_cap_rate_paid_is_inversely_related_to_implied_value():
    # Higher cap rate paid => lower implied value (cheaper pricing).
    cheap = _txn("A", 2024, 100_000, 0.07, 14.0)   # high cap rate paid
    rich = _txn("B", 2024, 100_000, 0.04, 25.0)    # low cap rate paid
    result = value_target_from_precedents(
        [cheap, rich], target_units=1000, target_forward_noi=10_000_000.0, target_ebitda=8_000_000.0,
        target_net_debt=0.0, lookback_years=None, multiple_range="min_max",
    )
    assert result.implied_equity_value_cap_rate_low == pytest.approx(10_000_000.0 / 0.07)
    assert result.implied_equity_value_cap_rate_high == pytest.approx(10_000_000.0 / 0.04)
    assert result.implied_equity_value_cap_rate_low < result.implied_equity_value_cap_rate_high
