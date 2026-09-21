import pytest

from real_estate_ma_toolkit.buyer_screening import Buyer, capacity_score, score_buyers


def _buyer(name="Test Buyer", buyer_type="Strategic", lo=100.0, hi=300.0) -> Buyer:
    return Buyer(
        name=name,
        buyer_type=buyer_type,
        description="Test description",
        equity_capacity_millions=1000.0,
        target_check_size_min_millions=lo,
        target_check_size_max_millions=hi,
        geographic_fit_score=4,
        asset_class_fit_score=4,
        thesis_alignment_score=4,
        leverage_appetite="Medium",
        notes="",
    )


def test_capacity_score_full_marks_inside_range():
    buyer = _buyer(lo=100.0, hi=300.0)
    assert capacity_score(buyer, 200.0) == pytest.approx(5.0)
    assert capacity_score(buyer, 100.0) == pytest.approx(5.0)
    assert capacity_score(buyer, 300.0) == pytest.approx(5.0)


def test_capacity_score_decays_above_range():
    buyer = _buyer(lo=100.0, hi=300.0)
    score_at_max = capacity_score(buyer, 300.0)
    score_above = capacity_score(buyer, 450.0)  # 50% over the max
    assert score_above < score_at_max
    assert score_above == pytest.approx(0.0, abs=1e-9)


def test_capacity_score_decays_below_range():
    buyer = _buyer(lo=100.0, hi=300.0)
    score_at_min = capacity_score(buyer, 100.0)
    score_below = capacity_score(buyer, 50.0)  # 50% under the min
    assert score_below < score_at_min


def test_score_buyers_rejects_weights_not_summing_to_one():
    buyers = [_buyer()]
    with pytest.raises(ValueError):
        score_buyers(buyers, implied_deal_size_millions=200.0, weights={"geographic_fit": 0.5, "asset_class_fit": 0.5, "thesis_alignment": 0.5, "capacity": 0.5})


def test_score_buyers_ranks_better_fit_higher():
    great_fit = Buyer("Great Fit", "Strategic", "", 1000, 100, 300, 5, 5, 5, "Medium", "")
    poor_fit = Buyer("Poor Fit", "Financial", "", 1000, 100, 300, 1, 1, 1, "Low", "")
    scored = score_buyers([poor_fit, great_fit], implied_deal_size_millions=200.0)
    assert scored.iloc[0]["name"] == "Great Fit"
    assert scored.iloc[0]["weighted_score"] > scored.iloc[1]["weighted_score"]
