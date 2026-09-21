import pytest

from real_estate_ma_toolkit.valuation_summary import MethodologyRange, overall_recommended_range


def test_overall_recommended_range_uses_intersection_when_overlapping():
    ranges = [
        MethodologyRange("A", 100.0, 300.0),
        MethodologyRange("B", 150.0, 250.0),
        MethodologyRange("C", 120.0, 280.0),
    ]
    low, high, mid = overall_recommended_range(ranges)
    # Intersection: highest low (150), lowest high (250)
    assert low == pytest.approx(150.0)
    assert high == pytest.approx(250.0)
    expected_mid = sum(r.midpoint() for r in ranges) / len(ranges)
    assert mid == pytest.approx(expected_mid)


def test_overall_recommended_range_falls_back_to_union_when_no_overlap():
    ranges = [
        MethodologyRange("A", 10.0, 20.0),
        MethodologyRange("B", 50.0, 60.0),
    ]
    low, high, _ = overall_recommended_range(ranges)
    assert low == pytest.approx(10.0)
    assert high == pytest.approx(60.0)
