"""
Unit tests for VietlottRandomnessTester.
"""

import pytest
from src.data_loader import VietlottDataLoader
from src.randomness_test import VietlottRandomnessTester


@pytest.fixture(scope="module")
def mega_df():
    return VietlottDataLoader("mega645").load_clean_data(force_refresh=False)


def test_chi_square_uniformity(mega_df):
    tester = VietlottRandomnessTester(mega_df, total_numbers=45, pick_count=6)
    res = tester.chi_square_uniformity_test(alpha=0.05)

    assert res["degrees_of_freedom"] == 44
    assert res["p_value"] > 0.05  # Should fail to reject uniformity
    assert res["is_uniform"] is True


def test_runs_test_independence(mega_df):
    tester = VietlottRandomnessTester(mega_df, total_numbers=45, pick_count=6)
    res = tester.runs_test_for_independence(alpha=0.05)

    assert res["total_tested_numbers"] == 45
    # With Bonferroni correction, no number should violate independence
    assert res["bonferroni_rejections_count"] == 0


def test_repeat_hypergeometric_test(mega_df):
    tester = VietlottRandomnessTester(mega_df, total_numbers=45, pick_count=6)
    res = tester.chi_square_repeat_test(alpha=0.05)

    assert res["degrees_of_freedom"] == 3
    assert res["is_consistent_with_hypergeom"] is True

