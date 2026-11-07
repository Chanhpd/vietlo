"""
Unit tests for VietlottAnalyzer.
"""

import pandas as pd
import pytest
from src.analyzer import VietlottAnalyzer
from src.data_loader import VietlottDataLoader


@pytest.fixture(scope="module")
def power_df():
    return VietlottDataLoader("power655").load_clean_data(force_refresh=False)


def test_frequency_stats(power_df):
    analyzer = VietlottAnalyzer(power_df, total_numbers=55, pick_count=6)
    freq_df = analyzer.get_frequency_stats()

    assert len(freq_df) == 55
    assert set(freq_df["number"]) == set(range(1, 56))
    assert freq_df["count"].sum() == len(power_df) * 6


def test_gap_stats(power_df):
    analyzer = VietlottAnalyzer(power_df, total_numbers=55, pick_count=6)
    gap_df = analyzer.get_gap_stats()

    assert len(gap_df) == 55
    assert all(gap_df["current_gap"] >= 0)
    assert all(gap_df["max_gap"] >= gap_df["min_gap"])
    assert gap_df["theoretical_mean_gap"].iloc[0] == pytest.approx(55 / 6.0, rel=1e-2)


def test_consecutive_repeats(power_df):
    analyzer = VietlottAnalyzer(power_df, total_numbers=55, pick_count=6)
    rep = analyzer.get_consecutive_draw_repeats()

    assert "repeat_table" in rep
    assert rep["total_transitions"] == len(power_df) - 1
    assert 0 <= rep["mean_observed_repeats"] <= 6
    # Theoretical mean is 6 * 6 / 55 = 0.6545
    assert rep["theoretical_mean_repeats"] == pytest.approx(0.655, abs=0.01)


def test_top_pairs_and_triplets(power_df):
    analyzer = VietlottAnalyzer(power_df, total_numbers=55, pick_count=6)
    pairs = analyzer.get_top_pairs(top_k=5)
    trips = analyzer.get_top_triplets(top_k=5)

    assert len(pairs) == 5
    assert len(trips) == 5
    assert list(pairs["count"]) == sorted(list(pairs["count"]), reverse=True)
    assert list(trips["count"]) == sorted(list(trips["count"]), reverse=True)
