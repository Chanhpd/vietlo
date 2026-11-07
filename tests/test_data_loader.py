"""
Unit tests for VietlottDataLoader.
"""

import pandas as pd
import pytest
from src.data_loader import VietlottDataLoader


def test_data_loader_power655():
    loader = VietlottDataLoader(lottery_code="power655", raw_dir="data/raw", processed_dir="data/processed")
    df = loader.load_clean_data(force_refresh=False)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000
    assert "main_numbers" in df.columns
    assert "bonus_number" in df.columns
    assert "sum_numbers" in df.columns
    assert "odd_count" in df.columns
    assert "even_count" in df.columns
    assert "day_of_week" in df.columns

    # Verify each main_numbers list has exactly 6 sorted numbers
    sample_draw = df["main_numbers"].iloc[0]
    assert len(sample_draw) == 6
    assert sample_draw == sorted(sample_draw)
    assert all(1 <= n <= 55 for n in sample_draw)


def test_data_loader_mega645():
    loader = VietlottDataLoader(lottery_code="mega645", raw_dir="data/raw", processed_dir="data/processed")
    df = loader.load_clean_data(force_refresh=False)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000
    sample_draw = df["main_numbers"].iloc[0]
    assert len(sample_draw) == 6
    assert all(1 <= n <= 45 for n in sample_draw)
    assert all(df["bonus_number"].isna())


def test_invalid_lottery_code():
    with pytest.raises(ValueError):
        VietlottDataLoader(lottery_code="invalid_lotto")

