"""
Unit tests for Strategies module.
"""

import pytest
from src.data_loader import VietlottDataLoader
from src.strategies import (
    ColdOverdueStrategy,
    HotFrequencyStrategy,
    HybridStrategy,
    MLRankingStrategy,
    RandomBaselineStrategy,
)


@pytest.fixture(scope="module")
def history_df():
    return VietlottDataLoader("power655").load_clean_data(force_refresh=False).iloc[:400]


def test_random_baseline(history_df):
    strat = RandomBaselineStrategy(total_numbers=55, pick_count=6, random_state=42)
    tickets = strat.generate_tickets(history_df, num_tickets=5)

    assert len(tickets) == 5
    for t in tickets:
        assert len(t) == 6
        assert len(set(t)) == 6
        assert all(1 <= x <= 55 for x in t)
        assert t == sorted(t)


def test_hot_frequency(history_df):
    strat = HotFrequencyStrategy(total_numbers=55, pick_count=6, lookback_window=50, random_state=42)
    tickets = strat.generate_tickets(history_df, num_tickets=4)

    assert len(tickets) == 4
    for t in tickets:
        assert len(t) == 6
        assert len(set(t)) == 6


def test_cold_overdue(history_df):
    strat = ColdOverdueStrategy(total_numbers=55, pick_count=6, random_state=42)
    tickets = strat.generate_tickets(history_df, num_tickets=4)

    assert len(tickets) == 4
    for t in tickets:
        assert len(t) == 6
        assert len(set(t)) == 6


def test_hybrid_strategy(history_df):
    strat = HybridStrategy(total_numbers=55, pick_count=6, lookback_window=50, random_state=42)
    tickets = strat.generate_tickets(history_df, num_tickets=5)

    assert len(tickets) == 5
    for t in tickets:
        assert len(t) == 6
        assert len(set(t)) == 6


def test_ml_ranker(history_df):
    strat = MLRankingStrategy(total_numbers=55, pick_count=6, train_window=100, model_type="logreg", random_state=42)
    tickets = strat.generate_tickets(history_df, num_tickets=3)

    assert len(tickets) == 3
    for t in tickets:
        assert len(t) == 6
        assert len(set(t)) == 6

