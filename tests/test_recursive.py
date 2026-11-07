"""
Unit tests for Recursive Adaptive Strategy.
"""

from src.data_loader import VietlottDataLoader
from src.strategies import RecursiveAdaptiveStrategy


def test_recursive_adaptive_strategy():
    df = VietlottDataLoader("power655").load_clean_data(force_refresh=False).iloc[:400]
    strat = RecursiveAdaptiveStrategy(total_numbers=55, pick_count=6, alpha=0.15, random_state=42)
    tickets = strat.generate_tickets(df, num_tickets=4)

    assert len(tickets) == 4
    for t in tickets:
        assert len(t) == 6
        assert len(set(t)) == 6
        assert all(1 <= x <= 55 for x in t)
        assert t == sorted(t)
