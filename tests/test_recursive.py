"""
Unit tests for Recursive Adaptive Strategy and Anti-Crowd Strategy.
"""

from src.data_loader import VietlottDataLoader
from src.strategies import GameTheoryAntiCrowdStrategy, RecursiveAdaptiveStrategy


def test_recursive_adaptive_strategy():
    df = VietlottDataLoader("power655").load_clean_data(force_refresh=False).iloc[:400]
    strat = RecursiveAdaptiveStrategy(total_numbers=55, pick_count=6, alpha=0.04, random_state=42)
    tickets = strat.generate_tickets(df, num_tickets=4)

    assert len(tickets) == 4
    for t in tickets:
        assert len(t) == 6
        assert len(set(t)) == 6
        assert all(1 <= x <= 55 for x in t)
        assert t == sorted(t)

    top_nums, weights = strat.get_top_candidates(df, top_k=10)
    assert len(top_nums) == 10
    assert len(set(top_nums)) == 10
    assert all(weights[n] > 0 for n in top_nums)


def test_game_theory_anti_crowd_strategy():
    df = VietlottDataLoader("power655").load_clean_data(force_refresh=False).iloc[:400]
    strat = GameTheoryAntiCrowdStrategy(total_numbers=55, pick_count=6, random_state=42)
    tickets = strat.generate_tickets(df, num_tickets=5)

    assert len(tickets) == 5
    for t in tickets:
        assert len(t) == 6
        assert len(set(t)) == 6
        assert sum(1 for x in t if x > 31) >= 4
