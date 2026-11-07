"""
Unit tests for VietlottBacktester.
"""

import pytest
from src.backtester import VietlottBacktester
from src.data_loader import VietlottDataLoader
from src.strategies import RandomBaselineStrategy


def test_evaluate_ticket_mega645():
    df = VietlottDataLoader("mega645").load_clean_data(force_refresh=False)
    bt = VietlottBacktester(df, lottery_code="mega645", total_numbers=45)

    actual = [5, 10, 15, 20, 25, 30]

    # Jackpot: 6 matches
    m, tier, prize = bt.evaluate_ticket([5, 10, 15, 20, 25, 30], actual, None)
    assert m == 6 and tier == "jackpot_1" and prize == 12000000000

    # First: 5 matches
    m, tier, prize = bt.evaluate_ticket([5, 10, 15, 20, 25, 31], actual, None)
    assert m == 5 and tier == "first" and prize == 10000000

    # Second: 4 matches
    m, tier, prize = bt.evaluate_ticket([5, 10, 15, 20, 26, 31], actual, None)
    assert m == 4 and tier == "second" and prize == 300000

    # Third: 3 matches
    m, tier, prize = bt.evaluate_ticket([5, 10, 15, 21, 26, 31], actual, None)
    assert m == 3 and tier == "third" and prize == 30000

    # None: 2 matches
    m, tier, prize = bt.evaluate_ticket([5, 10, 16, 21, 26, 31], actual, None)
    assert m == 2 and tier == "none" and prize == 0


def test_evaluate_ticket_power655():
    df = VietlottDataLoader("power655").load_clean_data(force_refresh=False)
    bt = VietlottBacktester(df, lottery_code="power655", total_numbers=55)

    actual = [1, 2, 3, 4, 5, 6]
    bonus = 7

    # Jackpot 1: 6 matches
    m, tier, prize = bt.evaluate_ticket([1, 2, 3, 4, 5, 6], actual, bonus)
    assert m == 6 and tier == "jackpot_1" and prize == 30000000000

    # Jackpot 2: 5 matches + bonus
    m, tier, prize = bt.evaluate_ticket([1, 2, 3, 4, 5, 7], actual, bonus)
    assert m == 5 and tier == "jackpot_2" and prize == 3000000000

    # First: 5 matches without bonus
    m, tier, prize = bt.evaluate_ticket([1, 2, 3, 4, 5, 8], actual, bonus)
    assert m == 5 and tier == "first" and prize == 40000000


def test_walk_forward_backtest_run():
    df = VietlottDataLoader("power655").load_clean_data(force_refresh=False)
    bt = VietlottBacktester(df, lottery_code="power655", total_numbers=55)

    strategies = [RandomBaselineStrategy(total_numbers=55, pick_count=6, random_state=42)]
    res = bt.run_backtest(strategies, test_draws=20, tickets_per_draw=2)

    assert res["test_draws_count"] == 20
    assert res["total_tickets_per_strategy"] == 40
    summary = res["summary_df"]
    assert len(summary) == 1
    assert summary["total_tickets"].iloc[0] == 40
    assert summary["total_cost_vnd"].iloc[0] == 400000

