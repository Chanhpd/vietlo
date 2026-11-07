"""
Unit tests for Expected Value (+EV) Calculator.
"""

from src.ev_calculator import VietlottEVCalculator


def test_ev_calculator_power655():
    calc = VietlottEVCalculator("power655")

    # At 30 billion (starter jackpot), EV must be strongly negative
    res_30 = calc.calculate_expected_value(30e9)
    assert not res_30["is_positive_ev"]
    assert res_30["total_expected_value_vnd"] < 5000.0

    # Break-even threshold should be > 200 billion
    assert res_30["break_even_jackpot_1_billion"] > 200.0

    # At 300 billion, EV must be positive
    res_300 = calc.calculate_expected_value(300e9)
    assert res_300["is_positive_ev"]
    assert res_300["total_expected_value_vnd"] > 10000.0


def test_ev_calculator_mega645():
    calc = VietlottEVCalculator("mega645")

    # At 12 billion, EV is negative
    res_12 = calc.calculate_expected_value(12e9)
    assert not res_12["is_positive_ev"]
    assert res_12["total_expected_value_vnd"] < 5000.0

    # At 120 billion, EV becomes positive
    res_120 = calc.calculate_expected_value(120e9)
    assert res_120["is_positive_ev"]

