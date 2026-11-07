"""
Backtest Engine & Evaluation Module for Vietlott Strategies.
Conducts strict walk-forward evaluation, calculates Vietlott prize payouts,
tracks hit rate, ROI, cumulative returns, and statistical superiority tests.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind

from src.strategies import BaseStrategy


class VietlottBacktester:
    def __init__(
        self,
        df: pd.DataFrame,
        lottery_code: str = "power655",
        total_numbers: int = 55,
        pick_count: int = 6,
        ticket_price: int = 10000,
        prize_structure: Optional[Dict[str, int]] = None
    ):
        self.df = df.copy()
        self.lottery_code = lottery_code.lower()
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.ticket_price = ticket_price

        # Standard Vietlott minimum guarantee prizes
        if prize_structure:
            self.prize_structure = prize_structure
        elif self.lottery_code == "power655":
            self.prize_structure = {
                "jackpot_1": 30000000000,
                "jackpot_2": 3000000000,
                "first": 40000000,
                "second": 500000,
                "third": 50000
            }
        else:  # mega645
            self.prize_structure = {
                "jackpot_1": 12000000000,
                "first": 10000000,
                "second": 300000,
                "third": 30000
            }

        self.theoretical_expected_hits = (self.pick_count * self.pick_count) / self.total_numbers

    def evaluate_ticket(
        self, ticket: List[int], actual_main: List[int], actual_bonus: Optional[int]
    ) -> Tuple[int, str, int]:
        """
        Evaluate a single 6-number ticket against actual draw result.
        Returns: (match_count, prize_tier_name, prize_amount)
        """
        t_set = set(ticket)
        main_set = set(actual_main)
        matches = len(t_set.intersection(main_set))

        tier = "none"
        prize = 0

        if self.lottery_code == "power655":
            if matches == 6:
                tier = "jackpot_1"
                prize = self.prize_structure["jackpot_1"]
            elif matches == 5 and (actual_bonus is not None and actual_bonus in t_set):
                tier = "jackpot_2"
                prize = self.prize_structure["jackpot_2"]
            elif matches == 5:
                tier = "first"
                prize = self.prize_structure["first"]
            elif matches == 4:
                tier = "second"
                prize = self.prize_structure["second"]
            elif matches == 3:
                tier = "third"
                prize = self.prize_structure["third"]
        else:  # mega 6/45
            if matches == 6:
                tier = "jackpot_1"
                prize = self.prize_structure["jackpot_1"]
            elif matches == 5:
                tier = "first"
                prize = self.prize_structure["first"]
            elif matches == 4:
                tier = "second"
                prize = self.prize_structure["second"]
            elif matches == 3:
                tier = "third"
                prize = self.prize_structure["third"]

        return matches, tier, prize

    def run_backtest(
        self,
        strategies: List[BaseStrategy],
        test_draws: int = 200,
        tickets_per_draw: int = 5
    ) -> Dict[str, Any]:
        """
        Conduct walk-forward out-of-sample backtesting.
        For each test draw t, feeds history up to t-1 to each strategy,
        generates tickets, and evaluates performance.
        """
        total_draws = len(self.df)
        if test_draws >= total_draws - 60:
            test_draws = total_draws - 60

        start_test_idx = total_draws - test_draws
        test_df = self.df.iloc[start_test_idx:].reset_index(drop=True)

        results: Dict[str, Dict[str, Any]] = {}
        for strat in strategies:
            results[strat.name] = {
                "strategy_name": strat.name,
                "all_matches": [],           # list of match count per ticket
                "draw_mean_matches": [],     # average match per draw
                "prizes_won": [],            # prize amount per ticket
                "draw_prizes_won": [],       # prize amount per draw
                "tier_counts": {
                    "jackpot_1": 0, "jackpot_2": 0, "first": 0, "second": 0, "third": 0, "none": 0
                },
                "match_counts": {k: 0 for k in range(7)},
                "cumulative_hits": [],
                "cumulative_spend": [],
                "cumulative_payout": [],
                "cumulative_profit": []
            }

        spend_per_draw = tickets_per_draw * self.ticket_price

        # Walk forward through each test draw
        for i in range(test_draws):
            curr_global_idx = start_test_idx + i
            history_df = self.df.iloc[:curr_global_idx]
            actual_row = self.df.iloc[curr_global_idx]

            actual_main = actual_row["main_numbers"]
            actual_bonus = actual_row["bonus_number"]

            for strat in strategies:
                tickets = strat.generate_tickets(history_df, num_tickets=tickets_per_draw)
                draw_matches = []
                draw_prize = 0

                for ticket in tickets:
                    m, tier, prize = self.evaluate_ticket(ticket, actual_main, actual_bonus)
                    draw_matches.append(m)
                    draw_prize += prize
                    results[strat.name]["all_matches"].append(m)
                    results[strat.name]["prizes_won"].append(prize)
                    results[strat.name]["match_counts"][m] += 1
                    if tier in results[strat.name]["tier_counts"]:
                        results[strat.name]["tier_counts"][tier] += 1

                mean_m = float(np.mean(draw_matches))
                results[strat.name]["draw_mean_matches"].append(mean_m)
                results[strat.name]["draw_prizes_won"].append(draw_prize)

                # Update cumulative series
                prev_hits = results[strat.name]["cumulative_hits"][-1] if results[strat.name]["cumulative_hits"] else 0
                prev_spend = results[strat.name]["cumulative_spend"][-1] if results[strat.name]["cumulative_spend"] else 0
                prev_payout = results[strat.name]["cumulative_payout"][-1] if results[strat.name]["cumulative_payout"] else 0

                curr_hits = prev_hits + sum(draw_matches)
                curr_spend = prev_spend + spend_per_draw
                curr_payout = prev_payout + draw_prize
                curr_profit = curr_payout - curr_spend

                results[strat.name]["cumulative_hits"].append(curr_hits)
                results[strat.name]["cumulative_spend"].append(curr_spend)
                results[strat.name]["cumulative_payout"].append(curr_payout)
                results[strat.name]["cumulative_profit"].append(curr_profit)

        # Compute summary metrics and comparisons
        summary_rows = []
        random_baseline_matches = None
        for strat in strategies:
            if "random" in strat.name.lower():
                random_baseline_matches = results[strat.name]["all_matches"]
                break

        for strat in strategies:
            data = results[strat.name]
            all_m = np.array(data["all_matches"])
            total_tickets = len(all_m)
            total_cost = total_tickets * self.ticket_price
            total_payout = sum(data["prizes_won"])
            net_profit = total_payout - total_cost
            roi = (net_profit / total_cost) * 100.0 if total_cost > 0 else 0.0

            avg_hits = float(np.mean(all_m)) if len(all_m) > 0 else 0.0
            std_hits = float(np.std(all_m)) if len(all_m) > 0 else 0.0
            winning_tickets = sum(1 for m in all_m if m >= 3)
            win_rate = (winning_tickets / total_tickets) * 100.0 if total_tickets > 0 else 0.0

            # Statistical significance vs Random Baseline
            p_val = 1.0
            t_stat = 0.0
            if random_baseline_matches is not None and strat.name != "Random Baseline":
                res = ttest_ind(all_m, random_baseline_matches, equal_var=False)
                t_stat = float(res.statistic)
                p_val = float(res.pvalue)

            summary_rows.append({
                "strategy": strat.name,
                "total_tickets": total_tickets,
                "total_cost_vnd": total_cost,
                "total_payout_vnd": total_payout,
                "net_profit_vnd": net_profit,
                "roi_pct": round(roi, 2),
                "avg_hits_per_ticket": round(avg_hits, 4),
                "theoretical_hits": round(self.theoretical_expected_hits, 4),
                "hits_diff_vs_theory": round(avg_hits - self.theoretical_expected_hits, 4),
                "win_rate_pct": round(win_rate, 2),
                "third_prizes": data["tier_counts"]["third"],
                "second_prizes": data["tier_counts"]["second"],
                "first_prizes": data["tier_counts"]["first"],
                "jackpots": data["tier_counts"]["jackpot_1"] + data["tier_counts"]["jackpot_2"],
                "t_stat_vs_random": round(t_stat, 3),
                "p_value_vs_random": round(p_val, 4),
                "beats_random_statistically": bool(p_val < 0.05 and avg_hits > self.theoretical_expected_hits)
            })

        summary_df = pd.DataFrame(summary_rows)

        return {
            "test_draws_count": test_draws,
            "tickets_per_draw": tickets_per_draw,
            "total_tickets_per_strategy": test_draws * tickets_per_draw,
            "date_start": str(test_df["date"].min().date()),
            "date_end": str(test_df["date"].max().date()),
            "theoretical_expected_hits": round(self.theoretical_expected_hits, 4),
            "summary_df": summary_df,
            "strategy_details": results
        }

