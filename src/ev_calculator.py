"""
Expected Value (+EV) and Break-Even Calculator for Vietlott.
Implements the mathematical syndicate strategy (Stefan Mandel / MIT Lottery Syndicate methodology).
Calculates when a lottery ticket mathematically flips from Negative EV (-EV) to Positive EV (+EV).
"""

from typing import Any, Dict
from scipy.special import comb


class VietlottEVCalculator:
    def __init__(self, lottery_code: str = "power655", ticket_price: int = 10000, tax_rate: float = 0.10):
        self.lottery_code = lottery_code.lower()
        self.ticket_price = ticket_price
        self.tax_rate = tax_rate

        if self.lottery_code == "power655":
            self.total_numbers = 55
            self.total_combos = comb(55, 6, exact=True)  # 28,989,675
            self.p_jackpot_1 = 1.0 / self.total_combos
            self.p_jackpot_2 = 1.0 / (self.total_combos / 6.0)  # 1 in 4,831,612.5
            self.p_first = (comb(6, 5) * comb(48, 1)) / self.total_combos      # 5 match, no bonus
            self.p_second = (comb(6, 4) * comb(49, 2)) / self.total_combos
            self.p_third = (comb(6, 3) * comb(49, 3)) / self.total_combos
            self.fixed_prizes = {
                "first": 40000000,
                "second": 500000,
                "third": 50000
            }
        else:  # mega645
            self.total_numbers = 45
            self.total_combos = comb(45, 6, exact=True)  # 8,145,060
            self.p_jackpot_1 = 1.0 / self.total_combos
            self.p_jackpot_2 = 0.0
            self.p_first = (comb(6, 5) * comb(39, 1)) / self.total_combos
            self.p_second = (comb(6, 4) * comb(39, 2)) / self.total_combos
            self.p_third = (comb(6, 3) * comb(39, 3)) / self.total_combos
            self.fixed_prizes = {
                "first": 10000000,
                "second": 300000,
                "third": 30000
            }

    def calculate_expected_value(
        self,
        current_jackpot_1: float,
        current_jackpot_2: float = 3000000000.0,
        estimated_tickets_sold: int = 1500000
    ) -> Dict[str, Any]:
        """
        Calculate expected monetary value of a single 10,000 VND ticket.
        Includes 10% personal income tax on prizes > 10,000,000 VND
        and probability of splitting the jackpot with other winners.
        """
        # Tax adjustments: prizes > 10,000,000 VND have 10% tax on the amount above 10m
        def net_prize(val: float) -> float:
            if val > 10000000:
                return 10000000 + (val - 10000000) * (1.0 - self.tax_rate)
            return val

        # Expected value from minor fixed prizes
        ev_third = self.p_third * net_prize(self.fixed_prizes["third"])
        ev_second = self.p_second * net_prize(self.fixed_prizes["second"])
        ev_first = self.p_first * net_prize(self.fixed_prizes["first"])
        fixed_ev = ev_third + ev_second + ev_first

        # Jackpot split probability adjustment (Poisson distribution)
        # Expected winners = tickets_sold * p_jackpot
        lambda_jp1 = estimated_tickets_sold * self.p_jackpot_1
        # Expected share factor = (1 - e^-lambda) / lambda ≈ 1 / (1 + lambda / 2)
        share_factor_1 = (1.0 - np.exp(-lambda_jp1)) / lambda_jp1 if lambda_jp1 > 0 else 1.0

        net_jp1 = net_prize(current_jackpot_1) * share_factor_1
        ev_jp1 = self.p_jackpot_1 * net_jp1

        ev_jp2 = 0.0
        if self.lottery_code == "power655":
            lambda_jp2 = estimated_tickets_sold * self.p_jackpot_2
            share_factor_2 = (1.0 - np.exp(-lambda_jp2)) / lambda_jp2 if lambda_jp2 > 0 else 1.0
            net_jp2 = net_prize(current_jackpot_2) * share_factor_2
            ev_jp2 = self.p_jackpot_2 * net_jp2

        total_ev = fixed_ev + ev_jp1 + ev_jp2
        net_ev = total_ev - self.ticket_price
        ev_ratio = (total_ev / self.ticket_price) * 100.0

        is_positive_ev = total_ev > self.ticket_price

        # Calculate break-even jackpot threshold
        target_from_jp = self.ticket_price - fixed_ev - ev_jp2
        if target_from_jp > 0:
            break_even_jp = (target_from_jp / self.p_jackpot_1) / (1.0 - self.tax_rate)
        else:
            break_even_jp = 30000000000.0

        return {
            "lottery": self.lottery_code,
            "current_jackpot_1_billion": round(current_jackpot_1 / 1e9, 2),
            "ticket_price": self.ticket_price,
            "fixed_prizes_ev": round(fixed_ev, 1),
            "jackpot_1_ev": round(ev_jp1, 1),
            "jackpot_2_ev": round(ev_jp2, 1),
            "total_expected_value_vnd": round(total_ev, 1),
            "net_expected_profit_vnd": round(net_ev, 1),
            "expected_return_pct": round(ev_ratio, 1),
            "is_positive_ev": is_positive_ev,
            "break_even_jackpot_1_billion": round(break_even_jp / 1e9, 1),
            "recommendation": (
                "KỲ VỌNG DƯƠNG (+EV): Jackpot đủ lớn để giá trị kỳ vọng vượt 10.000đ/vé. CÓ THỂ MUA (Kèm Wheeling System)."
                if is_positive_ev else
                f"KỲ VỌNG ÂM (-EV): Giá trị kỳ vọng chỉ đạt {total_ev:,.0f}đ / 10.000đ. TUYỆT ĐỐI KHÔNG MUA VÌ CHẮC CHẮN LỖ DÀI HẠN."
            )
        }


import numpy as np

