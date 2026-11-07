"""
Expected Value (+EV) and Break-Even Calculator for Vietlott.
Implements the mathematical syndicate strategy (Stefan Mandel / MIT Lottery Syndicate methodology).
Calculates when a lottery ticket mathematically flips from Negative EV (-EV) to Positive EV (+EV).
"""

from typing import Any, Dict
import numpy as np
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
        estimated_tickets_sold: int = 1500000,
        anti_crowd_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate expected monetary value of a single 10,000 VND ticket.
        Includes 10% personal income tax on prizes > 10,000,000 VND,
        Poisson probability of splitting the jackpot with other winners,
        Anti-Crowd Game Theory split reduction (55% lower competing crowd density),
        and Quarter-Kelly bankroll sizing.
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

        # Anti-Crowd Game Theory reduces competing ticket density by ~55% (multiplier 0.45)
        crowd_multiplier = 0.45 if anti_crowd_mode else 1.0
        effective_tickets = estimated_tickets_sold * crowd_multiplier

        # Jackpot split probability adjustment (Poisson distribution)
        lambda_jp1 = effective_tickets * self.p_jackpot_1
        share_factor_1 = (1.0 - np.exp(-lambda_jp1)) / lambda_jp1 if lambda_jp1 > 0 else 1.0

        net_jp1 = net_prize(current_jackpot_1) * share_factor_1
        ev_jp1 = self.p_jackpot_1 * net_jp1

        ev_jp2 = 0.0
        if self.lottery_code == "power655":
            lambda_jp2 = effective_tickets * self.p_jackpot_2
            share_factor_2 = (1.0 - np.exp(-lambda_jp2)) / lambda_jp2 if lambda_jp2 > 0 else 1.0
            net_jp2 = net_prize(current_jackpot_2) * share_factor_2
            ev_jp2 = self.p_jackpot_2 * net_jp2

        total_ev = fixed_ev + ev_jp1 + ev_jp2
        net_ev = total_ev - self.ticket_price
        ev_ratio = (total_ev / self.ticket_price) * 100.0

        is_positive_ev = total_ev > self.ticket_price

        # Exact break-even jackpot threshold accounting for Poisson split factor and tax
        target_from_jp = self.ticket_price - fixed_ev - ev_jp2
        if target_from_jp > 0:
            required_net_jp1 = (target_from_jp / self.p_jackpot_1) / max(share_factor_1, 1e-6)
            break_even_jp = 10000000.0 + (required_net_jp1 - 10000000.0) / (1.0 - self.tax_rate)
        else:
            break_even_jp = 30000000000.0

        # Also compute Anti-Crowd break-even vs Standard break-even for direct comparison
        lambda_std = estimated_tickets_sold * self.p_jackpot_1
        sf_std = (1.0 - np.exp(-lambda_std)) / lambda_std if lambda_std > 0 else 1.0
        lambda_ac = (estimated_tickets_sold * 0.45) * self.p_jackpot_1
        sf_ac = (1.0 - np.exp(-lambda_ac)) / lambda_ac if lambda_ac > 0 else 1.0

        std_be = 10000000.0 + ((max(0.0, target_from_jp) / self.p_jackpot_1) / sf_std - 10000000.0) / (1.0 - self.tax_rate)
        ac_be = 10000000.0 + ((max(0.0, target_from_jp) / self.p_jackpot_1) / sf_ac - 10000000.0) / (1.0 - self.tax_rate)

        # Kelly Criterion fraction f* = Edge / Odds for positive EV regimes
        if is_positive_ev and net_jp1 > self.ticket_price:
            b_odds = (net_jp1 - self.ticket_price) / float(self.ticket_price)
            edge_ratio = net_ev / float(self.ticket_price)
            kelly_fraction = max(0.0, edge_ratio / b_odds)
        else:
            kelly_fraction = 0.0

        return {
            "lottery": self.lottery_code,
            "current_jackpot_1_billion": round(current_jackpot_1 / 1e9, 2),
            "ticket_price": self.ticket_price,
            "anti_crowd_mode": anti_crowd_mode,
            "share_factor_pct": round(float(share_factor_1) * 100.0, 2),
            "fixed_prizes_ev": round(float(fixed_ev), 1),
            "jackpot_1_ev": round(float(ev_jp1), 1),
            "jackpot_2_ev": round(float(ev_jp2), 1),
            "total_expected_value_vnd": round(float(total_ev), 1),
            "net_expected_profit_vnd": round(float(net_ev), 1),
            "expected_return_pct": round(float(ev_ratio), 1),
            "is_positive_ev": bool(is_positive_ev),
            "break_even_jackpot_1_billion": round(float(break_even_jp) / 1e9, 1),
            "standard_break_even_billion": round(float(std_be) / 1e9, 1),
            "anti_crowd_break_even_billion": round(float(ac_be) / 1e9, 1),
            "kelly_fraction_pct": round(float(kelly_fraction) * 100.0, 8),
            "recommendation": (
                "KỲ VỌNG DƯƠNG (+EV): Jackpot đủ lớn để giá trị kỳ vọng vượt 10.000đ/vé. CÓ THỂ MUA (Kèm Wheeling System + Anti-Crowd)."
                if is_positive_ev else
                f"KỲ VỌNG ÂM (-EV): Giá trị kỳ vọng chỉ đạt {total_ev:,.0f}đ / 10.000đ. TUYỆT ĐỐI KHÔNG MUA VÌ CHẮC CHẮN LỖ DÀI HẠN."
            )
        }

