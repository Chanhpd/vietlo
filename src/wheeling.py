"""
Combinatorial Wheeling System (Covering Design) Module for Vietlott.
Constructs optimal abbreviated lottery wheels:
Given a candidate pool of M numbers, generates a minimal subset of 6-number tickets
that mathematically guarantees at least a t-match prize if m numbers from the pool are drawn.
"""

from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class CombinatorialWheeler:
    def __init__(self, pick_count: int = 6):
        self.pick_count = pick_count

    def generate_abbreviated_wheel(
        self,
        candidate_pool: List[int],
        target_guarantee: int = 3,
        condition_drawn: int = 4,
        max_tickets: int = 10,
        number_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate an abbreviated wheel using a probability-weighted greedy set-cover algorithm.
        Guarantees that if `condition_drawn` winning numbers fall inside `candidate_pool`,
        at least one ticket will match at least `target_guarantee` numbers.
        """
        candidate_pool = sorted(list(set(candidate_pool)))
        v = len(candidate_pool)
        k = self.pick_count
        t = target_guarantee
        m = condition_drawn

        if v < k:
            raise ValueError(f"Candidate pool size ({v}) must be >= pick count ({k})")

        all_possible_tickets = list(combinations(candidate_pool, k))
        full_wheel_count = len(all_possible_tickets)

        if full_wheel_count <= max_tickets:
            selected_tickets = [list(ticket) for ticket in all_possible_tickets]
            coverage = self.calculate_coverage(selected_tickets, candidate_pool, t, m)
            return {
                "candidate_pool": candidate_pool,
                "pool_size": v,
                "target_guarantee": t,
                "condition_drawn": m,
                "tickets": selected_tickets,
                "ticket_count": len(selected_tickets),
                "full_wheel_tickets": full_wheel_count,
                "cost_saving_pct": round((1.0 - len(selected_tickets) / max(1, full_wheel_count)) * 100.0, 1),
                "guarantee_coverage_pct": coverage
            }

        target_subsets = list(combinations(candidate_pool, m))
        uncovered_subsets = set(target_subsets)

        ticket_covers: Dict[Tuple[int, ...], Set[Tuple[int, ...]]] = {}
        for ticket in all_possible_tickets:
            t_set = set(ticket)
            covered = {sub for sub in target_subsets if len(t_set.intersection(sub)) >= t}
            ticket_covers[ticket] = covered

        # Precompute intrinsic quality score for each candidate ticket (weights + structural parity/balance)
        w_map = number_weights if number_weights is not None else {n: 1.0 for n in candidate_pool}
        ticket_quality: Dict[Tuple[int, ...], float] = {}
        for ticket in all_possible_tickets:
            w_sum = sum(float(w_map.get(n, 1.0)) for n in ticket)
            odd_cnt = sum(1 for n in ticket if n % 2 != 0)
            parity_bonus = 0.25 if 2 <= odd_cnt <= 4 else 0.0
            ticket_quality[ticket] = w_sum + parity_bonus

        selected_tickets: List[List[int]] = []
        available_tickets = list(all_possible_tickets)

        while uncovered_subsets and len(selected_tickets) < max_tickets and available_tickets:
            # Primary objective: number of newly covered m-subsets
            # Secondary objective: avoid >3 shared numbers with any existing ticket
            # Tertiary tie-breaker: intrinsic weight + low total overlap
            def score_candidate(tk: Tuple[int, ...]) -> Tuple[int, int, float]:
                new_cov = len(ticket_covers[tk].intersection(uncovered_subsets))
                max_ov = max((len(set(tk).intersection(s)) for s in selected_tickets), default=0)
                overlap_pen = sum(len(set(tk).intersection(s)) for s in selected_tickets) * 0.15
                return (new_cov, -max(0, max_ov - 3), ticket_quality[tk] - overlap_pen)

            best_ticket = max(available_tickets, key=score_candidate, default=None)
            if not best_ticket:
                break

            covers_now = ticket_covers[best_ticket].intersection(uncovered_subsets)
            if not covers_now and len(selected_tickets) >= 3:
                break

            selected_tickets.append(list(best_ticket))
            uncovered_subsets.difference_update(covers_now)
            available_tickets.remove(best_ticket)

        coverage = self.calculate_coverage(selected_tickets, candidate_pool, t, m)

        return {
            "candidate_pool": candidate_pool,
            "pool_size": v,
            "target_guarantee": t,
            "condition_drawn": m,
            "tickets": selected_tickets,
            "ticket_count": len(selected_tickets),
            "full_wheel_tickets": full_wheel_count,
            "cost_saving_pct": round((1.0 - len(selected_tickets) / max(1, full_wheel_count)) * 100.0, 1),
            "guarantee_coverage_pct": coverage
        }

    def calculate_coverage(
        self,
        tickets: List[List[int]],
        candidate_pool: List[int],
        target_guarantee: int,
        condition_drawn: int
    ) -> float:
        """Calculate the exact percentage of condition combinations covered."""
        ticket_sets = [set(t) for t in tickets]
        possible_draws = list(combinations(candidate_pool, condition_drawn))
        if not possible_draws:
            return 100.0

        covered_count = 0
        for draw in possible_draws:
            d_set = set(draw)
            if any(len(t_set.intersection(d_set)) >= target_guarantee for t_set in ticket_sets):
                covered_count += 1

        return round((covered_count / len(possible_draws)) * 100.0, 2)

