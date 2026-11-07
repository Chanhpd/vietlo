"""
Combinatorial Wheeling System (Covering Design) Module for Vietlott.
Constructs optimal abbreviated lottery wheels:
Given a candidate pool of M numbers, generates a minimal subset of 6-number tickets
that mathematically guarantees at least a t-match prize if m numbers from the pool are drawn.
"""

from itertools import combinations
from typing import Any, Dict, List, Set, Tuple

import numpy as np


class CombinatorialWheeler:
    def __init__(self, pick_count: int = 6):
        self.pick_count = pick_count

    def generate_abbreviated_wheel(
        self,
        candidate_pool: List[int],
        target_guarantee: int = 3,
        condition_drawn: int = 4,
        max_tickets: int = 10
    ) -> Dict[str, Any]:
        """
        Generate an abbreviated wheel using greedy set-cover algorithm.
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

        # All possible 6-number tickets from the candidate pool
        all_possible_tickets = list(combinations(candidate_pool, k))

        # If pool is small enough, return full wheel or slice
        if len(all_possible_tickets) <= max_tickets:
            selected_tickets = [list(ticket) for ticket in all_possible_tickets]
            coverage = self.calculate_coverage(selected_tickets, candidate_pool, t, m)
            return {
                "candidate_pool": candidate_pool,
                "pool_size": v,
                "target_guarantee": t,
                "condition_drawn": m,
                "tickets": selected_tickets,
                "ticket_count": len(selected_tickets),
                "guarantee_coverage_pct": coverage
            }

        # Subsets of size m that could be drawn from our pool
        target_subsets = list(combinations(candidate_pool, m))
        uncovered_subsets = set(target_subsets)

        # Precompute which ticket covers which m-subsets
        # A ticket of size 6 covers an m-subset if len(ticket.intersection(m_subset)) >= t
        ticket_covers: Dict[Tuple[int, ...], Set[Tuple[int, ...]]] = {}
        for ticket in all_possible_tickets:
            t_set = set(ticket)
            covered = {sub for sub in target_subsets if len(t_set.intersection(sub)) >= t}
            ticket_covers[ticket] = covered

        selected_tickets: List[List[int]] = []
        available_tickets = set(all_possible_tickets)

        while uncovered_subsets and len(selected_tickets) < max_tickets:
            # Pick ticket that covers the most remaining uncovered subsets
            best_ticket = max(
                available_tickets,
                key=lambda tk: len(ticket_covers[tk].intersection(uncovered_subsets)),
                default=None
            )
            if not best_ticket:
                break

            covers_now = ticket_covers[best_ticket].intersection(uncovered_subsets)
            if not covers_now and len(selected_tickets) >= 3:
                # Diminishing returns, pool already well covered
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

