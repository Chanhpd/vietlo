"""
Descriptive Statistics and Pattern Analysis for Vietlott Data.
Covers frequencies, hot/cold analysis, gap/recency distributions,
consecutive draw repeats, top pairs, and triplets.
"""

from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


class VietlottAnalyzer:
    def __init__(self, df: pd.DataFrame, total_numbers: int = 55, pick_count: int = 6):
        self.df = df.copy()
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.all_numbers = list(range(1, total_numbers + 1))
        self.total_draws = len(df)

    def get_frequency_stats(self, window: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate occurrence count, percentage, and ranking for each number.
        If window is specified, only use the latest `window` draws.
        """
        target_df = self.df.tail(window) if window else self.df
        n_draws = len(target_df)

        counts = Counter()
        for nums in target_df["main_numbers"]:
            counts.update(nums)

        rows = []
        expected_prob = self.pick_count / self.total_numbers
        expected_count = n_draws * expected_prob

        for num in self.all_numbers:
            c = counts[num]
            pct = (c / n_draws) * 100 if n_draws > 0 else 0
            rows.append({
                "number": num,
                "count": c,
                "frequency_pct": round(pct, 2),
                "expected_count": round(expected_count, 1),
                "diff_from_expected": round(c - expected_count, 1),
                "relative_ratio": round(c / expected_count, 3) if expected_count > 0 else 1.0
            })

        freq_df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
        freq_df["rank"] = range(1, len(freq_df) + 1)
        return freq_df

    def get_gap_stats(self) -> pd.DataFrame:
        """
        Calculate current gap (draws since last appearance), max gap, and mean gap
        for every number compared to theoretical expectation.
        Theoretical gap expectation: E[Gap] = total_numbers / pick_count
        """
        theoretical_mean_gap = self.total_numbers / self.pick_count
        records = []

        # Convert draws to list of sets for fast lookup
        draw_sets = [set(nums) for nums in self.df["main_numbers"]]
        total_draws = len(draw_sets)

        for num in self.all_numbers:
            # Find all draw indices where num appeared
            indices = [i for i, s in enumerate(draw_sets) if num in s]

            if not indices:
                records.append({
                    "number": num,
                    "current_gap": total_draws,
                    "max_gap": total_draws,
                    "min_gap": 0,
                    "mean_gap": 0,
                    "theoretical_mean_gap": round(theoretical_mean_gap, 2),
                    "appearances": 0,
                    "is_overdue": True
                })
                continue

            # Current gap = total_draws - 1 - last_seen_index
            last_seen = indices[-1]
            current_gap = (total_draws - 1) - last_seen

            # Gaps between appearances
            # A gap of 0 means it appeared in consecutive draws
            gaps = [indices[i] - indices[i-1] for i in range(1, len(indices))]
            if not gaps:
                # Appeared only once
                mean_gap = total_draws
                max_gap = total_draws
                min_gap = total_draws
            else:
                mean_gap = np.mean(gaps)
                max_gap = max(gaps)
                min_gap = min(gaps)

            # An overdue number has current_gap > 1.5 * theoretical_mean_gap
            is_overdue = current_gap > (1.5 * theoretical_mean_gap)

            records.append({
                "number": num,
                "current_gap": current_gap,
                "max_gap": max_gap,
                "min_gap": min_gap,
                "mean_gap": round(float(mean_gap), 2),
                "theoretical_mean_gap": round(theoretical_mean_gap, 2),
                "appearances": len(indices),
                "is_overdue": is_overdue
            })

        gap_df = pd.DataFrame(records).sort_values("current_gap", ascending=False).reset_index(drop=True)
        return gap_df

    def get_hot_cold_numbers(self, lookback: int = 50, top_n: int = 10) -> Dict[str, Any]:
        """
        Categorize numbers into Hot, Cold, Warm, and Overdue based on recent lookback window.
        """
        freq_df = self.get_frequency_stats(window=lookback)
        gap_df = self.get_gap_stats()

        # Merge recent frequency with gap
        merged = freq_df.merge(gap_df[["number", "current_gap", "max_gap", "mean_gap", "is_overdue"]], on="number")

        hot_df = merged.sort_values(by=["count", "current_gap"], ascending=[False, True]).head(top_n)
        cold_df = merged.sort_values(by=["count", "current_gap"], ascending=[True, False]).head(top_n)
        overdue_df = gap_df.sort_values(by="current_gap", ascending=False).head(top_n)

        return {
            "lookback_window": lookback,
            "hot_numbers": hot_df[["number", "count", "frequency_pct", "current_gap"]].to_dict(orient="records"),
            "cold_numbers": cold_df[["number", "count", "frequency_pct", "current_gap"]].to_dict(orient="records"),
            "overdue_numbers": overdue_df[["number", "current_gap", "max_gap", "mean_gap"]].to_dict(orient="records"),
            "summary_table": merged
        }

    def get_range_distribution(self, bin_size: int = 10) -> pd.DataFrame:
        """
        Analyze the distribution of winning numbers across intervals:
        e.g., 1-10, 11-20, 21-30, 31-40, 41-50, 51-55
        """
        bins = list(range(1, self.total_numbers + bin_size, bin_size))
        if bins[-1] < self.total_numbers + 1:
            bins.append(self.total_numbers + 1)

        counts = Counter()
        for nums in self.df["main_numbers"]:
            counts.update(nums)

        range_rows = []
        total_number_picks = self.total_draws * self.pick_count

        for i in range(len(bins) - 1):
            start = bins[i]
            end = min(bins[i+1] - 1, self.total_numbers)
            span_size = end - start + 1

            interval_count = sum(counts[n] for n in range(start, end + 1))
            pct = (interval_count / total_number_picks) * 100 if total_number_picks > 0 else 0
            expected_pct = (span_size / self.total_numbers) * 100

            range_rows.append({
                "range_label": f"{start:02d}-{end:02d}",
                "numbers_in_range": span_size,
                "total_occurrences": interval_count,
                "actual_pct": round(pct, 2),
                "expected_pct": round(expected_pct, 2),
                "diff_pct": round(pct - expected_pct, 2)
            })

        return pd.DataFrame(range_rows)

    def get_consecutive_draw_repeats(self) -> Dict[str, Any]:
        """
        Analyze how many numbers repeat from draw t-1 to draw t.
        Compare actual empirical repeat frequency with theoretical Hypergeometric distribution.
        """
        draw_sets = [set(nums) for nums in self.df["main_numbers"]]
        repeats = []

        for i in range(1, len(draw_sets)):
            overlap = len(draw_sets[i].intersection(draw_sets[i - 1]))
            repeats.append(overlap)

        repeat_counts = Counter(repeats)
        total_transitions = len(repeats)

        # Theoretical Hypergeometric distribution:
        # Drawing 6 numbers from total_numbers, where 6 numbers from draw t-1 are "successes"
        # N = total_numbers, K = 6, n = 6, k = overlap
        rows = []
        for k in range(self.pick_count + 1):
            obs = repeat_counts[k]
            obs_pct = (obs / total_transitions) * 100 if total_transitions > 0 else 0
            # hypergeom.pmf(k, M, n, N) in scipy:
            # M = total balls, n = number of white balls (successes in pop = 6), N = balls drawn = 6
            theo_prob = hypergeom.pmf(k, self.total_numbers, self.pick_count, self.pick_count)
            theo_pct = theo_prob * 100

            rows.append({
                "repeat_count": k,
                "observed_draws": obs,
                "observed_pct": round(obs_pct, 2),
                "theoretical_pct": round(theo_pct, 2),
                "expected_draws": round(theo_prob * total_transitions, 1)
            })

        mean_repeats = np.mean(repeats) if repeats else 0.0
        theoretical_mean = (self.pick_count * self.pick_count) / self.total_numbers

        return {
            "total_transitions": total_transitions,
            "mean_observed_repeats": round(float(mean_repeats), 3),
            "theoretical_mean_repeats": round(float(theoretical_mean), 3),
            "repeat_table": pd.DataFrame(rows),
            "at_least_one_repeat_pct": round(sum(obs for k, obs in repeat_counts.items() if k > 0) / total_transitions * 100, 2)
        }

    def get_top_pairs(self, top_k: int = 10) -> pd.DataFrame:
        """Find the pairs (combinations of 2) that appeared together most frequently."""
        pair_counts = Counter()
        for nums in self.df["main_numbers"]:
            for pair in combinations(sorted(nums), 2):
                pair_counts[pair] += 1

        # Theoretical probability of any pair: C(total-2, 4) / C(total, 6) = 6*5 / (total*(total-1))
        p_pair = (self.pick_count * (self.pick_count - 1)) / (self.total_numbers * (self.total_numbers - 1))
        expected_draws = self.total_draws * p_pair

        top_pairs = pair_counts.most_common(top_k)
        rows = []
        for (n1, n2), count in top_pairs:
            pct = (count / self.total_draws) * 100
            rows.append({
                "pair": f"({n1:02d}, {n2:02d})",
                "count": count,
                "actual_pct": round(pct, 2),
                "expected_draws": round(expected_draws, 2),
                "ratio_vs_expected": round(count / expected_draws, 2) if expected_draws > 0 else 1.0
            })
        return pd.DataFrame(rows)

    def get_top_triplets(self, top_k: int = 10) -> pd.DataFrame:
        """Find the triplets (combinations of 3) that appeared together most frequently."""
        triplet_counts = Counter()
        for nums in self.df["main_numbers"]:
            for trip in combinations(sorted(nums), 3):
                triplet_counts[trip] += 1

        # Theoretical probability of any triplet: C(6,3) / C(total, 3)
        p_trip = (self.pick_count * (self.pick_count - 1) * (self.pick_count - 2)) / (
            self.total_numbers * (self.total_numbers - 1) * (self.total_numbers - 2)
        )
        expected_draws = self.total_draws * p_trip

        top_trips = triplet_counts.most_common(top_k)
        rows = []
        for (n1, n2, n3), count in top_trips:
            pct = (count / self.total_draws) * 100
            rows.append({
                "triplet": f"({n1:02d}, {n2:02d}, {n3:02d})",
                "count": count,
                "actual_pct": round(pct, 2),
                "expected_draws": round(expected_draws, 2),
                "ratio_vs_expected": round(count / expected_draws, 2) if expected_draws > 0 else 1.0
            })
        return pd.DataFrame(rows)

    def get_parity_and_sum_stats(self) -> Dict[str, Any]:
        """Analyze odd/even counts and sum distribution."""
        odd_counts = self.df["odd_count"].value_counts().sort_index()
        total_draws = len(self.df)

        parity_rows = []
        # Theoretical hypergeometric for odd count:
        # Total numbers has odds and evens
        total_odds = (self.total_numbers + 1) // 2
        total_evens = self.total_numbers - total_odds

        for odd_k in range(7):
            obs = odd_counts.get(odd_k, 0)
            obs_pct = (obs / total_draws) * 100 if total_draws > 0 else 0
            theo_prob = hypergeom.pmf(odd_k, self.total_numbers, total_odds, self.pick_count)
            parity_rows.append({
                "odd_count": odd_k,
                "even_count": 6 - odd_k,
                "ratio_label": f"{odd_k}L-{6-odd_k}C",
                "observed_draws": obs,
                "observed_pct": round(obs_pct, 2),
                "theoretical_pct": round(theo_prob * 100, 2)
            })

        sum_stats = {
            "mean_sum": round(float(self.df["sum_numbers"].mean()), 2),
            "std_sum": round(float(self.df["sum_numbers"].std()), 2),
            "min_sum": int(self.df["sum_numbers"].min()),
            "max_sum": int(self.df["sum_numbers"].max()),
            "theoretical_mean_sum": round(float(self.pick_count * (self.total_numbers + 1) / 2.0), 2)
        }

        return {
            "parity_distribution": pd.DataFrame(parity_rows),
            "sum_stats": sum_stats
        }

