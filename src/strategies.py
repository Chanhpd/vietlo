"""
Prediction Strategies Module for Vietlott.
Includes:
1. Random Baseline Strategy
2. Hot Frequency Strategy (most frequent in recent window)
3. Cold / Overdue Strategy (longest gap since appearance)
4. Hybrid Strategy (balanced hot + cold + parity & sum constraints)
5. ML Ranking Strategy (Vectorized feature-based classifier ranking next-draw probabilities)
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class BaseStrategy(ABC):
    def __init__(self, total_numbers: int = 55, pick_count: int = 6, random_state: Optional[int] = 42):
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.rng = np.random.default_rng(random_state)
        self.all_numbers = list(range(1, total_numbers + 1))

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_tickets(self, history_df: pd.DataFrame, num_tickets: int = 5) -> List[List[int]]:
        """
        Generate num_tickets using ONLY historical data up to the current moment.
        Returns a list of tickets, where each ticket is a sorted list of pick_count ints.
        """
        pass

    def _ensure_unique_tickets(
        self, candidate_pool: List[int], weights: Optional[np.ndarray], num_tickets: int
    ) -> List[List[int]]:
        """Helper to sample distinct tickets without duplicates."""
        tickets: List[List[int]] = []
        ticket_tuples: Set[Tuple[int, ...]] = set()
        attempts = 0
        max_attempts = num_tickets * 100

        if weights is not None:
            weights = np.array(weights, dtype=float)
            total_w = weights.sum()
            if total_w <= 0:
                weights = None
            else:
                weights = weights / total_w

        while len(tickets) < num_tickets and attempts < max_attempts:
            attempts += 1
            if len(candidate_pool) < self.pick_count:
                ticket = sorted([int(x) for x in self.rng.choice(self.all_numbers, size=self.pick_count, replace=False)])
            else:
                ticket = sorted([int(x) for x in self.rng.choice(candidate_pool, size=self.pick_count, replace=False, p=weights)])

            t_tuple = tuple(ticket)
            if t_tuple not in ticket_tuples:
                ticket_tuples.add(t_tuple)
                tickets.append(ticket)

        # Fallback if attempts exceeded
        while len(tickets) < num_tickets:
            ticket = sorted([int(x) for x in self.rng.choice(self.all_numbers, size=self.pick_count, replace=False)])
            t_tuple = tuple(ticket)
            if t_tuple not in ticket_tuples:
                ticket_tuples.add(t_tuple)
                tickets.append(ticket)

        return tickets


class RandomBaselineStrategy(BaseStrategy):
    """Generates tickets purely by uniform random selection (theoretical baseline)."""
    @property
    def name(self) -> str:
        return "Random Baseline"

    def generate_tickets(self, history_df: pd.DataFrame, num_tickets: int = 5) -> List[List[int]]:
        return self._ensure_unique_tickets(self.all_numbers, None, num_tickets)


class HotFrequencyStrategy(BaseStrategy):
    """
    Selects numbers with the highest frequency in recent lookback window.
    Weights sampling by recent empirical frequency.
    """
    def __init__(self, total_numbers: int = 55, pick_count: int = 6, lookback_window: int = 50, random_state: Optional[int] = 42):
        super().__init__(total_numbers, pick_count, random_state)
        self.lookback_window = lookback_window

    @property
    def name(self) -> str:
        return f"Hot Frequency (W={self.lookback_window})"

    def generate_tickets(self, history_df: pd.DataFrame, num_tickets: int = 5) -> List[List[int]]:
        recent_df = history_df.tail(self.lookback_window)
        counts = Counter()
        for nums in recent_df["main_numbers"]:
            counts.update(nums)

        pool_size = min(max(15, self.pick_count * 3), self.total_numbers)
        most_common = counts.most_common()

        existing = {num for num, _ in most_common}
        for num in self.all_numbers:
            if num not in existing:
                most_common.append((num, 0))

        hot_candidates = [num for num, _ in most_common[:pool_size]]
        weights = [counts[num] + 1 for num in hot_candidates]

        return self._ensure_unique_tickets(hot_candidates, np.array(weights), num_tickets)


class ColdOverdueStrategy(BaseStrategy):
    """
    Selects numbers that have not appeared for the longest time.
    Weights sampling by the current gap of each number.
    """
    def __init__(self, total_numbers: int = 55, pick_count: int = 6, random_state: Optional[int] = 42):
        super().__init__(total_numbers, pick_count, random_state)

    @property
    def name(self) -> str:
        return "Cold / Overdue"

    def generate_tickets(self, history_df: pd.DataFrame, num_tickets: int = 5) -> List[List[int]]:
        draw_sets = [set(nums) for nums in history_df["main_numbers"]]
        total_draws = len(draw_sets)

        gaps = {}
        for num in self.all_numbers:
            indices = [i for i, s in enumerate(draw_sets) if num in s]
            if indices:
                gaps[num] = (total_draws - 1) - indices[-1]
            else:
                gaps[num] = total_draws

        sorted_by_gap = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
        pool_size = min(max(15, self.pick_count * 3), self.total_numbers)
        cold_candidates = [num for num, _ in sorted_by_gap[:pool_size]]
        weights = [gaps[num] + 1 for num in cold_candidates]

        return self._ensure_unique_tickets(cold_candidates, np.array(weights), num_tickets)


class HybridStrategy(BaseStrategy):
    """
    Balanced portfolio strategy:
    - 2 Hot numbers (frequent in last W draws)
    - 2 Cold numbers (longest overdue gap)
    - 2 Medium / Warm numbers
    - Filters: Max 1 repeat from previous draw, Balanced Odd/Even (2-4), Sum within 1.5 std.
    """
    def __init__(self, total_numbers: int = 55, pick_count: int = 6, lookback_window: int = 50, random_state: Optional[int] = 42):
        super().__init__(total_numbers, pick_count, random_state)
        self.lookback_window = lookback_window

    @property
    def name(self) -> str:
        return "Hybrid Balanced"

    def generate_tickets(self, history_df: pd.DataFrame, num_tickets: int = 5) -> List[List[int]]:
        if len(history_df) == 0:
            return self._ensure_unique_tickets(self.all_numbers, None, num_tickets)

        recent_df = history_df.tail(self.lookback_window)
        counts = Counter()
        for nums in recent_df["main_numbers"]:
            counts.update(nums)

        draw_sets = [set(nums) for nums in history_df["main_numbers"]]
        total_draws = len(draw_sets)
        last_draw_set = draw_sets[-1]

        gaps = {}
        for num in self.all_numbers:
            indices = [i for i, s in enumerate(draw_sets) if num in s]
            gaps[num] = (total_draws - 1) - indices[-1] if indices else total_draws

        sorted_by_freq = [num for num, _ in counts.most_common()]
        for num in self.all_numbers:
            if num not in sorted_by_freq:
                sorted_by_freq.append(num)

        sorted_by_gap = [num for num, _ in sorted(gaps.items(), key=lambda x: x[1], reverse=True)]

        tier_size = max(5, self.total_numbers // 3)
        hot_pool = sorted_by_freq[:tier_size]
        cold_pool = sorted_by_gap[:tier_size]
        warm_pool = [n for n in self.all_numbers if n not in hot_pool[:10] and n not in cold_pool[:10]]

        mean_sum = self.pick_count * (self.total_numbers + 1) / 2.0
        sum_margin = 40.0

        tickets: List[List[int]] = []
        ticket_tuples: Set[Tuple[int, ...]] = set()
        attempts = 0
        max_attempts = num_tickets * 150

        while len(tickets) < num_tickets and attempts < max_attempts:
            attempts += 1
            h_picks = self.rng.choice(hot_pool, size=min(2, len(hot_pool)), replace=False).tolist()
            c_picks = [x for x in self.rng.choice(cold_pool, size=min(4, len(cold_pool)), replace=False) if x not in h_picks][:2]
            existing = set(h_picks + c_picks)
            avail_warm = [x for x in warm_pool if x not in existing]
            if len(avail_warm) < 2:
                avail_warm = [x for x in self.all_numbers if x not in existing]
            w_picks = self.rng.choice(avail_warm, size=min(2, len(avail_warm)), replace=False).tolist()

            candidate = sorted(h_picks + c_picks + w_picks)
            if len(candidate) < self.pick_count:
                continue

            # Constraints:
            # 1. No more than 1 repeat from previous draw
            overlap = len(set(candidate).intersection(last_draw_set))
            if overlap > 1:
                continue

            # 2. Parity check: 2 to 4 odd numbers
            odd_cnt = sum(1 for x in candidate if x % 2 != 0)
            if odd_cnt < 2 or odd_cnt > 4:
                continue

            # 3. Sum check
            cand_sum = sum(candidate)
            if abs(cand_sum - mean_sum) > sum_margin:
                continue

            c_tuple = tuple(candidate)
            if c_tuple not in ticket_tuples:
                ticket_tuples.add(c_tuple)
                tickets.append(candidate)

        # Fallback if strict constraints couldn't produce enough tickets
        while len(tickets) < num_tickets:
            fallback = sorted(self.rng.choice(self.all_numbers, size=self.pick_count, replace=False).tolist())
            t_tuple = tuple(fallback)
            if t_tuple not in ticket_tuples:
                ticket_tuples.add(t_tuple)
                tickets.append(fallback)

        return tickets


class MLRankingStrategy(BaseStrategy):
    """
    High-Performance Vectorized Machine Learning Classifier Strategy.
    Constructs a tabular feature matrix using fast matrix operations:
    - Rolling frequency across multiple horizons (10, 30, 60)
    - Current gap and recurrence intervals
    - Lag indicators (t-1, t-2, t-3)
    - Day-of-week affinity
    Fits a Logistic Regression / Random Forest and scores numbers for the next draw.
    """
    def __init__(
        self,
        total_numbers: int = 55,
        pick_count: int = 6,
        train_window: int = 250,
        model_type: str = "logreg",  # 'logreg' is lightning fast and well-calibrated; 'rf' supported
        random_state: Optional[int] = 42
    ):
        super().__init__(total_numbers, pick_count, random_state)
        self.train_window = train_window
        self.model_type = model_type

    @property
    def name(self) -> str:
        return f"ML Ranker ({'RandomForest' if self.model_type == 'rf' else 'LogisticReg'})"

    def _build_feature_matrices(self, history_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast vectorized feature extraction.
        Returns:
            X_train: (N_train * total_numbers, n_features)
            y_train: (N_train * total_numbers,)
            X_next: (total_numbers, n_features) for the upcoming draw
        """
        M = len(history_df)
        N = self.total_numbers

        # 1. Binary indicator matrix (M, N)
        mat = np.zeros((M, N), dtype=np.float32)
        for t, nums in enumerate(history_df["main_numbers"]):
            for num in nums:
                mat[t, num - 1] = 1.0

        dows = history_df["day_of_week_num"].values

        # 2. Vectorized gaps calculation in a single forward pass
        gaps = np.zeros((M, N), dtype=np.float32)
        last_seen = np.full(N, -100, dtype=np.int32)
        for t in range(M):
            gaps[t, :] = t - last_seen
            last_seen[mat[t, :] == 1.0] = t

        # 3. Rolling frequencies using cumulative sum
        def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
            cumsum = np.cumsum(np.vstack([np.zeros((1, N), dtype=arr.dtype), arr]), axis=0)
            res = np.zeros_like(arr)
            for t in range(1, M + 1):
                start = max(0, t - window)
                res[t - 1, :] = (cumsum[t, :] - cumsum[start, :]) / max(1, t - start)
            return res

        f_10 = rolling_mean(mat, 10)
        f_30 = rolling_mean(mat, 30)
        f_60 = rolling_mean(mat, 60)

        # 4. Day-of-week affinity
        # Cumulative hits on same DOW
        dow_affinity = np.zeros((M, N), dtype=np.float32)
        dow_counts = Counter()
        dow_hits = {d: np.zeros(N, dtype=np.float32) for d in range(7)}
        for t in range(M):
            d = dows[t]
            if dow_counts[d] > 0:
                dow_affinity[t, :] = dow_hits[d] / dow_counts[d]
            else:
                dow_affinity[t, :] = self.pick_count / self.total_numbers
            dow_hits[d] += mat[t, :]
            dow_counts[d] += 1

        # 5. Assemble feature rows for training window
        # For training target draw t, features must be from draw t-1!
        start_t = max(65, M - self.train_window)
        X_list, y_list = [], []

        for t in range(start_t, M):
            # Features at t-1 predicting draw t
            prev_t = t - 1
            feat_matrix = np.column_stack([
                f_10[prev_t, :],
                f_30[prev_t, :],
                f_60[prev_t, :],
                gaps[prev_t, :],
                mat[prev_t, :],                              # lag 1
                mat[prev_t - 1, :] if prev_t >= 1 else np.zeros(N),  # lag 2
                mat[prev_t - 2, :] if prev_t >= 2 else np.zeros(N),  # lag 3
                dow_affinity[prev_t, :]
            ])
            X_list.append(feat_matrix)
            y_list.append(mat[t, :])

        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        # Features for upcoming draw (target = M, features from M-1)
        prev_t = M - 1
        X_next = np.column_stack([
            f_10[prev_t, :],
            f_30[prev_t, :],
            f_60[prev_t, :],
            gaps[prev_t, :],
            mat[prev_t, :],
            mat[prev_t - 1, :] if prev_t >= 1 else np.zeros(N),
            mat[prev_t - 2, :] if prev_t >= 2 else np.zeros(N),
            dow_affinity[prev_t, :]
        ])

        return X_train, y_train, X_next

    def generate_tickets(self, history_df: pd.DataFrame, num_tickets: int = 5) -> List[List[int]]:
        if len(history_df) < 70:
            return self._ensure_unique_tickets(self.all_numbers, None, num_tickets)

        X_train, y_train, X_next = self._build_feature_matrices(history_df)

        if self.model_type == "rf":
            model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42, n_jobs=1)
        else:
            model = LogisticRegression(max_iter=200, random_state=42, C=0.5)

        model.fit(X_train, y_train)

        # Predict probability of each number appearing in upcoming draw
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_next)[:, 1]
        else:
            proba = model.decision_function(X_next)

        pool_size = min(max(15, self.pick_count * 3), self.total_numbers)
        ranked_indices = np.argsort(proba)[::-1][:pool_size]
        candidate_numbers = [self.all_numbers[i] for i in ranked_indices]
        candidate_weights = np.maximum(proba[ranked_indices], 1e-4)

        return self._ensure_unique_tickets(candidate_numbers, candidate_weights, num_tickets)


class XGBoostRankingStrategy(BaseStrategy):
    """
    Advanced Gradient Boosting Strategy using XGBoost.
    Engineers multi-scale rolling statistics, gap acceleration, and interaction features.
    """
    def __init__(
        self,
        total_numbers: int = 55,
        pick_count: int = 6,
        train_window: int = 250,
        random_state: Optional[int] = 42
    ):
        super().__init__(total_numbers, pick_count, random_state)
        self.train_window = train_window

    @property
    def name(self) -> str:
        return "XGBoost Ranker"

    def generate_tickets(self, history_df: pd.DataFrame, num_tickets: int = 5) -> List[List[int]]:
        if len(history_df) < 70:
            return self._ensure_unique_tickets(self.all_numbers, None, num_tickets)

        # Reuse fast feature matrix builder from MLRankingStrategy
        ml_helper = MLRankingStrategy(self.total_numbers, self.pick_count, self.train_window)
        X_train, y_train, X_next = ml_helper._build_feature_matrices(history_df)

        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=42)

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_next)[:, 1]

        pool_size = min(max(15, self.pick_count * 3), self.total_numbers)
        ranked_indices = np.argsort(proba)[::-1][:pool_size]
        candidate_numbers = [self.all_numbers[i] for i in ranked_indices]
        candidate_weights = np.maximum(proba[ranked_indices], 1e-4)

        return self._ensure_unique_tickets(candidate_numbers, candidate_weights, num_tickets)


class MarkovTransitionStrategy(BaseStrategy):
    """
    First-Order Markov Chain Transition Strategy.
    Calculates the empirical conditional transition probability:
    P(number j in draw t | number i in draw t-1).
    Scores all numbers based on transitions from the exact combination of the previous draw.
    """
    def __init__(self, total_numbers: int = 55, pick_count: int = 6, random_state: Optional[int] = 42):
        super().__init__(total_numbers, pick_count, random_state)

    @property
    def name(self) -> str:
        return "Markov Transition"

    def generate_tickets(self, history_df: pd.DataFrame, num_tickets: int = 5) -> List[List[int]]:
        if len(history_df) < 20:
            return self._ensure_unique_tickets(self.all_numbers, None, num_tickets)

        draw_sets = [set(nums) for nums in history_df["main_numbers"]]
        M = len(draw_sets)
        N = self.total_numbers

        # Count pair transitions between t-1 and t
        transition_matrix = np.zeros((N + 1, N + 1), dtype=np.float32)
        appearances = Counter()

        for t in range(1, M):
            prev_set = draw_sets[t - 1]
            curr_set = draw_sets[t]
            for u in prev_set:
                appearances[u] += 1
                for v in curr_set:
                    transition_matrix[u, v] += 1.0

        # Conditional probability P(v in t | u in t-1)
        for u in range(1, N + 1):
            if appearances[u] > 0:
                transition_matrix[u, :] /= appearances[u]
            else:
                transition_matrix[u, :] = self.pick_count / self.total_numbers

        # Score next draw candidates given the latest draw
        last_drawn = draw_sets[-1]
        scores = np.zeros(N + 1, dtype=np.float32)
        for u in last_drawn:
            scores += transition_matrix[u, :]

        # Top scoring candidates
        sorted_indices = np.argsort(scores[1:])[::-1]  # 0-indexed corresponding to 1..N
        pool_size = min(max(15, self.pick_count * 3), self.total_numbers)
        candidate_numbers = [self.all_numbers[i] for i in sorted_indices[:pool_size]]
        candidate_weights = np.maximum(scores[candidate_numbers], 1e-4)

        return self._ensure_unique_tickets(candidate_numbers, candidate_weights, num_tickets)


class GameTheoryAntiCrowdStrategy(BaseStrategy):
    """
    Game Theory & Behavioral Economics Anti-Crowd Strategy.
    Human players heavily over-select:
    - Birthday dates (numbers 1 to 31)
    - Consecutive patterns or lucky digits (ending in 8, 9).
    This strategy selects combinations with >= 4 numbers > 31 and high entropy,
    maximizing the Expected Monetary Value (EV) by minimizing the probability
    of sharing the Jackpot with other ticket holders in the event of a win.
    """
    def __init__(self, total_numbers: int = 55, pick_count: int = 6, random_state: Optional[int] = 42):
        super().__init__(total_numbers, pick_count, random_state)
        # Numbers above calendar days
        self.high_numbers = [n for n in self.all_numbers if n > 31]

    @property
    def name(self) -> str:
        return "Game Theory (Anti-Crowd)"

    def generate_tickets(self, history_df: pd.DataFrame, num_tickets: int = 5) -> List[List[int]]:
        tickets: List[List[int]] = []
        ticket_tuples: Set[Tuple[int, ...]] = set()
        attempts = 0
        max_attempts = num_tickets * 150

        # Target: 4 or 5 numbers > 31, remainder from 1..31
        lower_numbers = [n for n in self.all_numbers if n <= 31]

        while len(tickets) < num_tickets and attempts < max_attempts:
            attempts += 1
            n_high = 4 if len(self.high_numbers) >= 4 else len(self.high_numbers)
            n_low = self.pick_count - n_high

            high_picks = self.rng.choice(self.high_numbers, size=n_high, replace=False).tolist()
            low_picks = self.rng.choice(lower_numbers, size=n_low, replace=False).tolist()

            ticket = sorted([int(x) for x in high_picks + low_picks])

            # Filter out simple arithmetic progressions or too many consecutive numbers
            consec = sum(1 for i in range(len(ticket) - 1) if ticket[i+1] == ticket[i] + 1)
            if consec > 1:
                continue

            t_tuple = tuple(ticket)
            if t_tuple not in ticket_tuples:
                ticket_tuples.add(t_tuple)
                tickets.append(ticket)

        while len(tickets) < num_tickets:
            ticket = sorted([int(x) for x in self.rng.choice(self.all_numbers, size=self.pick_count, replace=False)])
            t_tuple = tuple(ticket)
            if t_tuple not in ticket_tuples:
                ticket_tuples.add(t_tuple)
                tickets.append(ticket)

        return tickets
