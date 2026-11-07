"""
Randomness & Statistical Hypothesis Testing Module for Vietlott.
Includes Chi-Square Uniformity Test, Wald-Wolfowitz Runs Test for Independence,
and Hypergeometric Goodness-of-Fit tests.
"""

from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2, hypergeom, norm


class VietlottRandomnessTester:
    def __init__(self, df: pd.DataFrame, total_numbers: int = 55, pick_count: int = 6):
        self.df = df.copy()
        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.total_draws = len(df)
        self.all_numbers = list(range(1, total_numbers + 1))

    def chi_square_uniformity_test(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Chi-Square Goodness-of-Fit test for uniform distribution across all numbers.
        H0: All numbers 1..N have equal probability of being picked.
        H1: Some numbers are drawn with higher/lower probability than expected.
        """
        counts = Counter()
        for nums in self.df["main_numbers"]:
            counts.update(nums)

        expected_count = self.total_draws * (self.pick_count / self.total_numbers)
        observed = np.array([counts[num] for num in self.all_numbers])
        expected = np.full(self.total_numbers, expected_count)

        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        df_deg = self.total_numbers - 1
        p_value = float(chi2.sf(chi2_stat, df_deg))
        critical_value = float(chi2.ppf(1 - alpha, df_deg))
        is_uniform = p_value > alpha

        return {
            "test_name": "Chi-Square Uniformity Test",
            "null_hypothesis": "Các số từ 1 đến N xuất hiện với xác suất đồng đều (Uniform Distribution)",
            "chi2_stat": round(float(chi2_stat), 3),
            "degrees_of_freedom": df_deg,
            "p_value": round(p_value, 5),
            "critical_value_95pct": round(critical_value, 3),
            "alpha": alpha,
            "is_uniform": is_uniform,
            "conclusion": (
                "Chấp nhận H0 (Không thể bác bỏ tính ngẫu nhiên đều): Phân bố các số tuân theo ngẫu nhiên lý thuyết."
                if is_uniform else
                "Bác bỏ H0 (Có sai lệch thống kê): Tần suất một số số khác biệt có ý nghĩa thống kê so với phân bố đều."
            )
        }

    def runs_test_for_independence(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Wald-Wolfowitz Runs Test for independence across draws for each number.
        H0: The sequence of appearances of a number across draws is independent (random).
        H1: The sequence shows clustering (persistence) or over-alternation.
        """
        draw_sets = [set(nums) for nums in self.df["main_numbers"]]
        num_results = []

        for num in self.all_numbers:
            # Binary sequence: 1 if num in draw, 0 otherwise
            seq = np.array([1 if num in s else 0 for s in draw_sets])
            n1 = np.sum(seq == 1)
            n0 = np.sum(seq == 0)
            n = len(seq)

            if n1 == 0 or n0 == 0:
                continue

            # Count runs (a run is a maximal non-empty contiguous sub-sequence of same elements)
            diffs = np.diff(seq)
            runs_count = np.sum(diffs != 0) + 1

            # Expected runs and variance under independence
            mean_runs = ((2.0 * n0 * n1) / n) + 1.0
            variance_runs = (2.0 * n0 * n1 * (2.0 * n0 * n1 - n)) / ((n ** 2) * (n - 1))

            if variance_runs > 0:
                z_stat = (runs_count - mean_runs) / np.sqrt(variance_runs)
                p_value = 2.0 * norm.sf(abs(z_stat))
            else:
                z_stat = 0.0
                p_value = 1.0

            num_results.append({
                "number": num,
                "appearances": int(n1),
                "runs_count": int(runs_count),
                "expected_runs": round(float(mean_runs), 2),
                "z_stat": round(float(z_stat), 3),
                "p_value": round(float(p_value), 5),
                "is_independent": p_value > alpha
            })

        results_df = pd.DataFrame(num_results)

        # Multiple testing correction (Bonferroni threshold)
        bonferroni_alpha = alpha / self.total_numbers
        results_df["bonferroni_reject"] = results_df["p_value"] < bonferroni_alpha

        anomalies = results_df[results_df["bonferroni_reject"]]
        uncorrected_rejections = results_df[~results_df["is_independent"]]

        return {
            "test_name": "Wald-Wolfowitz Runs Test for Serial Independence",
            "null_hypothesis": "Chuỗi xuất hiện của từng số theo thời gian là độc lập (không có trí nhớ)",
            "total_tested_numbers": len(results_df),
            "uncorrected_rejections_count": len(uncorrected_rejections),
            "expected_false_positives_at_alpha_005": round(self.total_numbers * alpha, 1),
            "bonferroni_alpha": round(bonferroni_alpha, 6),
            "bonferroni_rejections_count": len(anomalies),
            "detail_table": results_df,
            "conclusion": (
                f"Không có số nào vi phạm tính độc lập sau khi hiệu chỉnh Bonferroni (ngưỡng {bonferroni_alpha:.5f}). "
                f"Số lượng p < 0.05 chưa hiệu chỉnh là {len(uncorrected_rejections)} "
                f"(hoàn toàn tương thích với kỳ vọng ngẫu nhiên ~{round(self.total_numbers * alpha, 1)} do hiện tượng false discovery)."
                if len(anomalies) == 0 else
                f"Phát hiện {len(anomalies)} số có p-value cực nhỏ vi phạm tính độc lập."
            )
        }

    def chi_square_repeat_test(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Chi-Square test comparing observed consecutive draw repeats with Hypergeometric distribution.
        """
        draw_sets = [set(nums) for nums in self.df["main_numbers"]]
        repeats = [len(draw_sets[i].intersection(draw_sets[i - 1])) for i in range(1, len(draw_sets))]
        repeat_counts = Counter(repeats)
        total_transitions = len(repeats)

        # Group bins: 0, 1, 2, >=3
        obs_grouped = [
            repeat_counts[0],
            repeat_counts[1],
            repeat_counts[2],
            sum(repeat_counts[k] for k in range(3, self.pick_count + 1))
        ]

        theo_p0 = hypergeom.pmf(0, self.total_numbers, self.pick_count, self.pick_count)
        theo_p1 = hypergeom.pmf(1, self.total_numbers, self.pick_count, self.pick_count)
        theo_p2 = hypergeom.pmf(2, self.total_numbers, self.pick_count, self.pick_count)
        theo_p3_plus = 1.0 - (theo_p0 + theo_p1 + theo_p2)

        exp_grouped = [p * total_transitions for p in [theo_p0, theo_p1, theo_p2, theo_p3_plus]]

        chi2_stat = sum((o - e) ** 2 / e for o, e in zip(obs_grouped, exp_grouped))
        df_deg = len(obs_grouped) - 1  # 4 bins - 1 = 3
        p_value = float(chi2.sf(chi2_stat, df_deg))

        is_random = p_value > alpha

        return {
            "test_name": "Consecutive Repeat Hypergeometric Test",
            "null_hypothesis": "Số lượng số lặp lại giữa 2 kỳ liên tiếp tuân theo phân bố Siêu bội (Hypergeometric)",
            "chi2_stat": round(float(chi2_stat), 3),
            "degrees_of_freedom": df_deg,
            "p_value": round(p_value, 5),
            "is_consistent_with_hypergeom": is_random,
            "conclusion": (
                "Chấp nhận H0: Tỷ lệ lặp lại giữa các kỳ liên tiếp hoàn toàn khớp với phân bố ngẫu nhiên lý thuyết."
                if is_random else
                "Bác bỏ H0: Tỷ lệ lặp lại sai lệch có ý nghĩa so với lý thuyết."
            )
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all randomness tests and return full battery report."""
        return {
            "uniformity_test": self.chi_square_uniformity_test(),
            "runs_test": self.runs_test_for_independence(),
            "repeat_test": self.chi_square_repeat_test()
        }

