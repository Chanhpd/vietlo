"""
Visualization Module for Vietlott Data Analysis and Backtesting.
Generates publication-quality charts saved to reports/figures/.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set global aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class VietlottVisualizer:
    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_frequencies(
        self, freq_df: pd.DataFrame, total_draws: int, total_numbers: int, pick_count: int = 6, filename: str = "frequencies.png"
    ) -> Path:
        """Plot frequency of each number compared to theoretical expectation."""
        fig, ax = plt.subplots(figsize=(14, 6))

        df_sorted = freq_df.sort_values("number").reset_index(drop=True)
        expected = total_draws * (pick_count / total_numbers)

        colors = ["#e74c3c" if c >= expected else "#3498db" for c in df_sorted["count"]]
        bars = ax.bar(df_sorted["number"], df_sorted["count"], color=colors, alpha=0.85, width=0.7)

        ax.axhline(expected, color="#2c3e50", linestyle="--", linewidth=1.5, label=f"Kỳ vọng lý thuyết ({expected:.1f} lần)")

        ax.set_title(f"Tần suất xuất hiện các số (Tổng số kỳ: {total_draws:,})", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Số", fontsize=11)
        ax.set_ylabel("Số lần xuất hiện", fontsize=11)
        ax.set_xticks(range(1, total_numbers + 1, 2 if total_numbers > 50 else 1))
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

        # Top 3 most frequent annotations
        top3 = freq_df.head(3)
        for _, row in top3.iterrows():
            ax.annotate(
                f"Top {int(row['rank'])}: Số {int(row['number'])}\n({int(row['count'])} lần)",
                xy=(row["number"], row["count"]),
                xytext=(row["number"], row["count"] + (expected * 0.05)),
                ha="center", fontsize=8, fontweight="bold", color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1)
            )

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_gaps(
        self, gap_df: pd.DataFrame, total_numbers: int, pick_count: int = 6, filename: str = "gaps.png"
    ) -> Path:
        """Plot current gap (draws since last seen) with overdue threshold."""
        fig, ax = plt.subplots(figsize=(14, 6))

        df_sorted = gap_df.sort_values("number").reset_index(drop=True)
        theo_mean = total_numbers / pick_count
        overdue_threshold = theo_mean * 1.5

        colors = ["#e67e22" if g >= overdue_threshold else "#1abc9c" for g in df_sorted["current_gap"]]
        ax.bar(df_sorted["number"], df_sorted["current_gap"], color=colors, alpha=0.85, width=0.7)

        ax.axhline(theo_mean, color="#2980b9", linestyle="--", linewidth=1.5, label=f"Khoảng cách TB lý thuyết ({theo_mean:.1f} kỳ)")
        ax.axhline(overdue_threshold, color="#d35400", linestyle=":", linewidth=1.8, label=f"Ngưỡng số lạnh/overdue ({overdue_threshold:.1f} kỳ)")

        ax.set_title("Số kỳ chưa xuất hiện (Current Gap / Recency)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Số", fontsize=11)
        ax.set_ylabel("Số kỳ chưa về", fontsize=11)
        ax.set_xticks(range(1, total_numbers + 1, 2 if total_numbers > 50 else 1))
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

        # Top overdue numbers annotation
        overdues = gap_df.head(3)
        for _, row in overdues.iterrows():
            ax.annotate(
                f"Lâu nhất: Số {int(row['number'])}\n({int(row['current_gap'])} kỳ)",
                xy=(row["number"], row["current_gap"]),
                xytext=(row["number"], row["current_gap"] + 3),
                ha="center", fontsize=8, fontweight="bold", color="#d35400",
                arrowprops=dict(arrowstyle="->", color="#d35400", lw=1)
            )

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_repeats_vs_theory(self, repeat_data: Dict[str, Any], filename: str = "consecutive_repeats.png") -> Path:
        """Plot observed consecutive repeats vs Hypergeometric theoretical expectation."""
        fig, ax = plt.subplots(figsize=(9, 5))

        df = repeat_data["repeat_table"].head(5)  # 0, 1, 2, 3, 4
        x = np.arange(len(df))
        width = 0.35

        ax.bar(x - width/2, df["observed_pct"], width, label="Thực tế (Empirical %)", color="#3498db", alpha=0.85)
        ax.bar(x + width/2, df["theoretical_pct"], width, label="Lý thuyết (Hypergeometric %)", color="#e74c3c", alpha=0.85)

        ax.set_title("Tỷ lệ số lặp lại giữa 2 kỳ liên tiếp: Thực tế vs Lý thuyết", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("Số lượng số lặp lại từ kỳ trước", fontsize=11)
        ax.set_ylabel("Tỷ lệ (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k} số" for k in df["repeat_count"]])
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

        # Add data labels
        for i in x:
            ax.text(i - width/2, df.loc[i, "observed_pct"] + 0.8, f"{df.loc[i, 'observed_pct']:.1f}%", ha="center", fontsize=8)
            ax.text(i + width/2, df.loc[i, "theoretical_pct"] + 0.8, f"{df.loc[i, 'theoretical_pct']:.1f}%", ha="center", fontsize=8)

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_parity_and_sums(
        self, parity_df: pd.DataFrame, sum_series: pd.Series, filename: str = "parity_and_sums.png"
    ) -> Path:
        """Plot Odd/Even breakdown and Sum distribution in a dual-panel figure."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Parity breakdown
        x = np.arange(len(parity_df))
        width = 0.35
        ax1.bar(x - width/2, parity_df["observed_pct"], width, label="Thực tế (%)", color="#2ecc71", alpha=0.85)
        ax1.bar(x + width/2, parity_df["theoretical_pct"], width, label="Lý thuyết (%)", color="#9b59b6", alpha=0.85)
        ax1.set_title("Phân bố Chẵn / Lẻ (Tỷ lệ Odd-Even)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Cơ cấu (Lẻ - Chẵn)", fontsize=10)
        ax1.set_ylabel("Tỷ lệ (%)", fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(parity_df["ratio_label"], rotation=30)
        ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

        # Panel 2: Sum distribution
        sns.histplot(sum_series, kde=True, ax=ax2, color="#34495e", bins=25, alpha=0.6)
        mean_val = sum_series.mean()
        ax2.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Trung bình: {mean_val:.1f}")
        ax2.set_title("Phân bố Tổng 6 số (Sum Distribution)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Tổng 6 số trong kỳ", fontsize=10)
        ax2.set_ylabel("Tần suất", fontsize=10)
        ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_backtest_cumulative_hits(
        self, backtest_results: Dict[str, Any], filename: str = "backtest_cumulative_hits.png"
    ) -> Path:
        """Plot cumulative number of matching numbers over the test draws."""
        fig, ax = plt.subplots(figsize=(12, 6))

        details = backtest_results["strategy_details"]
        n_draws = backtest_results["test_draws_count"]
        tickets_per_draw = backtest_results["tickets_per_draw"]
        theo_rate = backtest_results["theoretical_expected_hits"] * tickets_per_draw

        x = np.arange(1, n_draws + 1)
        colors = ["#2c3e50", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

        for idx, (s_name, s_data) in enumerate(details.items()):
            color = colors[idx % len(colors)]
            style = "-" if "random" not in s_name.lower() else "--"
            ax.plot(x, s_data["cumulative_hits"], label=s_name, color=color, linestyle=style, linewidth=2)

        # Theoretical line
        ax.plot(x, x * theo_rate, label=f"Kỳ vọng lý thuyết ({theo_rate:.2f} số/kỳ)", color="#7f8c8d", linestyle=":", linewidth=2)

        ax.set_title(f"Hiệu suất tích lũy số lượng số trúng (Backtest {n_draws} kỳ gần nhất)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Số kỳ đã trôi qua", fontsize=11)
        ax.set_ylabel("Tổng số lượt số trúng tích lũy (Cumulative Hits)", fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_backtest_roi_and_profit(
        self, backtest_results: Dict[str, Any], filename: str = "backtest_cumulative_pnl.png"
    ) -> Path:
        """Plot cumulative profit and loss (VND) for each strategy."""
        fig, ax = plt.subplots(figsize=(12, 6))

        details = backtest_results["strategy_details"]
        n_draws = backtest_results["test_draws_count"]
        x = np.arange(1, n_draws + 1)
        colors = ["#2c3e50", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

        for idx, (s_name, s_data) in enumerate(details.items()):
            color = colors[idx % len(colors)]
            # Convert profit to thousands or millions VND
            profits_k = np.array(s_data["cumulative_profit"]) / 1000.0  # in thousand VND
            ax.plot(x, profits_k, label=s_name, color=color, linewidth=2)

        ax.axhline(0, color="#7f8c8d", linestyle="-", linewidth=1.0)

        ax.set_title(f"Lợi nhuận ròng tích lũy (Net Profit/Loss) - Backtest {n_draws} kỳ", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Số kỳ đã trôi qua", fontsize=11)
        ax.set_ylabel("Lợi nhuận ròng tích lũy (Nghìn VNĐ)", fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

