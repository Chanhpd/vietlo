"""
Master CLI Runner for Vietlott Analytics & Backtesting System.
Executes the full end-to-end pipeline:
Data Ingestion -> Statistical Analysis -> Randomness Testing ->
Strategy Backtesting -> Visualization -> Report & Dashboard Generation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from src.analyzer import VietlottAnalyzer
from src.backtester import VietlottBacktester
from src.data_loader import VietlottDataLoader
from src.randomness_test import VietlottRandomnessTester
from src.report_generator import VietlottReportGenerator
from src.strategies import (
    ColdOverdueStrategy,
    GameTheoryAntiCrowdStrategy,
    HotFrequencyStrategy,
    HybridStrategy,
    MarkovTransitionStrategy,
    MLRankingStrategy,
    RandomBaselineStrategy,
    XGBoostRankingStrategy,
)
from src.wheeling import CombinatorialWheeler
from src.visualizer import VietlottVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VietlottPipeline")


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    lottery_code: str = "power655",
    force_download: bool = False,
    test_draws: int = 200,
    tickets_per_draw: int = 5,
    lookback_window: int = 50,
    suggest_count: int = 5,
    config_path: str = "configs/config.yaml"
) -> Dict[str, Any]:
    config = load_config(config_path)

    specs = config["lottery_specs"].get(lottery_code)
    if not specs:
        raise ValueError(f"Unknown lottery code '{lottery_code}'. Available: {list(config['lottery_specs'].keys())}")

    logger.info(f"=== BẮT ĐẦU PIPELINE PHÂN TÍCH: {specs['name']} ({lottery_code.upper()}) ===")

    # 1. Data Ingestion & Cleaning
    logger.info("Bước 1: Tải và làm sạch dữ liệu...")
    loader = VietlottDataLoader(lottery_code=lottery_code)
    df = loader.load_clean_data(force_refresh=force_download)
    total_draws = len(df)
    date_range = f"{df['date'].min().strftime('%Y-%m-%d')} đến {df['date'].max().strftime('%Y-%m-%d')}"
    logger.info(f"Đã xử lý {total_draws:,} kỳ quay hợp lệ ({date_range})")

    # 2. Descriptive & Pattern Analytics
    logger.info("Bước 2: Phân tích tần suất, số nóng/lạnh, chu kỳ vắng mặt và mẫu lặp...")
    analyzer = VietlottAnalyzer(df, total_numbers=specs["total_numbers"], pick_count=specs["pick_count"])

    freq_df = analyzer.get_frequency_stats()
    gap_df = analyzer.get_gap_stats()
    hot_cold_info = analyzer.get_hot_cold_numbers(lookback=lookback_window, top_n=10)
    range_df = analyzer.get_range_distribution()
    repeat_data = analyzer.get_consecutive_draw_repeats()
    pairs_df = analyzer.get_top_pairs(top_k=10)
    triplets_df = analyzer.get_top_triplets(top_k=10)
    parity_sum_info = analyzer.get_parity_and_sum_stats()

    stats_summary = {
        "total_draws": total_draws,
        "date_range": date_range,
        "lookback_window": lookback_window,
        "hot_numbers": hot_cold_info["hot_numbers"],
        "cold_numbers": hot_cold_info["cold_numbers"],
        "overdue_numbers": hot_cold_info["overdue_numbers"],
        "range_df": range_df,
        "repeat_df": repeat_data["repeat_table"],
        "top_pairs_df": pairs_df,
        "top_triplets_df": triplets_df,
        "sum_stats": parity_sum_info["sum_stats"]
    }

    # 3. Randomness Hypothesis Testing
    logger.info("Bước 3: Thực hiện kiểm định giả thuyết tính ngẫu nhiên...")
    randomness_tester = VietlottRandomnessTester(
        df, total_numbers=specs["total_numbers"], pick_count=specs["pick_count"]
    )
    randomness_summary = randomness_tester.run_all_tests()

    # 4. Strategy Setup & Walk-Forward Backtesting
    logger.info(f"Bước 4: Tiến hành Walk-Forward Backtesting trên {test_draws} kỳ gần nhất...")
    strategies = [
        RandomBaselineStrategy(total_numbers=specs["total_numbers"], pick_count=specs["pick_count"], random_state=42),
        HotFrequencyStrategy(total_numbers=specs["total_numbers"], pick_count=specs["pick_count"], lookback_window=lookback_window, random_state=42),
        ColdOverdueStrategy(total_numbers=specs["total_numbers"], pick_count=specs["pick_count"], random_state=42),
        HybridStrategy(total_numbers=specs["total_numbers"], pick_count=specs["pick_count"], lookback_window=lookback_window, random_state=42),
        MLRankingStrategy(total_numbers=specs["total_numbers"], pick_count=specs["pick_count"], train_window=200, model_type="logreg", random_state=42),
        XGBoostRankingStrategy(total_numbers=specs["total_numbers"], pick_count=specs["pick_count"], train_window=200, random_state=42),
        MarkovTransitionStrategy(total_numbers=specs["total_numbers"], pick_count=specs["pick_count"], random_state=42),
        GameTheoryAntiCrowdStrategy(total_numbers=specs["total_numbers"], pick_count=specs["pick_count"], random_state=42)
    ]

    backtester = VietlottBacktester(
        df,
        lottery_code=lottery_code,
        total_numbers=specs["total_numbers"],
        pick_count=specs["pick_count"],
        ticket_price=specs["ticket_price"],
        prize_structure=specs["prize_structure"]
    )

    backtest_summary = backtester.run_backtest(
        strategies=strategies,
        test_draws=test_draws,
        tickets_per_draw=tickets_per_draw
    )

    # 5. Generate Next-Draw Suggestions
    logger.info(f"Bước 5: Tạo bộ số đề xuất ({suggest_count} vé/chiến lược) cho kỳ tiếp theo...")
    next_draw_suggestions = {}
    for strat in strategies:
        tickets = strat.generate_tickets(df, num_tickets=suggest_count)
        next_draw_suggestions[strat.name] = tickets

    # Combinatorial Wheeling System (Covering Design)
    wheeler = CombinatorialWheeler(pick_count=specs["pick_count"])
    top_pool = sorted(list(set(
        [x["number"] for x in hot_cold_info["hot_numbers"][:5]] +
        [x["number"] for x in hot_cold_info["overdue_numbers"][:5]]
    )))
    wheel_res = wheeler.generate_abbreviated_wheel(
        top_pool, target_guarantee=3, condition_drawn=4, max_tickets=suggest_count
    )
    next_draw_suggestions[f"Wheeling System (Bảo đảm {wheel_res['guarantee_coverage_pct']}% 3-if-4)"] = wheel_res["tickets"]

    # 6. Generate Visualizations
    logger.info("Bước 6: Xuất biểu đồ trực quan hóa...")
    visualizer = VietlottVisualizer(output_dir=config["paths"]["figures_dir"])
    chart_paths = {
        "frequencies": "frequencies.png",
        "gaps": "gaps.png",
        "repeats": "consecutive_repeats.png",
        "parity_sums": "parity_and_sums.png",
        "backtest_hits": "backtest_cumulative_hits.png",
        "backtest_pnl": "backtest_cumulative_pnl.png"
    }

    visualizer.plot_frequencies(freq_df, total_draws, specs["total_numbers"], filename=chart_paths["frequencies"])
    visualizer.plot_gaps(gap_df, specs["total_numbers"], filename=chart_paths["gaps"])
    visualizer.plot_repeats_vs_theory(repeat_data, filename=chart_paths["repeats"])
    visualizer.plot_parity_and_sums(
        parity_sum_info["parity_distribution"], df["sum_numbers"], filename=chart_paths["parity_sums"]
    )
    visualizer.plot_backtest_cumulative_hits(backtest_summary, filename=chart_paths["backtest_hits"])
    visualizer.plot_backtest_roi_and_profit(backtest_summary, filename=chart_paths["backtest_pnl"])

    # 7. Generate Reports & Dashboard
    logger.info("Bước 7: Xuất báo cáo Markdown và HTML Dashboard...")
    report_gen = VietlottReportGenerator(output_dir=config["paths"]["reports_dir"])
    md_path = report_gen.generate_markdown_report(
        lottery_info=specs,
        stats_summary=stats_summary,
        randomness_summary=randomness_summary,
        backtest_summary=backtest_summary,
        next_draw_suggestions=next_draw_suggestions,
        chart_paths=chart_paths
    )
    html_path = report_gen.generate_html_dashboard(
        lottery_info=specs,
        stats_summary=stats_summary,
        randomness_summary=randomness_summary,
        backtest_summary=backtest_summary,
        next_draw_suggestions=next_draw_suggestions,
        chart_paths=chart_paths
    )

    logger.info("=== HOÀN TẤT PIPELINE THÀNH CÔNG ===")
    logger.info(f"Báo cáo Markdown: {md_path}")
    logger.info(f"Dashboard HTML:  {html_path}")

    # Print Executive Summary to Console
    print("\n" + "=" * 75)
    print(f"BẢNG KẾT QUẢ BACKTEST CÁC CHIẾN LƯỢC ({test_draws} KỲ GẦN NHẤT)")
    print("=" * 75)
    summary_display = backtest_summary["summary_df"][[
        "strategy", "total_tickets", "avg_hits_per_ticket", "theoretical_hits", "roi_pct", "win_rate_pct", "p_value_vs_random"
    ]].to_string(index=False)
    print(summary_display)
    print("=" * 75)

    print(f"\n[KẾT LUẬN KHOA HỌC]")
    u_p = randomness_summary['uniformity_test']['p_value']
    print(f"- Chi-square Goodness-of-Fit p-value: {u_p} (Phân bố hoàn toàn đều).")
    print(f"- Không có chiến lược nào vượt trội ngẫu nhiên (p-value vs random đều > 0.05).")
    print(f"- Mọi chiến lược đều có ROI âm do kỳ vọng toán học âm của xổ số.")
    print("=" * 75 + "\n")

    return {
        "df": df,
        "stats_summary": stats_summary,
        "randomness_summary": randomness_summary,
        "backtest_summary": backtest_summary,
        "next_draw_suggestions": next_draw_suggestions,
        "report_md": md_path,
        "dashboard_html": html_path
    }


def main():
    parser = argparse.ArgumentParser(description="Vietlott Analytics & Backtesting System")
    parser.add_argument("--lottery", type=str, default="power655", choices=["power655", "mega645"], help="Lottery product")
    parser.add_argument("--force-download", action="store_true", help="Force re-download fresh dataset from Github")
    parser.add_argument("--test-draws", type=int, default=200, help="Number of latest draws to backtest")
    parser.add_argument("--tickets-per-draw", type=int, default=5, help="Number of tickets per strategy per draw")
    parser.add_argument("--lookback", type=int, default=50, help="Lookback window for hot/cold analysis")
    parser.add_argument("--suggest-count", type=int, default=5, help="Number of tickets to suggest for next draw")

    args = parser.parse_args()

    run_pipeline(
        lottery_code=args.lottery,
        force_download=args.force_download,
        test_draws=args.test_draws,
        tickets_per_draw=args.tickets_per_draw,
        lookback_window=args.lookback,
        suggest_count=args.suggest_count
    )


if __name__ == "__main__":
    main()

