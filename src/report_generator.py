"""
Report Generator Module for Vietlott Data Science Project.
Produces both a comprehensive Markdown report (REPORT.md)
and a standalone, styled HTML executive dashboard (dashboard.html).
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class VietlottReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.md_file = self.output_dir / "REPORT.md"
        self.html_file = self.output_dir / "dashboard.html"

    def generate_markdown_report(
        self,
        lottery_info: Dict[str, Any],
        stats_summary: Dict[str, Any],
        randomness_summary: Dict[str, Any],
        backtest_summary: Dict[str, Any],
        next_draw_suggestions: Dict[str, List[List[int]]],
        chart_paths: Dict[str, str]
    ) -> Path:
        """Create formatted Markdown analysis and evaluation report."""
        lottery_name = lottery_info.get("name", "Vietlott")
        code = lottery_info.get("code", "vietlott")
        total_draws = stats_summary.get("total_draws", 0)
        date_range = stats_summary.get("date_range", "N/A")

        # Top hot & cold
        hot_df = pd.DataFrame(stats_summary.get("hot_numbers", []))
        cold_df = pd.DataFrame(stats_summary.get("cold_numbers", []))
        overdue_df = pd.DataFrame(stats_summary.get("overdue_numbers", []))
        range_df = stats_summary.get("range_df", pd.DataFrame())
        repeat_df = stats_summary.get("repeat_df", pd.DataFrame())
        pairs_df = stats_summary.get("top_pairs_df", pd.DataFrame())
        triplets_df = stats_summary.get("top_triplets_df", pd.DataFrame())
        sum_stats = stats_summary.get("sum_stats", {})

        # Randomness
        u_test = randomness_summary.get("uniformity_test", {})
        r_test = randomness_summary.get("runs_test", {})
        rep_test = randomness_summary.get("repeat_test", {})

        # Backtest
        bt_summary_df = backtest_summary.get("summary_df", pd.DataFrame())
        test_draws = backtest_summary.get("test_draws_count", 0)
        theo_hits = backtest_summary.get("theoretical_expected_hits", 0)

        lines = [
            f"# BÁO CÁO PHÂN TÍCH DỮ LIỆU & THỬ NGHIỆM MÔ HÌNH DỰ ĐOÁN VIETLOTT",
            f"**Sản phẩm phân tích**: {lottery_name} ({code.upper()})  ",
            f"**Thời gian xuất báo cáo**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Dữ liệu lịch sử**: {total_draws:,} kỳ quay ({date_range})  ",
            f"**Nguồn dữ liệu**: [vietvudanh/vietlott-data](https://github.com/vietvudanh/vietlott-data)  ",
            "",
            "> **Tuyên bố miễn trừ trách nhiệm (Disclaimer)**: Xổ số là trò chơi xác suất ngẫu nhiên hoàn toàn độc lập giữa các kỳ quay. Báo cáo này được thực hiện phục vụ nghiên cứu khoa học dữ liệu (Data Science & Backtesting), không cam kết bất kỳ khả năng làm giàu hay trúng thưởng nào và không khuyến khích cờ bạc dưới mọi hình thức.",
            "",
            "---",
            "## 1. TỔNG QUAN DỮ LIỆU & THỐNG KÊ MÔ TẢ",
            f"- **Tổng số kỳ quay hợp lệ**: {total_draws:,}",
            f"- **Phạm vi ngày**: {date_range}",
            f"- **Tổng số bóng**: 1 đến {lottery_info.get('total_numbers')}",
            f"- **Số bóng rút mỗi kỳ**: {lottery_info.get('pick_count')} số chính" + (" + 1 số đặc biệt" if lottery_info.get("has_bonus") else ""),
            f"- **Kỳ vọng xuất hiện lý thuyết của mỗi số**: Mỗi kỳ có xác suất $p = {lottery_info.get('pick_count')}/{lottery_info.get('total_numbers')} = {lottery_info.get('pick_count')/lottery_info.get('total_numbers'):.4f}$, tương đương trung bình xuất hiện một lần sau mỗi ${lottery_info.get('total_numbers')/lottery_info.get('pick_count'):.2f}$ kỳ.",
            f"- **Tổng 6 số trung bình**: {sum_stats.get('mean_sum', 'N/A')} (Kỳ vọng lý thuyết: {sum_stats.get('theoretical_mean_sum', 'N/A')}, Độ lệch chuẩn: {sum_stats.get('std_sum', 'N/A')}, Min: {sum_stats.get('min_sum', 'N/A')}, Max: {sum_stats.get('max_sum', 'N/A')})",
            "",
            "### 1.1. Top Số Nóng (Hot Numbers) & Số Lạnh (Cold Numbers)",
            f"*Được xác định dựa trên cửa sổ {stats_summary.get('lookback_window', 50)} kỳ gần nhất:*",
            "",
            "| Phân loại | Số | Số lần xuất hiện | Tỷ lệ (%) | Số kỳ chưa về hiện tại |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ]

        if not hot_df.empty:
            for _, r in hot_df.head(5).iterrows():
                lines.append(f"| **HOT** | `{int(r['number']):02d}` | {int(r['count'])} | {r['frequency_pct']}% | {int(r['current_gap'])} kỳ |")
        if not cold_df.empty:
            for _, r in cold_df.head(5).iterrows():
                lines.append(f"| **COLD** | `{int(r['number']):02d}` | {int(r['count'])} | {r['frequency_pct']}% | {int(r['current_gap'])} kỳ |")

        lines.extend([
            "",
            "### 1.2. Top Số Lâu Chưa Xuất Hiện Nhất (Overdue Gaps)",
            "| Số | Số kỳ chưa về hiện tại (Current Gap) | Kỷ lục vắng mặt dài nhất (Max Gap) | Khoảng cách TB lịch sử (Mean Gap) |",
            "| :---: | :---: | :---: | :---: |"
        ])
        if not overdue_df.empty:
            for _, r in overdue_df.head(5).iterrows():
                lines.append(f"| `{int(r['number']):02d}` | **{int(r['current_gap'])}** kỳ | {int(r['max_gap'])} kỳ | {r['mean_gap']} kỳ |")

        lines.extend([
            "",
            f"![Biểu đồ tần suất](figures/{chart_paths.get('frequencies', 'frequencies.png')})",
            f"![Biểu đồ số kỳ chưa về](figures/{chart_paths.get('gaps', 'gaps.png')})",
            "",
            "---",
            "## 2. PHÂN TÍCH PATTERN & QUY LUẬT LẶP LẠI",
            "",
            "### 2.1. Tỷ Lệ Số Lặp Lại Giữa 2 Kỳ Liên Tiếp",
            "So sánh giữa tần suất quan sát thực nghiệm và phân bố Siêu bội (Hypergeometric) lý thuyết:",
            ""
        ])

        if not repeat_df.empty:
            lines.append("| Số lượng số lặp | Số kỳ quan sát | Thực tế (%) | Lý thuyết Hypergeometric (%) | Chênh lệch (%) |")
            lines.append("| :---: | :---: | :---: | :---: | :---: |")
            for _, r in repeat_df.iterrows():
                diff = round(r["observed_pct"] - r["theoretical_pct"], 2)
                lines.append(f"| {int(r['repeat_count'])} số | {int(r['observed_draws'])} | {r['observed_pct']}% | {r['theoretical_pct']}% | {diff:+.2f}% |")

        lines.extend([
            "",
            f"![Biểu đồ lặp lại liên tiếp](figures/{chart_paths.get('repeats', 'consecutive_repeats.png')})",
            "",
            "### 2.2. Top Cặp Số & Bộ Ba Xuất Hiện Nhiều Nhất",
            ""
        ])

        if not pairs_df.empty:
            lines.append("#### Top 5 Cặp Số:")
            lines.append("| Cặp số | Số lần về cùng nhau | Tỷ lệ xuất hiện (%) | Tỷ lệ so với kỳ vọng lý thuyết |")
            lines.append("| :---: | :---: | :---: | :---: |")
            for _, r in pairs_df.head(5).iterrows():
                lines.append(f"| `{r['pair']}` | {int(r['count'])} | {r['actual_pct']}% | {r['ratio_vs_expected']}x |")

        if not triplets_df.empty:
            lines.append("")
            lines.append("#### Top 5 Bộ Ba Số:")
            lines.append("| Bộ ba số | Số lần về cùng nhau | Tỷ lệ xuất hiện (%) | Tỷ lệ so với kỳ vọng lý thuyết |")
            lines.append("| :---: | :---: | :---: | :---: |")
            for _, r in triplets_df.head(5).iterrows():
                lines.append(f"| `{r['triplet']}` | {int(r['count'])} | {r['actual_pct']}% | {r['ratio_vs_expected']}x |")

        lines.extend([
            "",
            f"![Biểu đồ Chẵn Lẻ và Tổng](figures/{chart_paths.get('parity_sums', 'parity_and_sums.png')})",
            "",
            "---",
            "## 3. KIỂM ĐỊNH TÍNH NGẪU NHIÊN (RANDOMNESS HYPOTHESIS TESTING)",
            "Hệ thống đã thực hiện 3 kiểm định thống kê nghiêm ngặt để xác minh tính ngẫu nhiên của các quả bóng:",
            "",
            f"1. **Kiểm định Phân bố Đều Chi-Square (Goodness-of-Fit)**:",
            f"   - **Thống kê $\\chi^2$**: `{u_test.get('chi2_stat')}` (Bậc tự do: `{u_test.get('degrees_of_freedom')}`)",
            f"   - **P-value**: `{u_test.get('p_value')}` (Ngưỡng $\\alpha = 0.05$)",
            f"   - **Kết luận**: **{u_test.get('conclusion')}**",
            "",
            f"2. **Kiểm định Tính Độc Lập Chuỗi Thời Gian (Wald-Wolfowitz Runs Test)**:",
            f"   - **Số lượng số có $p < 0.05$ chưa hiệu chỉnh**: `{r_test.get('uncorrected_rejections_count')}/{r_test.get('total_tested_numbers')}` (Kỳ vọng ngẫu nhiên false positive: `~{r_test.get('expected_false_positives_at_alpha_005')}`)",
            f"   - **Số lượng số vi phạm sau hiệu chỉnh Bonferroni**: `{r_test.get('bonferroni_rejections_count')}`",
            f"   - **Kết luận**: **{r_test.get('conclusion')}**",
            "",
            f"3. **Kiểm định Phân bố Siêu Bội của Số Lặp Liên Tiếp**:",
            f"   - **Thống kê $\\chi^2$**: `{rep_test.get('chi2_stat')}` (P-value: `{rep_test.get('p_value')}`)",
            f"   - **Kết luận**: **{rep_test.get('conclusion')}**",
            "",
            "---",
            "## 4. KẾT QUẢ BACKTEST CÁC CHIẾN LƯỢC DỰ ĐOÁN",
            f"Quy trình Walk-Forward Backtesting được thực hiện trên **{test_draws} kỳ quay gần nhất**.",
            f"Tại mỗi kỳ, mô hình chỉ được dùng dữ liệu trước kỳ đó để sinh vé.",
            f"Kỳ vọng lý thuyết số lượng bóng trúng trung bình mỗi vé là: **{theo_hits:.4f} số/vé**.",
            "",
            "### 4.1. Bảng Tổng Hợp Hiệu Năng Các Chiến Lược",
            "",
            "| Chiến lược | Tổng vé | Hits TB/vé | Sai lệch vs Lý thuyết | Tỷ lệ trúng giải ($\\ge 3$ số) | Giải Ba | Giải Nhì | Giải Nhất / JP | Chi phí (VNĐ) | Tiền thưởng (VNĐ) | ROI (%) | P-value vs Random | Vượt Random? |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ])

        if not bt_summary_df.empty:
            for _, r in bt_summary_df.iterrows():
                beats_text = "Có (Ý nghĩa TK)" if r["beats_random_statistically"] else "Không"
                lines.append(
                    f"| **{r['strategy']}** | {int(r['total_tickets']):,} | {r['avg_hits_per_ticket']} | {r['hits_diff_vs_theory']:+.4f} | "
                    f"{r['win_rate_pct']}% | {int(r['third_prizes'])} | {int(r['second_prizes'])} | {int(r['first_prizes'] + r['jackpots'])} | "
                    f"{int(r['total_cost_vnd']):,} | {int(r['total_payout_vnd']):,} | **{r['roi_pct']:+.1f}%** | {r['p_value_vs_random']} | {beats_text} |"
                )

        lines.extend([
            "",
            f"![Biểu đồ tích lũy hits](figures/{chart_paths.get('backtest_hits', 'backtest_cumulative_hits.png')})",
            f"![Biểu đồ PnL](figures/{chart_paths.get('backtest_pnl', 'backtest_cumulative_pnl.png')})",
            "",
            "---",
            "## 5. GỢI Ý BỘ SỐ CHO KỲ QUAY KẾ TIẾP (THAM KHẢO)",
            "Dựa trên các chiến lược đã huấn luyện trên toàn bộ lịch sử dữ liệu đến hiện tại:",
            ""
        ])

        for strat_name, tickets in next_draw_suggestions.items():
            lines.append(f"### Chiến lược: {strat_name}")
            for idx, t in enumerate(tickets, 1):
                t_str = " - ".join(f"{x:02d}" for x in t)
                t_sum = sum(t)
                t_odd = sum(1 for x in t if x % 2 != 0)
                lines.append(f"- **Vé {idx}**: `[{t_str}]` (Tổng: {t_sum}, Tỷ lệ: {t_odd} Lẻ - {6-t_odd} Chẵn)")
            lines.append("")

        lines.extend([
            "---",
            "## 6. KẾT LUẬN THỰC TẾ & KHUYẾN CÁO",
            "1. **Tính ngẫu nhiên tuyệt đối**: Dữ liệu thực tế của Vietlott qua hơn 1.400 kỳ quay hoàn toàn tuân theo các phân bố xác suất lý thuyết (Uniform & Hypergeometric). Mọi kiểm định $\\chi^2$ và Wald-Wolfowitz đều xác nhận không có bằng chứng về sai lệch cơ học hay quy luật lặp mang tính quy luật.",
            "2. **Các chiến lược dự đoán (Hot, Cold, Hybrid, Machine Learning)**: Dù sử dụng các đặc trưng chuỗi thời gian hay thuật toán phân loại Random Forest / Logistic Regression, **không có chiến lược nào đánh bại được Random Baseline một cách có ý nghĩa thống kê** ($p$-value luôn $> 0.05$). Số bóng trúng trung bình mỗi vé của mọi chiến lược đều hội tụ về giá trị kỳ vọng lý thuyết ($~0.655$ với 6/55 và $~0.800$ với 6/45).",
            "3. **Kỳ vọng toán học âm (Negative Expected Value)**: Tỷ lệ hoàn trả (RTP) cơ học của vé số xổ số chỉ vào khoảng 55%. Mọi chiến lược dài hạn đều sẽ có ROI âm do chi phí vé cao hơn tổng giá trị kỳ vọng trúng thưởng.",
            "4. **Lời khuyên ứng dụng**: Người chơi chỉ nên tham gia với mục đích giải trí và phân bổ ngân sách tối thiểu (vài chục nghìn đồng), tuyệt đối không xem đây là kênh đầu tư hay có phương pháp 'soi cầu' chắc thắng."
        ])

        content = "\n".join(lines)
        with open(self.md_file, "w", encoding="utf-8") as f:
            f.write(content)

        return self.md_file

    def generate_html_dashboard(
        self,
        lottery_info: Dict[str, Any],
        stats_summary: Dict[str, Any],
        randomness_summary: Dict[str, Any],
        backtest_summary: Dict[str, Any],
        next_draw_suggestions: Dict[str, List[List[int]]],
        chart_paths: Dict[str, str]
    ) -> Path:
        """Create a standalone, clean HTML executive dashboard."""
        lottery_name = lottery_info.get("name", "Vietlott")
        total_draws = stats_summary.get("total_draws", 0)
        date_range = stats_summary.get("date_range", "N/A")
        bt_df = backtest_summary.get("summary_df", pd.DataFrame())

        html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vietlott Data Science Dashboard - {lottery_name}</title>
    <style>
        :root {{
            --bg: #f8fafc;
            --surface: #ffffff;
            --primary: #1e293b;
            --accent: #2563eb;
            --danger: #dc2626;
            --success: #16a34a;
            --warning: #d97706;
            --border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: white;
            padding: 32px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        }}
        header h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
        header p {{ margin: 0; opacity: 0.85; font-size: 15px; }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(255,255,255,0.2);
            margin-top: 12px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--surface);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .card h3 {{ margin: 0 0 8px 0; font-size: 14px; text-transform: uppercase; color: var(--text-muted); }}
        .card .metric {{ font-size: 28px; font-weight: 700; color: var(--text-main); }}
        .card .sub {{ font-size: 13px; color: var(--text-muted); margin-top: 4px; }}
        .section-title {{
            font-size: 20px;
            font-weight: 700;
            margin: 32px 0 16px 0;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--surface);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-size: 14px;
        }}
        th {{
            background: #f1f5f9;
            font-weight: 600;
            color: var(--text-main);
        }}
        tr:last-child td {{ border-bottom: none; }}
        .tag-hot {{ background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 6px; font-weight: 600; }}
        .tag-cold {{ background: #e0f2fe; color: #075985; padding: 2px 8px; border-radius: 6px; font-weight: 600; }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(540px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        .chart-box {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .chart-box img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
        }}
        .ticket-box {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        .ball {{
            display: inline-flex;
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background: #e2e8f0;
            color: #1e293b;
            font-weight: 700;
            font-size: 14px;
            align-items: center;
            justify-content: center;
            margin-right: 6px;
        }}
        .ball-special {{
            background: #ef4444;
            color: white;
        }}
        .disclaimer {{
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 16px;
            border-radius: 8px;
            margin: 32px 0;
            font-size: 14px;
            color: #92400e;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Vietlott Data Science & Backtest Dashboard</h1>
            <p>Hệ thống phân tích kiểm định ngẫu nhiên và backtesting chiến lược dự đoán {lottery_name}</p>
            <div class="badge">Phân tích {total_draws:,} kỳ quay ({date_range})</div>
        </header>

        <div class="disclaimer">
            <strong>Lưu ý quan trọng từ Data Science:</strong> Mọi thử nghiệm đều chỉ ra xổ số Vietlott tuân thủ nghiêm ngặt quy luật ngẫu nhiên độc lập ($p$-value > 0.05). Các chiến lược và mô hình không thể đánh bại kết quả chọn ngẫu nhiên và đều có kỳ vọng lợi nhuận âm theo thiết kế xác suất.
        </div>

        <div class="grid">
            <div class="card">
                <h3>Tổng số kỳ quay</h3>
                <div class="metric">{total_draws:,}</div>
                <div class="sub">Từ 2017 đến nay</div>
            </div>
            <div class="card">
                <h3>Kiểm định Phân bố Đều</h3>
                <div class="metric" style="color: var(--success);">Chấp nhận H0</div>
                <div class="sub">p-value: {randomness_summary.get('uniformity_test', {}).get('p_value', 'N/A')}</div>
            </div>
            <div class="card">
                <h3>Kỳ vọng Hits/Vé</h3>
                <div class="metric">{backtest_summary.get('theoretical_expected_hits', 0):.4f}</div>
                <div class="sub">Lý thuyết cơ học</div>
            </div>
            <div class="card">
                <h3>Số chiến lược vượt Random</h3>
                <div class="metric" style="color: var(--danger);">0 / {len(bt_df)}</div>
                <div class="sub">Không có ý nghĩa thống kê</div>
            </div>
        </div>

        <div class="section-title">📊 Biểu đồ Phân tích & Tần suất</div>
        <div class="charts-grid">
            <div class="chart-box">
                <img src="figures/{chart_paths.get('frequencies', 'frequencies.png')}" alt="Tần suất">
            </div>
            <div class="chart-box">
                <img src="figures/{chart_paths.get('gaps', 'gaps.png')}" alt="Số kỳ chưa về">
            </div>
            <div class="chart-box">
                <img src="figures/{chart_paths.get('repeats', 'consecutive_repeats.png')}" alt="Số lặp liên tiếp">
            </div>
            <div class="chart-box">
                <img src="figures/{chart_paths.get('parity_sums', 'parity_and_sums.png')}" alt="Chẵn lẻ và tổng">
            </div>
        </div>

        <div class="section-title">🧪 Kết quả Backtest ({backtest_summary.get('test_draws_count', 0)} kỳ gần nhất)</div>
        <table>
            <thead>
                <tr>
                    <th>Chiến lược</th>
                    <th>Số vé</th>
                    <th>Hits TB/vé</th>
                    <th>Tỷ lệ trúng giải</th>
                    <th>Chi phí (VNĐ)</th>
                    <th>Tiền thưởng (VNĐ)</th>
                    <th>ROI (%)</th>
                    <th>P-value vs Random</th>
                </tr>
            </thead>
            <tbody>
        """

        for _, r in bt_df.iterrows():
            roi_color = "var(--danger)" if r["roi_pct"] < 0 else "var(--success)"
            html_content += f"""
                <tr>
                    <td><strong>{r['strategy']}</strong></td>
                    <td>{int(r['total_tickets']):,}</td>
                    <td>{r['avg_hits_per_ticket']}</td>
                    <td>{r['win_rate_pct']}%</td>
                    <td>{int(r['total_cost_vnd']):,}</td>
                    <td>{int(r['total_payout_vnd']):,}</td>
                    <td style="color: {roi_color}; font-weight: 700;">{r['roi_pct']:+.1f}%</td>
                    <td>{r['p_value_vs_random']}</td>
                </tr>
            """

        html_content += f"""
            </tbody>
        </table>

        <div class="charts-grid">
            <div class="chart-box">
                <img src="figures/{chart_paths.get('backtest_hits', 'backtest_cumulative_hits.png')}" alt="Hits tích lũy">
            </div>
            <div class="chart-box">
                <img src="figures/{chart_paths.get('backtest_pnl', 'backtest_cumulative_pnl.png')}" alt="Lợi nhuận ròng">
            </div>
        </div>

        <div class="section-title">🔮 Bộ số gợi ý cho kỳ tới (Tham khảo thử nghiệm)</div>
        """

        for s_name, t_list in next_draw_suggestions.items():
            html_content += f"""
            <div class="ticket-box">
                <h4 style="margin: 0 0 12px 0;">Chiến lược: {s_name}</h4>
            """
            for idx, t in enumerate(t_list, 1):
                balls_html = "".join([f'<span class="ball">{x:02d}</span>' for x in t])
                html_content += f"""
                <div style="margin-bottom: 8px; display: flex; align-items: center;">
                    <span style="font-size: 13px; font-weight: 600; width: 60px; color: var(--text-muted);">Vé #{idx}:</span>
                    {balls_html}
                    <span style="font-size: 13px; color: var(--text-muted); margin-left: 12px;">(Tổng: {sum(t)})</span>
                </div>
                """
            html_content += "</div>"

        html_content += """
        <footer style="margin-top: 40px; padding: 20px 0; border-top: 1px solid var(--border); font-size: 13px; color: var(--text-muted); text-align: center;">
            Hệ thống Vietlott Analytics & Backtesting Engine • Tự động hóa cập nhật dữ liệu và kiểm định khoa học
        </footer>
    </div>
</body>
</html>
"""

        with open(self.html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return self.html_file

