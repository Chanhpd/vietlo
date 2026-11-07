"""
Streamlit Web Application & Live Monitoring Dashboard for Vietlott Analytics.
Deployable to Streamlit Community Cloud, HuggingFace Spaces, or Local server.
Features:
1. Live Daily Evaluation & Drift Monitoring
2. Positive Expected Value (+EV) Break-Even Calculator
3. Combinatorial Wheeling System Generator
4. Comprehensive Historical Analytics & Predictions
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout and style
st.set_page_config(
    page_title="Vietlott Data Science & Live EV Engine",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.analyzer import VietlottAnalyzer
from src.data_loader import VietlottDataLoader
from src.ev_calculator import VietlottEVCalculator
from src.randomness_test import VietlottRandomnessTester
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
from src.tracker import DailyModelTracker
from src.wheeling import CombinatorialWheeler

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .ball-badge {
        display: inline-flex;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background-color: #1e293b;
        color: white;
        font-weight: 700;
        font-size: 14px;
        align-items: center;
        justify-content: center;
        margin-right: 6px;
    }
    .ball-special {
        background-color: #ef4444 !important;
    }
    .ev-banner-pos {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 16px;
        border-radius: 6px;
        color: #065f46;
        margin-bottom: 16px;
    }
    .ev-banner-neg {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 16px;
        border-radius: 6px;
        color: #991b1b;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar controls
st.sidebar.title("⚙️ Cấu hình hệ thống")
lottery_choice = st.sidebar.selectbox("Chọn sản phẩm Vietlott:", ["Power 6/55", "Mega 6/45"])
lottery_code = "power655" if lottery_choice == "Power 6/55" else "mega645"
total_numbers = 55 if lottery_code == "power655" else 45

force_refresh = st.sidebar.button("🔄 Cập nhật dữ liệu từ GitHub")


@st.cache_data(ttl=600)
def load_lottery_data(code: str, refresh: bool):
    loader = VietlottDataLoader(code)
    return loader.load_clean_data(force_refresh=refresh)


df = load_lottery_data(lottery_code, force_refresh)
latest_draw = df.iloc[-1]
latest_id = latest_draw["id"]
latest_date = latest_draw["date"].date()
latest_main = latest_draw["main_numbers"]
latest_bonus = latest_draw["bonus_number"]

# Header
st.title(f"🎰 Vietlott Live Analytics & Decision Engine: {lottery_choice}")
st.caption(f"Cập nhật mới nhất: Kỳ quay **#{latest_id}** ngày **{latest_date}** ({latest_draw['day_of_week']})")

# Main Navigation Tabs
tab_ev, tab_tracker, tab_wheel, tab_analytics = st.tabs([
    "💰 1. Phương pháp tính có lời (+EV)",
    "🎯 2. Giám sát độ chính xác liên tục",
    "🛡️ 3. Phủ tổ hợp (Wheeling 3-if-4)",
    "📊 4. Thống kê & Phân tích chu kỳ"
])

# ==========================================
# TAB 1: EXPECTED VALUE & BREAK-EVEN CALCULATOR
# ==========================================
with tab_ev:
    st.subheader("💡 Phương pháp đầu tư có lời toán học: Positive Expected Value (+EV)")
    st.markdown("""
    > **Nguyên tắc cốt lõi (Stefan Mandel & MIT Lottery Syndicate)**:  
    > Trong xổ số, **tuyệt đối không bao giờ mua mỗi ngày**, vì kỳ vọng toán học cơ học là âm (-45%).  
    > **Phương pháp duy nhất có lời toán học**: Chỉ giải ngân khi giải Jackpot tích lũy (Rollover) vượt qua **Điểm hòa vốn (Break-even Threshold)** khiến giá trị kỳ vọng của 1 vé 10.000đ **vượt qua 10.000đ (+EV)**.
    """)

    col_inp1, col_inp2 = st.columns(2)
    with col_inp1:
        default_jp = 100.0 if lottery_code == "power655" else 40.0
        jackpot_input = st.number_input(
            f"Nhập giá trị Jackpot 1 hiện tại (Tỷ VNĐ):",
            min_value=12.0 if lottery_code == "mega645" else 30.0,
            max_value=600.0,
            value=default_jp,
            step=5.0
        )
    with col_inp2:
        ticket_sales = st.slider("Ước lượng số vé người khác mua (ảnh hưởng xác suất chia giải):", 500000, 4000000, 1500000, 100000)

    ev_calc = VietlottEVCalculator(lottery_code=lottery_code)
    ev_res = ev_calc.calculate_expected_value(jackpot_input * 1e9, estimated_tickets_sold=ticket_sales)

    # Display EV status banner
    if ev_res["is_positive_ev"]:
        st.markdown(f"""
        <div class="ev-banner-pos">
            <h4>✅ KHUYẾN NGHỊ: CÓ THỂ MUA (KỲ VỌNG TOÁN HỌC DƯƠNG +EV)</h4>
            Giá trị kỳ vọng của 1 vé 10.000đ hiện tại là <strong>{ev_res['total_expected_value_vnd']:,.0f} VNĐ</strong> 
            (Lợi nhuận kỳ vọng <strong>+{ev_res['net_expected_profit_vnd']:,.0f} VNĐ/vé</strong>). 
            Tỷ suất sinh lời kỳ vọng: <strong>{ev_res['expected_return_pct']}%</strong>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ev-banner-neg">
            <h4>⛔ KHUYẾN NGHỊ: TUYỆT ĐỐI KHÔNG MUA (KỲ VỌNG ÂM -EV)</h4>
            Giá trị kỳ vọng của 1 vé 10.000đ hiện chỉ đạt <strong>{ev_res['total_expected_value_vnd']:,.0f} VNĐ</strong> 
            (Mỗi vé mua vào chắc chắn chịu lỗ kỳ vọng <strong>{ev_res['net_expected_profit_vnd']:,.0f} VNĐ</strong>).<br>
            👉 <em>Điểm hòa vốn Jackpot cần đạt: <strong>≥ {ev_res['break_even_jackpot_1_billion']} Tỷ VNĐ</strong> thì mới bắt đầu có lời toán học.</em>
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá vé niêm yết", "10,000 VNĐ")
    c2.metric("Giá trị kỳ vọng 1 vé", f"{ev_res['total_expected_value_vnd']:,.0f} VNĐ")
    c3.metric("Điểm hòa vốn Jackpot 1", f"≥ {ev_res['break_even_jackpot_1_billion']} Tỷ VNĐ")
    c4.metric("Kỳ vọng giải phụ cố định", f"{ev_res['fixed_prizes_ev']:,.0f} VNĐ")


# ==========================================
# TAB 2: LIVE ACCURACY TRACKING & DRIFT MONITOR
# ==========================================
with tab_tracker:
    st.subheader("🎯 Kiểm tra độ chính xác liên tục (Live Ground-Truth Evaluation)")
    st.markdown("""
    Mỗi ngày trước giờ quay, hệ thống tự động chốt dự đoán vào sổ cái kiểm toán (`pending_predictions.json`). 
    Ngay sau khi có kết quả lúc 18h30, hệ thống đối soát trực tiếp để đo lường: **Mô hình có thực sự duy trì độ chính xác hay rơi về mốc ngẫu nhiên?**
    """)

    tracker = DailyModelTracker(lottery_code)
    eval_res = tracker.evaluate_latest_draw()

    st.write(f"**Kết quả kỳ quay vừa mở thưởng (#{eval_res['latest_draw_id']})**:")
    balls_html = "".join([f'<span class="ball-badge">{x:02d}</span>' for x in eval_res['actual_numbers']])
    if lottery_code == "power655" and latest_bonus:
        balls_html += f'<span class="ball-badge ball-special">{int(latest_bonus):02d}</span> (Số đặc biệt)'
    st.markdown(balls_html, unsafe_allow_html=True)

    summary_df = tracker.get_performance_summary()
    if not summary_df.empty:
        st.subheader("📈 Bảng theo dõi hiệu năng thực chiến (Out-of-sample Tracker)")
        st.dataframe(summary_df, width="stretch")

        theo_rate = 0.655 if lottery_code == "power655" else 0.800
        best_model = summary_df.sort_values("avg_hits", ascending=False).iloc[0]
        st.info(
            f"Chiến lược có Hits TB cao nhất gần đây: **{best_model['strategy']}** "
            f"({best_model['avg_hits']} bóng/vé so với mốc ngẫu nhiên {theo_rate})."
        )
    else:
        st.info(f"Đã lưu bản ghi dự đoán cho kỳ kế tiếp **#{eval_res['new_prediction_for_draw']}**. Kết quả đối soát sẽ xuất hiện ngay sau kỳ quay tiếp theo!")

    st.write("---")
    st.subheader(f"🔮 Dự đoán đã lưu cho kỳ tiếp theo (#{eval_res['new_prediction_for_draw']}):")
    for s_name, tickets in eval_res["fresh_predictions"].items():
        with st.expander(f"Chiến lược: {s_name} ({len(tickets)} vé)"):
            for idx, t in enumerate(tickets, 1):
                b_html = "".join([f'<span class="ball-badge">{x:02d}</span>' for x in t])
                st.markdown(f"**Vé #{idx}**: {b_html} *(Tổng: {sum(t)})*", unsafe_allow_html=True)


# ==========================================
# TAB 3: COMBINATORIAL WHEELING SYSTEM
# ==========================================
with tab_wheel:
    st.subheader("🛡️ Hệ thống Phủ Tổ Hợp Rút Gọn (Abbreviated Wheeling System)")
    st.markdown("""
    **Cách tiết kiệm chi phí thông minh nhất**: Thay vì mua bao 10 tốn **2.100.000 VNĐ** (210 vé), thuật toán 
    **Covering Design** nén thành **4 đến 5 vé** (40.000đ - 50.000đ) nhưng **bảo đảm 100%**: 
    *Nếu 4 trong số các số bạn chọn xuất hiện, bạn chắc chắn 100% trúng tối thiểu 1 giải thưởng (3-if-4 guarantee).*
    """)

    default_pool = "08, 15, 20, 22, 31, 32, 34, 41, 47, 52" if lottery_code == "power655" else "01, 08, 10, 19, 26, 37, 38, 39, 43, 45"
    pool_str = st.text_input("Nhập danh sách 8 đến 12 số tiềm năng (cách nhau bởi dấu phẩy):", default_pool)

    try:
        user_pool = [int(x.strip()) for x in pool_str.split(",") if x.strip().isdigit()]
        user_pool = sorted(list(set([x for x in user_pool if 1 <= x <= total_numbers])))
    except Exception:
        user_pool = []

    if len(user_pool) < 6:
        st.warning("Vui lòng nhập tối thiểu 6 số hợp lệ.")
    else:
        st.write(f"Tập số của bạn ({len(user_pool)} số): `{user_pool}`")
        wheeler = CombinatorialWheeler(pick_count=6)
        wheel_out = wheeler.generate_abbreviated_wheel(user_pool, target_guarantee=3, condition_drawn=4, max_tickets=6)

        st.success(
            f"✅ Đã tạo thành công **{wheel_out['ticket_count']} vé tối ưu** "
            f"với độ phủ bảo đảm: **{wheel_out['guarantee_coverage_pct']}%**!"
        )

        for i, t in enumerate(wheel_out["tickets"], 1):
            t_html = "".join([f'<span class="ball-badge">{x:02d}</span>' for x in t])
            st.markdown(f"**Vé #{i}**: {t_html} *(Tổng: {sum(t)})*", unsafe_allow_html=True)


# ==========================================
# TAB 4: HISTORICAL ANALYTICS & DRIFT
# ==========================================
with tab_analytics:
    st.subheader(f"📊 Thống kê chu kỳ & Phân tích chuyên sâu: {lottery_choice}")
    analyzer = VietlottAnalyzer(df, total_numbers=total_numbers, pick_count=6)

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("🔥 **Top số nóng (Xuất hiện nhiều nhất 50 kỳ qua)**:")
        hot_df = pd.DataFrame(analyzer.get_hot_cold_numbers(lookback=50)["hot_numbers"]).head(6)
        st.dataframe(hot_df, width="stretch")

    with col_b:
        st.write("❄️ **Top số câm (Vắng bóng lâu nhất hiện tại)**:")
        cold_df = pd.DataFrame(analyzer.get_hot_cold_numbers(lookback=50)["overdue_numbers"]).head(6)
        st.dataframe(cold_df, width="stretch")

    # Display pre-generated figures if available
    st.write("---")
    st.subheader("📈 Biểu đồ trực quan hóa dữ liệu")
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        if os.path.exists("reports/figures/frequencies.png"):
            st.image("reports/figures/frequencies.png", caption="Tần suất xuất hiện của các số")
        if os.path.exists("reports/figures/consecutive_repeats.png"):
            st.image("reports/figures/consecutive_repeats.png", caption="Tỷ lệ số lặp lại giữa 2 kỳ liên tiếp")

    with fig_col2:
        if os.path.exists("reports/figures/gaps.png"):
            st.image("reports/figures/gaps.png", caption="Số kỳ chưa về hiện tại (Gaps)")
        if os.path.exists("reports/figures/backtest_cumulative_hits.png"):
            st.image("reports/figures/backtest_cumulative_hits.png", caption="Tích lũy số bóng trúng qua backtest")
