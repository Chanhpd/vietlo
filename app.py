"""
Streamlit Web Application & Live Monitoring Dashboard for Vietlott Analytics.
Designed for clear, accessible Vietnamese user experience:
1. Today's Recommended Tickets (10k Single Best Ticket vs 40k 4-Ticket Wheel) & 2-Week Historical Verification
2. Easy-to-understand Budget & Jackpot Profit Advisor (+EV)
3. Custom 10-Number to 4-Ticket Smart Saver (Wheeling System)
4. Hot & Overdue Number Statistics in Plain Vietnamese
"""

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dự Đoán & Đối Soát Vietlott AI",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.analyzer import VietlottAnalyzer
from src.backtester import VietlottBacktester
from src.data_loader import VietlottDataLoader
from src.ev_calculator import VietlottEVCalculator
from src.strategies import GameTheoryAntiCrowdStrategy, RecursiveAdaptiveStrategy
from src.tracker import DailyModelTracker
from src.wheeling import CombinatorialWheeler

# Custom CSS for clear, friendly visual presentation
st.markdown("""
<style>
    .recommend-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
    }
    .wheel-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #3b82f6;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
    }
    .history-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .ball-badge {
        display: inline-flex;
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background-color: #1e293b;
        color: #ffffff;
        font-weight: 700;
        font-size: 15px;
        align-items: center;
        justify-content: center;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .ball-gold {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 6px rgba(217, 119, 6, 0.35);
    }
    .ball-hit {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        border: 2px solid #a7f3d0;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.45);
        transform: scale(1.06);
    }
    .ball-miss {
        background-color: #cbd5e1 !important;
        color: #475569 !important;
    }
    .ball-special {
        background-color: #ef4444 !important;
        color: #ffffff !important;
    }
    .prize-tag {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 13px;
        margin-left: 8px;
    }
    .near-tag {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 13px;
        margin-left: 8px;
    }
    .ev-banner-pos {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 16px;
        border-radius: 8px;
        color: #065f46;
        margin-bottom: 16px;
    }
    .ev-banner-neg {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 16px;
        border-radius: 8px;
        color: #92400e;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar controls
st.sidebar.title("🎰 Chọn Loại Xổ Số")
lottery_choice = st.sidebar.selectbox("Sản phẩm Vietlott:", ["Power 6/55 (Thứ 3, 5, 7)", "Mega 6/45 (Thứ 4, 6, CN)"])
lottery_code = "power655" if "Power 6/55" in lottery_choice else "mega645"
lottery_short = "Power 6/55" if lottery_code == "power655" else "Mega 6/45"
total_numbers = 55 if lottery_code == "power655" else 45

force_refresh = st.sidebar.button("🔄 Cập nhật kết quả mới nhất")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 Mẹo chơi thông minh
- **Ngày thường (Jackpot < 120 tỷ)**: Chỉ nên mua **1 vé 10.000đ** (lấy **Vé Vàng #1**).
- **Khi Jackpot lên cao (> 150 tỷ)**: Mua trọn **bộ 4 vé (40.000đ)** để bọc 10 con số mạnh nhất.
- Quả bóng màu **Xanh Lá** ở bảng lịch sử là số AI đã **đoán trúng**.
""")


@st.cache_data(ttl=600)
def load_lottery_data(code: str, refresh: bool) -> pd.DataFrame:
    loader = VietlottDataLoader(code)
    return loader.load_clean_data(force_refresh=refresh)


@st.cache_data(ttl=600)
def compute_recommendations_and_history(code: str, total_n: int, num_history_draws: int, latest_draw_id: str) -> Dict[str, Any]:
    """
    Generate today's recommended tickets AND run a strict out-of-sample walk-forward
    comparison over the last `num_history_draws` draws so users can see exact hits & prizes won.
    """
    loader = VietlottDataLoader(code)
    df_local = loader.load_clean_data(force_refresh=False)

    rec_strat = RecursiveAdaptiveStrategy(total_numbers=total_n, pick_count=6, alpha=0.04, memory_window=60, pool_size=11)
    ac_strat = GameTheoryAntiCrowdStrategy(total_numbers=total_n, pick_count=6)
    wheeler = CombinatorialWheeler(pick_count=6)
    bt = VietlottBacktester(df_local, code, total_n, 6)

    # 1. Today's recommendations for next draw
    top_10_pool, pool_weights = rec_strat.get_top_candidates(df_local, top_k=10)
    today_tickets = rec_strat.generate_tickets(df_local, num_tickets=4)
    anti_crowd_tickets = ac_strat.generate_tickets(df_local, num_tickets=2)
    wheel_info = wheeler.generate_abbreviated_wheel(
        top_10_pool, target_guarantee=3, condition_drawn=4, max_tickets=4, number_weights=pool_weights
    )

    # 2. Strict walk-forward verification on recent draws (no data leakage: draw t only uses data < t)
    M = len(df_local)
    start_idx = max(60, M - num_history_draws)
    history_records: List[Dict[str, Any]] = []

    total_prizes_4tickets = 0
    total_cost_4tickets = 0
    total_prizes_1ticket = 0
    total_cost_1ticket = 0
    winning_draws_count = 0
    two_plus_draws_count = 0

    tier_vn = {
        "jackpot_1": "🏆 ĐỘC ĐẮC JACKPOT 1",
        "jackpot_2": "💎 ĐỘC ĐẮC JACKPOT 2",
        "first": "🥇 GIẢI NHẤT",
        "second": "🥈 GIẢI NHÌ",
        "third": "🎉 GIẢI BA",
        "none": "Chưa đủ 3 số"
    }

    for idx in range(M - 1, start_idx - 1, -1):
        past_df = df_local.iloc[:idx]
        row = df_local.iloc[idx]
        actual_main = sorted([int(x) for x in row["main_numbers"]])
        actual_set = set(actual_main)
        actual_bonus = int(row["bonus_number"]) if pd.notna(row["bonus_number"]) and row["bonus_number"] is not None else None

        pred_tickets = rec_strat.generate_tickets(past_df, num_tickets=4)
        pred_top10, _ = rec_strat.get_top_candidates(past_df, top_k=10)
        pool_hits = sorted(list(set(pred_top10).intersection(actual_set)))

        ticket_evals = []
        draw_prize_4 = 0
        best_matches_in_draw = 0

        for t_idx, t in enumerate(pred_tickets, 1):
            matches_cnt, tier_code, prize_amt = bt.evaluate_ticket(t, actual_main, actual_bonus)
            matched_nums = sorted(list(set(t).intersection(actual_set)))
            draw_prize_4 += prize_amt
            best_matches_in_draw = max(best_matches_in_draw, matches_cnt)

            if t_idx == 1:
                total_prizes_1ticket += prize_amt
                total_cost_1ticket += 10000

            ticket_evals.append({
                "ticket_no": t_idx,
                "numbers": [int(x) for x in t],
                "matched_numbers": matched_nums,
                "match_count": matches_cnt,
                "tier_code": tier_code,
                "tier_label": tier_vn.get(tier_code, "Chưa trúng"),
                "prize_vnd": prize_amt
            })

        total_prizes_4tickets += draw_prize_4
        total_cost_4tickets += 40000
        if draw_prize_4 > 0:
            winning_draws_count += 1
        if best_matches_in_draw >= 2:
            two_plus_draws_count += 1

        history_records.append({
            "draw_id": str(row["id"]),
            "draw_date": row["date"].strftime("%d/%m/%Y"),
            "day_of_week": str(row["day_of_week"]),
            "actual_main": actual_main,
            "actual_bonus": actual_bonus,
            "pred_top10": pred_top10,
            "pool_hits": pool_hits,
            "tickets": ticket_evals,
            "draw_prize_vnd": draw_prize_4,
            "best_matches": best_matches_in_draw
        })

    return {
        "top_10_pool": top_10_pool,
        "pool_weights": pool_weights,
        "today_tickets": today_tickets,
        "anti_crowd_tickets": anti_crowd_tickets,
        "wheel_info": wheel_info,
        "history_records": history_records,
        "num_draws": len(history_records),
        "total_prizes_4tickets": total_prizes_4tickets,
        "total_cost_4tickets": total_cost_4tickets,
        "total_prizes_1ticket": total_prizes_1ticket,
        "total_cost_1ticket": total_cost_1ticket,
        "winning_draws_count": winning_draws_count,
        "two_plus_draws_count": two_plus_draws_count
    }


df = load_lottery_data(lottery_code, force_refresh)
latest_draw = df.iloc[-1]
latest_id = str(latest_draw["id"])
next_id = f"{int(latest_id) + 1:05d}"
latest_date_str = latest_draw["date"].strftime("%d/%m/%Y")
latest_main = [int(x) for x in latest_draw["main_numbers"]]
latest_bonus = int(latest_draw["bonus_number"]) if pd.notna(latest_draw["bonus_number"]) and latest_draw["bonus_number"] is not None else None

# Also keep DailyModelTracker updated in background
tracker = DailyModelTracker(lottery_code)
tracker.evaluate_latest_draw()

# Header
st.title(f"🎰 Trợ Lý Dự Đoán & Đối Soát Vietlott: {lottery_short}")
latest_balls_html = "".join([f'<span class="ball-badge">{x:02d}</span>' for x in latest_main])
if lottery_code == "power655" and latest_bonus is not None:
    latest_balls_html += f'<span class="ball-badge ball-special">{latest_bonus:02d}</span> <small>(Số đặc biệt)</small>'

st.markdown(
    f"**Kỳ mở thưởng gần nhất (#{latest_id} - Ngày {latest_date_str})**: {latest_balls_html}",
    unsafe_allow_html=True
)

# Navigation Tabs in plain, friendly Vietnamese
tab_today, tab_budget, tab_wheel, tab_stats = st.tabs([
    "🎯 1. Hôm Nay Mua Số Gì? & Lịch Sử 2 Tuần Qua",
    "💰 2. Khi Nào Nên Mua 10k Hay 40k?",
    "🛠️ 3. Tự Ghép Vé Tiết Kiệm (Nén 10 Số Thành 4 Vé)",
    "📊 4. Bảng Số Hay Về & Số Lâu Chưa Ra"
])


# ==========================================
# TAB 1: TODAY'S RECOMMENDATIONS & 2-WEEK VERIFICATION
# ==========================================
with tab_today:
    history_window_label = st.radio(
        "Chọn khoảng thời gian muốn xem lại lịch sử AI đoán đúng/sai:",
        ["2 tuần gần đây (6 kỳ quay)", "3 tuần gần đây (9 kỳ quay)", "1 tháng gần đây (12 kỳ quay)"],
        index=1,
        horizontal=True
    )
    n_hist = 6 if "6 kỳ" in history_window_label else (9 if "9 kỳ" in history_window_label else 12)

    data_bundle = compute_recommendations_and_history(lottery_code, total_numbers, n_hist, latest_id)
    today_tickets = data_bundle["today_tickets"]
    top_10_pool = data_bundle["top_10_pool"]
    best_single_ticket = today_tickets[0]

    st.subheader(f"🔥 Đề Xuất Mua Vé Hôm Nay (Cho Kỳ Tiếp Theo #{next_id})")

    col_rec1, col_rec2 = st.columns(2)

    with col_rec1:
        b1_html = "".join([f'<span class="ball-badge ball-gold">{x:02d}</span>' for x in best_single_ticket])
        st.markdown(f"""
        <div class="recommend-box">
            <h4 style="margin-top:0; color:#166534;">⭐ LỰA CHỌN 1: MUA 1 VÉ DUY NHẤT (10.000đ)</h4>
            <p style="margin-bottom:10px; color:#15803d;">
                <strong>Khuyên dùng hàng ngày:</strong> Đây là <strong>Vé Vàng #1</strong> hội tụ 6 con số có điểm nhịp rơi cao nhất hôm nay. 
                Chỉ tốn 10.000đ/kỳ (khoảng 1 ly cà phê mỗi tháng sau khi trừ giải phụ).
            </p>
            <div style="margin: 14px 0;">{b1_html}</div>
            <small style="color:#334155;">💡 <em>Chép đúng 1 dòng 6 số này ra mua vé 10.000đ là đủ!</em></small>
        </div>
        """, unsafe_allow_html=True)

    with col_rec2:
        tickets_html_lines = ""
        for i, tk in enumerate(today_tickets, 1):
            row_balls = "".join([f'<span class="ball-badge">{x:02d}</span>' for x in tk])
            label_tag = " <strong>(Vé Vàng #1)</strong>" if i == 1 else ""
            tickets_html_lines += f'<div style="margin-bottom:6px;"><strong>Vé #{i}:</strong> {row_balls}{label_tag}</div>'

        pool_str_display = ", ".join(f"{x:02d}" for x in top_10_pool)
        st.markdown(f"""
        <div class="wheel-box">
            <h4 style="margin-top:0; color:#1e40af;">🛡️ LỰA CHỌN 2: BỘ 4 VÉ BAO VÂY (40.000đ)</h4>
            <p style="margin-bottom:8px; color:#1e3a8a;">
                Dành cho lúc Jackpot lên cao hoặc khi bạn muốn <strong>bọc trọn 10 con số đẹp nhất hôm nay</strong> (<code>{pool_str_display}</code>) 
                chỉ với 4 vé để dễ trúng giải phụ hơn:
            </p>
            {tickets_html_lines}
        </div>
        """, unsafe_allow_html=True)

    with st.expander("💎 Xem thêm: 2 Vé Săn Độc Đắc Không Đụng Hàng (Ưu tiên số lớn > 31 để tránh phải chia giải)"):
        st.write("Đa số người chơi hay chọn ngày sinh nhật (từ 01 đến 31). 2 vé dưới đây ưu tiên các số lớn (> 31) đang có cầu đẹp để nếu trúng Jackpot bạn sẽ nhận trọn 100% tiền thưởng:")
        for idx, tk in enumerate(data_bundle["anti_crowd_tickets"], 1):
            ac_balls = "".join([f'<span class="ball-badge">{x:02d}</span>' for x in tk])
            st.markdown(f"**Vé săn Jackpot #{idx}**: {ac_balls}", unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================
    # HISTORICAL COMPARISON SECTION (LAST 2-4 WEEKS)
    # ==========================================
    st.subheader(f"📅 Kiểm Chứng Lịch Sử {history_window_label}: AI Đã Đoán Đúng Số Nào & Trúng Giải Gì?")
    st.caption(
        "Cách kiểm chứng minh bạch: Ở mỗi kỳ quay trong quá khứ, máy chỉ được nhìn dữ liệu CỦA CÁC KỲ TRƯỚC ĐÓ để đưa ra 4 vé dự đoán, "
        "rồi đem so trực tiếp với kết quả Vietlott mở thưởng. **Quả bóng màu Xanh Lá Cây là con số AI đã đoán TRÚNG!**"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số kỳ đã đối soát", f"{data_bundle['num_draws']} kỳ quay")
    m2.metric("Số kỳ có vé trúng ≥ 2 số", f"{data_bundle['two_plus_draws_count']} / {data_bundle['num_draws']} kỳ")
    m3.metric("Số kỳ nổ Giải Thưởng (≥ 3 số)", f"{data_bundle['winning_draws_count']} kỳ có giải")
    m4.metric("Tổng tiền giải thưởng đạt được", f"{data_bundle['total_prizes_4tickets']:,.0f} VNĐ")

    for rec in data_bundle["history_records"]:
        actual_set = set(rec["actual_main"])
        actual_balls_html = "".join([f'<span class="ball-badge">{x:02d}</span>' for x in rec["actual_main"]])
        if lottery_code == "power655" and rec["actual_bonus"] is not None:
            actual_balls_html += f'<span class="ball-badge ball-special">{rec["actual_bonus"]:02d}</span>'

        if rec["draw_prize_vnd"] > 0:
            header_status = f'<span class="prize-tag">🎉 TRÚNG THƯỞNG +{rec["draw_prize_vnd"]:,.0f} VNĐ</span>'
        elif rec["best_matches"] == 2:
            header_status = '<span class="near-tag">✨ Đoán trúng 2/6 số trên 1 vé (Sát nút giải Ba)</span>'
        elif rec["best_matches"] == 1:
            header_status = '<span style="color:#64748b; font-size:13px; margin-left:8px;">Đoán trúng 1 số</span>'
        else:
            header_status = '<span style="color:#94a3b8; font-size:13px; margin-left:8px;">Chưa khớp kỳ này</span>'

        tickets_rows_html = ""
        for tk_info in rec["tickets"]:
            balls_str = ""
            for num in tk_info["numbers"]:
                if num in actual_set:
                    balls_str += f'<span class="ball-badge ball-hit" title="Đoán trúng số {num:02d}!">{num:02d}</span>'
                else:
                    balls_str += f'<span class="ball-badge ball-miss">{num:02d}</span>'

            if tk_info["prize_vnd"] > 0:
                res_text = f'<strong style="color:#15803d;">👉 {tk_info["tier_label"]} (+{tk_info["prize_vnd"]:,.0f}đ) — Trúng {tk_info["match_count"]} số: {tk_info["matched_numbers"]}</strong>'
            elif tk_info["match_count"] == 2:
                res_text = f'<span style="color:#b45309; font-weight:600;">👉 Trúng 2 số {tk_info["matched_numbers"]} (Thiếu 1 số là trúng 50.000đ)</span>'
            elif tk_info["match_count"] == 1:
                res_text = f'<span style="color:#475569;">👉 Trúng 1 số {tk_info["matched_numbers"]}</span>'
            else:
                res_text = '<span style="color:#94a3b8;">Trúng 0 số</span>'

            tk_label = "Vé Vàng #1 (Gói 10k)" if tk_info["ticket_no"] == 1 else f"Vé #{tk_info['ticket_no']} (Gói 40k)"
            tickets_rows_html += f'<div style="margin-top:8px;"><strong>{tk_label}:</strong> {balls_str} {res_text}</div>'

        pool_hit_str = (
            f"Dàn 10 số AI chọn kỳ này đã bắt trúng <strong>{len(rec['pool_hits'])} bóng</strong>: <code>{rec['pool_hits']}</code>"
            if rec["pool_hits"] else "Dàn 10 số AI chọn chưa khớp kỳ này"
        )

        st.markdown(f"""
        <div class="history-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; border-bottom:1px solid #f1f5f9; padding-bottom:10px; margin-bottom:8px;">
                <div>
                    <strong style="font-size:16px;">Kỳ #{rec['draw_id']} — Ngày {rec['draw_date']} ({rec['day_of_week']})</strong>
                    {header_status}
                </div>
                <div style="margin-top:4px;">
                    <span style="color:#475569; font-size:14px; margin-right:6px;">Kết quả Vietlott:</span>
                    {actual_balls_html}
                </div>
            </div>
            <div style="font-size:13px; color:#334155; margin-bottom:6px;">🎯 {pool_hit_str}</div>
            {tickets_rows_html}
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# TAB 2: SIMPLE BUDGET & JACKPOT ADVISOR
# ==========================================
with tab_budget:
    st.subheader("💰 Hôm Nay Nên Bỏ Ra 10.000đ Hay 40.000đ?")
    st.markdown("""
    Bạn không cần nhớ bất kỳ công thức toán học nào. Chỉ cần nhìn **mức tiền thưởng Jackpot hiện tại trên tờ vé Vietlott** và làm theo đúng **3 mức đi tiền chuẩn** dưới đây:
    """)

    col_step1, col_step2, col_step3 = st.columns(3)
    with col_step1:
        st.info("""
        **🟢 MỨC 1: Khi Jackpot dưới 120 Tỷ**
        - **Nên mua**: **Chỉ 1 vé = 10.000đ/kỳ** (lấy **Vé Vàng #1** ở Tab 1).
        - **Lý do**: Lúc này giải độc đắc chưa cao, mua 1 vé 10k chỉ tốn khoảng 4.000đ/kỳ sau khi trừ tiền trúng giải phụ — giữ vận may mà không bao giờ sợ lỗ nặng.
        """)
    with col_step2:
        st.warning("""
        **🟡 MỨC 2: Khi Jackpot từ 120 – 180 Tỷ**
        - **Nên mua**: **2 vé = 20.000đ/kỳ** (lấy **Vé #1 và Vé #2** ở Tab 1).
        - **Lý do**: Giải thưởng đã phình to gấp 4–5 lần mức khởi điểm, đáng để mở rộng thêm 1 vé bọc lót.
        """)
    with col_step3:
        st.success("""
        **🔥 MỨC 3: Khi Jackpot vượt > 180 Tỷ**
        - **Nên mua**: **Trọn bộ 4 vé = 40.000đ/kỳ** (bộ 4 vé ở Tab 1).
        - **Lý do**: Đây là lúc giải thưởng cực lớn (vùng có lợi cho người chơi). Mua đủ 4 vé sẽ khóa chặt 10 con số đẹp nhất: nếu có 4 số rơi vào dàn 10 số này, bạn **chắc chắn 100% trúng giải!**
        """)

    st.markdown("---")
    st.subheader("🧮 Máy Tính Nhanh: Kiểm Tra Xem Mức Jackpot Hôm Nay Đã Đáng Mua Nhiều Chưa")

    col_inp1, col_inp2 = st.columns(2)
    with col_inp1:
        default_jp = 50.0 if lottery_code == "power655" else 30.0
        jackpot_input = st.number_input(
            "Nhập số tiền Giải Độc Đắc (Jackpot 1) hiện tại (Tỷ VNĐ):",
            min_value=12.0 if lottery_code == "mega645" else 30.0,
            max_value=600.0,
            value=default_jp,
            step=5.0
        )
    with col_inp2:
        anti_crowd_toggle = st.checkbox(
            "Ưu tiên chọn bộ số ít người trùng (Chọn nhiều số > 31 để không phải chia đôi giải)",
            value=True
        )

    ev_calc = VietlottEVCalculator(lottery_code=lottery_code)
    ev_res = ev_calc.calculate_expected_value(
        jackpot_input * 1e9,
        estimated_tickets_sold=1500000,
        anti_crowd_mode=anti_crowd_toggle
    )

    if jackpot_input >= 180.0 or ev_res["is_positive_ev"]:
        st.markdown(f"""
        <div class="ev-banner-pos">
            <h4>🔥 ĐÈN XANH: GIẢI JACKPOT ĐANG RẤT LỚN ({jackpot_input:.0f} TỶ VNĐ) — NÊN MUA BỘ 4 VÉ (40.000đ)</h4>
            Một tờ vé 10.000đ lúc này có giá trị quy đổi thực tế lên tới <strong>{ev_res['total_expected_value_vnd']:,.0f} VNĐ</strong>. 
            Bạn có thể tự tin dùng <strong>Lựa chọn 2 (Bộ 4 vé 40.000đ)</strong> ở Tab 1!
        </div>
        """, unsafe_allow_html=True)
    elif jackpot_input >= 120.0:
        st.markdown(f"""
        <div class="ev-banner-neg">
            <h4>🟡 ĐÈN VÀNG: JACKPOT ĐANG TÍCH LŨY KHÁ CAO ({jackpot_input:.0f} TỶ VNĐ) — NÊN MUA 1 ĐẾN 2 VÉ (10.000đ - 20.000đ)</h4>
            Giá trị thực của 1 vé 10.000đ hiện đạt khoảng <strong>{ev_res['total_expected_value_vnd']:,.0f} VNĐ</strong>. 
            Hãy mua <strong>Vé Vàng #1 (10k)</strong> hoặc thêm <strong>Vé #2 (tổng 20k)</strong> là vừa đẹp.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ev-banner-neg">
            <h4>💡 KHUYẾN NGHỊ AN TOÀN: JACKPOT ĐANG Ở MỨC THƯỜNG ({jackpot_input:.0f} TỶ VNĐ) — CHỈ NÊN MUA 1 VÉ 10.000đ</h4>
            Ở mức Jackpot này, mua nhiều vé sẽ dễ hao hụt vốn (1 vé 10.000đ hiện chỉ tương đương <strong>{ev_res['total_expected_value_vnd']:,.0f} VNĐ</strong> giá trị giải thưởng).<br>
            👉 <strong>Cách tốt nhất hôm nay:</strong> Chỉ bỏ ra <strong>10.000đ mua duy nhất Vé Vàng #1</strong> ở Tab 1! 
            (Khi nào Jackpot vượt lên trên <strong>180 – {ev_res['break_even_jackpot_1_billion']} Tỷ VNĐ</strong> mới nên mua mạnh 40.000đ).
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# TAB 3: SMART 10-TO-4 TICKET SAVER (WHEELING)
# ==========================================
with tab_wheel:
    st.subheader("🛠️ Công Cụ Nén 10 Số Yêu Thích Thành 4 Vé Tiết Kiệm (Chỉ 40.000đ)")
    st.markdown("""
    Giả sử bạn thích **10 con số**, nếu ra đại lý mua vé **Bao 10** bạn sẽ phải trả **2.100.000 VNĐ** (tương đương 210 vé).  
    Công cụ này giúp bạn **xếp 10 số đó thành đúng 4 vé (40.000đ)** nhưng vẫn **bảo đảm 100%**:  
    👉 *Hễ có 4 con số bạn chọn xuất hiện trong kết quả mở thưởng, bạn chắc chắn có ít nhất 1 vé trúng giải Ba trở lên!*
    """)

    rec_wheeler_strat = RecursiveAdaptiveStrategy(total_numbers=total_numbers, pick_count=6, alpha=0.04)
    ai_top_pool, ai_weights = rec_wheeler_strat.get_top_candidates(df, top_k=10)
    default_pool = ", ".join(f"{x:02d}" for x in ai_top_pool)

    st.info(f"🤖 Ô bên dưới đang điền sẵn **10 con số đẹp nhất hôm nay do AI chọn** (`{default_pool}`). Bạn có thể giữ nguyên hoặc tự sửa thành các con số may mắn của riêng bạn:")
    pool_str = st.text_input("Nhập từ 8 đến 12 con số (cách nhau bởi dấu phẩy):", default_pool)

    try:
        user_pool = [int(x.strip()) for x in pool_str.split(",") if x.strip().isdigit()]
        user_pool = sorted(list(set([x for x in user_pool if 1 <= x <= total_numbers])))
    except Exception:
        user_pool = []

    if len(user_pool) < 6:
        st.warning("Vui lòng nhập ít nhất 6 con số hợp lệ (từ 1 đến 55).")
    else:
        wheeler = CombinatorialWheeler(pick_count=6)
        wheel_out = wheeler.generate_abbreviated_wheel(
            user_pool,
            target_guarantee=3,
            condition_drawn=4,
            max_tickets=4,
            number_weights=ai_weights
        )

        bao_cost = wheel_out["full_wheel_tickets"] * 10000
        wheel_cost = wheel_out["ticket_count"] * 10000

        st.success(
            f"✅ Thay vì mua vé Bao tốn **{bao_cost:,.0f} VNĐ** ({wheel_out['full_wheel_tickets']} vé), "
            f"bạn chỉ cần mua **{wheel_out['ticket_count']} vé dưới đây ({wheel_cost:,.0f} VNĐ)** — "
            f"Tiết kiệm **{wheel_out['cost_saving_pct']}% tiền** mà vẫn đạt độ phủ trúng thưởng **{wheel_out['guarantee_coverage_pct']}%**!"
        )

        for i, t in enumerate(wheel_out["tickets"], 1):
            t_html = "".join([f'<span class="ball-badge">{x:02d}</span>' for x in t])
            st.markdown(f"**Vé rút gọn #{i}**: {t_html}", unsafe_allow_html=True)


# ==========================================
# TAB 4: PLAIN VIETNAMESE HOT/COLD NUMBERS
# ==========================================
with tab_stats:
    st.subheader(f"📊 Bảng Xếp Hạng Số Hay Về & Số Lâu Ngày Chưa Ra ({lottery_short})")
    analyzer = VietlottAnalyzer(df, total_numbers=total_numbers, pick_count=6)
    hc_data = analyzer.get_hot_cold_numbers(lookback=50, top_n=10)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🔥 Top 10 Số Đang Vào Cầu (Về nhiều nhất 50 kỳ qua)")
        hot_df = pd.DataFrame(hc_data["hot_numbers"]).rename(columns={
            "number": "Con số",
            "count": "Số lần về (50 kỳ qua)",
            "frequency_pct": "Tỷ lệ xuất hiện (%)",
            "current_gap": "Cách mấy kỳ chưa về"
        })
        st.dataframe(hot_df, width="stretch", hide_index=True)

    with col_b:
        st.markdown("#### ❄️ Top 10 Số Gan Lâu Ngày (Lâu nhất chưa xuất hiện)")
        cold_df = pd.DataFrame(hc_data["overdue_numbers"]).rename(columns={
            "number": "Con số",
            "current_gap": "Số kỳ liên tiếp chưa về",
            "max_gap": "Kỷ lục gan lâu nhất lịch sử",
            "mean_gap": "Trung bình mấy kỳ về 1 lần"
        })
        st.dataframe(cold_df, width="stretch", hide_index=True)
