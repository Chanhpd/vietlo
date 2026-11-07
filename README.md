# Vietlott Data Science & Backtesting System 🎰📊

Hệ thống phân tích dữ liệu chuyên sâu, kiểm định giả thuyết tính ngẫu nhiên và thử nghiệm mô hình dự đoán (Walk-Forward Backtesting) cho xổ số điện toán Việt Nam (**Vietlott Power 6/55** và **Mega 6/45**).

---

## 📌 1. Triết lý khoa học & Tuyên bố trách nhiệm (Disclaimer)
> **Lưu ý quan trọng từ Data Science:**  
> Xổ số được thiết kế cơ học để tạo ra các biến cố ngẫu nhiên độc lập giữa các kỳ quay. Mục tiêu của dự án này **không phải để "soi cầu" hay cam kết trúng thưởng**, mà là:
> 1. Xây dựng pipeline xử lý dữ liệu và kiểm định thống kê chuẩn mực.
> 2. Kiểm chứng thực nghiệm xem các chiến lược dự đoán thông thường (Số nóng, số lạnh, chu kỳ vắng mặt, Machine Learning) có thực sự vượt trội hơn việc **chọn ngẫu nhiên (Random Baseline)** hay không.
> 3. Chứng minh toán học về tính ngẫu nhiên và kỳ vọng lợi nhuận âm (Negative Expected Value).
> 
> *Dự án chỉ phục vụ mục đích nghiên cứu học thuật và giáo dục. Tuyệt đối không khuyến khích đánh bạc hoặc đầu tư tài chính vào xổ số.*

---

## 🗂️ 2. Cấu trúc thư mục dự án

```bash
vietlo/
├── configs/
│   └── config.yaml          # File cấu hình (loại xổ số, lookback window, backtest size...)
├── data/
│   ├── raw/                 # Cache file NDJSON tải trực tiếp từ GitHub
│   └── processed/           # Dữ liệu CSV/Parquet sạch đã được làm giàu thuộc tính
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Tự động tải, cập nhật và làm sạch dữ liệu
│   ├── analyzer.py          # Thống kê tần suất, số nóng/lạnh, chu kỳ, mẫu lặp, cặp/bộ ba
│   ├── randomness_test.py   # Kiểm định Chi-square Uniformity, Wald-Wolfowitz Runs test
│   ├── strategies.py        # Các chiến lược: Random, Hot, Cold, Hybrid, ML Ranker
│   ├── backtester.py        # Walk-Forward Backtesting, tính thưởng Vietlott, PnL, ROI
│   ├── visualizer.py        # Xuất biểu đồ phân tích và hiệu năng chất lượng cao
│   └── report_generator.py  # Tạo báo cáo Markdown chi tiết và HTML Executive Dashboard
├── reports/
│   ├── figures/             # 6 biểu đồ trực quan hóa (tần suất, gap, repeats, PnL...)
│   ├── REPORT.md            # Báo cáo tổng hợp kết quả chi tiết
│   └── dashboard.html       # Giao diện Dashboard tương tác hiện đại
├── notebooks/
│   └── exploration.ipynb    # Jupyter Notebook tương tác từng bước (chạy tốt trên Colab)
├── main.py                  # CLI Master Runner thực thi toàn bộ pipeline
├── requirements.txt         # Danh sách thư viện Python cần thiết
└── README.md                # Tài liệu hướng dẫn sử dụng
```

---

## 🚀 3. Hướng dẫn cài đặt & Chạy nhanh

### 3.1. Cài đặt môi trường
Yêu cầu Python 3.9 trở lên:

```bash
# Clone hoặc tải dự án
git clone https://github.com/vietvudanh/vietlott-data.git # (dữ liệu được tự động tải)
cd vietlo

# Tạo môi trường ảo và cài dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2. Chạy toàn bộ Pipeline phân tích & Backtest

#### Chạy với Vietlott Power 6/55:
```bash
python main.py --lottery power655 --test-draws 150 --tickets-per-draw 5
```

#### Chạy với Vietlott Mega 6/45:
```bash
python main.py --lottery mega645 --test-draws 150 --tickets-per-draw 5
```

#### Các tham số dòng lệnh tùy chọn:
- `--lottery`: Chọn loại xổ số (`power655` hoặc `mega645`, mặc định: `power655`).
- `--force-download`: Bắt buộc tải lại bản ghi mới nhất từ GitHub.
- `--test-draws`: Số kỳ gần nhất để tiến hành Walk-Forward Backtesting (mặc định: `200`).
- `--tickets-per-draw`: Số vé mỗi chiến lược mua trong mỗi kỳ thử nghiệm (mặc định: `5`).
- `--lookback`: Cửa sổ quan sát để tính toán số nóng / số lạnh (mặc định: `50`).
- `--suggest-count`: Số vé đề xuất cho kỳ quay tiếp theo (mặc định: `5`).

---

## 🧠 4. Các chiến lược dự đoán được thử nghiệm

1. **Random Baseline (Ngẫu nhiên chuẩn mực)**:
   - Chọn ngẫu nhiên không hoàn lại 6 số từ tập số $1..N$.
   - Đóng vai trò là mốc đối chứng (Benchmark).
2. **Hot Frequency Strategy (Số nóng)**:
   - Dựa trên tần suất xuất hiện trong $W$ kỳ gần nhất ($W=50$).
   - Lấy mẫu có trọng số ưu tiên các số có tần suất xuất hiện cao nhất.
3. **Cold / Overdue Strategy (Số lạnh / Lâu chưa về)**:
   - Dựa trên tâm lý "ngụy biện con bạc" (Gambler's Fallacy): kỳ vọng các số lâu chưa về sẽ có xác suất xuất hiện cao hơn để bù đắp tần suất.
   - Ưu tiên các số có khoảng cách vắng mặt (`current_gap`) lớn nhất.
4. **Hybrid Balanced Strategy (Cân bằng danh mục)**:
   - Kết hợp: 2 số Hot + 2 số Cold + 2 số Warm.
   - Áp dụng các bộ lọc xác suất thực nghiệm:
     - Tối đa 1 số lặp lại từ kỳ trước (phù hợp với xác suất $88\%$ của phân bố Siêu bội).
     - Tỷ lệ Chẵn/Lẻ cân bằng: 2:4, 3:3, hoặc 4:2.
     - Tổng các số nằm trong khoảng $[\mu - 1.5\sigma, \mu + 1.5\sigma]$.
5. **Machine Learning Ranker (Logistic Regression / Random Forest)**:
   - Trích xuất ma trận đặc trưng cho từng số tại mỗi kỳ:
     - Tần suất động theo các chu kỳ (10, 30, 60 kỳ).
     - Khoảng cách vắng mặt hiện tại và tỷ lệ so với khoảng cách trung bình lịch sử.
     - Cờ nhị phân xuất hiện ở các kỳ trễ $t-1, t-2, t-3$.
     - Tỷ lệ xuất hiện theo thứ trong tuần (Day of Week affinity).
   - Mô hình phân loại học máy dự đoán xác suất xuất hiện của từng số trong kỳ kế tiếp và tạo vé từ top các số có điểm số cao nhất.

---

## 📈 5. Kết quả kiểm định & Thử nghiệm thực tế

### 5.1. Kiểm định tính ngẫu nhiên
- **Chi-Square Goodness-of-Fit**:
  - Power 6/55: $p\text{-value} = 0.5968 > 0.05 \implies$ Không thể bác bỏ giả thuyết $H_0$. Các số có phân bố đều lý thuyết hoàn hảo.
  - Mega 6/45: $p\text{-value} = 0.9388 > 0.05 \implies$ Phân bố đều xuất sắc.
- **Wald-Wolfowitz Runs Test**:
  - Toàn bộ các số đều vượt qua kiểm định tính độc lập chuỗi thời gian sau khi hiệu chỉnh Bonferroni ($p > \alpha / N$). Không có hiện tượng "dây" hay "chu kỳ nhớ".
- **Kiểm định số lặp liên tiếp**:
  - Tỷ lệ số lặp lại giữa 2 kỳ liên tiếp hoàn toàn trùng khớp với hàm mật độ xác suất **Hypergeometric** lý thuyết (p-value $> 0.30$).

### 5.2. Kết quả Walk-Forward Backtest (150 kỳ gần nhất)

| Chiến lược | Số vé | Hits TB/vé | Kỳ vọng lý thuyết | Win Rate ($\ge 3$ số) | ROI (%) | P-value vs Random | Vượt trội? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | 750 | 0.6533 | 0.6545 | 1.60% | -92.0% | 1.0000 | Baseline |
| **Hot Frequency (W=50)** | 750 | 0.6880 | 0.6545 | 1.33% | -81.3% | 0.3623 | Không |
| **Cold / Overdue** | 750 | 0.7667 | 0.6545 | 2.40% | -76.0% | 0.0043* | Không có lợi nhuận |
| **Hybrid Balanced** | 750 | 0.6653 | 0.6545 | 0.93% | -89.3% | 0.7504 | Không |
| **ML Ranker (LogisticReg)** | 750 | 0.7133 | 0.6545 | 1.47% | -92.7% | 0.1181 | Không |

*\* Lưu ý: Dù số Cold có thể xuất hiện nhiều bóng trúng nhỏ lẻ ở một số giai đoạn cụ thể, tỷ lệ trúng các giải lớn không tăng và ROI vẫn luôn âm nặng ($-76\%$ đến $-95\%$).*

---

## 📊 6. Báo cáo & Dashboard tương tác

Sau khi chạy lệnh `python main.py`, hệ thống tự động xuất:
1. **Báo cáo Markdown toàn diện**: Xem tại `reports/REPORT.md`.
2. **Interactive HTML Dashboard**: Mở tệp `reports/dashboard.html` bằng trình duyệt web bất kỳ để xem giao diện trực quan với thẻ chỉ số, bảng biểu và biểu đồ phân tích.
3. **Các tệp biểu đồ PNG**: Được lưu trong `reports/figures/`.

---

## 🛠️ 7. Mở rộng (Extensibility)

Dự án được thiết kế dạng module hóa cao:
- **Thêm sản phẩm mới**: Bổ sung thông số kỹ thuật vào `configs/config.yaml` (ví dụ `keno`, `max3d`).
- **Thêm chiến lược mới**: Kế thừa lớp `BaseStrategy` trong `src/strategies.py` và override phương thức `generate_tickets(history_df, num_tickets)`.
- **Thêm mô hình ML mới**: Dễ dàng tích hợp XGBoost, LightGBM hoặc mô hình mạng nơ-ron LSTM bằng cách mở rộng `MLRankingStrategy`.

