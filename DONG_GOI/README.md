# README — Hệ thống định danh người dùng qua hành vi điện thoại

Đồ án môn **Nhập môn Khoa học dữ liệu (IT4142)**.

## 1. Mô tả ngắn

Dự án cài đặt một pipeline Học máy (Machine Learning) hoàn chỉnh cho bài toán định danh người dùng **Closed-set 15 người dùng** dựa trên dữ liệu chuyển động cảm biến thu thập từ điện thoại (Gia tốc kế & Con quay hồi chuyển) và log gõ phím. Dự án xây dựng và so sánh 3 mô hình chính:
* **Random Forest** (trên 65 đặc trưng thống kê thủ công).
* **SVM RBF** (trên 65 đặc trưng thống kê thủ công sau chuẩn hóa).
* **CNN 1D** (học sâu trực tiếp trên chuỗi thời gian).

## 2. Yêu cầu môi trường

* **Hệ điều hành**: Windows 10/11, Linux (đã kiểm thử tốt trên Ubuntu 24.04), macOS.
* **Phiên bản Python**: Python 3.10+ (khuyến nghị 3.12).
* **Bộ nhớ RAM**: &ge; 4 GB.
* **Tự động hỗ trợ CUDA** nếu máy có card đồ họa Nvidia, nếu không sẽ tự động fallback chạy trên CPU.

## 3. Cài đặt

1. Giải nén tệp zip mã nguồn (ví dụ: `nmkhdl_source.zip` hoặc tương đương).
2. Di chuyển vào thư mục dự án:
   ```bash
   cd NMKHDL
   ```
3. Tạo môi trường ảo Python (Virtual Environment):
   ```bash
   python -m venv venv
   ```
4. Kích hoạt môi trường ảo:
   * **Windows (PowerShell)**:
     ```powershell
     venv\Scripts\Activate.ps1
     ```
   * **Windows (CMD)**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   * **Linux / macOS**:
     ```bash
     source venv/bin/activate
     ```
5. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## 4. Chuẩn bị dữ liệu

Đặt thư mục dữ liệu thô `rootdata - Copy` (chứa các tệp dữ liệu cảm biến dạng `.csv` của 15 người dùng) vào thư mục gốc của dự án (cùng cấp với `src` và `scripts`):
```
NMKHDL/
├── rootdata - Copy/              # Dữ liệu cảm biến thô (15 user)
│   ├── userA/
│   │   ├── session_1.csv
│   │   └── ...
│   ├── userB/
│   │   └── ...
│   └── ...
├── src/                          # Mã nguồn chính
├── scripts/                      # Kịch bản chạy nhanh
└── ...
```

## 5. Hướng dẫn chạy chương trình

### 5.1 Chạy nhanh toàn bộ pipeline (CNN 1D + RF + SVM RBF)
Kịch bản chạy nhanh sẽ tự động thực hiện tiền xử lý dữ liệu, huấn luyện cả 3 mô hình bằng phương pháp 3-fold Cross Validation và ghi lại tóm tắt kết quả.
```bash
# Huấn luyện và đánh giá 3 mô hình
python -m scripts.run_simple_pipeline

# Vẽ biểu đồ so sánh kết quả
python -m scripts.plot_simple_results
```
* **Đầu ra**:
  * Tóm tắt kết quả: `reports/simple_results.json`
  * Biểu đồ so sánh: `reports/figures/simple_model_comparison.png`, `simple_fold_macro_f1.png` và `simple_rf_accuracy.png`.

### 5.2 Chạy từng bước chi tiết (Phân tích sâu)
```bash
# Bước 1: Quét dữ liệu thô, sinh manifest chỉ mục
python -m src.io
# → data/interim/manifest.csv

# Bước 2: Lọc ButterWorth lọc nhiễu + chia cửa sổ tín hiệu
python -m src.preprocess
# → data/processed/windows.npz

# Bước 3: Trích xuất đặc trưng gõ phím Keystroke bổ sung
python -m src.features
# → data/interim/feat_keystroke.csv

# Bước 4: Huấn luyện mô hình CNN 1D chính thức (80 epochs, 3 folds)
python -m src.train --epochs 80 --folds 3
# → models/fold{0,1,2}/model.pt

# Bước 5: Đánh giá chi tiết, chạy RF & SVM baseline, vẽ Confusion Matrix
python -m src.evaluate
# → reports/evaluate_results.json và các biểu đồ confusion matrix trong reports/figures/
```

## 6. Cấu trúc thư mục đóng gói

```
NMKHDL/
├── BAOCAO/
│   ├── bao_cao.pdf            # Báo cáo kết quả đồ án môn học (PDF)
│   └── figures/               # Các biểu đồ minh họa dùng trong báo cáo
├── src/                       # Mã nguồn cốt lõi của hệ thống
│   ├── config.py              # Tham số cấu hình bộ lọc, cửa sổ
│   ├── io.py                  # Xử lý nhập xuất và quản lý chỉ mục manifest
│   ├── preprocess.py          # Bộ lọc số Butterworth và phân đoạn tín hiệu
│   ├── features.py            # Trích xuất 65 đặc trưng thời gian & tần số
│   ├── datasets.py            # Quản lý Dataset PyTorch và Data Augmentations
│   ├── models.py              # Định nghĩa kiến trúc CNN, ResNet...
│   ├── train.py               # Chạy quy trình huấn luyện 3-fold CV cho CNN
│   └── evaluate.py            # Đánh giá RF, SVM baseline và gom kết quả CNN
├── scripts/                   # Kịch bản chạy và vẽ biểu đồ
│   ├── run_simple_pipeline.py # Huấn luyện và đánh giá nhanh 3 mô hình
│   ├── plot_simple_results.py # Vẽ biểu đồ từ simple_results.json
│   └── plot_report_extra.py   # Vẽ các biểu đồ phân tích sâu trong báo cáo PDF
├── experiments/configs/       # File YAML chứa siêu tham số thí nghiệm
├── notebooks/                 # Thử nghiệm trực quan bằng Jupyter Notebook
├── reports/                   # Thư mục lưu trữ kết quả và biểu đồ đầu ra
│   ├── figures/               # Biểu đồ Confusion Matrix và Per-activity F1
│   ├── evaluate_results.json  # Kết quả đánh giá chi tiết
│   └── simple_results.json    # Kết quả của simple pipeline
├── requirements.txt           # Danh sách các gói thư viện Python sử dụng
└── readme.txt                 # Hướng dẫn nộp bài (Plain Text)
```

## 7. Kết quả thí nghiệm chính

Đánh giá bằng phương pháp 3-fold Cross Validation trên 15 người dùng (Closed-set):

| Mô hình | Đặc trưng đầu vào | Học máy / Học sâu | Macro-F1 trung bình | Accuracy trung bình |
| :--- | :--- | :--- | :---: | :---: |
| **Random Forest (RF)** | 65 đặc trưng thủ công | Học máy truyền thống | **0.7434** &plusmn; 0.0465 | **0.7482** |
| **SVM (RBF)** | 65 đặc trưng thủ công | Học máy truyền thống | **0.6550** &plusmn; 0.0456 | **0.6740** |
| **CNN 1D** | Chuỗi tín hiệu thô | Học sâu (Deep Learning) | **0.6272** &plusmn; 0.0336 | **0.6385** |

* **Nhận xét**: Mô hình Random Forest trên đặc trưng thống kê được thiết kế thủ công đạt hiệu suất cao nhất và ổn định nhất. Mô hình học sâu CNN 1D cần lượng dữ liệu lớn và đa dạng hơn để có thể tự động học các đặc trưng tối ưu hơn.
