========================================================================
           HƯỚNG DẪN CÀI ĐẶT & CHẠY CHƯƠNG TRÌNH ĐỒ ÁN MÔHỌC
           Môn học: Nhập môn Khoa học dữ liệu (IT4142)
  Đề tài: Phát triển hệ thống định danh người dùng qua hành vi sử dụng điện thoại
========================================================================

Đồ án thực hiện bài toán định danh người dùng (Closed-set 15 người dùng)
dựa trên dữ liệu cảm biến (Gia tốc kế & Con quay hồi chuyển) và log gõ phím.
Hệ thống cài đặt và so sánh 3 mô hình: CNN 1D, Random Forest, và SVM (RBF).

------------------------------------------------------------------------
1. CẤU TRÚC THƯ MỤC NỘP BÀI (DONG_GOI)
------------------------------------------------------------------------
NMKHDL/
├── BAOCAO/
│   ├── bao_cao.pdf            # Báo cáo kết quả đồ án hoàn chỉnh (PDF)
│   └── figures/               # Các hình ảnh, biểu đồ minh họa dùng trong báo cáo
├── src/                       # Mã nguồn cốt lõi của hệ thống
│   ├── __init__.py
│   ├── config.py              # Các tham số cấu hình (tần số, lọc, cửa sổ...)
│   ├── io.py                  # Quét dữ liệu thô, xây dựng manifest.csv
│   ├── preprocess.py          # Lọc tín hiệu (lowpass/highpass), cắt cửa sổ
│   ├── features.py            # Trích xuất 65 đặc trưng cảm biến & keystroke
│   ├── datasets.py            # Định nghĩa Dataset PyTorch & Augmentations
│   ├── models.py              # Định nghĩa cấu trúc mô hình CNN 1D, ResNet1D, TCN...
│   ├── train.py               # Huấn luyện 3-fold Cross Validation cho CNN
│   └── evaluate.py            # Huấn luyện RF, SVM baseline và tổng hợp kết quả
├── scripts/                   # Các kịch bản chạy nhanh và vẽ biểu đồ
│   ├── __init__.py
│   ├── run_simple_pipeline.py # Chạy nhanh pipeline 3 mô hình (CNN, RF, SVM)
│   ├── plot_simple_results.py # Vẽ biểu đồ so sánh kết quả từ simple_results.json
│   └── plot_report_extra.py   # Vẽ các biểu đồ phân tích sâu trong báo cáo PDF
├── experiments/
│   └── configs/               # Lưu file cấu hình thí nghiệm (.yaml)
├── notebooks/
│   └── simple_pipeline.ipynb  # Jupyter Notebook chạy thử nghiệm trực quan
├── reports/                   # Kết quả chạy thử nghiệm và biểu đồ kết quả đầu ra
│   ├── figures/               # Biểu đồ Confusion Matrix, Per-activity F1...
│   ├── evaluate_results.json  # Kết quả chi tiết của 3 mô hình
│   └── simple_results.json    # Kết quả chạy từ kịch bản đơn giản
├── requirements.txt           # Danh sách các thư viện Python cần dùng
└── readme.txt                 # File hướng dẫn này (định dạng Plain Text)

------------------------------------------------------------------------
2. YÊU CẦU MÔI TRƯỜNG & HỆ ĐIỀU HÀNH
------------------------------------------------------------------------
- Hệ điều hành hỗ trợ: Windows 10/11, Linux (đã kiểm thử tốt trên Ubuntu 24.04), macOS.
- Phiên bản Python khuyến nghị: Python 3.10 trở lên (đã kiểm thử tốt trên Python 3.12).
- Cấu hình phần cứng tối thiểu: RAM >= 4GB, CPU Intel Core i3 / Ryzen 3 trở lên.
- Thiết bị hỗ trợ tăng tốc CUDA (nếu có), nếu không chương trình sẽ tự động chạy trên CPU.

------------------------------------------------------------------------
3. HƯỚNG DẪN CÀI ĐẶT
------------------------------------------------------------------------
Bước 1: Giải nén tệp zip mã nguồn (ví dụ: `nmkhdl_source.zip`)
Bước 2: Di chuyển vào thư mục dự án
    cd NMKHDL

Bước 3: Tạo môi trường ảo Python (Virtual Environment) để tránh xung đột thư viện
    python -m venv venv

Bước 4: Kích hoạt môi trường ảo
    - Trên Windows (PowerShell):
        venv\Scripts\Activate.ps1
    - Trên Windows (CMD):
        venv\Scripts\activate.bat
    - Trên Linux / macOS:
        source venv/bin/activate

Bước 5: Cài đặt các thư viện cần thiết
    pip install -r requirements.txt

------------------------------------------------------------------------
4. CHUẨN BỊ DỮ LIỆU
------------------------------------------------------------------------
Để chương trình chạy được, bạn cần chuẩn bị thư mục dữ liệu thô (raw data)
và đặt nó ở thư mục gốc của dự án (cùng cấp với thư mục `src` và `scripts`).

- Tên thư mục dữ liệu thô: `rootdata - Copy`
- Cấu trúc bên trong thư mục dữ liệu thô:
  rootdata - Copy/
  ├── userA/
  │   ├── session_1.csv
  │   └── ...
  ├── userB/
  │   └── ...
  └── [Tổng cộng 15 thư mục con từ userA đến userQ, loại bỏ các chữ cái không có]

------------------------------------------------------------------------
5. HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH
------------------------------------------------------------------------

LỰA CHỌN A: CHẠY NHANH TOÀN BỘ PIPELINE (KHUYẾN NGHỊ)
Kịch bản chạy nhanh sẽ tự động thực hiện huấn luyện cả 3 mô hình (CNN 1D, Random Forest, SVM RBF)
trên tập dữ liệu đã chuẩn bị bằng phương pháp 3-fold Cross Validation, sau đó ghi lại báo cáo tóm tắt.

Lệnh chạy:
    python -m scripts.run_simple_pipeline

Kết quả sẽ được ghi vào tệp: `reports/simple_results.json`

Sau đó, bạn có thể chạy kịch bản vẽ biểu đồ so sánh giữa 3 mô hình từ kết quả vừa lưu:
    python -m scripts.plot_simple_results

Các biểu đồ kết quả sẽ được lưu vào thư mục: `reports/figures/`
- `simple_model_comparison.png` (So sánh macro-F1 trung bình)
- `simple_fold_macro_f1.png`    (Chi tiết F1 qua từng fold)
- `simple_rf_accuracy.png`      (So sánh Accuracy qua các fold)


LỰA CHỌN B: CHẠY TỪNG BƯỚC CHI TIẾT
Nếu muốn chạy thủ công từng phần để kiểm tra chi tiết dữ liệu trung gian:

Bước 1: Quét dữ liệu thô, lọc bỏ tệp lỗi và sinh chỉ mục dữ liệu
    python -m src.io
    --> Đầu ra: `data/interim/manifest.csv`

Bước 2: Tiền xử lý DSP (Lọc Butterworth Lowpass/Highpass) và cắt cửa sổ tín hiệu
    python -m src.preprocess
    --> Đầu ra: `data/processed/windows.npz` (chứa các tensor đã xử lý)

Bước 3: Trích xuất đặc trưng gõ phím Keystroke (nếu cần phân tích thêm)
    python -m src.features
    --> Đầu ra: `data/interim/feat_keystroke.csv`

Bước 4: Huấn luyện mô hình CNN 1D bằng 3-fold CV (lưu checkpoints của từng fold)
    python -m src.train --epochs 80 --folds 3
    --> Đầu ra: `models/fold0/model.pt`, `models/fold1/model.pt`...

Bước 5: Huấn luyện Random Forest, SVM (RBF) baseline và tổng hợp kết quả chi tiết
    python -m src.evaluate
    --> Đầu ra: `reports/evaluate_results.json` và các ma trận nhầm lẫn (Confusion Matrix) trong `reports/figures/`

------------------------------------------------------------------------
6. CẤU TRÚC MÃ NGUỒN & VAI TRÒ CỦA CÁC LỚP, PHƯƠNG THỨC CHÍNH
------------------------------------------------------------------------
- `src/config.py`: Định nghĩa các hằng số cấu hình toàn cục.
  + Các tham số bộ lọc: `LOWPASS_CUTOFF = 20.0` Hz, `HIGHPASS_GRAVITY = 0.3` Hz.
  + Kích thước cửa sổ: `WINDOW_SIZE = 128` (tương đương 2.56 giây với fs = 50Hz).
  + Trượt cửa sổ: `WINDOW_STRIDE = 64` (overlap 50%).

- `src/io.py`: Tiền xử lý và lập chỉ mục dữ liệu.
  + `build_manifest()`: Quét qua toàn bộ thư mục dữ liệu, lọc bỏ các file CSV bị rỗng hoặc hỏng, ghi lại đường dẫn và nhãn tương ứng vào `manifest.csv`.

- `src/preprocess.py`: Tiền xử lý tín hiệu số (DSP) và cắt cửa sổ.
  + `butter_filter()`: Thiết kế bộ lọc số Butterworth bậc 4 thông qua hàm `scipy.signal.filtfilt` (lọc hai chiều giúp không bị méo pha thời gian).
  + `preprocess_file()`: Lọc nhiễu, khử gia tốc trọng trường để lấy gia tốc chuyển động thực tế (Body Acceleration), resample tín hiệu về tần số 50 Hz chuẩn.
  + `segment_windows()`: Cắt tín hiệu thành các cửa sổ có độ dài 128 mẫu với overlap 50%.

- `src/features.py`: Trích xuất đặc trưng thủ công phục vụ mô hình truyền thống (RF, SVM).
  + `extract_features_1d()`: Trích xuất 65 đặc trưng thống kê trên miền thời gian (mean, std, min, max, energy, skewness, kurtosis, correlation...) và miền tần số (spectral entropy, spectral energy, peak frequencies của FFT) cho 6 kênh tín hiệu.

- `src/datasets.py`: Quản lý dữ liệu PyTorch và Tăng cường dữ liệu (Data Augmentation).
  + Lớp `InertialDataset`: Kế thừa `torch.utils.data.Dataset`, thực hiện chuẩn hoá dữ liệu z-score động và áp dụng các phép tăng cường ngẫu nhiên trên tập Train.
  + `mixup_batch()` & `cutmix_batch()`: Thực hiện phối hợp mẫu nhằm cải thiện khả năng tổng quát hoá của mô hình học sâu.

- `src/models.py`: Định nghĩa kiến trúc các mô hình học sâu.
  + Lớp `CNN1D`: Mô hình CNN 1 chiều baseline với các tầng Conv1D, BatchNorm1D, ReLU và Dropout.
  + Lớp `ResNet1D`: Mô hình ResNet với các khối skip-connection giúp huấn luyện mô hình sâu hiệu quả hơn.

- `src/train.py`: Quản lý huấn luyện mạng nơ-ron.
  + `run_cv()`: Thực hiện phân chia dữ liệu bằng `StratifiedGroupKFold` để chia theo tệp ghi (tránh rò rỉ dữ liệu), huấn luyện mô hình CNN qua từng fold sử dụng bộ tối ưu Adam và bộ lập lịch OneCycleLR.

- `src/evaluate.py`: Đánh giá hiệu năng và so sánh.
  + `rf_baseline()`: Huấn luyện Random Forest Classifier với 65 đặc trưng thủ công qua 3 fold.
  + `svm_baseline()`: Huấn luyện Support Vector Machine (SVM) với kernel RBF sau khi đi qua chuẩn hoá đặc trưng `StandardScaler` trên từng fold huấn luyện.
  + `cnn_predictions()`: Gom dự đoán từ các checkpoint CNN đã lưu để đánh giá tổng hợp.
  + `plot_confusion()`: Vẽ ma trận nhầm lẫn Confusion Matrix cho các mô hình.

------------------------------------------------------------------------
7. TÓM TẮT KẾT QUẢ THÍ NGHIỆM ĐẠT ĐƯỢC
------------------------------------------------------------------------
Đánh giá 3-fold Cross Validation trên tập closed-set 15 người dùng cho kết quả:
- Random Forest (RF):
  + Macro-F1 trung bình: 0.7434 ± 0.0465
  + Accuracy trung bình: 0.7482
- SVM (RBF) Classifier:
  + Macro-F1 trung bình: 0.6550 ± 0.0456
  + Accuracy trung bình: 0.6740
- CNN 1D (Deep Learning):
  + Macro-F1 trung bình: 0.6272 ± 0.0336
  + Accuracy trung bình: 0.6385

Nhận xét quan trọng:
- Mô hình Random Forest trên tập đặc trưng thủ công đạt hiệu năng vượt trội và ổn định nhất.
- CNN 1D đạt hiệu năng thấp hơn do lượng dữ liệu huấn luyện còn giới hạn, việc tự học đặc trưng từ chuỗi tín hiệu thô gặp khó khăn hơn so với việc tận dụng các đặc trưng vật lý/thống kê được thiết kế sẵn.

========================================================================
                                HẾT
========================================================================
