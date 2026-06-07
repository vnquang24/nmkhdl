"""Cấu hình toàn cục cho pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "rootdata - Copy"
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

for d in [INTERIM_DIR, PROCESSED_DIR, SPLITS_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Sampling
FS = 50.0  # Hz, xác nhận từ timestamp_ns
WINDOW_SEC = 2.56
WINDOW_SIZE = int(WINDOW_SEC * FS)  # 128
WINDOW_STRIDE = WINDOW_SIZE // 2     # 64 (overlap 50%)

# Channels: 6 = acc(x,y,z) + gyro(x,y,z)
INERTIAL_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
N_CHANNELS = len(INERTIAL_CHANNELS)

# User split
ACTIVITIES = ["sitting", "standing", "walking"]
# Closed-set: toàn bộ 15 user trong dataset đều dùng cho train/validation.
# Không giữ lại user unknown; danh sách 12 known user sẽ tự suy ra trong code.

# Filter
LOWPASS_CUTOFF = 20.0  # Hz
HIGHPASS_GRAVITY = 0.3  # Hz để tách trọng lực

SEED = 42
