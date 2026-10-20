# Face Anti-Spoofing (Chống Giả Mạo Khuôn Mặt)

Hệ thống nhận diện và phân loại khuôn mặt thật / giả (Face Anti-Spoofing) theo thời gian thực sử dụng mạng nơ-ron tích chập (LivenessNet) kết hợp OpenCV YuNet và TensorFlow Lite tăng tốc phần cứng XNNPACK.

---

## 1. Chạy thử nghiệm Camera nhanh nhất

Cách đơn giản nhất trên Windows là nhấp đúp vào file batch:
```cmd
chay_test_camera.bat
```

Menu tương tác sẽ hiển thị cho phép bạn chọn:
- **`[0]` (Mặc định - Enter):** Mở ngay **Camera USB SC-M200** (Full HD / 720p 30 FPS).
- **`[1]`:** Mở **Camera liền máy** (Laptop Built-in Webcam).
- **`[2]`:** Khởi động giao diện Web (`http://127.0.0.1:8080`).

Hoặc chạy trực tiếp bằng dòng lệnh:
```powershell
# Chạy với Camera USB SC-M200 (mặc định)
.\venv\Scripts\python.exe run.py --camera 0

# Chạy với Camera Laptop
.\venv\Scripts\python.exe run.py --camera 1

# Menu chọn camera tương tác
.\venv\Scripts\python.exe run.py --choose-camera
```

### Các phím bấm khi cửa sổ Camera đang mở
- **`[0]`**: Chuyển ngay sang Camera USB SC-M200.
- **`[1]`**: Chuyển ngay sang Camera liền máy Laptop.
- **`[C]`**: Đổi qua lại luân phiên giữa 2 camera.
- **`[Q]` hoặc `[ESC]`**: Đóng ứng dụng và thoát.

---

## 2. Trạng thái nhận diện (Chỉ hiển thị 2 màu Xanh / Đỏ)

Hệ thống đã được tinh chỉnh sang chế độ quyết định nhị phân tức thì (`FastBinaryDecision` - loại bỏ hoàn toàn các khung màu vàng trung gian):
- **Khung XANH LÁ (`REAL | xx.x%`)**: Khuôn mặt người thật.
- **Khung ĐỎ (`FAKE | xx.x%`)**: Khuôn mặt giả mạo (màn hình điện thoại, máy tính, ảnh in).
- **Không có khung**: Không phát hiện khuôn mặt trong khung hình.

Tốc độ xử lý đạt **30–31 FPS mượt mà theo thời gian thực** nhờ cơ chế đa luồng tách biệt giữa luồng đọc camera (`_capture_loop`), luồng AI suy luận (`_ai_loop`) và luồng hiển thị giao diện. Hệ thống có cơ chế tự phục hồi chống mất kết nối khi di chuyển/cầm rung lắc camera USB.

---

## 3. Các Model trong dự án

Dự án bảo tồn nguyên vẹn model gốc đồng thời cung cấp model mới được tối ưu cho camera ngoại vi:

| Model | File Trọng Số | Label Encoder | Bản Siêu Nhẹ TFLite | Biểu đồ | Ghi chú |
|---|---|---|---|---|---|
| **cam4k (Mới nhất - Mặc định)** | [`liveness_cam4k.h5`](liveness_cam4k.h5) | [`le_cam4k.pickle`](le_cam4k.pickle) | [`web_models/liveness_cam4k.tflite`](web_models/liveness_cam4k.tflite) | [`plot_cam4k.png`](plot_cam4k.png) | Huấn luyện trên 91 video camera SC-M200, 18 epoch (`best_epoch = 14`, val loss 0.0941, accuracy 98.88%) |
| **Gốc (Nguyên bản)** | [`liveness.h5`](liveness.h5) | [`le.pickle`](le.pickle) | [`web_models/liveness.tflite`](web_models/liveness.tflite) | [`plot.png`](plot.png) | Model ban đầu của dự án |

*Lưu ý: Mặc định `run.py` và `pipeline.py` tự động nhận diện và sử dụng bản tăng tốc phần cứng XNNPACK `.tflite` tương ứng giúp thời gian suy luận chỉ mất ~0.25 ms/khung hình.*

---

## 4. Dữ liệu huấn luyện (Dataset)

- **`video cam4k/`**: 91 video 4K quay trên Camera SC-M200 phân chia độc lập theo người:
  - `train/`: 30 video real, 33 video fake.
  - `validate/`: 7 video real, 7 video fake.
  - `test/`: 7 video real, 7 video fake.
- **`dataset_cam4k/`**: 26.094 ảnh khuôn mặt (crop 32x32) trích xuất từ `video cam4k/` với bước nhảy frame `--skip 5`:
  - `train/`: 16.268 ảnh.
  - `validate/`: 5.015 ảnh.
  - `test/`: 4.811 ảnh.
- **`dataset/` & `videos/`**: Bộ dữ liệu mẫu ban đầu được giữ nguyên vẹn.

---

## 5. Các lệnh Huấn luyện và Trích xuất dữ liệu

```powershell
# Trích xuất khuôn mặt từ bộ video cam4k vào dataset_cam4k
.\venv\Scripts\python.exe scripts/extract_cam4k.py --skip 5

# Huấn luyện model liveness_cam4k
.\venv\Scripts\python.exe scripts/train_cam4k.py --epochs 18
```

Xem chi tiết quy trình huấn luyện, tiêu chí đánh giá và các thử nghiệm tại [docs/TRAINING.md](docs/TRAINING.md).

---

## 6. API Tích hợp & Bản Web

### API Server cục bộ
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-api.txt
.\venv\Scripts\python.exe api_app.py --model liveness_cam4k.h5 --labels le_cam4k.pickle
```
API phục vụ tại `http://127.0.0.1:8000`. Xem tài liệu chi tiết tại [docs/API.md](docs/API.md).

### Ứng dụng Web
```powershell
.\venv\Scripts\python.exe web_app.py
```
Truy cập qua trình duyệt tại `http://127.0.0.1:8080`.

---

## 7. Cấu trúc thư mục chính

- `chay_test_camera.bat`: Phím tắt 1-click test camera trên Windows.
- `run.py`: Ứng dụng camera chính, hỗ trợ đa luồng real-time, phím tắt đổi cam và báo cáo benchmark.
- `pipeline.py`: Pipeline tích hợp YuNet detector, tiền xử lý và thuật toán phân loại tức thì `FastBinaryDecision`.
- `inference_runtime.py`: Runtime tải và thực thi TFLite XNNPACK siêu nhẹ.
- `liveness_cam4k.h5`, `le_cam4k.pickle`: Model Keras và nhãn mới nhất.
- `web_models/`: Chứa các model TFLite (`liveness_cam4k.tflite`, `liveness.tflite`) cho Web và suy luận tốc độ cao.
- `scripts/`: Chứa script trích xuất dữ liệu (`extract_cam4k.py`), huấn luyện (`train_cam4k.py`), và các công cụ benchmark.
- `docs/`: Tài liệu chi tiết về API ([`docs/API.md`](docs/API.md)) và Huấn luyện ([`docs/TRAINING.md`](docs/TRAINING.md)).
