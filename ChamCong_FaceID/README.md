# FaceID Lab — nhận diện khuôn mặt từ video

Nhận diện khuôn mặt cho điểm danh: quay vài video ngắn cho mỗi người, trích đặc trưng khuôn mặt
bằng mô hình có sẵn, rồi nhận diện realtime qua webcam. Người không có trong danh sách hiện là
"Người lạ". Không cần huấn luyện mô hình hay gán nhãn từng ảnh.

- **Phát hiện + nhận diện:** InsightFace `buffalo_l` (SCRFD + ArcFace ResNet-50), chạy bằng ONNX Runtime.
- **So khớp:** mỗi người một vector trung bình (prototype); khuôn mặt được nhận khi cosine ≥ ngưỡng (mặc định 0.55).
- **Giao diện:** web chạy ở `http://localhost:8000` (FastAPI) hoặc cửa sổ OpenCV.

## Luồng làm việc

```
video/<tên người>/*.mp4 ──► [Colab GPU] preprocessing.py ──► face_embeddings.npz
                                                               │
                         relabel.py (sửa nhãn) ◄───────────────┤
                         calibrate_threshold.py (chọn ngưỡng) ◄┘
                                                               │
                                                               ▼
                                  face_prototypes.npz ──► app.py (web) / real_time_inferent.py
```

Trích embedding chạy nhanh trên GPU của Colab. Mọi bước sau chỉ cần CPU, chạy được trên laptop thường.

## Cấu trúc thư mục

| File | Vai trò |
|---|---|
| `colab_extract.ipynb` | Notebook chạy trên Google Colab để trích embedding bằng GPU |
| `preprocessing.py` | Đọc video, lọc frame xấu, trích embedding, dựng prototype |
| `face_runtime.py` | Khởi tạo InsightFace, tự chọn GPU/CPU |
| `relabel.py` | Sửa nhãn trong `face_embeddings.npz` và dựng lại prototype (không cần GPU) |
| `calibrate_threshold.py` | Đánh giá theo từng frame và đề xuất ngưỡng cosine |
| `app.py`, `web/` | Giao diện web: nhận diện, đăng ký người mới, đánh giá |
| `real_time_inferent.py` | Nhận diện bằng cửa sổ OpenCV (không cần trình duyệt) |
| `requirements.txt` | Thư viện cho máy có GPU NVIDIA |
| `requirements-cpu.txt` | Thư viện cho máy chỉ có CPU |
| `ui_design/` | Bản thiết kế giao diện (mã nguồn canvas) |

Các file `face_embeddings.npz`, `face_prototypes.npz`, `video_data/` và `logs/` chứa dữ liệu khuôn mặt
nên **không** được đưa lên GitHub (đã có trong `.gitignore`).

---

## Bước 1 — Chuẩn bị video

Đặt video theo cấu trúc **một thư mục cho mỗi người**; tên thư mục là tên hiển thị:

```
video_data/
├── nguyen_van_a/   a_1.mp4, a_2.mp4, a_3.mp4
├── tran_thi_b/     b_1.mp4, b_2.mp4
└── ...
```

Cách quay để có dữ liệu tốt:

- **Mỗi người ít nhất 2 video** (nên 3–4), mỗi video khoảng 10 giây. Người chỉ có 1 video vẫn dùng được nhưng không đánh giá được.
- **Chỉ một người trong khung hình.** Frame có nhiều khuôn mặt bị bỏ; video toàn nhiều người sẽ có 0 embedding.
- Đủ sáng, giữ máy yên (frame mờ bị loại), mặt chiếm phần lớn khung hình.
- Nhìn thẳng rồi xoay nhẹ trái/phải (dưới 35°); đổi chỗ ngồi hoặc ánh sáng giữa các video.
- Định dạng `.mp4`, `.avi`, `.mov` hoặc `.mkv`.

Video để thẳng trong `video_data/` (không có thư mục người) sẽ được tự xếp theo tên file, nhưng có thể
đoán sai (vd. `real-1.mp4` thành người tên `1`). Nên tự xếp thư mục trước.

Tải thư mục `video_data` lên Google Drive.

## Bước 2 — Trích embedding trên Google Colab (GPU miễn phí)

1. Mở https://colab.research.google.com → **File › Upload notebook** → chọn `colab_extract.ipynb`.
2. **Runtime › Change runtime type › T4 GPU**.
3. Chạy lần lượt từng ô. Ô cài đặt có sẵn bước cài cuDNN 9.10: bản cuDNN có sẵn trên Colab khiến mô hình
   âm thầm chạy bằng CPU rất chậm.
4. Ở ô `VIDEO_SRC`, sửa đường dẫn tới thư mục video trên Drive, vd. `'/content/drive/MyDrive/video_data'`.
   - Thư mục được **người khác chia sẻ** cho bạn: trên Drive, chuột phải thư mục › **Sắp xếp › Thêm lối tắt** vào My Drive, rồi dùng đường dẫn của lối tắt.
5. Ô `preprocessing.py` in thống kê từng video, ví dụ:
   ```
   [INFO] Đang xử lý video: Nga/Nga 2.mp4 (Label: Nga)
    -> Embeddings: 55; thống kê loại: {'no_face': 0, 'multiple_faces': 0, 'low_score': 0, 'small': 0, 'blurry': 55, 'pose': 10}
   ```
   Video có `Embeddings` rất thấp nên quay lại (xem lý do ở `blurry`, `multiple_faces`, ...).
   Log không được có dòng `Falling back to ['CPUExecutionProvider']`; nếu có, GPU chưa được dùng.
6. Ô cuối tải về `face_embeddings.npz`, `face_prototypes.npz`, `preprocessing_log.txt`.
   Nếu trình duyệt không tự tải, mở biểu tượng 📁 bên trái Colab, chuột phải từng file › **Download**.
7. Chép các file vừa tải vào thư mục dự án trên máy.

> Nếu `preprocessing.py` dừng với lỗi "Không trích xuất được embedding đạt chất lượng cho: ...",
> `face_embeddings.npz` vẫn đã được lưu. Dùng `relabel.py` ở bước 4 để tạo prototype từ những người còn lại.

## Bước 3 — Cài đặt trên máy (Windows)

Cần Python 3.10–3.12. Tạo môi trường riêng trong thư mục dự án:

```bash
python -m venv .venv
```

Máy **không có GPU NVIDIA** (vd. Intel Iris Xe):

```bash
.venv\Scripts\python -m pip install -r requirements-cpu.txt
```

Máy **có GPU NVIDIA**: làm theo chú thích đầu file `requirements.txt`, rồi cài thêm `fastapi` và `uvicorn[standard]`
như trong `requirements-cpu.txt` nếu muốn dùng giao diện web.

Lần chạy đầu tiên sẽ tự tải mô hình `buffalo_l` (~280 MB) vào `%USERPROFILE%\.insightface\models`.

## Bước 4 — Sửa nhãn và chọn ngưỡng (CPU, vài giây)

Xem mỗi video giống ai nhất (so với các video còn lại, không so với chính nó):

```bash
.venv\Scripts\python relabel.py
```

Nếu có nhãn sai, đổi theo nhãn cũ hoặc theo đường dẫn video, rồi lưu:

```bash
.venv\Scripts\python relabel.py --map 1=thanh --map real/real.mp4=thanh --save
```

Đánh giá theo từng frame và tìm ngưỡng:

```bash
.venv\Scripts\python calibrate_threshold.py --max-far 0.01
```

Bảng kết quả cho từng ngưỡng: tỉ lệ người quen được nhận đúng, bị gọi nhầm tên, và người lạ bị nhận
nhầm (mô phỏng bằng cách bỏ hẳn từng người khỏi danh sách). Chọn ngưỡng tại đó người lạ gần như không
bị nhận nhầm, cộng thêm khoảng 0.05 dự phòng; sửa `COSINE_THRESHOLD` trong `preprocessing.py` rồi ghi lại prototype:

```bash
.venv\Scripts\python relabel.py --save
```

## Bước 5 — Chạy demo

### Giao diện web (khuyên dùng)

```bash
.venv\Scripts\python -m uvicorn app:app --port 8000
```

Mở **http://localhost:8000** bằng Chrome hoặc Edge và cho phép quyền camera.

- **Nhận diện:** khung xanh kèm tên là người quen, khung cam "Người lạ" là người không có trong danh sách.
  Thanh trượt đổi ngưỡng ngay (chỉ trong phiên đó). Nhật ký điểm danh ghi vào `logs/attendance_<ngày>.csv`,
  mỗi người tối đa một lần mỗi 30 giây; tải CSV bằng link trong trang.
- **Đăng ký:** nhập tên → quay 2–4 đoạn 6 giây (mỗi đoạn lấy 30 frame) → kiểm tra trùng → lưu.
  Người mới được thêm vào `face_embeddings.npz` và `face_prototypes.npz`, nhận diện được ngay.
  Đăng ký qua web chỉ dùng mô hình phát hiện + nhận diện nên không lọc góc nghiêng như ở Colab.
- **Đánh giá:** số liệu và biểu đồ theo ngưỡng, tính lại từ `face_embeddings.npz` mỗi lần mở server.

Trên CPU laptop tốc độ khoảng 3–6 FPS, đủ cho điểm danh.

Muốn mở trên điện thoại hoặc gửi link tạm cho người khác (máy vẫn phải bật):
cài [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) rồi chạy

```bash
cloudflared tunnel --url http://localhost:8000
```

và dùng link `https://...trycloudflare.com` được in ra. Ai có link đều mở được, nên chỉ bật khi cần.

### Cửa sổ OpenCV (không cần trình duyệt)

```bash
.venv\Scripts\python real_time_inferent.py
```

Nhấn `q` để thoát. Tên có dấu tiếng Việt không hiển thị được trong cửa sổ OpenCV; giao diện web thì được.

---

## Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'insightface'` | Đang dùng Python khác (vd. Anaconda). Chạy bằng `.venv\Scripts\python ...` |
| `CUDAExecutionProvider không khả dụng ... Fallback sang CPU` | Bình thường trên máy không có GPU NVIDIA |
| Colab rất chậm, log có `CUDNN_STATUS_NOT_SUPPORTED` hoặc `Falling back to ['CPUExecutionProvider']` | Chạy lại ô cài đặt (có `nvidia-cudnn-cu12==9.10.2.21`) rồi chạy lại `preprocessing.py` |
| Một người có `Embeddings: 0` | Video có nhiều người trong khung hình, quá mờ hoặc quá tối: quay lại |
| Trang web báo "Không mở được webcam" | Cho phép quyền camera trên trình duyệt; đóng ứng dụng khác đang dùng camera. Webcam chỉ chạy trên `localhost` hoặc HTTPS |
| Người lạ bị nhận thành người quen | Tăng ngưỡng; chạy `calibrate_threshold.py` để chọn lại |
| Người quen hay bị "Người lạ" | Quay thêm video ở điều kiện ánh sáng giống nơi demo, đăng ký thêm qua web |
| `UnicodeEncodeError: 'charmap' codec ...` khi chạy script | Đặt `$env:PYTHONIOENCODING="utf-8"` trong PowerShell trước khi chạy |

## Quyền riêng tư

Embedding, prototype, nhật ký và ảnh chụp là dữ liệu sinh trắc học của người trong danh sách:
hãy xin phép trước khi thu thập, không đưa lên GitHub hay chia sẻ công khai, và xóa khi không còn dùng.
