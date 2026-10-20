# Tích hợp API chống giả mạo

API nhận ảnh từ hệ thống của bạn và trả đánh giá thật/giả cùng vị trí khuôn mặt.
Phần nhận diện danh tính vẫn do dự án của bạn xử lý. API không trả tên người,
không lưu ảnh và không tự cấp quyền đăng nhập/chấm công.

## Chạy trên máy

Cài vào môi trường Python 3.11 của dự án:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-api.txt
.\venv\Scripts\python.exe api_app.py --model outputs/retrain-20260915/liveness.h5 --labels outputs/retrain-20260915/le.pickle
```

Lệnh trên chọn rõ model huấn luyện lại ngày 15/9. Nếu không truyền `--model`
và `--labels`, API dùng `liveness.h5` và `le.pickle` ở thư mục gốc. Không tự
chọn checkpoint chỉ vì có file mới. `--cpu-threads 2` là mặc định.
Model được tải và warmup một lần lúc khởi động, dùng cùng `Pipeline` với `run.py`.

- API: `http://127.0.0.1:8000`
- Trạng thái và mã SHA256 model: `GET /v1/health`
- OpenAPI để import vào Postman: `GET /openapi.json`
- Server dùng Waitress, hỗ trợ Windows và Linux:
  <https://flask.palletsprojects.com/en/stable/deploying/waitress/>

Một process giữ model và lịch sử các phiên trong RAM. Chạy thêm worker độc lập
sẽ không chia sẻ phiên; cần định tuyến cố định phiên đến cùng worker hoặc triển khai
kho trạng thái chung trước khi mở rộng. Dịch vụ hiện phù hợp chạy cạnh backend nhận diện
trên cùng máy. Đổi web thành API không tự làm model chính xác hơn hoặc máy chủ mạnh hơn.

## Hợp đồng request/response

| Phương thức | Đường dẫn | Công dụng |
|---|---|---|
| POST | `/v1/analyze` | Một ảnh độc lập, kết quả ngay |
| POST | `/v1/sessions` | Tạo phiên nhiều khung hình; trả `session_id` |
| POST | `/v1/sessions/{session_id}/frames` | Gửi từng khung hình theo thứ tự |
| DELETE | `/v1/sessions/{session_id}` | Kết thúc phiên |

Request ảnh là **JPEG nhị phân**, `Content-Type: image/jpeg`, không cần base64
hoặc multipart. Tối đa 2.000.000 byte; mỗi chiều từ 32 đến 1920 pixel.
Client Python tự giảm cạnh dài xuống 960 pixel, áp dụng cho cả ảnh ngang và dọc.
Gửi ảnh màu chứa đầy đủ một mặt, chưa vẽ khung/chữ. API tự tìm và cắt mặt;
không gửi ảnh crop 32×32 vì đó chỉ là kích thước đầu vào nội bộ của CNN.

Response mẫu có cấu trúc sau (các số chỉ minh họa):

```json
{
  "request_id": "request-id",
  "mode": "temporal",
  "status": "real",
  "faces": [{
    "box": [120, 80, 310, 330],
    "status": "real",
    "scores": {"fake": 0.04, "real": 0.96},
    "quality": null,
    "track_id": 1,
    "detector_score": 0.97
  }],
  "image": {"width": 640, "height": 480, "sha256": "sha256-cua-jpeg"},
  "model": {"id": "liveness-...", "sha256": "sha256-cua-model", "labels": ["fake", "real"]},
  "processing_ms": 40.0,
  "total_ms": 42.0
}
```

`box` là `[trái, trên, phải, dưới]` trên ảnh JPEG gửi đến API. API đọc theo
hướng pixel đã mã hóa, không xoay ảnh theo EXIF. `image.sha256` giúp đối chiếu
đúng ảnh và `request_id` giúp đối chiếu request; các trường này không phải chữ ký xác thực.
`track_id` chỉ theo dõi vị trí trong phiên, không phải mã danh tính một người.

| `status` cấp ngoài cùng | Cách hệ thống nhận diện xử lý |
|---|---|
| `real` | Model dự đoán thật; kết hợp với danh tính của chính ảnh/mặt này |
| `fake` | Từ chối lượt kiểm tra này |
| `checking` | Chưa đủ lịch sử, tiếp tục gửi khung hình |
| `uncertain` | Chưa nhất quán, chưa chấp nhận |
| `adjust` | Hướng dẫn điều chỉnh theo `faces[0].quality` |
| `no_face` | Yêu cầu đưa mặt vào khung |
| `multiple_faces` | Yêu cầu chỉ một người trong khung; không tự chọn người |

`scores` có thể là null khi chưa đủ dữ liệu hoặc chất lượng ảnh không đạt.
Điểm số chưa được hiệu chỉnh thành xác suất đúng thực tế.

Phiên mặc định cần tối thiểu 6 quan sát trong ít nhất 0,4 giây, dùng quy tắc tổng hợp
giống camera local. Gửi tuần tự, chỉ một request đang xử lý cho mỗi luồng, bỏ khung cũ
nếu máy chủ bận. Không cần đợi đủ khung rồi gửi đồng loạt. Phiên hết hạn sau 30 giây
không có ảnh hợp lệ, tối đa 128 phiên. Lịch sử bị xóa khi mất mặt, chất lượng không đạt,
nhiều mặt hoặc lỗi xử lý ảnh; khoảng cách giữa quan sát quá 0,5 giây cũng làm học lại lịch sử.

API trả 429 nếu model đang bận, kèm `Retry-After: 1`; ứng dụng giữ hiển thị camera
độc lập với request và gửi khung mới nhất khi thử lại. Các lỗi khác: 401 sai API key,
404 phiên hết hạn/không tồn tại, 413 ảnh quá nặng, 415 sai Content-Type, 422 ảnh lỗi,
503 đầy phiên. Lỗi có dạng `{"error":{"code":"...","message":"..."},"request_id":"..."}`.
Lỗi bị server HTTP hoặc proxy chặn trước ứng dụng (ví dụ request quá nặng) có thể
chỉ có HTTP status và văn bản; client cũng xử lý trường hợp này.
Không biến lỗi/timeout thành kết quả `real`.

## Gắn vào hệ thống nhận diện có sẵn

Chép `clients/antispoof_client.py` vào dự án Python cần tích hợp hoặc import khi
chạy từ thư mục dự án này. Client dùng `requests`, `numpy`, `opencv-python`.
Với Java, C#, Node.js hoặc Postman, gọi cùng các endpoint HTTP theo OpenAPI.

Luồng đề xuất: lấy khung hình -> API chống giả mạo -> đối chiếu danh tính trên
cùng khung và cùng mặt -> dự án chủ quyết định chấp nhận.

```python
from clients.antispoof_client import AntiSpoofClient, encode_jpeg
import cv2
import numpy as np

with AntiSpoofClient() as api:
    session_id = api.create_session()
    try:
        # frame_bgr là khung hình mới từ camera của dự án hiện có.
        jpeg = encode_jpeg(frame_bgr)
        result = api.frame(session_id, jpeg)
        if result['status'] == 'real':
            same_frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            x1, y1, x2, y2 = result['faces'][0]['box']
            face_for_identity = same_frame[y1:y2, x1:x2]
            # Gọi bộ nhận diện của dự án với face_for_identity ở đây.
            # Chỉ chấp nhận khi danh tính cũng khớp theo chính sách của dự án.
        # checking/adjust/uncertain: lấy frame mới và gọi api.frame tiếp.
    finally:
        api.close_session(session_id)
```

Ví dụ trên minh họa một lượt gọi trong vòng lặp camera có sẵn, không mở một camera thứ hai.
Giữ cùng `session_id` cho một người trong một lượt kiểm tra; tạo phiên mới khi đổi người.
Không tái sử dụng kết quả `real` cũ cho ảnh hoặc người khác. Nếu cả hai model chạy trong
cùng tiến trình Python, có thể gọi `Pipeline` trực tiếp để bỏ phần HTTP.

Ảnh đơn phù hợp thử tích hợp, nhưng một ảnh hay việc lặp lại ảnh trong phiên chưa chứng minh
người đang có mặt trực tiếp. Hệ thống hiện chưa có thử thách chớp mắt/quay đầu hay kiểm tra
độ sâu; cần kiểm thử tấn công bằng ảnh và video trên dữ liệu độc lập trước khi dùng cho xác thực.

## Kiểm thử và đo tốc độ

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe scripts/benchmark_api.py --image outputs/camera_optimized.jpg --model outputs/retrain-20260915/liveness.h5 --labels outputs/retrain-20260915/le.pickle --iterations 30 --output outputs/api-benchmark.json
```

Benchmark dùng cùng JPEG và model, warmup trước, đối chiếu nhãn/khung/điểm số rồi đo median
và p95 (95% request nhanh hơn mốc này). File output phải là tên mới.
Ảnh mẫu hiện là ảnh camera đã lưu có chú thích: chỉ dùng đo tốc độ/tính tương đương,
không dùng tính độ chính xác. Không suy tốc độ nhiều người dùng từ một luồng request.

Kết quả đo trên máy ngày 23/9/2026, model huấn luyện lại, 2 luồng CPU,
ảnh `camera_optimized.jpg` 640×567, 30 lượt mỗi cách sau 5 lượt warmup:

| Cách gọi | Median | P95 | Tốc độ tuần tự |
|---|---:|---:|---:|
| Gọi pipeline trực tiếp, gồm giải mã JPEG | 49,14 ms | 57,84 ms | 19,93 ảnh/giây |
| Gọi API qua localhost | 55,85 ms | 61,62 ms | 17,81 ảnh/giây |

Chênh lệch trung vị khoảng 6,70 ms; vị trí mặt, trạng thái và điểm số khớp giữa hai cách.
Bằng chứng: `outputs/api-benchmark-20260923.json`. Kiểm tra thêm qua HTTP thật:
phiên mới trả `checking`, phiên đủ quan sát trả `real`, phiên khác bắt đầu lại từ
`checking`, phiên đã xóa trả 404, JPEG hỏng trả 422. Việc lặp ảnh để thử phiên chỉ
kiểm tra hợp đồng API, không chứng minh khả năng phát hiện replay.

## Dùng từ máy khác

Thiết lập `ANTISPOOF_API_KEY` bằng giá trị bí mật riêng, chạy `--host 0.0.0.0`
và gửi `Authorization: Bearer <key>` từ backend. Client dùng `allow_remote=True` khi
chủ động chọn máy chủ khác. Không nhúng key vào JavaScript công khai.
Dùng HTTPS và kiểm soát truy cập mạng khi triển khai bên ngoài máy; không mở cổng hoặc
deploy dịch vụ tự động chỉ bằng việc chạy thử local. Bản API không cần giao diện web.
