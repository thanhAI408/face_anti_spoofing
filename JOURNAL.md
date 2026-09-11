# Weekly Journal — RGB Face Anti-Spoofing

## Tuần tái cấu trúc nền tảng

### Mục tiêu

- Chuyển project môn học thành cấu trúc đủ rõ để tiếp tục làm đồ án.
- Chọn các luồng khả thi với webcam phổ thông.
- Thiết kế lại evaluation để kết quả không bị cao ảo do frame leakage.

### Đã hoàn thành

- Audit code cũ từ lịch sử Git và xác định giới hạn CNN 32×32/random split.
- Tham khảo hướng texture/gradient, auxiliary supervision, domain generalization, challenge-response và metric PAD.
- Xây pipeline ba signal + quality gate + decision state.
- Xây data/training/evaluation tooling và bộ tài liệu theo phong cách P232.

### Khó khăn và cách xử lý

| Khó khăn | Cách xử lý |
|---|---|
| Dataset/model cũ không nằm trong repo | Không bịa kết quả; để trạng thái pending và cung cấp command tái lập |
| Nhiều frame cùng video gần trùng | Sample FPS + average-hash filter + video-level metrics |
| Challenge có thể gây false reject | Calibration, consecutive frames, timeout trả `RETRY` |
| Signal có thể thiếu/lỗi | Reliability + minimum evidence coverage + `UNCERTAIN` |
| Phần cứng hạn chế | ONNX/OpenCV CPU + MediaPipe; bỏ thermal/depth/IR |

### Bài học

- Với PAD, protocol dữ liệu quan trọng không kém backbone.
- “Nhiều luồng” chỉ có giá trị khi được ablation và không cùng lặp một lỗi.
- Accuracy frame-level không đủ để chứng minh hệ thống hoạt động ngoài video đã thu.
- Trạng thái không chắc chắn là một phần của thiết kế an toàn, không phải lỗi giao diện.

### Kế hoạch tuần sau

1. Khôi phục video gốc và lập data inventory.
2. Thu bổ sung print/replay theo ma trận điều kiện.
3. Train baseline, phân tích error và chốt threshold.
4. Chạy manual webcam test trên đúng máy demo.
5. Cập nhật journal bằng số thật, ảnh/video evidence và quyết định keep/reject từng signal.

## Mẫu cho các tuần tiếp theo

### Mục tiêu tuần này

- ...
### Đã hoàn thành

- ...

### Đóng góp trong tuần

| Thành viên | Vai trò | Việc làm | Bằng chứng/commit |
|---|---|---|---|
| ... | ... | ... | ... |

### Khó khăn và giải pháp

- ...

### Quyết định kỹ thuật

- ...

### Bài học

- ...

### Kế hoạch tuần sau

- ...
