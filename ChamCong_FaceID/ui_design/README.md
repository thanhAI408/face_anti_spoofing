# Thiết kế UI/UX demo — FaceID Lab

Bản thiết kế giao diện cho demo nhận diện khuôn mặt, gồm 3 màn hình:

| File | Màn hình |
|---|---|
| `Main.dc.html` | Nhận diện trực tiếp: khung webcam, thanh trượt ngưỡng, nhật ký điểm danh |
| `Enroll.dc.html` | Đăng ký người mới: thông tin → quay video (kiểm tra chất lượng frame) → xác nhận |
| `Report.dc.html` | Đánh giá mô hình: biểu đồ theo ngưỡng, dữ liệu theo người |
| `canvas.json` | Bố cục 3 màn hình trên canvas |

Xem và chỉnh sửa trực tiếp tại canvas: https://claude.ai/artifact/WC6qXNgisogVii64bDvcT4

Các file `.dc.html` là mã nguồn của canvas, cần runtime của canvas (`support.js`) để hiển thị,
nên mở trực tiếp bằng trình duyệt sẽ không đúng. Số liệu ngưỡng và chất lượng video lấy từ
`calibrate_threshold.py` và `preprocessing_log.txt`; nhật ký điểm danh là dữ liệu minh họa.
