# Face Anti-Spoofing

## Chạy camera

```powershell
.\venv\Scripts\python.exe run.py --camera 1
```

Camera index 1 đã được xác minh trên máy này. Bỏ `--camera` để thử tự chọn nguồn có hình; việc này không xác minh được danh tính thiết bị. Q / Esc hoặc đóng cửa sổ để thoát.

### Trạng thái

- Xanh lá: model dự đoán REAL.
- Đỏ: model dự đoán FAKE.
- Vàng: đang kiểm tra, chưa chắc chắn hoặc cần điều chỉnh ảnh.
- Không có khung: không phát hiện được mặt.

`score` là đầu ra model, chưa được hiệu chỉnh thành xác suất đúng thực tế. Các ngưỡng mặc định là ngưỡng kỹ thuật ban đầu, chưa được tối ưu trên tập validation độc lập.

## Thay đổi luồng xử lý

1. YuNet 2023mar tìm mặt và 5 điểm mốc, với NMS tích hợp và ngưỡng phát hiện 0.85.
2. Kiểm tra kích thước, mặt vượt mép ảnh, điểm mốc, độ sáng và độ nét. Đây là kiểm tra sơ bộ, không bảo đảm phát hiện mọi trường hợp che mặt.
3. Cắt ảnh gốc chưa vẽ chữ/khung. Giữ tiền xử lý model cũ: BGR, 32x32, float32 / 255. Batch inference cho các mặt hợp lệ.
4. Ghép khung theo IoU; mỗi khung có lịch sử riêng. Cần ít nhất 6 quan sát, kéo dài ít nhất 0.4 giây; điểm trung bình >=0.8, >=80% phiếu đồng thuận, dự đoán hiện tại cũng phải đồng thuận.
5. Xóa lịch sử khi mất mặt hoặc chất lượng không đạt. Theo dõi hình học không phải nhận dạng danh tính: trường hợp thay người cùng vị trí mà không mất mặt vẫn cần kiểm thử thêm.
6. Vẽ giao diện sau inference, có thanh trạng thái và FPS xử lý.

Các ngưỡng có thể chỉnh bằng `--confidence` (tìm mặt), `--threshold` (phân loại). Không dùng điểm detector như điểm chống giả mạo.

## Kiểm thử

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe run.py --input videos\real-1.mp4 --headless --max-frames 90 --report outputs\optimized_real.json --snapshot outputs\optimized_real.jpg
.\venv\Scripts\python.exe run.py --input videos\fake-1.mp4 --headless --max-frames 90 --report outputs\optimized_fake.json --snapshot outputs\optimized_fake.jpg
```

Kết quả kiểm thử 2026-09-15:

| Video | Khung | Real | Fake | Checking | Adjust | Không mặt |
|---|---:|---:|---:|---:|---:|---:|
| real-1.mp4 | 90 | 83 | 0 | 7 | 0 | 0 |
| fake-1.mp4 | 90 | 0 | 20 | 19 | 49 | 2 |

Đây là smoke/regression test trên dữ liệu sẵn có, không phải kiểm thử độc lập. Không tính các khung Adjust/Checking/Không mặt là phân loại đúng. Tỷ lệ đưa ra quyết định video giả còn thấp và cần đánh giá thêm.

Bốn ảnh lỗi người dùng cung cấp: không phát hiện khung phụ ở vai/nền; phát hiện mặt chính ở hai ảnh, không phát hiện mặt ở hai ảnh che mặt. Ảnh chụp màn hình chứa khung/chữ cũ nên không thay thế thử nghiệm video gốc.

## Cấu trúc

- `run.py`: camera/video, giao diện, CLI và báo cáo.
- `pipeline.py`: YuNet, kiểm tra ảnh, tiền xử lý và tổng hợp theo thời gian.
- `tests/test_pipeline.py`: kiểm tra nền đồng màu, tiền xử lý, mất mặt, nhiều mặt và tốc độ khung hình.
- `face_extract.py`: trích xuất dataset bằng SSD cũ.
- `train_liveness.py`, `model/livenessnet.py`: huấn luyện CNN.
- `liveness.h5`, `le.pickle`: model và encoder hiện có; chưa thay hoặc huấn luyện lại.

## Giới hạn cần giải quyết khi huấn luyện lại

Dataset train/validation đã tách theo video, chưa tách theo người và chưa có test độc lập. Detector mới thay đổi vùng cắt so với dữ liệu huấn luyện SSD cũ; cần đánh giá toàn bộ trước khi kết luận chất lượng tăng. Trích xuất dữ liệu cho lần train mới nên dùng cùng detector và kiểm tra chất lượng với inference.

Encoder cũ dùng scikit-learn 1.6.1 còn môi trường dùng 1.7.0; vẫn có cảnh báo phiên bản. Thứ tự fake/real được kiểm tra khi tải model. Không tải model/encoder pickle không tin cậy.

## Nguồn model YuNet

- OpenCV Zoo: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- File: face_detection_yunet_2023mar.onnx (tương thích OpenCV 4.x).
- SHA256: 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
- Giấy phép MIT được lưu tại `face_detector/YUNET-LICENSE.txt`.
