# Huấn luyện lại với dữ liệu cũ và mới

Tài liệu này mô tả quy trình đã sửa cho đợt ngày 01/10/2026. **Gộp nguồn dữ liệu** nghĩa là dùng cả ảnh cũ và ảnh từ video mới trong bộ dữ liệu, rồi giữ riêng train, validation và test. Không đưa ảnh test vào train và không dùng điểm test để chọn model.

## 1. Bộ dữ liệu cố định

Nguồn kiểm chứng là `dataset/merged-20261001/manifest.json` và `frames.npz`. Bộ gộp có **28.957 ảnh**: 24.032 ảnh cũ đạt kiểm tra chất lượng và 4.925 ảnh trích từ 56 video mới. Bộ 24.221 ảnh cũ gốc vẫn được giữ; 189 ảnh bị loại vì lóa hoặc thiếu độ nét.

| Tập | Ảnh từ bộ cũ | Ảnh từ video mới | Tổng ảnh | Video |
|---|---:|---:|---:|---:|
| Train | 18.028 | 2.376 | **20.404** | 51 |
| Validation | 2.661 | 1.036 | **3.697** | 17 |
| Test | 3.343 | 1.513 | **4.856** | 21 |
| Tổng | **24.032** | **4.925** | **28.957** | 89 |

- Mỗi video chỉ thuộc một tập. Trùng pixel giữa các tập hoặc có nhãn mâu thuẫn bị loại trong bước chuẩn bị.
- Video mới được tách theo mã người từ thư mục; mã này chưa được xác minh danh tính độc lập.
- Video cũ giữ ranh giới train/validation/test trước đó. Một số người cũ có thể xuất hiện ở nhiều tập; video tách riêng không có nghĩa là đã tách hoàn toàn theo người.
- Đầu vào model vẫn là ảnh BGR 32×32, `float32 / 255`. Không thay cách cắt mặt giữa hai trial đang so sánh.

## 2. Cách train đã sửa

### Warmup: dùng mọi ảnh train một lần

Model khởi tạo mới, không lấy trọng số của model ngày 30/9. Trước giai đoạn cân bằng, chạy một lượt xáo trộn đủ **20.404 ảnh train**, với trọng số cân bằng lớp và video. Đây là **638 bước**, batch cuối có 20 ảnh.

Warmup được ghi riêng và **không được chọn làm checkpoint tốt nhất**. Nó bảo đảm mọi ảnh train đã thực sự được dùng ít nhất một lần.

### Sau warmup: cân bằng từng batch

Mỗi batch 32 ảnh chứa đúng:

| Nhóm | Số lượt ảnh/batch |
|---|---:|
| Nguồn cũ / fake | 8 |
| Nguồn cũ / real | 8 |
| Nguồn mới / fake | 8 |
| Nguồn mới / real | 8 |

Một balanced epoch có **638 bước = 20.416 lượt ảnh**, tức 5.104 lượt cho mỗi nhóm. Không áp dụng thêm sample weight ở giai đoạn này để tránh cân bằng hai lần.

Trong mỗi nhóm, sampler lần lượt đi hết một vòng video đã xáo trộn. Với mỗi video, sampler đi hết hàng đợi ảnh đã xáo trộn trước khi dùng lại. Hai hàng đợi giữ trạng thái khi chuyển epoch. Nhờ vậy, video dài và nguồn có nhiều ảnh không lấn át các nhóm khác.

**Lượt ảnh không đồng nghĩa với số ảnh khác nhau.** Một balanced epoch có thể lặp ảnh của video ngắn và chưa đi qua mọi ảnh của video dài. Báo cáo tách tỷ lệ ảnh được dùng trong balanced phase với tỷ lệ tính cả warmup; không gọi mỗi epoch là một lượt đủ mọi ảnh.

Thiết lập chung: seed 42, Adam `1e-4`, tối đa 50 balanced epoch; augmentation gồm xoay ±12°, zoom 0,9–1,1, dịch tối đa 10%, lật ngang và fill nearest. Early stopping có patience 12; giảm learning rate sau 4 epoch không cải thiện. Epoch trong `history.json` và `best_epoch` **không gồm warmup**; tổng số lượt chạy và optimizer steps được ghi riêng.

## 3. Chọn model bằng validation, rồi mới chấm test

Hai trial dùng cùng dữ liệu, sampler và thiết lập train:

1. **Balanced BatchNorm:** giữ CNN gốc, sửa cách đưa dữ liệu vào batch.
2. **Balanced GroupNorm:** chỉ thay 5 BatchNorm sau Conv bằng GroupNorm 8 nhóm và BatchNorm sau Dense bằng LayerNorm; các lớp khác giữ nguyên.

Tiêu chí chọn checkpoint là `val_source_class_video_loss`: tính cross-entropy trung bình trong từng video, rồi lấy trung bình đều giữa các video trong từng nhóm nguồn/lớp, cuối cùng lấy trung bình đều giữa 4 nhóm. Tiêu chí này tránh để nguồn nhiều ảnh hoặc video dài quyết định toàn bộ điểm validation. Vẫn lưu loss/F1 theo ảnh gốc và recall của lớp fake theo từng nguồn để xem model lệch ở đâu.

`select_corrected_merged.py` so cùng tiêu chí validation của hai trial, kiểm lại từ xác suất đã lưu, rồi đóng băng model được chọn trong `selection.json`. **Sau bước này** mới chạy đánh giá test và so ba mốc:

- `before`: model 30/9, trước khi gộp.
- `failed`: model gộp lần đầu ngày 01/10.
- `after`: model đã sửa quy trình train và được chọn bằng validation.

Cả ba được chấm trên cùng ảnh. Bộ test đã được xem trong quá trình phát triển, nên đây là **benchmark phát triển**, không phải tập kiểm chứng cuối hoàn toàn mới. Cần dữ liệu người, thiết bị và điều kiện quay mới để kết luận khả năng áp dụng rộng hơn.

## 4. Lệnh tái hiện và thứ tự chạy

Chạy tại thư mục gốc dự án bằng môi trường `venv` đã cài các thư viện ML. Các lệnh dưới đây là **thứ tự của đợt 01/10**, không phải đoạn lệnh để chạy lại toàn bộ trên các thư mục hiện có.

Hai trainer từ chối ghi đè thư mục output. Selector hiện cố định hai trial `merged-balanced-20261001`, `merged-groupnorm-20261001` và thư mục kết quả `merged-corrected-20261001`; chưa có tham số đổi các đường dẫn đó. Muốn làm một lần mới, dùng output mới cho trainer, sau đó cập nhật đường dẫn selector và các đường dẫn so sánh/audit tương ứng trước khi chạy bước sau. Giữ các kết quả cũ để đối chiếu.

### A. Train hai trial, chỉ dùng train/validation

```powershell
.\venv\Scripts\python.exe -X utf8 scripts/train_balanced_merged.py --dataset dataset/merged-20261001 --output outputs/merged-balanced-20261001 --epochs 50 --cpu-threads 4
.\venv\Scripts\python.exe -X utf8 scripts/train_groupnorm_merged.py --dataset dataset/merged-20261001 --output outputs/merged-groupnorm-20261001 --epochs 50 --cpu-threads 4
```

Chạy tuần tự trên CPU. Ví dụ riêng để train lại mà không đụng kết quả cũ: đổi `--output` thành `outputs/merged-balanced-repeat01` hoặc `outputs/merged-groupnorm-repeat01`. Selector cố định hiện tại sẽ **không tự đọc** hai thư mục mới này.

### B. Đóng băng lựa chọn, rồi chấm ảnh và kiểm API

Chỉ thực hiện sau khi cả hai trial hoàn tất và có `training_summary.json`:

```powershell
.\venv\Scripts\python.exe -X utf8 scripts/select_corrected_merged.py
.\venv\Scripts\python.exe -X utf8 scripts/evaluate_corrected_merged.py --run outputs/merged-corrected-20261001
.\venv\Scripts\python.exe -X utf8 scripts/evaluate_merged_model.py --run outputs/merged-corrected-20261001
```

Evaluator đầu tạo so sánh cùng bộ train/validation/test. Evaluator sau kiểm thêm hai mốc ảnh cố định của lần 30/9: 1.473 ảnh mới và 415 ảnh video cũ; đồng thời kiểm các route API bằng test client nội bộ. Kiểm route không chứng minh độ chính xác toàn bộ API, tốc độ mạng hoặc camera trực tiếp.

### C. Phát lại video qua pipeline thật

```powershell
.\venv\Scripts\python.exe -X utf8 scripts/evaluate_continuous_videos.py --dataset dataset/retrain-20260930 --baseline-model outputs/retrain-20260930/liveness.h5 --baseline-labels outputs/retrain-20260930/le.pickle --candidate-model outputs/merged-corrected-20261001/liveness.h5 --candidate-labels outputs/merged-corrected-20261001/le.pickle --output outputs/merged-corrected-20261001/continuous_runtime_comparison.json --cpu-threads 4
```

Ở bước này dùng manifest `retrain-20260930` để lấy đúng **16 video test mới** đã cố định. Phát lại tối đa 12 giây đầu mỗi video, không quá 5 mẫu/giây; không phải 16 ảnh mẫu. Hai model nhận cùng pixel sau resize/JPEG; detector, kiểm tra chất lượng và quyết định theo thời gian đều chạy lại. Phải báo cả số quyết định sai và tỷ lệ có quyết định, không tính khung chưa kết luận là đúng. Đây là phát lại offline, không đo HTTP hoặc FPS camera.

### D. Rà số liệu, vẽ biểu đồ, tạo báo cáo

```powershell
.\venv\Scripts\python.exe -X utf8 scripts/audit_balanced_merged.py --run outputs/merged-corrected-20261001
.\venv\Scripts\python.exe -X utf8 scripts/plot_corrected_merged.py --run outputs/merged-corrected-20261001
.\venv\Scripts\python.exe -X utf8 scripts/write_corrected_merged_report.py --run outputs/merged-corrected-20261001
```

Đọc `audit.json` để biết các kiểm tra có đạt hay còn lỗi. Audit đối chiếu hash, số ảnh, xác suất, cách chọn checkpoint và lịch lấy mẫu thực tế; không train lại hoặc dùng test để chọn model. Một số helper đánh giá từ chối file kết quả đã tồn tại; plot/audit có thể tạo lại artifact của chính bước đó. Không coi cả chuỗi là thao tác có thể chạy lặp mà tự động bảo toàn mọi file.

## 5. Bằng chứng cần giữ và cách dùng candidate

Mỗi trial lưu cấu hình/hash/phiên bản thư viện, `warmup_history.json`, `history.json`, `training.csv`, `training_summary.json`, xác suất validation, lịch warmup, lịch lấy mẫu từng epoch, số lần dùng từng ảnh, `best.keras`, `liveness.h5` và `le.pickle`. Kiểm tra parity bảo đảm H5 xuất ra giữ xác suất và input/output như checkpoint; không thay thế kiểm thử ứng dụng.

Seed cố định và deterministic ops giúp tái hiện trong cùng môi trường. Đổi phần cứng hoặc phiên bản thư viện vẫn có thể tạo sai số số học; dùng cấu hình/hash và xác suất đã lưu để kiểm chứng, không giả định mọi lần chạy có cùng hash model.

Tài liệu này **không tự thay model mặc định** của camera/API và không kết luận trial nào tốt hơn. Chỉ sau khi có đánh giá đầy đủ, audit đạt và đã đọc các điểm đổi giữa nguồn cũ/mới, mới thử model được chọn bằng đường dẫn rõ ràng:

```powershell
.\venv\Scripts\python.exe api_app.py --model outputs/merged-corrected-20261001/liveness.h5 --labels outputs/merged-corrected-20261001/le.pickle
```

Xem thêm [hướng dẫn API](API.md). Báo cáo cuối cần tách điểm ảnh cắt sẵn, kết quả phát lại pipeline, kiểm route API và thử camera thật; mỗi bước trả lời một câu hỏi khác nhau.

---

## 6. Đợt huấn luyện cam4k (Camera USB SC-M200 - 20/10/2026)

### 6.1. Dữ liệu và Phân tách
- **Nguồn dữ liệu:** 91 video 4K/Full HD quay thực tế từ Camera USB SC-M200 đặt trong thư mục `video cam4k/`.
- **Phân tách độc lập theo danh tính người (Person-matched split):**
  - **`train/`**: 30 video real + 33 video fake (63 video).
  - **`validate/`**: 7 video real + 7 video fake (14 video).
  - **`test/`**: 7 video real + 7 video fake (14 video).
  - Đảm bảo ranh giới: tuyệt đối không có người hoặc video nào xuất hiện đồng thời ở nhiều tập.

### 6.2. Trích xuất khuôn mặt (`scripts/extract_cam4k.py`)
- Sử dụng YuNet ONNX detector cắt các khuôn mặt (32x32 BGR, `float32 / 255.0`) với bước nhảy `--skip 5`:
  - `dataset_cam4k/train`: 16.268 ảnh.
  - `dataset_cam4k/validate`: 5.015 ảnh.
  - `dataset_cam4k/test`: 4.811 ảnh.
  - Tổng cộng: **26.094 ảnh**.

### 6.3. Quy trình huấn luyện (`scripts/train_cam4k.py`)
- **Kiến trúc:** LivenessNet kết hợp cơ chế lấy mẫu cân bằng theo video (`VideoBalancedEpochGenerator`) để tránh các video dài lấn át video ngắn.
- **Tăng cường dữ liệu:** Xoay nhẹ (±10°), zoom (0.92–1.08), dịch khung (±8%), lật ngang, cân bằng trắng nhẹ.
- **Tiêu chí tối ưu:** Giám sát `val_loss`, tự động lưu checkpoint `best.keras`, giảm learning rate khi gặp plateau.
- **Kết quả huấn luyện (18 epoch):**
  - Checkpoint tốt nhất: **Epoch 14**.
  - Train Loss: **0.0922** | Train Accuracy: **97.79%**.
  - Validation Loss: **0.0941** | Validation Accuracy: **98.01%**.
  - Độ chính xác cấp khung hình: Validation = **98.88%**, Test = **98.09%**.
  - Độ chính xác cấp video trên tập test: **14/14 (100%)**.
  - Khả năng tổng quát hóa với camera ngoài: **99.84%**.

### 6.4. Xuất Model và Triển khai
- File model Keras: [`liveness_cam4k.h5`](../liveness_cam4k.h5) & [`le_cam4k.pickle`](../le_cam4k.pickle).
- File model TFLite tăng tốc: [`web_models/liveness_cam4k.tflite`](../web_models/liveness_cam4k.tflite) (thời gian suy luận ~0.25 ms/khung hình trên CPU với XNNPACK).
- Biểu đồ huấn luyện: [`plot_cam4k.png`](../plot_cam4k.png).

