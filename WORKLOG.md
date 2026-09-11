# Worklog — RGB Face Anti-Spoofing

Worklog ghi thay đổi có bằng chứng trong repository. Thành viên nhóm bổ sung owner/commit cho các giai đoạn thu dữ liệu, huấn luyện và demo.

## 11/09/2026 — Audit baseline và tái cấu trúc 2.0

### Phạm vi đã kiểm tra

- Repository trên ổ đĩa chỉ còn `face_extract.py`, README sơ sài và OpenCV SSD detector.
- Lịch sử Git cho thấy baseline cũ gồm CNN Keras 32×32, random image split, `run.py` webcam và label encoder.
- Dataset khoảng 10.500 ảnh, model `.h5` và `le.pickle` không có trong workspace nên chưa thể train/evaluate lại.

### Đã hoàn thành

- Tách package theo detector, quality, landmarks, challenge, motion, texture, fusion và pipeline.
- Chuyển runtime model sang ONNX/OpenCV DNN, training dùng MobileNetV3-Small.
- Thêm challenge-response ngẫu nhiên dựa trên landmark và temporal motion cue.
- Thêm fail-closed decision policy, reliability, evidence coverage và temporal smoothing.
- Thêm data layout, face crop/dedup, subject-disjoint split và leakage audit.
- Thêm APCER/BPCER/ACER, AUC/EER, frame/video aggregation và report hash.
- Thêm unit tests, CI, README, architecture, method selection, dataset/evaluation protocol và project report.

### Bằng chứng

- Source: `src/face_antispoofing/`, `scripts/`, `tests/`.
- Architecture: `ARCHITECTURE.md`.
- Test command: `python -m unittest discover -s tests -v`.
- Kết quả ban đầu: 33 test pass trên bundled Python 3.12; camera/model integration chưa chạy do thiếu runtime dependencies, dataset và ONNX artifact.

### Quyết định kỹ thuật

- Không giữ random frame split vì data leakage.
- Không đưa rPPG/depth/thermal vào default runtime.
- Không coi challenge timeout là `SPOOF`.
- Không công bố metric khi chưa có data/model thật.

## Backlog thực nghiệm

- [ ] Gán owner và khôi phục/thu lại video có metadata.
- [ ] Tạo manifest và audit số subject/video/frame từng split.
- [ ] Train LBP baseline và MobileNetV3-Small.
- [ ] Tune threshold trên validation, khóa test.
- [ ] Chạy P1–P4 và ablation.
- [ ] Ghi false accept/false reject cases.
- [ ] Quay demo và thêm evidence.
- [ ] Cập nhật report actual, commit/hash và kết luận RQ1–RQ4.
