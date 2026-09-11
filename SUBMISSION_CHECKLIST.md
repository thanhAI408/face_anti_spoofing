# Bản đồ nộp đồ án

Trang này nối từng hạng mục cần chấm với code, tài liệu và bằng chứng. Các ô kết quả thực nghiệm chỉ được đánh dấu sau khi dataset/model được đưa vào máy và chạy đúng protocol.

## Deliverables

| Hạng mục | Vị trí | Trạng thái |
|---|---|---|
| Source code đã tái cấu trúc | `src/face_antispoofing/` | Có |
| Cấu hình threshold/weight | `config/default.toml` | Có baseline, cần tune |
| Architecture diagram | `ARCHITECTURE.md` | Có |
| Lựa chọn phương pháp có tài liệu tham khảo | `docs/METHOD_SELECTION.md` | Có |
| Data collection/split protocol | `docs/DATASET_PROTOCOL.md` | Có |
| Script trích xuất + manifest | `scripts/prepare_dataset.py` | Có |
| Script training + ONNX export | `scripts/train_texture.py` | Có |
| Script evaluation chuẩn PAD | `scripts/evaluate_texture.py` | Có |
| Evaluation protocol | `docs/EVALUATION.md` | Có |
| Evaluation result thật | `reports/evaluation/` | Chưa có dataset/model để chạy |
| Unit tests | `tests/` | 33/33 pass bằng `unittest` |
| Manual webcam evidence | `eval/manual_challenge_cases.csv` + ảnh/video | Chưa thực hiện |
| Báo cáo đồ án | `docs/PROJECT_REPORT.md` | Có bản kỹ thuật, chờ điền kết quả |
| Demo video | Link do nhóm cập nhật | Chưa có |

## Bản đồ yêu cầu tới code

| Yêu cầu | File/symbol | Bằng chứng nên nộp |
|---|---|---|
| Nhận webcam và hiển thị kết quả | `cli.py:run_demo` | Video demo có FPS/latency |
| Phát hiện đúng một khuôn mặt | `detector.py:OpenCVSSDDetector` | Case 0/1/2 khuôn mặt |
| Kiểm soát chất lượng | `quality.py:QualityGate` | Case tối, mờ, xa, gần |
| Luồng CNN/texture | `texture.py`, `train_texture.py` | ONNX hash + report test |
| Xét point/landmark | `landmarks.py:extract_features` | Overlay/demo chớp/quay |
| Thử thách ngẫu nhiên | `challenge.py:ChallengeEngine` | Ghi lại nhiều thứ tự action |
| Chuyển động theo thời gian | `motion.py:LandmarkMotionAnalyzer` | Ablation có/không motion |
| Hợp nhất đa luồng | `fusion.py:FusionEngine` | Ablation + test fail-closed |
| Không rò rỉ train/test | `dataset.py:assert_no_subject_leakage` | Manifest audit |
| Metric PAD | `evaluation/metrics.py` | APCER/BPCER/ACER frame + video |

## Trình tự demo đề xuất (3–4 phút)

1. Mở sơ đồ kiến trúc và nói threat model: ảnh in + replay trên điện thoại/màn hình.
2. Cho webcam thấy `NO_FACE`, sau đó mặt quá xa/mờ trả `RETRY`.
3. Người thật hoàn thành hai thử thách ngẫu nhiên và nhận `LIVE`.
4. Đưa ảnh in: texture/challenge không qua, hệ thống không trả `LIVE`.
5. Đưa video replay: trình bày kết quả thực tế, không cắt bỏ case fail.
6. Mở evaluation report có model/manifest hash và metric video-level.
7. Kết thúc bằng giới hạn: chưa bảo vệ camera injection/3D mask cao cấp.

## Điều kiện trước khi tuyên bố hoàn thành thực nghiệm

- [ ] Dataset có consent và data card.
- [ ] Có ít nhất ba subject để tách train/val/test; khuyến nghị nhiều hơn đáng kể.
- [ ] Không có subject/video/session trùng split.
- [ ] Threshold chỉ tune trên validation.
- [ ] Báo cáo cả APCER và BPCER; không chỉ báo accuracy.
- [ ] Có kết quả video-level và attack-specific.
- [ ] Có ablation cho texture-only, challenge-only, texture+challenge và full fusion.
- [ ] Có latency/FPS trên đúng laptop demo.
- [ ] Kết quả manual có cả lỗi/uncertain, không chỉ ảnh pass.
- [ ] README thay trạng thái “chưa có kết quả” bằng link báo cáo thật.
