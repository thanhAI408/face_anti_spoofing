# Giao thức đánh giá

## 1. Mục tiêu

Đánh giá phải trả lời bốn câu hỏi riêng:

1. Texture model có phân biệt live/spoof trên người chưa thấy không?
2. Nó có tổng quát sang session, ánh sáng, thiết bị và loại attack khác không?
3. Thêm challenge/motion/fusion có thực sự giảm lỗi so với texture-only không?
4. Pipeline có chạy đủ nhanh và dễ dùng trên đúng laptop + webcam demo không?

Không dùng accuracy duy nhất để trả lời cả bốn câu.

## 2. Nhãn và confusion matrix

Trong code:

```text
0 = spoof / attack
1 = live / bona fide
model output order = [spoof_logit, live_logit]
```

| Ground truth \ Prediction | SPOOF | LIVE |
|---|---:|---:|
| Spoof | TN | FP (attack được chấp nhận nhầm) |
| Live | FN (người thật bị từ chối nhầm) | TP |

Vì positive class trong code là live, `FP` tương ứng lỗi bảo mật quan trọng: spoof bị nhận là live.

## 3. Metric

```text
APCER = attack classified as live / all attack presentations
      = FP / (FP + TN)

BPCER = bona fide classified as spoof / all bona fide presentations
      = FN / (FN + TP)

ACER  = (APCER + BPCER) / 2

Precision_live = TP / (TP + FP)
Recall_live    = TP / (TP + FN)
F1_live        = 2TP / (2TP + FP + FN)
```

APCER/BPCER theo tinh thần ISO/IEC 30107-3:2023; ACER là metric tổng hợp thường dùng trong benchmark FAS. Báo cáo thêm ROC-AUC và EER nhưng không thay thế error rate tại threshold triển khai.

## 4. Đơn vị đánh giá

Frame-level được lưu để debug model. Video-level là kết quả chính:

```text
video_live_score = mean(frame_live_score của cùng source_video)
video_prediction = video_live_score >= threshold
```

Lý do: hàng trăm frame liên tiếp không phải hàng trăm thử nghiệm độc lập. Báo cáo chỉ frame-level dễ làm sample count và độ chắc chắn trông lớn hơn thực tế.

## 5. Chọn threshold

1. Train model trên `train`.
2. Early stopping và chọn checkpoint bằng `validation ACER`.
3. Tìm threshold làm ACER validation thấp nhất.
4. Khóa model + threshold.
5. Chạy test một lần và ghi hash.

Không chọn threshold bằng test và không dùng ảnh demo đẹp để tune.

## 6. Các protocol bắt buộc

### P1 — Subject-disjoint internal

Train/validation/test không chung subject. Báo cáo frame/video APCER, BPCER, ACER, ROC-AUC, EER và attack-specific APCER.

### P2 — Session/device holdout

Nếu có từ hai camera hoặc đủ nhiều session, giữ toàn bộ một device/session condition làm test. Protocol này đo domain shift thực tế hơn random split.

### P3 — Attack holdout

Train không có một biến thể attack cụ thể, ví dụ replay từ laptop hoặc glossy print; test riêng biến thể đó. Kết quả cho biết khả năng chống attack chưa thấy.

### P4 — End-to-end webcam

Thực hiện manual test trên đúng máy demo, ghi video màn hình hoặc ảnh evidence và hoàn thành `eval/manual_challenge_cases.csv`.

### P5 — External/cross-dataset

Train bằng dữ liệu tự thu, khóa toàn bộ model/threshold rồi test trên partition chính thức của Replay-Attack hoặc OULU-NPU nếu được cấp quyền. Báo cáo riêng, không trộn với P1. Đây là bằng chứng trực tiếp cho câu hỏi “model có chỉ học camera/background của nhóm hay không?”.

## 7. Ablation đa luồng

| Cấu hình | Mục tiêu |
|---|---|
| Texture only | Baseline thụ động |
| Challenge only | Chứng minh giới hạn của active signal |
| Texture + challenge | Kiểm tra hai signal chính có bổ sung nhau không |
| Texture + challenge + motion | Chứng minh motion có giá trị; nếu không, bỏ motion |
| Full fusion không smoothing | Đo tác dụng của temporal smoothing |

Giữ nguyên test cases và threshold policy hợp lệ cho từng cấu hình. Không kết luận “nhiều luồng tốt hơn” nếu ablation không cải thiện APCER/BPCER hoặc làm usability xấu rõ rệt.

## 8. Manual cases tối thiểu

| Case | Input | Kỳ vọng |
|---|---|---|
| M-01 | Không có mặt | `NO_FACE` |
| M-02 | Hai người cùng khung | `RETRY: multiple_faces` |
| M-03 | Mặt quá tối | `RETRY: too_dark` |
| M-04 | Mặt mờ | `RETRY: too_blurry` |
| M-05 | Người thật hoàn thành challenge | `LIVE` khi đủ evidence |
| M-06 | Ảnh in đứng yên | Không được `LIVE` |
| M-07 | Ảnh in được lắc/quay | Không được `LIVE` |
| M-08 | Ảnh tĩnh trên điện thoại | Không được `LIVE` |
| M-09 | Video replay có chớp mắt | Không được `LIVE` nếu challenge không khớp ngẫu nhiên |
| M-10 | Video replay có quay đầu | Không được `LIVE` nếu challenge không khớp ngẫu nhiên |
| M-11 | Người thật đeo kính | Đo BPCER/usability, không mặc định pass |
| M-12 | Challenge timeout | `RETRY`, không cáo buộc `SPOOF` |

## 9. Chỉ tiêu nghiệm thu đề xuất

Đây là target của đồ án, chưa phải kết quả hiện tại:

| Chỉ tiêu | Target đề xuất | Cách đo |
|---|---:|---|
| Video ACER nội bộ | ≤ 10% | P1 test |
| Print APCER | ≤ 10% | P1/P3 |
| Replay APCER | ≤ 15% | P1/P3 |
| BPCER người thật | ≤ 10% | P1 + M-05/M-11 |
| Challenge success người thật | ≥ 90% trong tối đa 2 lần | P4 |
| End-to-end FPS | ≥ 15 FPS | Laptop demo, 640×480 |
| P95 decision latency | Ghi số đo, không ước lượng | P4 |
| Unit test | 100% pass | CI/local |

Nếu không đạt, báo cáo đúng số và phân tích failure cases; không hạ target sau khi xem test mà không giải trình.

## 10. Lệnh evaluation

```powershell
python scripts/evaluate_texture.py `
  --manifest data/processed/manifest.csv `
  --model artifacts/models/texture_mobilenetv3.onnx `
  --threshold auto `
  --output-dir reports/evaluation
```

Báo cáo sinh ra phải chứa:

- Timestamp UTC.
- Command.
- Manifest SHA-256.
- Model SHA-256.
- Threshold chọn trên validation.
- Frame/video metrics.
- Confusion matrix.
- APCER từng attack type.
- Median/P95 model latency.
- Phần diễn giải giới hạn.

## 11. Evaluation report checklist

- [ ] Không có subject leakage.
- [ ] Test chưa được dùng để chọn model/threshold.
- [ ] Có số subject/video/frame từng split.
- [ ] Có confusion matrix và error case cụ thể.
- [ ] Có video-level result.
- [ ] Có attack-specific APCER.
- [ ] Có ablation đa luồng.
- [ ] Có latency/FPS trên máy demo.
- [ ] Có manifest/model hash.
- [ ] Không trộn target với actual.

Nguồn tiêu chuẩn: <https://www.iso.org/standard/79520.html>
