# Lựa chọn phương pháp chống giả mạo

## Phạm vi và threat model

Bản đồ án ưu tiên camera RGB phổ thông và hai kiểu tấn công có thể tự thu thập hợp pháp:

- Print attack: ảnh khuôn mặt in trên giấy phẳng/cong, giấy thường hoặc bóng.
- Replay attack: ảnh/video phát lại qua điện thoại, tablet hoặc laptop với nhiều độ sáng/khoảng cách.

Hệ thống không tuyên bố giải quyết hoàn toàn 3D mask, deepfake tương tác realtime, virtual-camera injection hay malware can thiệp trước OpenCV. Đây là ranh giới quan trọng giữa PAD ở mức hình ảnh và bảo mật thiết bị đầu cuối.

## So sánh phương pháp

| Phương pháp | Phần cứng | Chi phí tính toán | Điểm mạnh | Rủi ro/điểm yếu | Quyết định |
|---|---|---:|---|---|---|
| Binary CNN một frame 32×32 | RGB | Thấp | Dễ train/demo | Mất chi tiết; dễ học background/device; cross-domain yếu | Thay thế |
| MobileNetV3 texture 224×224 | RGB | Thấp-vừa | Nhẹ, pretrained, xuất ONNX CPU; giữ nhiều dấu vết không gian | Vẫn phụ thuộc domain và dữ liệu | Chọn làm passive branch |
| CDCN/depth-supervised FAS | RGB khi inference | Vừa-cao | Học gradient/độ sâu giả, mạnh về fine-grained cue | Training/loss/dữ liệu phức tạp hơn cho nguồn lực đồ án | Baseline nghiên cứu giai đoạn 2 |
| LBP/color texture | RGB | Thấp | Baseline cổ điển, dễ giải thích | Kém trước camera/ánh sáng mới | Dùng làm baseline so sánh nếu còn thời gian |
| Chớp mắt đơn lẻ | RGB | Thấp | Chặn ảnh tĩnh đơn giản | Replay video có thể chứa blink | Không dùng riêng lẻ |
| Random challenge-response | RGB | Thấp | Tăng entropy, dùng blink/quay đầu, không cần sensor | Gây bất tiện; replay/deepfake tương tác vẫn có thể thích ứng | Chọn làm active gate |
| Landmark temporal non-rigid motion | RGB | Thấp | Tận dụng chuỗi frame; bổ sung cue hình học | Heuristic, MediaPipe có thể dao động trên màn hình | Chọn với trọng số thấp |
| Optical flow toàn mặt/nền | RGB | Vừa | Nhìn được chuyển động phẳng so với 3D | Nhạy camera motion, background và tuning | Backlog/ablation |
| rPPG từ màu da | RGB | Vừa, cần nhiều frame | Tín hiệu sinh lý không tiếp xúc | Rất nhạy ánh sáng, auto exposure, nén và motion; tăng thời gian | Chưa dùng trong quyết định chính |
| Depth/IR/thermal | Sensor riêng | Cao | Cue vật lý mạnh hơn cho mask/planar attack | Không phù hợp ngân sách và yêu cầu webcam cơ bản | Loại khỏi phạm vi |

## Căn cứ nghiên cứu

### Texture và gradient

CDCN cho thấy gradient cục bộ và chi tiết tần số là cue quan trọng cho PAD, đồng thời nhấn mạnh vấn đề thay đổi môi trường. Bản hiện tại chọn MobileNetV3-Small vì dễ huấn luyện, export và chạy CPU hơn; CDCN là baseline học thuật hợp lý cho vòng thí nghiệm sau, không nên chép nguyên kiến trúc khi chưa có đủ protocol/compute.

- Yu et al., *Searching Central Difference Convolutional Networks for Face Anti-Spoofing*, CVPR 2020: <https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.html>
- Wang et al., *Deep Spatial Gradient and Temporal Depth Learning for Face Anti-Spoofing*, CVPR 2020: <https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Deep_Spatial_Gradient_and_Temporal_Depth_Learning_for_Face_Anti-Spoofing_CVPR_2020_paper.html>

### Auxiliary supervision và temporal cue

Depth/rPPG auxiliary supervision giúp mô hình học cue gần bản chất liveness hơn binary label. Tuy nhiên rPPG webcam cần điều kiện ổn định và depth supervision cần quy trình tạo depth map/kiến trúc phức tạp. Đồ án giữ temporal cue ở mức landmark + challenge để có bản chạy được, rồi mới so sánh với auxiliary model.

- Liu et al., *Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision*, CVPR 2018: <https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0615.pdf>

### Domain generalization

Nút thắt thực tế là domain shift do resolution, blur, sensor và ánh sáng. Vì vậy nâng backbone mà vẫn random-split frame không giải quyết đúng vấn đề. Protocol của dự án bắt buộc tách subject/video và khuyến nghị thêm device/session holdout.

- Sun et al., *Rethinking Domain Generalization for Face Anti-Spoofing*, CVPR 2023: <https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.html>
- Kim et al., *Advancing Cross-Domain Generalizability in Face Anti-Spoofing: Insights, Design and Metrics*, CVPRW 2024: <https://openaccess.thecvf.com/content/CVPR2024W/FAS2024/html/Kim_Advancing_Cross-Domain_Generalizability_in_Face_Anti-Spoofing_Insights_Design_and_Metrics_CVPRW_2024_paper.html>

### Landmark và challenge-response

MediaPipe Face Landmarker hỗ trợ landmark chuẩn hóa, blendshape và transformation matrix trong chế độ ảnh/video/live stream. Code dùng Face Mesh 468 điểm đóng gói trong MediaPipe để giảm bước tải model; EAR/yaw proxy và state machine nằm trong code của dự án.

- Google AI Edge, *Face Landmarker options*: <https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerOptions>
- Widjaya & Wicaksana, *Liveness Detection with Randomized Challenge-Response for Face Recognition Anti-Spoofing*: <https://www.ijicic.org/ijicic-190208.pdf>

Challenge-response chỉ là một lớp. Replay có sẵn hành động đúng hoặc deepfake tương tác có thể qua nếu attacker biết thử thách; vì vậy `challenge-only` được đánh dấu rõ là chế độ demo yếu.

### Metric chuẩn PAD

ISO/IEC 30107-3:2023 là tiêu chuẩn testing/reporting PAD. Repository dùng APCER, BPCER và ACER bên cạnh ROC-AUC/EER, đồng thời báo cáo từng attack type và kết quả theo video.

- ISO/IEC 30107-3:2023: <https://www.iso.org/standard/79520.html>

## Vì sao không “càng nhiều luồng càng tốt”

Nhiều luồng chỉ giúp khi lỗi của chúng khác nhau và mỗi luồng đã được kiểm chứng. Một heuristic yếu có thể làm ACER xấu hơn hoặc tạo cảm giác an toàn giả. Do đó dự án có:

1. Reliability cho từng observation.
2. Minimum evidence coverage trước khi approve.
3. Ablation để đo texture-only, challenge-only, texture+challenge và full fusion.
4. Vùng `UNCERTAIN/RETRY` thay vì buộc binary decision.
5. Weight/threshold tune trên validation, không chỉnh bằng test/demo đẹp.

## Roadmap nghiên cứu có thứ tự

1. Làm sạch protocol 10.500 ảnh cũ và thu lại metadata video/subject/session.
2. Chạy MobileNetV3 texture baseline + LBP baseline.
3. Đánh giá challenge và full fusion trên print/replay thật.
4. Ablation motion; bỏ luồng nếu không cải thiện APCER/BPCER có ý nghĩa.
5. Thử CDCN/auxiliary depth nếu GPU và thời gian cho phép.
6. Chỉ thử rPPG như nhánh nghiên cứu, không đưa vào production default trước khi cross-light/cross-device test đạt.
