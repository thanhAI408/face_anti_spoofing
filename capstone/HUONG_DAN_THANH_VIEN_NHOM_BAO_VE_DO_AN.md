# Hướng Dẫn Phân Công, Làm Slide, Thuyết Trình & Bảo Vệ Đồ Án Tốt Nghiệp (4 Thành Viên)

**Đề tài:** *Real-Time Passive Face Anti-Spoofing Using Aligned Lightweight Convolutional Networks for Biometric Attendance Verification*  
**Giảng viên hướng dẫn:** ThS. Nguyễn Ngọc Huy (`MSc. Nguyen Ngoc Huy`)  
**Học kỳ:** Fall 2026 — Ngành Trí tuệ Nhân tạo (AI), Đại học FPT

---

## 1. Bảng Phân Công Độc Lập 100% Cho 4 Thành Viên (Không Trùng Lặp)

Toàn bộ khóa luận (`6 Chương`) và mã nguồn hệ thống được chia thành **4 phần độc lập theo luồng dọc của hệ thống** (`Đặt vấn đề & Lý thuyết` $\to$ `Dữ liệu & Tiền xử lý` $\to$ `Kiến trúc & Thuật toán` $\to$ `Thực nghiệm, Kết luận & Live Demo`). Mỗi thành viên nắm trọn các chương LaTeX và các file code riêng của mình:

| Thứ tự nói | Thành viên | Mã SV | File LaTeX (`capstone/`) phụ trách | File Code phụ trách | Nội dung Thuyết trình & Slide | Thời lượng |
| :---: | :--- | :---: | :--- | :--- | :--- | :---: |
| **Phần 1** | **Nguyễn Văn Thành** | `HE181962` | `Preamble/*`, **Chapter 01**, **Chapter 02** | `pipeline.py`, `README.md` | Đặt vấn đề, Mục tiêu, Cơ sở lý thuyết Quang học & Toán học (`Slide 1–5`) | ~4.0 phút |
| **Phần 2** | **Nguyễn Chiến Thắng** | `HE186312` | **Chapter 03**, `references.bib`, `main.tex` | `scripts/extract_cam4k.py`, `face_extract.py`, `ChamCong_FaceID/preprocessing.py` | Thu thập dữ liệu 3 giai đoạn (`Cam4K`), Tiền xử lý 5 bước, Lọc chất lượng 6 tầng & Chia tập chống rò rỉ (`Slide 6–10`) | ~4.0 phút |
| **Phần 3** | **Nguyễn Hoàng Tuyên** | `HE180882` | **Chapter 04** | `model/livenessnet.py`, `scripts/train_cam4k.py`, `train_liveness.py`, `inference_runtime.py`, `api_app.py`, `ChamCong_FaceID/liveness/liveness_gate.py` | Luồng hệ thống, Kiến trúc `LivenessNet`, Huấn luyện cân bằng + `WiSE-FT`, Tối ưu `LiteRT XNNPACK`, Đa luồng Camera, REST API & `LivenessGate` (`Slide 11–15`) | ~4.5 phút |
| **Phần 4** | **Lê Thanh Phương** | `HE186753` | **Chapter 05**, **Chapter 06** | `run.py`, `web_app.py`, `ChamCong_FaceID/app.py`, `ChamCong_FaceID/calibrate_threshold.py`, `outputs/cam4k-20261020/` | Kết quả thực nghiệm (`Cam4K` & Đa nguồn), Độ trễ CPU, Phân tích lỗi, Ablation Study, Kết luận & **Live Demo trực tiếp** (`Slide 16–20`) | ~5.5 phút |

---

## 2. Bảng "Số Liệu Vàng" Thống Nhất Toàn Nhóm (Bắt Buộc Khớp 100%)

Khi làm slide và trả lời Hội đồng, cả 4 thành viên **bắt buộc dùng thống nhất các con số dưới đây** (trích xuất từ `outputs/cam4k-20261020/evaluation.json` và Chương 3–5 của khóa luận):

| Hạng mục | Con số chuẩn trong Khóa luận & Code | Ý nghĩa ngắn gọn để giải thích |
| :--- | :--- | :--- |
| **Kích thước ảnh đầu vào `LivenessNet`** | **`32 × 32 × 3` BGR** (`float32` trong `[0, 1]`) | Làm mờ chi tiết định danh cá nhân (chống học vẹt mặt người), giữ lại màu da/vệt lóa màn hình và chạy cực nhanh trên CPU. |
| **Kích thước ảnh đầu vào `ArcFace`** | **`112 × 112 × 3`** (căn chỉnh 5 điểm mốc mắt-mũi-miệng) | Trích xuất vector đặc trưng `512 chiều` chuẩn hóa $\ell_2$, ngưỡng nhận diện Cosine $\tau = 0.55$ ($\text{FAR} = 0.00\%$). |
| **Tổng số tham số mạng `LivenessNet`** | **`545,154` tham số** (`543,298` trainable + `1,856` BN) | Dung lượng file `.h5` là `2.15 MB`, file phẳng `.tflite` (`LiteRT float32`) là **`2.18 MB`**. |
| **Bộ dữ liệu mục tiêu `Cam4K` (Cohort III)** | **`91` video** (`44` real, `47` fake) $\to$ **`26,094` ảnh mặt** | Quét từ `140,707` frame gốc trên camera **USB SC-M200** với bước nhảy `--skip 5` (`6 Hz`). Chia độc lập theo người: Train `63` video (`16,268` ảnh), Val `14` video (`5,015` ảnh), Test `14` video (`4,811` ảnh). |
| **Bộ dữ liệu đa nguồn (Cohort IIb Merged)** | **`89` video** (`33` legacy + `56` mới) $\to$ **`28,957` ảnh mặt** | Dùng để chứng minh lỗi lệch thống kê `BatchNorm` khi gộp dữ liệu lệch (`52.75×` loss weight) và cách khắc phục bằng hàng đợi `4×8` + `GroupNorm` ($G=8$). |
| **Hiệu năng mức Frame trên tập Test `Cam4K`** | **Accuracy `98.09%`** (`4,719 / 4,811` ảnh đúng) | `Macro-F1 = 98.09%`, **`APCER = 3.76%`** (`89/2,365` ảnh giả bị lọt ở mức đơn frame), **`BPCER = 0.12%`** (`3/2,446` ảnh thật bị từ chối nhầm), **`ACER = 1.94%`** (vượt mục tiêu `< 2.50%`). |
| **Hiệu năng mức Video trên `Cam4K`** | **`100.00%` (`14 / 14` video Test & `14 / 14` video Val)** | Khi lấy trung bình xác suất theo video hoặc qua bộ lọc trượt thời gian, hệ thống phân loại đúng tuyệt đối `100%` tất cả các video. |
| **Kiểm tra chéo trên Webcam Laptop ngoài** | **Accuracy `99.84%`** (`599 / 600` ảnh đúng, `ACER = 0.17%`) | Nhờ trộn trọng số **`WiSE-FT`**: $\theta^*_{\text{head}} = 0.75\,\theta_{\text{cam4k}} + 0.25\,\theta_{\text{anchor}}$, model không bị quên các dòng webcam cũ. |
| **Tốc độ suy luận trên CPU (Ryzen 7 6800H)** | **`0.163 ms` / mặt** (trung vị $P_{50}$ với `LiteRT XNNPACK`) | Nhanh gấp **`198 lần`** so với gọi `model.predict()` trực tiếp trong Keras (`32.4 ms`). Luồng camera trực tiếp đạt **`30.4 FPS`**. |

---

## 3. Hướng Dẫn Chi Tiết Cho Thành Viên 1: Nguyễn Văn Thành (`HE181962` — Chương 1 & 2)

Phần của Thành đã có trọn bộ tài liệu hoàn chỉnh được đóng gói sẵn trong repo:
- **Slide PowerPoint / Canva (`.pptx`)**: [`deliverables/Slide_BaoVe_Phan1_NguyenVanThanh_5Slides.pptx`](../deliverables/Slide_BaoVe_Phan1_NguyenVanThanh_5Slides.pptx)
- **Kịch bản thuyết trình Tiếng Anh + Giải nghĩa Tiếng Việt + 4 câu hỏi phản biện**: [`capstone/Kich_Ban_Thuyet_Trinh_5_Slide_Phan1_NguyenVanThanh.md`](Kich_Ban_Thuyet_Trinh_5_Slide_Phan1_NguyenVanThanh.md)
- **Bản Teleprompter HTML đọc & giải thích từng câu Chương 1 & 2**: [`capstone/Thanh_BienDich_ThuyetTrinh_Chap1_Chap2.html`](Thanh_BienDich_ThuyetTrinh_Chap1_Chap2.html) *(Mở trực tiếp bằng trình duyệt Chrome/Edge trên máy tính hoặc điện thoại)*.

---

## 4. Hướng Dẫn Chi Tiết Cho Thành Viên 2: Nguyễn Chiến Thắng (`HE186312` — Chương 3: Data Processing)

### 4.1. Các file cần đọc & nắm vững
- **File khóa luận LaTeX**: [`capstone/Chapters/Chapter 03 - Data processing.tex`](Chapters/Chapter%2003%20-%20Data%20processing.tex) (kèm kiểm tra định dạng [`capstone/main.tex`](main.tex) và [`capstone/references.bib`](references.bib)).
- **Hình vẽ có sẵn để chèn vào Slide**:
  - [`capstone/Images/fig_dataset_samples.jpg`](Images/fig_dataset_samples.jpg) (*Hình 3.1: 10 mẫu ảnh mặt `32×32` thực tế từ tập dữ liệu: hàng trên viền xanh là người thật, hàng dưới viền đỏ là giả mạo qua màn hình*).
  - [`capstone/Images/fig_dataset_statistics.jpg`](Images/fig_dataset_statistics.jpg) (*Hình 3.2: Biểu đồ phân bố cân bằng giữa Real/Fake trên 3 tập Train/Val/Test và phân bố số lượng ảnh trích xuất trên mỗi video của `Cam4K`*).
- **File code phụ trách**: [`scripts/extract_cam4k.py`](../scripts/extract_cam4k.py), [`face_extract.py`](../face_extract.py), [`ChamCong_FaceID/preprocessing.py`](../ChamCong_FaceID/preprocessing.py).

### 4.2. Giải thích bản chất toàn bộ Chương 3 (Đọc để hiểu sâu 100%)
1. **Mục 3.1 — Ba giai đoạn tiến hóa của Dữ liệu (`Three Dataset Cohorts` - Bảng 3.1 & 3.2)**:
   - **Cohort I (`Legacy Baseline` — 33 video, 24,221 ảnh mặt)**: Bộ dữ liệu ban đầu. Khi kiểm toán, nhóm phát hiện các nghiên cứu cũ thường trộn ngẫu nhiên các frame (`random frame split`), khiến các frame sát nhau của cùng 1 video nằm ở cả tập Train và Test $\to$ độ chính xác ảo trên 99% nhưng gặp người lạ là sai. Khi chia lại theo video (`video-disjoint`), nhóm thấy rõ cần mở rộng thêm nhiều người.
   - **Cohort II (`Multi-Subject & Merged Benchmark` — thêm 56 video mới, gộp thành 89 video / 28,957 ảnh mặt)**: Giúp kiểm tra khả năng tổng quát hóa qua nhiều người và phát hiện 2 "bẫy" dữ liệu: (a) lệch định dạng nén `.mov` vs `.mp4`, và (b) lệch thống kê `BatchNorm` khi gộp nhiều nguồn.
   - **Cohort III (`Cam4K Benchmark` — 91 video `.mp4` quay trực tiếp trên camera mục tiêu USB SC-M200)**: Quét `140,707` frame gốc với bước nhảy `--skip 5` thu được **`26,094` ảnh mặt `32×32`** (`12,618` real / `13,476` fake). Chia độc lập theo người (`Person-Disjoint Split`):
     - `Train`: `63` video (`30` real, `33` fake) $\to$ `16,268` ảnh (`47.35%` real / `52.65%` fake).
     - `Validation`: `14` video (`7` real, `7` fake) $\to$ `5,015` ảnh (`49.23%` real / `50.77%` fake).
     - `Test`: `14` video (`7` real, `7` fake) $\to$ `4,811` ảnh (`50.84%` real / `49.16%` fake).
     - Kèm thêm bộ kiểm tra chéo `600` ảnh (`300` real, `300` fake) trên webcam liền máy laptop để kiểm chứng model không bị học lệch vào 1 camera duy nhất.
2. **Mục 3.2 — Quy trình Tiền xử lý & Làm sạch 5 bước (`Data Preprocessing and Cleaning`)**:
   - **Bước 1 — Lấy mẫu theo thời gian (`Temporal Subsampling --skip 5`)**: Video 30 FPS có các frame liên tiếp gần như giống hệt nhau. Chỉ lấy mỗi 5 frame 1 lần (`6 Hz`) giúp giảm từ `140,707` frame xuống `26,094` ảnh mặt trong `452.6 giây` mà vẫn giữ đủ mọi góc nghiêng đầu, dùng bộ dò mặt `Caffe ResNet-10 SSD` (ngưỡng tin cậy `0.50`).
   - **Bước 2 — Tô đen mặt người cầm điện thoại ở rìa phải (`Spatial Masking of Contaminating Presenter Faces`)**: Đây là phát hiện thực tế rất đắt giá của nhóm! Trong 3 video quay góc rộng (`fake_REC-0001-A`, `0002-A`, `0003-A`), người cầm điện thoại đứng lọt một phần vào rìa phải khung hình. Khi màn hình điện thoại bị nghiêng lóa sáng, bộ dò mặt lại bắt nhầm **mặt thật của người cầm điện thoại** rồi lưu vào thư mục `fake` (làm bẩn nhãn huấn luyện). Nhóm xử lý triệt để bằng cách tô đen $35\%$ rìa phải khung hình ($I[:, \lfloor 0.65W \rfloor : W] \leftarrow 0$) trước khi dò mặt trên các video góc rộng này.
   - **Bước 3 — Bộ lọc chất lượng quang học 6 tầng (`Six-Stage Optical Quality Filtering`)**:
     1. *Kích thước tối thiểu*: $\min(w, h) \ge 65\text{ px}$ cho chống giả mạo ($\ge 80\text{ px}$ cho đăng ký ArcFace).
     2. *Không bị cắt lẹm biên*: Loại bỏ các khung mặt chạm mép ảnh ($x_1 < 0, y_1 < 0, x_2 > W, y_2 > H$).
     3. *Hình học 5 điểm mốc (`YuNet landmarks`)*: Khoảng cách 2 mắt $d_{\text{eye}} \ge 0.12\,w$ (loại góc quay ngang quá gắt).
     4. *Ngưỡng tối thiểu ánh sáng*: Độ sáng trung bình $\mu_{\text{gray}} \ge 35$ (trên thang `[0, 255]`, loại ảnh tối đen).
     5. *Ngưỡng trần chống cháy sáng*: Tỷ lệ điểm ảnh bão hòa ($I_{\text{gray}} > 245$) không vượt quá $45\%$.
     6. *Chống mờ nhòe bằng phương sai Laplacian*: Chuẩn hóa về `128×128` xám và loại bỏ ảnh có $\sigma^2_{\text{Lap}} < 12.0$ ($< 45.0$ khi đăng ký nhân viên). Bộ lọc này đã tự động loại bỏ `189` ảnh lỗi (`158` ảnh lóa trắng và `31` ảnh nhòe chuyển động) trong tập cũ.
   - **Bước 4 — Chuẩn hóa & Tăng cường dữ liệu cân bằng trắng (`Photometric White-Balance Augmentation`)**: Chia pixel cho `255.0` về `[0, 1]`. Chỉ tăng cường trên tập `Train` (xoay $\pm 10^\circ$, dịch/zoom $8\%$, lật ngang $50\%$, và **nhiễu cân bằng trắng từng kênh màu BGR** $\tilde{X}_c = \text{clip}(\alpha_c X_c + \delta_c, 0, 1)$ với xác suất $35\%$ để mô phỏng sự thay đổi màu giữa các webcam).
3. **Mục 3.3 — Tại sao dùng đồng thời 2 tỷ lệ `32×32` và `112×112`? (`Dual-Scale Representation`)**:
   - `32×32×3` cho `LivenessNet`: Đóng vai trò bộ lọc thông thấp xóa các chi tiết nhận dạng cá nhân (nốt ruồi, dáng lông mày) để mạng CNN không học vẹt mặt người quen, chỉ tập trung vào sắc độ da và nhiễu màn hình; kèm thêm 4 đặc trưng thống kê màu $YC_rC_b$/HSV ở vùng da trung tâm khi bật chế độ chống ảnh giấy in (`fusion=True`).
   - `112×112×3` cho `ArcFace` (`ChamCong_FaceID`): Giữ độ nét cao để trích xuất vector `512 chiều`, lưu cả prototype theo từng video và vector trung tâm của từng nhân viên.
4. **Mục 3.4 — Chiến lược Chia tập & Kiểm toán chống rò rỉ (`Data Validation and Splitting Strategy`)**:
   - Chia vật lý theo từng người trước khi cắt frame (`63` train / `14` val / `14` test).
   - Kiểm tra tự động bằng code trong `split.json` đảm bảo giao của 3 tập bằng rỗng ($\mathcal{V}_{\text{train}} \cap \mathcal{V}_{\text{val}} = \emptyset, \dots$) và kiểm tra mã băm `SHA-256` của từng ảnh để đảm bảo `0` ảnh trùng lặp, cố định `seed = 42`.

### 4.3. Gợi ý 5 Slide & Kịch bản nói Tiếng Anh cho Nguyễn Chiến Thắng (Phần 2)

- **Slide 6 (Mở đầu Chương 3 — Ba giai đoạn dữ liệu & Bảng 3.1/3.2)**:
  - *Nội dung hiển thị*: Bảng tiến hóa 3 Cohort (`Legacy 33 vids` $\to$ `Merged 89 vids` $\to$ `Cam4K 91 vids / 26,094 crops`) + Bảng chia tập `Cam4K` (`63 Train / 14 Val / 14 Test`) + Hình 3.2 (`fig_dataset_statistics.jpg`).
  - *Lời nói Tiếng Anh gợi ý*:
    > “Thank you, Thanh. Hello teachers, my name is **Nguyen Chien Thang (`HE186312`)**, and I will present **Chapter 3: Data Processing**.
    > In anti-spoofing, clean data is just as important as model design. As shown in Table 3.1, we built three dataset cohorts. After finding frame-split leakage in the 33-video legacy set and container differences in the 89-video merged set, we recorded our main **`Cam4K` benchmark** directly on the target **USB SC-M200 camera**. From `91` videos and `140,707` raw frames, we extracted **`26,094` clean face crops**, balanced nearly `50–50` between real and fake across `63` training, `14` validation, and `14` unseen test videos.”
- **Slide 7 (Hình ảnh mẫu thực tế & Phát hiện lỗi người cầm điện thoại — Spatial Masking)**:
  - *Nội dung hiển thị*: Hình 3.1 (`fig_dataset_samples.jpg`) + Minh họa kỹ thuật lấy mẫu `--skip 5` (`6 Hz`) và tô đen rìa phải $I[:, \lfloor 0.65W \rfloor : W] \leftarrow 0$ trên video góc rộng.
  - *Lời nói Tiếng Anh gợi ý*:
    > “Figure 3.1 shows real `32×32` crops from our dataset. During extraction with a 5-frame stride, we discovered an important label bug in wide-angle 4K attack videos: when the phone screen tilted and glared, the face detector accidentally picked up the **real face of the person holding the phone** on the right edge, saving a real face into the `fake` folder. To fix this completely, we added an automated **right-edge zero mask** covering the rightmost `35%` of wide-angle attack videos before detection, ensuring `100%` clean spoof labels.”
- **Slide 8 (Bộ lọc chất lượng quang học 6 tầng & Tăng cường cân bằng trắng)**:
  - *Nội dung hiển thị*: 6 quy tắc lọc chất lượng (Box $\ge 65\text{ px}$, không chạm viền, khoảng cách mắt $d_{\text{eye}} \ge 0.12w$, độ sáng $\mu_{\text{gray}} \ge 35$, điểm lóa $\le 45\%$, phương sai độ nét $\sigma^2_{\text{Lap}} \ge 12.0$) + Augmentation White-Balance Jitter (`35%`).
  - *Lời nói Tiếng Anh gợi ý*:
    > “Next, every detected face must pass our **six-stage optical quality filter**: minimum size of `65` pixels, no border clipping, valid eye distance, brightness floor of `35`, glare ceiling of `45%`, and Laplacian sharpness variance above `12.0`. This filter automatically removed `189` blurry or over-glared images from the legacy set. For training, we apply geometric jitter and **per-channel white-balance jitter** (`±8%` gain) so the model handles different camera color temperatures.”
- **Slide 9 (Biểu diễn 2 tỷ lệ `32×32` vs `112×112` & Chống rò rỉ dữ liệu Person-Disjoint)**:
  - *Nội dung hiển thị*: Sơ đồ so sánh `32×32×3` (`LivenessNet` — ẩn danh tính, giữ sắc độ da/lóa kính) vs `112×112×3` (`ArcFace` — giữ chi tiết nhận dạng `512-D`) + Quy trình kiểm tra rò rỉ (`split.json`, `SHA-256`, `seed = 42`) và chuyển lời cho **Nguyễn Hoàng Tuyên (Chương 4)**.
  - *Lời nói Tiếng Anh gợi ý*:
    > “Finally, why do we use two image sizes? We use **`32×32`** for `LivenessNet` to blur out personal identity traits so the model learns skin and screen texture instead of memorizing faces, while keeping sharp **`112×112`** aligned crops for `ArcFace` employee recognition. All splits are strictly separated by person before frame extraction and verified by `SHA-256` hashing with zero overlap. Next, **Nguyen Hoang Tuyen** will present **Chapter 4: Methodology**. Thank you!”

### 4.4. Bốn câu hỏi Hội đồng hay hỏi xoáy phần Chương 3 (Thắng cần nắm)
1. **“Tại sao các em lấy bước nhảy `--skip 5` mà không lấy toàn bộ các frame trong video để có nhiều dữ liệu hơn?”**
   - *Trả lời*: Dạ ở tốc độ quay `30 FPS`, 5 frame liên tiếp chỉ cách nhau `0.16 giây` nên góc mặt và ánh sáng gần như trùng lặp hoàn toàn. Nếu lấy hết `140,707` frame thì dữ liệu bị dư thừa nặng, thời gian train lâu gấp 5 lần và dễ overfit vào các đoạn đứng yên. Lấy `--skip 5` (`6 frame/giây`) vừa thu được `26,094` ảnh mặt đa dạng góc nghiêng, vừa rút ngắn thời gian trích xuất xuống chỉ còn `452.6 giây` ạ.
2. **“Em giải thích rõ hơn lỗi ‘người cầm điện thoại’ (Contaminating Presenter Face) được phát hiện và xử lý như thế nào?”**
   - *Trả lời*: Dạ khi bọn em kiểm tra các ảnh được cắt tự động từ 3 video giả mạo quay góc rộng (`fake_REC-0001/0002/0003-A`), bọn em thấy có những ảnh thực chất là mặt thật của bạn cầm điện thoại đứng ở góc phải. Lý do là khi màn hình điện thoại bị bóng đèn chiếu lóa, điểm tin cậy (`confidence score`) của mặt trong điện thoại tụt xuống thấp hơn mặt người cầm ở rìa phải. Vì vậy trong `extract_cam4k.py`, bọn em đặt mặt nạ xóa đen vùng từ `0.65 × chiều rộng` đến mép phải khung hình trước khi chạy SSD detector, đảm bảo chỉ bắt đúng khuôn mặt trên màn hình giả mạo ạ.
3. **“Làm sao nhóm chứng minh chắc chắn tập Train và tập Test không bị lọt cùng một người (Data Leakage)?”**
   - *Trả lời*: Dạ bọn em chia thư mục vật lý `train/` (63 video), `validate/` (14 video) và `test/` (14 video) theo từng người tình nguyện **trước khi** chạy script cắt ảnh. Sau đó script tự động xuất file `split.json` kiểm tra giao tập hợp tên video bằng rỗng ($\mathcal{V}_{\text{train}} \cap \mathcal{V}_{\text{test}} = \emptyset$) và kiểm tra mã băm `SHA-256` trên từng mảng pixel để cam kết `0%` trùng lặp giữa các tập ạ.
4. **“Tại sao ngưỡng độ nét Laplacian $\sigma^2_{\text{Lap}}$ khi lọc anti-spoofing là `12.0` nhưng khi đăng ký nhân viên ArcFace lại nâng lên `45.0`?”**
   - *Trả lời*: Dạ vì khi người dùng đứng điểm danh thực tế, việc đầu hơi chuyển động nhẹ tạo độ mờ vừa phải ($\sigma^2_{\text{Lap}}$ từ `12` đến `40`) là bình thường, cổng `LivenessNet` ở kích thước `32×32` vẫn phân biệt tốt nên chỉ cần chặn ảnh quá nhòe ($< 12.0$). Ngược lại, bước **đăng ký hồ sơ gốc (`Enrollment`)** cho `ArcFace` dùng lâu dài nên cần chọn những frame sắc nét nhất ($\ge 45.0$) để tạo vector chuẩn (`prototype`) chính xác nhất ạ.

---

## 5. Hướng Dẫn Chi Tiết Cho Thành Viên 3: Nguyễn Hoàng Tuyên (`HE180882` — Chương 4: Methodology)

### 5.1. Các file cần đọc & nắm vững
- **File khóa luận LaTeX**: [`capstone/Chapters/Chapter 04 - Methodology.tex`](Chapters/Chapter%2004%20-%20Methodology.tex).
- **Hình vẽ có sẵn để chèn vào Slide**:
  - [`capstone/Images/fig_system_workflow.jpg`](Images/fig_system_workflow.jpg) (*Hình 4.1: Kiến trúc tổng thể 2 giai đoạn — Stage 1 Huấn luyện & Xuất LiteRT offline; Stage 2 Suy luận thời gian thực phục vụ 3 kênh: Desktop UI, REST API, và Web Chấm công*).
  - [`capstone/Images/fig_livenessnet_arch.jpg`](Images/fig_livenessnet_arch.jpg) (*Hình 4.2: Sơ đồ kiến trúc mạng `LivenessNet` 3 khối tích chập $32 \to 64 \to 128$, tổng `545,154` tham số và cơ chế trộn trọng số `WiSE-FT`*).
- **File code phụ trách**: [`model/livenessnet.py`](../model/livenessnet.py), [`scripts/train_cam4k.py`](../scripts/train_cam4k.py), [`train_liveness.py`](../train_liveness.py), [`inference_runtime.py`](../inference_runtime.py), [`pipeline.py`](../pipeline.py), [`api_app.py`](../api_app.py), [`ChamCong_FaceID/liveness/liveness_gate.py`](../ChamCong_FaceID/liveness/liveness_gate.py).

### 5.2. Giải thích bản chất toàn bộ Chương 4 (Đọc để hiểu sâu 100%)
1. **Mục 4.1 — Luồng hệ thống tổng thể 2 giai đoạn (`Overall Workflow` - Hình 4.1)**:
   - **Stage 1 (Offline)**: Video `Cam4K` + Đa nguồn $\to$ Cắt mặt & Lọc chất lượng 6 tầng $\to$ Huấn luyện cân bằng theo video trên khung xương tích chập cố định (`frozen universal backbone`) $\to$ Trộn trọng số đầu phân loại bằng `WiSE-FT` $\to$ Đóng băng sang `liveness_cam4k.tflite` (`2.18 MB`).
   - **Stage 2 (Online Runtime)**: Dùng chung 1 nhân suy luận (`pipeline.py` + `LiteModel`) gồm dò mặt `YuNet ONNX` $\to$ nối vết đa khuôn mặt `IoU` $\to$ chấm điểm CNN bằng `XNNPACK` (`0.163 ms`) $\to$ làm mượt theo thời gian. Nhân này phục vụ đồng thời 3 kênh: (A) App Desktop `run.py` `30.4 FPS`, (B) Microservice `api_app.py`, và (C) Web Chấm công `ChamCong_FaceID`.
2. **Mục 4.2 — Thiết kế Kiến trúc Mạng & Hệ thống (`System Design and Architecture`)**:
   - **Kiến trúc `LivenessNet` (Bảng 4.1 & Hình 4.2)**:
     - Đầu vào `32×32×3`.
     - **Conv Block 1**: 1 lớp `Conv2D(32, 3×3)` + `BatchNorm/GroupNorm` + `MaxPool(3×3)` + `Dropout(0.25)` $\to$ hạ kích thước từ `32×32` xuống `10×10×32`.
     - **Conv Block 2**: 2 lớp `Conv2D(64, 3×3)` + Norm + `MaxPool(2×2)` + `Dropout(0.25)` $\to$ `5×5×64`.
     - **Conv Block 3**: 2 lớp `Conv2D(128, 3×3)` + Norm + `MaxPool(2×2)` + `Dropout(0.25)` $\to$ `2×2×128`.
     - **Classifier Head**: `Flatten` (`2×2×128 = 512`) $\to$ `Dense(512)` + `ReLU` + Norm + `Dropout(0.50 / 0.15)` $\to$ `Dense(2, Softmax)` xuất ra $[P_{\text{fake}}, P_{\text{real}}]$. Tổng cộng **`545,154` tham số** (riêng lớp `Dense(512)` chiếm `262,656` tham số = `48.2%`).
     - Hỗ trợ công tắc `norm_type="batch"` hoặc `norm_type="group"` (`GroupNorm` với $G=8$ nhóm trên Conv + `LayerNorm` trên Dense để khử hoàn toàn hiện tượng trôi trung bình trượt khi train đa nguồn).
   - **Biên dịch `LiteRT` & tại sao KHÔNG lượng tử hóa `int8` (`Frozen LiteRT & XNNPACK`)**:
     - Gọi `model.predict()` trong Keras 3 tốn `~32.4 ms` mỗi lần do chi phí quản lý đồ thị Python. Nhóm đóng băng đồ thị sang FlatBuffer `.tflite` chạy qua tập lệnh vector **XNNPACK AVX2/FMA** trên CPU chỉ mất **`0.163 ms`** (`198×` nhanh hơn).
     - **Điểm nhấn kỹ thuật**: Nhóm **cố ý giữ nguyên `float32` (`2.18 MB`) chứ không nén `int8`**, vì các dấu hiệu giả mạo màn hình ở kích thước `32×32` có biên độ chênh lệch số thực rất nhỏ — nén `int8` sẽ làm nhiễu các chi tiết này, trong khi bản `float32` đã đạt `0.163 ms` và chỉ nặng `2.18 MB`.
   - **Kiến trúc 3 luồng tách biệt trong `run.py` (`Decoupled Multi-Threaded Camera Runtime`)**:
     - Luồng 1 (`_capture_loop`): Đọc camera ở chuẩn nén `MJPG 1280×720 30 FPS`, đặt `buffer_size = 1` để luôn ghi đè frame mới nhất (không bao giờ bị trễ hình), tự động chịu lỗi tới `50` lần đứt gói tin USB trước khi tự khởi động lại camera, và chặn các camera ảo (`DroidCam` màn hình màu trơn có độ lệch chuẩn pixel $\sigma_{\text{frame}} < 2.0$).
     - Luồng 2 (`_ai_loop`): Thu nhỏ cạnh dài về $\le 480\text{ px}$ để dò mặt `YuNet` cực nhanh, sau đó cắt mặt từ **khung hình gốc độ phân giải cao** và chấm điểm theo lô (`batch`).
     - Luồng 3 (`UI loop`): Vẽ khung Xanh (`REAL`) / Đỏ (`FAKE`) mượt mà ở `30.4 FPS`.
   - **Kiến trúc REST API (`api_app.py` - Bảng 4.2)**:
     - Cung cấp 6 endpoint `/v1/health`, `/openapi.json`, `/v1/analyze` (đơn ảnh), `/v1/sessions` (tạo phiên `30s TTL`, tối đa `64` phiên LRU), `/v1/sessions/{id}/frames` (trả về 7 trạng thái an toàn), `DELETE /v1/sessions/{id}`.
     - 2 cơ chế bảo vệ thực chiến: (1) Đọc ảnh bằng `cv2.IMREAD_IGNORE_ORIENTATION` để thẻ xoay EXIF của điện thoại không làm ngược mặt; (2) Dùng khóa không chờ `infer_lock.acquire(blocking=False)` trả về ngay `HTTP 429 Too Many Requests` khi quá tải thay vì xếp hàng các frame cũ.
   - **Kiến trúc Chấm công & Kỹ thuật căn chỉnh lại khung mặt (`Detector ROI Re-Alignment` trong `LivenessGate`)**:
     - Hệ thống chấm công `ChamCong_FaceID` dùng `InsightFace SCRFD` để tìm mặt. Tuy nhiên khung mặt của `SCRFD` cắt hẹp phần trán/cằm hơn so với khung mặt của `SSD/YuNet` lúc huấn luyện `LivenessNet`.
     - Giải pháp trong `liveness_gate.py`: Lấy khung `SCRFD` **mở rộng ra 4 phía thêm `15%`**, sau đó chạy `YuNet` ngay bên trong vùng mở rộng đó để lấy lại đúng khung mặt chuẩn như lúc train, rồi mới đưa vào `liveness_cam4k.tflite`. Chỉ khi `LivenessGate == REAL` **VÀ** `ArcFace Cosine >= 0.55` (kèm chống chấm trùng `30 giây`) thì mới ghi vào file điểm danh CSV.
3. **Mục 4.3 — Hai Thuật toán cốt lõi (`Algorithm 1` & `Algorithm 2`)**:
   - **Algorithm 1 (`Video-Balanced Aligned Training & WiSE-FT`)**:
     - Thay vì bốc ngẫu nhiên theo ảnh (làm video dài lấn át video ngắn), mỗi batch `32` ảnh bốc chính xác `16 ảnh real + 16 ảnh fake`, trong đó chọn đều video trước rồi mới chọn ảnh trong video ($v \sim \text{Uniform}(\mathcal{V}_{\text{train}, c})$).
     - Đóng băng 3 khối tích chập (`frozen backbone`), chỉ tinh chỉnh đầu phân loại `512-unit head` trong `18` epoch (chọn checkpoint tốt nhất ở **Epoch 14**), sau đó nội suy trọng số **`WiSE-FT`**: $\theta^*_{\text{head}} = 0.75\,\theta^{\text{best}}_{\text{head}} + 0.25\,\theta^{\text{anchor}}_{\text{head}}$.
   - **Algorithm 2 (`Multi-Face IoU Tracking & Dual-Policy Temporal Decision`)**:
     - Nối vết khuôn mặt giữa các frame bằng độ chồng lấp $\text{IoU} \ge \tau_{\text{IoU}}$.
     - Hỗ trợ 2 chế độ:
       1. `FastBinaryDecision` (dùng cho Camera trực tiếp & Chấm công): Cửa sổ trượt `0.45 giây`, gán trọng số ưu tiên frame mới tăng tuyến tính $w_k \in [0.5, 1.0]$ $\to$ phản hồi tức thì xanh/đỏ mà không bị nhấp nháy khi chớp mắt.
       2. `TemporalHysteresis` (dùng cho REST API từ xa): Cửa sổ `1.50 giây`, loại bỏ các frame mờ/lóa (`adjust`) trước khi đưa vào cửa sổ, yêu cầu tối thiểu `6 frame sạch` trong $\ge 0.40\text{ s}$ với tỷ lệ đồng thuận $\ge 80\%$ mới chốt kết quả (`fail-closed`).

### 5.3. Gợi ý 5 Slide & Kịch bản nói Tiếng Anh cho Nguyễn Hoàng Tuyên (Phần 3)

- **Slide 11 (Tổng quan Luồng Hệ thống 2 giai đoạn — Hình 4.1)**:
  - *Lời nói Tiếng Anh gợi ý*:
    > “Thank you, Thang. Hello teachers, my name is **Nguyen Hoang Tuyen (`HE180882`)**, and I will present **Chapter 4: Methodology**.
    > Figure 4.1 shows our two-stage system workflow. In **Stage 1 (Offline)**, we train `LivenessNet` using video-balanced sampling, blend the classifier head with `WiSE-FT`, and export a frozen `2.18 MB` `LiteRT` model. In **Stage 2 (Online)**, a single shared CPU engine runs `YuNet` detection, `IoU` face tracking, and `0.163 ms` `XNNPACK` inference to power three applications simultaneously: our `30 FPS` desktop monitor, our REST API microservice, and our WebSocket attendance system.”
- **Slide 12 (Kiến trúc mạng `LivenessNet` & Công tắc `BatchNorm` / `GroupNorm` — Hình 4.2 & Bảng 4.1)**:
  - *Lời nói Tiếng Anh gợi ý*:
    > “Figure 4.2 and Table 4.1 detail our **`LivenessNet`** architecture. Taking a `32×32×3` BGR patch, three convolutional blocks expand channels from `32` to `64` to `128` while pooling spatial size down to `2×2×128`, flattening into a `512`-unit classifier head. The entire model has only **`545,154` parameters**. To support multi-source datasets without `BatchNorm` drift, our architecture includes a switch to replace `BatchNorm` with `GroupNorm` (`8` groups) and `LayerNorm`.”
- **Slide 13 (Thuật toán 1: Huấn luyện cân bằng theo Video + Trộn trọng số `WiSE-FT` & Tối ưu `LiteRT XNNPACK`)**:
  - *Lời nói Tiếng Anh gợi ý*:
    > “In Algorithm 1, instead of sampling images unevenly, each 32-crop batch picks exactly `16` real and `16` fake crops uniformly across videos so long videos do not dominate training. We freeze the universal convolutional backbone, train the `512`-unit head for `18` epochs—reaching best validation loss at **Epoch 14**—and mix **`75%` new Cam4K weights with `25%` anchor weights** via `WiSE-FT`. Finally, we compile the model into an **unquantized `float32` LiteRT flatbuffer** running on CPU via `XNNPACK`, avoiding `int8` quantization noise while achieving `0.163 ms` latency.”
- **Slide 14 (Thuật toán 2: Bám vết `IoU`, 2 Chế độ làm mượt thời gian & Kiến trúc Đa luồng / REST API)**:
  - *Lời nói Tiếng Anh gợi ý*:
    > “To stop frame flickering, Algorithm 2 tracks multiple faces using `IoU` matching and supports two decision modes: **`FastBinaryDecision`** uses recency weights from `0.5` to `1.0` over a `0.45-second` window for instant `30 FPS` camera response; while **`TemporalHysteresis`** requires at least `6` clean frames with `80%` agreement for secure REST API sessions. In `run.py`, three separate threads for USB capture, AI inference, and UI display keep the camera smooth at `30.4 FPS` and auto-recover from USB cable drops.”
- **Slide 15 (Tích hợp Chấm công `ChamCong_FaceID` & Kỹ thuật `Detector ROI Re-Alignment` — Chuyển lời sang Chương 5)**:
  - *Lời nói Tiếng Anh gợi ý*:
    > “Finally, when integrating our model into the `ChamCong_FaceID` attendance system, we solved a critical detector mismatch: `InsightFace` uses `SCRFD`, which crops faces tighter on the forehead and chin than `YuNet`. Therefore, our **`LivenessGate`** expands each `SCRFD` box by **`15%`** and re-runs `YuNet` inside that window so `LivenessNet` always receives properly aligned crops. Check-in is logged only if **`LivenessGate` says REAL** and **`ArcFace` similarity is at least `0.55`**. Next, **Le Thanh Phuong** will present **Chapter 5, Chapter 6, and our Live Demo**. Thank you!”

### 5.4. Bốn câu hỏi Hội đồng hay hỏi xoáy phần Chương 4 (Tuyên cần nắm)
1. **“Tại sao khi xuất sang `.tflite` (`LiteRT`), nhóm không lượng tử hóa xuống `int8` (`INT8 Quantization`) cho nhẹ hơn nữa mà lại giữ `float32`?”**
   - *Trả lời*: Dạ bọn em đã thử nghiệm và nhận thấy hai lý do thực tế ạ: Thứ nhất, ở kích thước ảnh rất nhỏ `32×32`, các dấu hiệu giả mạo như vệt sóng Moiré hay độ chuyển màu da tạo ra sự khác biệt kích hoạt (`activation`) rất nhỏ ở phần thập phân — nếu ép về `int8` thì sai số làm tròn (`quantization noise`) sẽ làm mờ các dấu hiệu này. Thứ hai, ngay cả khi giữ nguyên `float32`, file model cũng chỉ nặng **`2.18 MB`** và nhờ tập lệnh vector `XNNPACK AVX2` trên CPU, thời gian chạy chỉ mất **`0.163 ms`** (tức hơn 5,900 mặt/giây), đã vượt xa yêu cầu thời gian thực mà không cần đánh đổi độ chính xác ạ.
2. **“Tại sao trong `train_cam4k.py` (Algorithm 1), nhóm lại đóng băng (`freeze`) các lớp tích chập mà chỉ train phần đầu phân loại (`Classifier Head`)?”**
   - *Trả lời*: Dạ trong mạng `LivenessNet` (`545,154` tham số), riêng lớp `Dense(512)` ở đầu phân loại đã chiếm tới `262,656` tham số (`48.2%`), trong khi 3 khối tích chập đầu tiên chỉ làm nhiệm vụ trích xuất các đặc trưng bề mặt cơ bản (cạnh viền, độ lóa, sắc độ màu) đã được học rất tốt trên tập dữ liệu đa nguồn. Việc đóng băng các lớp tích chập giúp mạng không bị *overfit* vào 63 video của tập `Cam4K`, làm độ lệch giữa train loss và val loss ở Epoch 14 chỉ còn $\Delta\mathcal{L} = 0.0019$, đồng thời cho phép trộn tuyến tính trọng số đầu phân loại bằng `WiSE-FT` cực kỳ hiệu quả ạ.
3. **“Giải thích sự khác nhau giữa `FastBinaryDecision` và `TemporalHysteresis`? Khi nào dùng cái nào?”**
   - *Trả lời*: Dạ cả hai đều dùng cửa sổ trượt trên từng khuôn mặt được nối vết bằng `IoU`, nhưng phục vụ 2 ngữ cảnh khác nhau:
     - `FastBinaryDecision` dùng cho **màn hình Camera trực tiếp & Web Chấm công**: cửa sổ ngắn `0.45 giây`, các frame mới nhất có trọng số cao gấp đôi frame cũ ($w_k$ từ `0.5` đến `1.0`) và quyết định ngay ở ngưỡng `0.50` để giao diện chỉ hiện Xanh (`REAL`) hoặc Đỏ (`FAKE`) tức thì, không làm nhân viên phải đứng chờ.
     - `TemporalHysteresis` dùng cho **REST API xác thực từ xa (`/v1/sessions`)**: yêu cầu khắt khe hơn — tự động bỏ qua các frame bị mờ/lóa (`adjust`), đòi hỏi tích lũy đủ ít nhất `6 frame sạch` trong tối thiểu `0.40 giây` và phải có $\ge 80\%$ số frame đồng thuận cùng điểm trung bình $\ge 0.80$ mới mở khóa (`fail-closed`).
4. **“Trong `api_app.py`, tại sao lại trả về ngay `HTTP 429` bằng `infer_lock.acquire(blocking=False)` thay vì cho các request xếp hàng chờ (`queue`)?”**
   - *Trả lời*: Dạ vì trong bài toán xác thực sinh trắc học thời gian thực, nếu có nhiều request ập đến cùng lúc mà bắt xếp hàng chờ trong queue, thì các frame hình bị nằm chờ vài giây sẽ trở thành **frame cũ (`stale frames`)** không còn phản ánh đúng người đang đứng trước camera hiện tại nữa, dễ tạo lỗ hổng đánh tráo. Nên bọn em dùng non-blocking mutex để xử lý dứt điểm frame hiện tại trong `~38 ms` và báo client gửi lại frame mới nhất sau `1 giây` (`Retry-After: 1`) ạ.

---

## 6. Hướng Dẫn Chi Tiết Cho Thành Viên 4: Lê Thanh Phương (`HE186753` — Chương 5, Chương 6 & Live Demo)

### 6.1. Các file cần đọc & nắm vững
- **File khóa luận LaTeX**: [`capstone/Chapters/Chapter 05 - Result analysis.tex`](Chapters/Chapter%2005%20-%20Result%20analysis.tex) và [`capstone/Chapters/Chapter 06 - Conclusion.tex`](Chapters/Chapter%2006%20-%20Conclusion.tex).
- **Hình vẽ có sẵn để chèn vào Slide**:
  - [`capstone/Images/fig_cam4k_learning_curves.jpg`](Images/fig_cam4k_learning_curves.jpg) (*Hình 5.1: Đường cong hội tụ Loss và Accuracy qua 18 epoch trên `Cam4K`, đánh dấu điểm tối ưu tại Epoch 14*).
  - [`capstone/Images/fig_confusion_matrices.jpg`](Images/fig_confusion_matrices.jpg) (*Hình 5.2: Ma trận nhầm lẫn trên tập Validation 5,015 ảnh và tập Test độc lập 4,811 ảnh*).
  - [`capstone/Images/fig_video_probabilities.jpg`](Images/fig_video_probabilities.jpg) (*Hình 5.3: Biểu đồ xác suất trung bình mức Video trên 14 video Test độc lập — đúng tuyệt đối `14/14 = 100%`*).
  - [`capstone/Images/fig_qualitative_outputs.jpg`](Images/fig_qualitative_outputs.jpg) (*Hình 5.4: Kết quả nhận diện trực quan trên 4 tình huống thực tế ở `30.4 FPS`*).
  - [`capstone/Images/fig_ablation_and_latency.jpg`](Images/fig_ablation_and_latency.jpg) (*Hình 5.5: Biểu đồ Ablation Study so sánh các phương pháp chuẩn hóa A1–A4 và biểu đồ độ trễ CPU*).
- **File code & kết quả phụ trách**: [`run.py`](../run.py), [`web_app.py`](../web_app.py), [`ChamCong_FaceID/app.py`](../ChamCong_FaceID/app.py), [`ChamCong_FaceID/calibrate_threshold.py`](../ChamCong_FaceID/calibrate_threshold.py), [`outputs/cam4k-20261020/evaluation.json`](../outputs/cam4k-20261020/evaluation.json).

### 6.2. Giải thích bản chất toàn bộ Chương 5 & Chương 6 (Đọc để hiểu sâu 100%)
1. **Mục 5.1 & 5.2 — Thiết lập thực nghiệm & Tiêu chuẩn đánh giá (`Experimental Setup & Evaluation Strategy`)**:
   - Chạy hoàn toàn trên **CPU văn phòng** (`AMD Ryzen 7 6800H`, `16 GB RAM`, không dùng GPU rời).
   - Đánh giá ở 4 cấp độ: (1) Mức từng ảnh cắt (`Frame-level`: `5,015` ảnh Val + `4,811` ảnh Test `Cam4K` + `4,856` ảnh Test đa nguồn + `600` ảnh webcam laptop ngoài); (2) Mức gộp cả video (`Video-level`: `14` video Test `Cam4K` + `16` video Cohort IIa kèm kiểm định `2,000` lần Bootstrap); (3) Phiên stream liên tục `12 giây/video` (`942` frame); và (4) Đo độ trễ CPU / hiệu chuẩn ngưỡng chấm công.
   - Bộ chỉ số chuẩn quốc tế **ISO/IEC 30107-3**:
     - $\text{APCER}$ (*Attack Presentation Classification Error Rate*): Tỷ lệ ảnh/video giả mạo bị nhận nhầm thành thật (nguy cơ lọt gian lận).
     - $\text{BPCER}$ (*Bona Fide Presentation Classification Error Rate*): Tỷ lệ người thật bị từ chối nhầm thành giả (gây phiền hà phải quét lại).
     - $\text{ACER} = \frac{\text{APCER} + \text{BPCER}}{2}$: Sai số trung bình tổng hợp (mục tiêu đề tài: $< 2.50\%$).
2. **Mục 5.3 — Các kết quả Định lượng cốt lõi (`Quantitative Results` - Hình 5.1 $\to$ 5.3 & Bảng 5.1 $\to$ 5.4)**:
   - **Độ hội tụ huấn luyện (Hình 5.1)**: Qua `18` epoch, loss giảm mượt mà và đạt cực tiểu trên tập Validation tại **Epoch 14** ($\mathcal{L}_{\text{val}} = 0.0941$, $\mathcal{L}_{\text{train}} = 0.0922$, độ lệch $\Delta\mathcal{L} = 0.0019$ cực nhỏ chứng tỏ không bị overfit). Sau khi trộn trọng số `WiSE-FT` ($\lambda = 0.75$), độ chính xác Validation tăng từ `98.01%` lên **`98.88%`**.
   - **Kết quả mức Frame & Ma trận nhầm lẫn trên `Cam4K` (Bảng 5.1 & Hình 5.2)**:
     - Tập Validation (`5,015` ảnh): Accuracy **`98.88%`**, `APCER = 0.31%` (`8/2,546`), `BPCER = 1.94%` (`48/2,469`), **`ACER = 1.13%`**.
     - Tập Test độc lập (`4,811` ảnh của những người chưa từng xuất hiện trong tập Train/Val): Accuracy **`98.09%`** (`4,719/4,811` ảnh đúng), `Macro-F1 = 98.09%`, **`APCER = 3.76%`** (`89/2,365`), **`BPCER = 0.12%`** (chỉ sai đúng **`3` ảnh** trên tổng số `2,446` ảnh người thật!), **`ACER = 1.94%`** (vượt xa mục tiêu `< 2.50%`).
   - **Kết quả mức Video & Kiểm tra chéo giữa các Camera (Hình 5.3 & Bảng 5.2)**:
     - Trên cả `14/14` video Validation và **`14/14` video Test độc lập**, độ chính xác mức video đạt tuyệt đối **`100.00%`**. Tất cả 7 video người thật đều đạt $\bar{P}_{\text{real}} \ge 97.76\%$; 6/7 video giả mạo có $\bar{P}_{\text{fake}} > 93.0\%$, và ngay cả video giả mạo khó nhất (`fake_REC-0041-A`) vẫn đạt $\bar{P}_{\text{fake}} = 75.41\%$ (cách ngưỡng quyết định `50%` hơn `25%` biên an toàn).
     - Kiểm tra chéo trên **`600` ảnh webcam laptop ngoài** (không chỉnh lại ngưỡng): Đạt **`99.84%` accuracy** (`299/300` ảnh giả bị chặn, `300/300` ảnh thật qua cửa, `ACER = 0.17%`).
   - **Độ trễ CPU & Hiệu chuẩn ngưỡng Chấm công (Bảng 5.3 & Bảng 5.4)**:
     - Suy luận mạng `LiteModel` (`32×32`, `XNNPACK`): trung vị **`0.163 ms`** (`P95 = 0.182 ms`, xử lý `~5,980 mặt/giây`), chỉ chiếm `< 0.5%` quỹ thời gian `33.3 ms` của khung hình `30 FPS`.
     - Toàn bộ pipeline 1 ảnh (Giải mã JPEG + YuNet + CNN): `37.84 ms` trung vị (`40.61 ms` qua HTTP Waitress). Giao diện Camera đa luồng `run.py` đạt ổn định **`30.4 FPS`**.
     - Hiệu chuẩn ngưỡng Cosine `ArcFace` (Bảng 5.4): Chọn ngưỡng tối ưu **`$\tau = 0.55$`** giúp tỷ lệ nhận nhầm người lạ bằng không (**`Impostor FAR = 0.00%`**) trong khi tỷ lệ nhận đúng nhân viên theo video đạt **`100.00%`**. Khi dí video điện thoại quay mặt nhân viên vào hệ thống, `ArcFace` đơn thuần cho điểm tương đồng tới `0.72–0.84` (bị lừa hoàn toàn), nhưng cổng `LivenessGate` đã chặn đứng **`100%`** (`FAKE - TỪ CHỐI`).
3. **Mục 5.4 & 5.5 — Kết quả Định tính & Phân tích lỗi (`Qualitative Results & Error Analysis`)**:
   - **Tại sao có 89 frame giả bị nhầm ở mức đơn frame?**: Hơn một nửa số frame lỗi này tập trung ở đúng 1 video `fake_REC-0041-A` (quay màn hình OLED mật độ điểm ảnh rất cao ở khoảng cách `~0.65 m` không có bóng đèn trần phản chiếu làm giảm hiện tượng sọc Moiré ở từng frame lẻ). Tuy nhiên trên toàn video đó, hơn `75.41%` số frame vẫn bị bắt là `fake`, nên bộ lọc trượt thời gian chặn đứng hoàn toàn.
   - **Tại sao có 3 frame thật bị nhầm (`BPCER = 0.12%`)?**: Xảy ra đúng vào khoảnh khắc người dùng quay đầu nhanh làm ảnh bị nhòe chuyển động kết hợp bóng đèn huỳnh quang chiếu lóa trên trán trong `0.15 giây` (4–5 frame), sau đó `FastBinaryDecision` giữ ổn định nhãn `REAL`.
4. **Mục 5.6 & 5.7 — So sánh với các nghiên cứu trước & Nghiên cứu Ablation 3 nhân tố (Bảng 5.5, Bảng 5.6 & Hình 5.5)**:
   - **Ablation Nhân tố A (Chuẩn hóa & Lấy mẫu trên tập gộp 89 video)**: Gộp dữ liệu lệch với trọng số phạt `52.75×` + `BatchNorm` (A2) làm lỗi lọt giả `APCER` tăng vọt từ `7.88%` lên **`16.16%`**. Đổi sang hàng đợi cân bằng `4×8` + `GroupNorm/LayerNorm` (A4) kéo `APCER` giảm gần 4 lần xuống còn **`4.44%`** (`Accuracy = 95.22%`, `ACER = 4.82%`).
   - **Ablation Nhân tố B (Làm mượt thời gian)**: Gộp theo chuỗi thời gian giúp nâng độ chính xác trên `Cam4K` từ `98.09%` (đơn frame) lên **`100.00%`** (video), và trong `942` frame phát lại liên tục suốt `12 giây` trên 16 video, hệ thống đạt **`741` quyết định đúng, `0` quyết định sai (`0.0%` lỗi)**.
   - **Ablation Nhân tố C (Biên dịch đồ thị tĩnh)**: Đóng băng sang `LiteRT XNNPACK` giảm thời gian chạy từ `32.4 ms` xuống **`0.163 ms`** (**gấp `198 lần`**).
5. **Chương 6 — Tổng kết, Hạn chế & Hướng phát triển (`Conclusion`)**:
   - **4 hạn chế trung thực**: (1) Tập trung chủ yếu vào tấn công phát lại qua màn hình điện tử và ảnh in thường, chưa kiểm thử mặt nạ silicon 3D đắt tiền hoặc giấy in lụa cao cấp; (2) Số lượng video kiểm thử độc lập ở mức chuỗi là `14` video trên `Cam4K` (`16` video trên Cohort IIa); (3) Phụ thuộc ánh sáng khả kiến RGB (hạn chế khi phòng tối đen $\mu < 35$ hoặc góc nghiêng đầu $> 35^\circ$); (4) Màn hình OLED độ phân giải siêu cao đặt ở góc không phản광 làm hẹp biên độ đơn frame.
   - **4 hướng phát triển**: (1) Kết hợp **Hybrid Thụ động + Chủ động có điều kiện** (chỉ yêu cầu chớp mắt/đổi màu màn hình khi điểm tự tin rơi vào vùng nghi ngờ `0.35–0.85`); (2) Bổ sung nhánh biến đổi tần số `FFT` và tập dữ liệu mặt nạ 3D; (3) Đưa model `2.18 MB` chạy trực tiếp trên trình duyệt bằng `WebAssembly / WebGPU`; (4) Tự động thích nghi miền liên tục (`Continual WiSE-FT`) khi lắp thêm camera mới.

### 6.3. Gợi ý 5 Slide & Kịch bản nói Tiếng Anh cho Lê Thanh Phương (Phần 4 + Live Demo)

- **Slide 16 (Hội tụ huấn luyện & Kết quả mức Frame trên `Cam4K` — Hình 5.1, Hình 5.2 & Bảng 5.1)**:
  - *Lời nói Tiếng Anh gợi ý*:
    > “Thank you, Tuyen. Hello teachers, my name is **Le Thanh Phuong (`HE186753`)**, and I will present **Chapter 5, Chapter 6, and our Live Demo**.
    > Figure 5.1 shows our training convergence over `18` epochs on `Cam4K`, reaching lowest validation loss at **Epoch 14** with almost zero gap between training and validation. In Table 5.1 and Figure 5.2, on **`4,811` unseen test crops**, our model achieves **`98.09%` accuracy** and an **`ACER` of `1.94%`**—beating our `< 2.5%` target. Notably, real-user false rejection (`BPCER`) is only **`0.12%`**—just `3` misclassified frames out of `2,446` real frames.”
- **Slide 17 (Kết quả mức Video `14/14 = 100%`, Kiểm tra chéo Webcam & Phân tích lỗi — Hình 5.3 & Bảng 5.2)**:
  - *Lời nói Tiếng Anh gợi ý*:
    > “In real check-ins, we evaluate video streams rather than single frames. As shown in Figure 5.3 and Table 5.2, our model classifies **`100%`—all `14` out of `14` unseen test videos—correctly** with wide safety margins. Even the hardest OLED replay video (`fake_REC-0041-A`) scores `75.41%` fake. Moreover, when tested on `600` crops from an external laptop webcam without retuning, the same model retains **`99.84%` accuracy**, proving that `WiSE-FT` prevents forgetting across cameras.”
- **Slide 18 (Nghiên cứu Ablation 3 nhân tố & Bảng tốc độ CPU — Hình 5.5, Bảng 5.3, Bảng 5.4 & Bảng 5.6)**:
  - *Lời nói Tiếng Anh gợi ý*:
    > “Figure 5.5 and Table 5.6 summarize three ablation studies: First, replacing weighted `BatchNorm` with `4×8` balanced sampling and `GroupNorm` cuts spoof error by nearly four times, from `16.16%` down to **`4.44%`**. Second, temporal smoothing achieves **`0%` committed error** across `942` streaming frames. Third, compiling to `LiteRT XNNPACK` speeds up CPU inference **`198` times**, from `32.4 ms` down to **`0.163 ms`**, running live at **`30.4 FPS`**. For `ArcFace` attendance, Table 5.4 shows that a cosine threshold of **`0.55`** gives **`0%` stranger false acceptance** and **`100%` employee recall**.”
- **Slide 19 (Kết quả Định tính, Tổng kết Chương 6, Hạn chế & Hướng phát triển — Hình 5.4)**:
  - *Lời nói Tiếng Anh gợi ý*:
    > “In Chapter 6, we conclude that our lightweight `545K`-parameter framework successfully protects CPU webcam attendance in real time. Our current limitations are 3D silicone masks, matte photo prints, extreme darkness, and ángulos beyond `35` degrees. In future work, we plan to add conditional active challenges only for uncertain scores, frequency-domain branches, and browser-side `WebAssembly` inference.”
- **Slide 20 (Trình diễn Live Demo trực tiếp trước Hội đồng — `60 giây`)**:
  - *Cách thực hiện Live Demo ăn điểm tuyệt đối*:
    1. Mở sẵn giao diện Web Chấm công (`Chay_Cham_Cong_FaceID.bat` tại `http://localhost:8000`) hoặc cửa sổ Camera `30.4 FPS` (`chay_test_camera.bat`).
    2. **Kịch bản 1 (Người thật đứng vào camera)**: Thành hoặc Phương đứng trước camera $\to$ Hệ thống hiện khung **XANH LÁ (`REAL | ~99%`)**, nhận diện đúng tên nhân viên và ghi nhận điểm danh thành công vào bảng lịch sử CSV.
    3. **Kịch bản 2 (Tấn công giả mạo bằng điện thoại)**: Giơ điện thoại mở sẵn video/ảnh chụp khuôn mặt của chính nhân viên đó trước camera $\to$ Dù khuôn mặt giống hệt, cổng `LivenessGate` lập tức khóa khung **ĐỎ (`FAKE - TỪ CHỐI`)** trong `< 0.5 giây` và chặn không cho ghi điểm danh!
  - *Lời nói Tiếng Anh khi Demo*:
    > “Now, we would like to show a live demonstration of our system. First, when a real team member stands in front of the camera, `LivenessGate` verifies real skin at `30 FPS`, and `ArcFace` logs the attendance immediately. Next, when we hold up a smartphone playing a high-resolution video of the same member, our `LivenessGate` immediately detects the screen attack, turns red (`FAKE - REJECTED`), and blocks the fraudulent check-in. Thank you for listening, and we welcome your questions!”

### 6.4. Bốn câu hỏi Hội đồng hay hỏi xoáy phần Chương 5–6 (Phương cần nắm)
1. **“Trên tập Test `Cam4K`, tại sao `APCER` ở mức đơn frame là `3.76%` (`89` frame bị lọt), nhưng sang mức Video lại đạt `100%` (`0%` lỗi)?”**
   - *Trả lời*: Dạ `89` frame giả bị nhầm ở mức đơn lẻ chủ yếu nằm ở video `fake_REC-0041-A` khi người tấn công cầm điện thoại màn hình OLED ở khoảng cách `~0.65m` trùng độ sáng tường nên một số frame lẻ bị giảm vân Moiré. Tuy nhiên ngay trong video khó nhất đó, vẫn có tới **`75.41%`** số frame bị phát hiện là `FAKE`. Khi hệ thống chạy thực tế theo chuỗi thời gian (trung bình trượt cửa sổ `0.45s` hoặc `6 frame`), đa số áp đảo các frame `FAKE` sẽ quyết định nhãn của cả phiên, giúp chặn đúng `100%` (`14/14` video) ạ.
2. **“Làm sao nhóm chọn được con số ngưỡng Cosine $\tau = 0.55$ cho nhận diện `ArcFace` trong hệ thống chấm công?”**
   - *Trả lời*: Dạ bọn em không chọn cảm tính mà viết script `calibrate_threshold.py` chạy thực nghiệm quét ngưỡng từ `0.40` đến `0.65` (Bảng 5.4) theo 2 giao thức: *Leave-One-Video-Out* (đo tỷ lệ nhận đúng nhân viên ở video mới) và *Leave-One-Person-Out* (giả lập người lạ chưa đăng ký bước vào). Ở ngưỡng `0.50` vẫn còn `0.12%` nguy cơ nhận nhầm người lạ, còn lên `0.60` thì nhân viên hơi nghiêng đầu bị từ chối mất `4.9%`. Tại điểm **`$\tau = 0.55$**, tỷ lệ lọt người lạ giảm về **`0.00%`** trong khi tỷ lệ nhận diện nhân viên theo phiên video vẫn đạt **`100.00%`** ạ.
3. **“Nếu kẻ gian dùng một bức ảnh in giấy nhám (matte paper) không bị bóng lóa như màn hình điện thoại thì hệ thống xử lý thế nào?”**
   - *Trả lời*: Dạ bên cạnh mạng CNN `LivenessNet`, trong `pipeline.py` bọn em thiết kế thêm mô-đun kiểm tra quang học trên không gian màu $YC_rC_b$ và HSV (`fusion=True`) đo độ phân tán sắc đỏ của máu dưới da $\sigma(C_r)$, độ bão hòa màu $\mu(S)$ và độ phủ bạc màu của giấy in ($P_{92}(Y) - P_{50}(Y)$). Giấy in nhám tuy không có bóng kính nhưng bị mất độ sâu sắc đỏ tự nhiên của da người nên sẽ bị bộ lọc sắc độ này hạ điểm. Tuy nhiên với các loại giấy in chuyên dụng cao cấp hoặc mặt nạ 3D, trong Chương 6 bọn em cũng đề xuất hướng ghép thêm thử thách chủ động có điều kiện (`Conditional Challenge`) khi điểm tự tin nằm ở vùng lưỡng lự `0.35–0.85` ạ.
4. **“Tại sao nhóm đánh giá tốc độ trên CPU mà không dùng GPU?”**
   - *Trả lời*: Dạ mục tiêu thực tế của đề tài là triển khai cho các quầy lễ tân, lớp học và văn phòng vừa và nhỏ — nơi máy tính điểm danh thường chỉ là mini-PC hoặc laptop văn phòng không có card đồ họa NVIDIA rời. Nhờ kiến trúc `32×32` nhẹ và biên dịch `LiteRT XNNPACK`, hệ thống chạy hoàn toàn trên CPU vẫn đạt `0.163 ms/mặt` và `30.4 FPS` mượt mà, giúp tiết kiệm chi phí phần cứng cho doanh nghiệp ạ.

---

## 7. Hướng Dẫn Clone & Chạy Hệ Thống Nhanh Cho Các Thành Viên Trong Nhóm

Bất kỳ thành viên nào trong nhóm muốn tải toàn bộ dự án về máy cá nhân để chạy thử hoặc chuẩn bị máy dự phòng cho buổi bảo vệ chỉ cần làm đúng 2 bước:

### Bước 1: Clone repository về máy
```powershell
git clone https://github.com/thanhAI408/face_anti_spoofing.git
cd face_anti_spoofing
```

### Bước 2: Chạy ngay bằng các file `.bat` tự động (1-Click trên Windows)
1. **Chạy ứng dụng Camera Chống Giả Mạo `30.4 FPS` (`run.py`) hoặc Web Liveness (`port 8080`)**:
   - Nhấp đúp vào file **`chay_test_camera.bat`** ở thư mục gốc.
   - *(Nếu máy chưa có môi trường ảo `venv`, file `.bat` sẽ tự động tạo `venv` và cài đặt thư viện trong lần chạy đầu tiên).*
   - Nhấn `Enter` (hoặc số `0`) để mở Camera USB, số `1` để mở Webcam Laptop, hoặc số `2` để mở giao diện Web tại `http://127.0.0.1:8080`.
2. **Chạy Hệ thống Web Điểm Danh Nhân Viên tích hợp Chống Giả Mạo (`ChamCong_FaceID` tại `port 8000`)**:
   - Nhấp đúp vào file **`Chay_Cham_Cong_FaceID.bat`** ở thư mục gốc (hoặc `ChamCong_FaceID\1_Chay_Web_Cham_Cong.bat`).
   - *(Nếu máy chưa có `ChamCong_FaceID\.venv`, script sẽ tự động tạo `.venv`, cài đặt `requirements-cpu.txt` và nạp sẵn hồ sơ nhân viên mẫu).*
   - Trình duyệt sẽ tự động mở **`http://localhost:8000`** với đầy đủ 3 tab: **Điểm danh Realtime (có cổng chặn `LivenessGate`)**, **Đăng ký nhân viên mới**, và **Báo cáo & Biểu đồ đánh giá ngưỡng**.
