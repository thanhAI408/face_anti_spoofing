# Bộ 5 Slide Bảo Vệ Đồ Án (Tiếng Anh Dễ Nói • An Toàn Canva) — Phần 1: Chương 1 & 2 (Nguyễn Văn Thành • HE181962)

> [!IMPORTANT]
> Toàn bộ 5 slide và kịch bản dưới đây đã được **viết bằng Tiếng Anh kỹ thuật trong sáng, câu gọn, từ vựng quen thuộc và dễ phát âm/giải thích**:
> - **File `.pptx` tải lên Canva / PowerPoint**: [`deliverables/Slide_BaoVe_Phan1_NguyenVanThanh_5Slides.pptx`](../deliverables/Slide_BaoVe_Phan1_NguyenVanThanh_5Slides.pptx)
> - **File Teleprompter HTML đọc & giải thích từng câu Chương 1 & 2**: [`capstone/Thanh_BienDich_ThuyetTrinh_Chap1_Chap2.html`](Thanh_BienDich_ThuyetTrinh_Chap1_Chap2.html)

---

## 1. Xem Trước Cả 5 Slide Tiếng Anh (`1920 × 1080`)

### Slide 1 — Cover, Team Roster & Live Demo on USB SC-M200 Camera (`30.4 FPS`)
![Slide 1](../deliverables/slide_previews_phan1/slide_1.png)

### Slide 2 — Why Webcam Attendance Needs Anti-Spoofing & Three Main Challenges
![Slide 2](../deliverables/slide_previews_phan1/slide_2.png)

### Slide 3 — Table 1.1 Method Comparison, Key Targets & Four Main Contributions
![Slide 3](../deliverables/slide_previews_phan1/slide_3.png)

### Slide 4 — Real `32×32` Cam4K Crops, Figure 2.1 Light Differences & 3-Step Separation
![Slide 4](../deliverables/slide_previews_phan1/slide_4.png)

### Slide 5 — GroupNorm, Weight Mixing (`WiSE-FT`), 4-Step Pipeline & Bridge to Chapter 3
![Slide 5](../deliverables/slide_previews_phan1/slide_5.png)

---

## 2. Kịch Bản Nói Cực Dễ Thuộc (Câu Ngắn • Kèm Gợi Ý Phát Âm & Hiểu Bản Chất)

### Slide 1 — Mở Đầu & Giới Thiệu Kết Quả Chạy Thật *(~35–40 giây)*
- **Lời nói Tiếng Anh (câu ngắn, từ quen thuộc):**
  > “Good morning/afternoon teachers. My name is **Nguyen Van Thanh (`HE181962`)**. Today, I will start our capstone presentation on **Real-Time Passive Face Anti-Spoofing for Biometric Attendance**, supervised by **Mr. Nguyen Ngoc Huy**.
  >
  > On the right is our real system running at **`30.4 FPS`** on a normal USB camera. It lets real users pass into `ArcFace`, and immediately blocks phone video attacks on a standard CPU in **`0.163` milliseconds**, with **`1.94%` error** and **`14 out of 14`** test videos correct. In the first five slides, I will present Chapter 1 and Chapter 2.”
- **Từ cần lưu ý cách đọc:** `Passive` *(pát-xịp)* • `Anti-Spoofing` *(an-ti spúp-phinh)* • `Biometric` *(bai-ô-mét-trích)*.
- **Hiểu bản chất bằng Tiếng Việt (để nói tự nhiên không cần học vẹt):**
  > Chào thầy cô, giới thiệu tên mình và thầy Huy hướng dẫn. Chỉ tay sang cột phải: đây là ảnh chụp hệ thống của nhóm đang chạy thật ở `30,4 FPS` trên webcam USB thường — người thật (`REAL 100%`) thì cho đi tiếp vào nhận diện `ArcFace`, còn giơ điện thoại chiếu video giả (`FAKE`) là chặn ngay trên CPU văn phòng chỉ mất `0,163 ms`, tỷ lệ lỗi chung `ACER` chỉ `1,94%` và đúng trọn vẹn `14/14` video kiểm thử.

---

### Slide 2 — Vấn Đề Của ArcFace & 3 Bài Toán Nhóm Giải Quyết *(~50 giây)*
- **Lời nói Tiếng Anh (câu ngắn, dễ nói):**
  > “In Chapter 1, why do normal webcam check-in systems need anti-spoofing? Models like **ArcFace** are great at recognizing *who you are*, but they **cannot tell real human skin from a flat phone screen**. If someone holds up a printed photo or plays a 4K phone video, ArcFace still gives a high match score above `0.55` and accepts the fake check-in.
  >
  > Infrared and 3D cameras are too expensive for normal classrooms, while asking users to blink slows check-in by `3 to 5 seconds` and can still be tricked by recorded videos. Therefore, we place a fast **`LivenessNet` security gate** right before ArcFace, and solve **three main challenges**:
  > - **First**, stopping the model from memorizing faces and handling color changes across webcams;
  > - **Second**, running under `1` millisecond on CPU at a smooth `30 FPS` without label flickering;
  > - **Third**, matching face crop boxes between `SCRFD` and `YuNet`.”
- **Từ cần lưu ý cách đọc:** `Infrared` *(in-phờ-ra-rét — hồng ngoại)* • `Flickering` *(phlích-cơ-rinh — nhấp nháy nhãn)*.
- **Hiểu bản chất bằng Tiếng Việt:**
  > `ArcFace` chỉ so nét mặt xem *"đây là ai"* chứ không biết mặt đó bằng da thịt hay màn hình điện thoại, nên kẻ gian chiếu video 4K vẫn qua cửa (`Cosine ≥ 0,55`). Lắp camera hồng ngoại/3D thì tốn tiền, bắt nháy mắt/quay đầu thì mất `3–5 giây` xếp hàng mà lấy video quay sẵn cảnh nháy mắt dí vào vẫn lừa được. Vì vậy nhóm đặt cổng gác **`LivenessNet`** chặn ngay trước `ArcFace` và giải quyết 3 việc rất thực tế (`I`, `II`, `III`) ở góc dưới phải.

---

### Slide 3 — So Sánh Các Hướng Cũ, Mục Tiêu Đạt Được & 4 Đóng Góp *(~55 giây)*
- **Lời nói Tiếng Anh (câu ngắn, dễ nói):**
  > “Table 1.1 on the left compares four common RGB approaches. Traditional features like `LBP` are fast on CPU, but fail easily when camera lighting or phone screens change. Meanwhile, 3D depth CNNs like `CDCN` are too heavy and slow without a GPU. That is why we choose **Small-Patch Light CNNs**, and directly fix their two main issues: `BatchNorm` drift and frame flicker.
  >
  > On the right, our system beats all three project targets: **`1.94%` ACER error**, **`100%` accuracy on 14 test videos**, and **`0.163 ms` CPU speed**, through **four main contributions**:
  > - **One**, fixing `BatchNorm` drift with balanced `4×8` batches and `GroupNorm`, cutting spoof error from `16.16%` down to `4.44%`;
  > - **Two**, collecting our 91-video `Cam4K` dataset split by person, and mixing weights with `WiSE-FT`;
  > - **Three**, converting our `2.15 MB` model to `LiteRT` with 6-frame score smoothing;
  > - **Four**, building an aligned `LivenessGate` using `YuNet` inside `SCRFD` face boxes.”
- **Từ cần lưu ý cách đọc:** `Approaches` *(ơ-prâu-chịt — các hướng tiếp cận)* • `Contributions` *(con-tri-biu-shần — đóng góp)*.
- **Hiểu bản chất bằng Tiếng Việt:**
  > Nhìn bảng trái: thuật toán cổ điển `LBP` nhẹ nhưng đổi ánh sáng là sai; mạng đoán độ sâu `CDCN` chính xác hơn nhưng nặng, không có GPU không chạy nổi thời gian thực → nhóm chọn **mạng CNN nhỏ học trên vùng mặt nhỏ (`32×32`)** và tự sửa 2 bệnh của nó là lệch `BatchNorm` và nháy khung hình. Bên phải là 3 con số vượt chỉ tiêu kèm 4 đóng góp đánh số `01 → 04`; hộp xám dưới trái nhắc rõ phạm vi đồ án (chống phát lại video 2D & ảnh in trên 1 webcam thường, không xét mặt nạ silicon 3D).

---

### Slide 4 — Vì Sao Camera 2D Bắt Được Giả Mạo & Vì Sao Thu Nhỏ Về `32 × 32`? *(~55 giây)*
- **Lời nói Tiếng Anh (câu ngắn, dễ nói):**
  > “Moving to Chapter 2: how can a normal 2D camera spot fake faces without a 3D sensor? Figure 2.1 on the right shows **three clear light differences**:
  > - **First, Real Human Skin** absorbs and softens incoming light, creating smooth skin tones and gentle 3D shadows along the nose and cheeks.
  > - **Second, Phone and Tablet Screens** have an RGB pixel grid that clashes with the camera sensor to form wavy **Moiré lines**, plus sharp white glass glare.
  > - **Third, Printed Paper Photos** scatter light flatly, losing warm skin depth and natural 3D shadows.
  >
  > On the left are real **`32 × 32`** face crops from our `Cam4K` dataset. Why resize to `32 × 32`? Because tiny `32 × 32` images blur out personal details like eyes and moles, **stopping the CNN from memorizing people's faces**, while keeping screen glare and color clues clear at under `0.2` milliseconds on CPU.”
- **Từ cần lưu ý cách đọc:** `Moiré` *(moa-rê — vân gợn sóng khi chụp màn hình)* • `Glare` *(gle — vệt lóa trắng trên mặt kính)* • `Blur out` *(blơ ao — làm mờ đi)* • Chữ nhỏ dưới ảnh mẫu `Bona Fide (Real)` *(bô-na phai-đê = thuật ngữ chuẩn ISO chỉ Người thật)*.
- **Hiểu bản chất bằng Tiếng Việt:**
  > Slide này cực dễ ăn điểm vì rất trực quan:
  > 1. **3 dấu hiệu ánh sáng (`Figure 2.1`)**: Da thật mềm và thấm sáng nên màu da chuyển êm, có bóng mũi/má 3D; Màn hình điện thoại có lưới điểm ảnh RGB đụng với cảm biến camera tạo sọc sóng **Moiré** và bóng đèn lóa trắng trên kính; Ảnh giấy thì phẳng lì, bệt sáng.
  > 2. **Tại sao ép ảnh về `32×32`?**: Để làm nhòe mắt, mũi, nốt ruồi — ép mạng CNN không thể "học vẹt" mặt người quen trong tập train mà bắt buộc phải nhìn chất liệu da hay vệt lóa màn hình. Dải dưới phải nói thêm lợi ích của việc tách riêng 3 bước (Phát hiện mặt → Chống giả → Nhận diện ArcFace): công ty thêm nhân viên mới không bao giờ phải train lại model chống giả.

---

### Slide 5 — Ba Ý Toán Học Cốt Lõi, 4 Bước Hệ Thống & Nhường Lời Sang Chương 3 *(~55 giây)*
- **Lời nói Tiếng Anh (câu ngắn, dễ nói):**
  > “My final slide summarizes three core math ideas behind our system:
  > - **First, Per-Image Normalization (`GroupNorm`)**: When merging datasets with `52×` class weights, normal `BatchNorm` averages drift across cameras and double spoof error to `16.16%`. `GroupNorm` normalizes *inside each single image*, cutting spoof error down to **`4.44%`**.
  > - **Second, Weight Mixing (`WiSE-FT`)**: We mix **`25%`** of our general webcam weights with **`75%`** of our new `Cam4K` camera weights into one single model file—keeping `98.88%` on `Cam4K` and `99.84%` on other webcams with **`0 ms`** extra delay.
  > - **Third, ISO Error Metrics**: We evaluate using `ACER`—the average of fake-accepted (`APCER`) and real-rejected (`BPCER`)—alongside Laplacian blur filtering and `IoU` face tracking across our 4-step pipeline below.
  >
  > Finally, to avoid the four data pitfalls in the bottom dark bar—especially putting frames from the same video into both train and test—clean data processing is essential. Next, **Nguyen Chien Thang** will present **Chapter 3: Data Processing**. Thank you!”
- **Từ cần lưu ý cách đọc:** `WiSE-FT` *(đọc từng chữ: oai - ép - ti)* • `Laplacian` *(la-pờ-lát-xi-ần)* • `Pitfalls` *(pít-phôn — các bẫy lỗi dễ mắc)*.
- **Hiểu bản chất bằng Tiếng Việt:**
  > 3 hộp trắng ở trên tương ứng 3 công thức dễ giải thích nhất:
  > 1. **`GroupNorm`**: Tính trung bình và phương sai *ngay bên trong từng bức ảnh lẻ* thay vì tính chung cả lô như `BatchNorm`, nhờ vậy gộp nhiều bộ dữ liệu không bị lệch thống kê (hạ lỗi lọt giả từ `16,16%` xuống `4,44%`).
  > 2. **`WiSE-FT` (`0.25 · θ(base) + 0.75 · θ(cam4k)`)**: Cộng trực tiếp `25%` bộ trọng số webcam chung với `75%` bộ trọng số camera mới thành 1 file duy nhất → vừa giỏi trên camera mới (`98,88%`), vừa không quên webcam cũ (`99,84%`), không tốn thêm thời gian chạy.
  > 3. **`ACER = (APCER + BPCER)/2`**: Trung bình cộng của tỷ lệ lọt giả mạo và tỷ lệ từ chối nhầm người thật; kèm lọc ảnh mờ bằng phương sai `Laplacian` và nối vết khuôn mặt bằng `IoU`.
  > Chỉ xuống dải đen dưới cùng (4 cái bẫy dữ liệu, nguy hiểm nhất là cắt frame ngẫu nhiên làm cùng 1 video lọt vào cả tập train lẫn test) rồi nhường mic mượt mà cho **Nguyễn Chiến Thắng (Chương 3)**.

---

## 3. Bí Kíp Trả Lời Nhanh 4 Câu Hội Đồng Hay Hỏi Nhất Ở Phần Của Thành

1. **“Tại sao nhóm lại resize ảnh khuôn mặt nhỏ tới mức `32 × 32` pixel? Nhỏ vậy có mất hết thông tin không?”**
   - **Cách trả lời gọn:** Dạ không những không mất dấu hiệu giả mạo mà đây là thiết kế cố ý ạ. Nếu để ảnh lớn như `112×112` hay `224×224`, mạng CNN rất dễ nhìn vào mắt, mũi, nốt ruồi rồi *học thuộc lòng mặt của các bạn tình nguyện viên* trong tập train (gặp người mới là đoán sai). Khi thu về `32×32`, các nét nhận dạng cá nhân bị làm mờ đi, ép mạng chỉ học được độ mịn màu da, vệt lóa kính và nhiễu màn hình — đồng thời giúp tốc độ chạy trên CPU đạt `0,163 ms`.

2. **“Giải thích ngắn gọn tại sao `BatchNorm` lại bị lỗi khi gộp dữ liệu, còn `GroupNorm` thì sửa được?”**
   - **Cách trả lời gọn:** Dạ khi gộp dữ liệu nhiều nguồn, số lượng mẫu bị lệch nặng nên bọn em đặt trọng số phạt (`loss weight`) gấp `52,75` lần cho lớp ít mẫu. Lúc train, `BatchNorm` lấy trung bình chung trên cả `batch` gồm nhiều camera khác nhau nên bộ trung bình trượt (`running mean/variance`) bị kéo lệch hẳn so với lúc chạy thực tế 1 camera, làm lỗi lọt giả tăng gấp đôi lên `16,16%`. Khi đổi sang `GroupNorm`, việc chuẩn hóa được tính **khép kín bên trong từng bức ảnh riêng lẻ**, không dính dáng tới các ảnh khác trong batch nên triệt tiêu được hiện tượng lệch này và hạ lỗi xuống `4,44%` ạ.

3. **“Công thức `θ* = 0.25 · θ(base) + 0.75 · θ(cam4k)` (WiSE-FT) giúp ích gì so với fine-tune thông thường?”**
   - **Cách trả lời gọn:** Dạ nếu chỉ fine-tune `100%` trên camera `SC-M200` mới (`Cam4K`), model sẽ bị *quên kiến thức cũ* — tức là chạy rất tốt trên camera `SC-M200` nhưng đem sang webcam laptop khác dễ bị sai. Vì hai model có cùng kiến trúc, bọn em trộn tuyến tính `75%` trọng số mới với `25%` trọng số gốc trước khi xuất ra 1 file `.tflite` duy nhất: nhờ vậy vừa giữ được `98,88%` trên camera mới, vừa giữ `99,84%` trên webcam ngoài mà thời gian chạy vẫn giữ nguyên `0,163 ms` (không phải chạy 2 model cùng lúc) ạ.

4. **“Tại sao trong hệ thống điểm danh phải dùng thêm `YuNet` cắt lại ảnh bên trong `SCRFD` (`LivenessGate`)?”**
   - **Cách trả lời gọn:** Dạ vì `ArcFace` dùng bộ dò `SCRFD` vốn cắt khung mặt hơi rộng (lấy cả một phần trán/cổ/nền), trong khi mạng `LivenessNet` lúc train lại được học trên khung mặt cắt sát của `SSD` và `YuNet`. Nếu lấy thẳng khung rộng của `SCRFD` đưa vào `LivenessNet` thì tỷ lệ vùng mặt bị lệch làm giảm độ chính xác; nên bọn em cho `YuNet` (rất nhẹ, `< 250 KB`) cắt lại đúng khung mặt sát bên trong vùng đó trước khi chấm điểm thật/giả ạ.
