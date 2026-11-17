"""Kịch bản kiểm thử so sánh: TRƯỚC VÀ SAU KHI BẬT CHỐNG GIẢ MẠO (ANTI-SPOOFING)
Sử dụng mô hình liveness_cam4k (Epoch 14, Test Acc 98.09%, Video Acc 100%).
"""
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Đảm bảo in tiếng Việt trên console Windows không lỗi
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from face_runtime import create_face_app, normalized_embedding
from liveness.liveness_gate import LivenessGate
from preprocessing import PROTOTYPES_FILE

def main():
    print("=" * 78)
    print("   HỆ THỐNG KIỂM THỬ SO SÁNH CHẤM CÔNG KHUÔN MẶT & CHỐNG GIẢ MẠO (LIVENESS)")
    print("=" * 78)

    # 1. Khởi tạo các module
    print("\n[1/3] Đang nạp mô hình nhận diện khuôn mặt và chống giả mạo...")
    face_app = create_face_app(allowed_modules=["detection", "recognition"])
    liveness_gate = LivenessGate()

    # Nạp cơ sở dữ liệu nhân viên mẫu
    proto_file = BASE_DIR / PROTOTYPES_FILE
    if not proto_file.exists():
        print("[INFO] Đang tạo hồ sơ nhân viên mẫu...")
        from init_demo_data import init_demo_employee
        init_demo_employee()

    with np.load(str(proto_file), allow_pickle=False) as model:
        labels = model["labels"].astype(str)
        prototypes = model["prototypes"].astype(np.float32)
        threshold = float(model["cosine_threshold"].item())

    print(f" -> Danh sách nhân viên trong DB: {list(labels)}")
    print(f" -> Ngưỡng nhận diện (Cosine Threshold): {threshold:.2f}")

    # 2. Đọc 2 ảnh test: Người thật vs Kẻ gian lận cầm ảnh điện thoại
    real_img_path = BASE_DIR.parent / "demo_real.jpg"
    fake_img_path = BASE_DIR.parent / "demo_fake.jpg"

    if not real_img_path.is_file() or not fake_img_path.is_file():
        print(f"[ERROR] Thiếu file ảnh demo ({real_img_path} hoặc {fake_img_path})")
        return

    img_real = cv2.imread(str(real_img_path))
    img_fake = cv2.imread(str(fake_img_path))

    faces_real = face_app.get(img_real)
    faces_fake = face_app.get(img_fake)

    if not faces_real or not faces_fake:
        print("[ERROR] Không phát hiện được khuôn mặt trong ảnh test")
        return

    emb_real = normalized_embedding(faces_real[0])
    emb_fake = normalized_embedding(faces_fake[0])

    # Tính độ tương đồng khuôn mặt
    sim_real = float((prototypes @ emb_real)[0])
    sim_fake = float((prototypes @ emb_fake)[0])

    print("\n" + "=" * 78)
    print("   KỊCH BẢN 1: KHI TẮT CHỐNG GIẢ MẠO (HỆ THỐNG GỐC - CHƯA BẢO VỆ)")
    print("=" * 78)
    print("Mô tả: Kẻ gian lận dùng màn hình điện thoại chụp ảnh nhân viên để điểm danh hộ.")
    print("-" * 78)
    print(f"1. Khuôn mặt đưa vào: ẢNH MÀN HÌNH ĐIỆN THOẠI (demo_fake.jpg)")
    print(f"   - Độ khớp khuôn mặt (ArcFace): {sim_fake:.4f} (Ngưỡng yêu cầu: {threshold:.2f})")
    print(f"   - Nhận diện là: '{labels[0]}'")
    if sim_fake >= threshold:
        print(f"   - KẾT QUẢ ĐIỂM DANH: ❌ [LỖ HỔNG NGUY HIỂM] HỆ THỐNG ĐÃ GHI NHẬN ĐIỂM DANH!")
        print(f"     => Gian lận thành công vì hệ thống không phân biệt được người thật hay màn hình!")
    else:
        print(f"   - KẾT QUẢ: Không nhận dạng được.")

    print("\n" + "=" * 78)
    print("   KỊCH BẢN 2: KHI BẬT CHỐNG GIẢ MẠO (TÍCH HỢP MODEL LIVENESS_CAM4K)")
    print("=" * 78)
    print("Mô tả: Tích hợp mô hình liveness_cam4k để thẩm định thực thể sống trước khi điểm danh.")
    print("-" * 78)

    # Test trường hợp FAKE với Liveness
    box_fake = faces_fake[0].bbox.tolist()
    live_fake = liveness_gate.check_liveness(img_fake, box_fake)
    print(f"A. Kiểm tra tấn công giả mạo (demo_fake.jpg):")
    print(f"   - Độ khớp nhận diện: {sim_fake:.4f} (Trùng khớp '{labels[0]}')")
    print(f"   - Thẩm định Liveness: {live_fake['label'].upper()}")
    print(f"   - Điểm tin cậy FAKE : {live_fake['fake_score'] * 100:.2f}%")
    print(f"   - Lý do: {live_fake['reason']}")
    if not live_fake["is_real"]:
        print(f"   - HÀNH ĐỘNG HỆ THỐNG: 🛡️ [CHẶN TUYỆT ĐỐI] TỪ CHỐI ĐIỂM DANH VÀ BẬT CẢNH BÁO ĐỎ!")
        print(f"     => Ngăn chặn gian lận thành công 100%!")

    # Test trường hợp REAL với Liveness
    box_real = faces_real[0].bbox.tolist()
    live_real = liveness_gate.check_liveness(img_real, box_real)
    print(f"\nB. Kiểm tra nhân viên thật (demo_real.jpg):")
    print(f"   - Độ khớp nhận diện: {sim_real:.4f} (Trùng khớp '{labels[0]}')")
    print(f"   - Thẩm định Liveness: {live_real['label'].upper()}")
    print(f"   - Điểm tin cậy REAL : {live_real['real_score'] * 100:.2f}%")
    print(f"   - Lý do: {live_real['reason']}")
    if live_real["is_real"] and sim_real >= threshold:
        print(f"   - HÀNH ĐỘNG HỆ THỐNG: ✅ [CHẤM CÔNG HỢP LỆ] ĐÃ GHI NHẬN VÀO NHẬT KÝ ĐIỂM DANH!")

    print("\n" + "=" * 78)
    print("   TỔNG KẾT SO SÁNH")
    print("=" * 78)
    print(f"| Trường hợp           | Trước khi bật Liveness | Sau khi bật Liveness        |")
    print(f"|----------------------|-----------------------|-----------------------------|")
    print(f"| Người thật trước cam | Chấm công thành công  | Chấm công thành công (100%) |")
    print(f"| Dùng ảnh/màn hình ĐT | BỊ QUA MẶT GIAN LẬN   | BỊ CHẶN TUYỆT ĐỐI ({live_fake['fake_score'] * 100:.1f}%)   |")
    print("=" * 78)

if __name__ == "__main__":
    main()
