"""Chương trình nhận diện khuôn mặt và chống giả mạo realtime bằng OpenCV (Desktop Window).
Hỗ trợ chọn camera (chỉ số 0, 1, 2... tương ứng với webcam tích hợp hoặc SC-M200).

Cách dùng:
    python real_time_inferent.py --camera 0
    python real_time_inferent.py --camera 1
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Reconfigure console encoding
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from face_runtime import create_face_app, normalized_embedding
from liveness.liveness_gate import LivenessGate
from preprocessing import PROTOTYPES_FILE
from app import attendance

def parse_args():
    parser = argparse.ArgumentParser(description="Chấm công FaceID Real-Time OpenCV")
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số camera (0: cam mặc định, 1: cam phụ / SC-M200,...)")
    parser.add_argument("--no-liveness", action="store_true", help="Tắt module chống giả mạo liveness (chế độ kiểm thử)")
    return parser.parse_args()

def main():
    args = parse_args()
    liveness_enabled = not args.no_liveness

    proto_path = BASE_DIR / PROTOTYPES_FILE
    if not proto_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {proto_path}; hãy chạy init_demo_data.py hoặc preprocessing.py trước.")

    with np.load(str(proto_path), allow_pickle=False) as model:
        prototype_labels = model["labels"].astype(str)
        prototypes = model["prototypes"].astype(np.float32)
        cosine_threshold = float(model["cosine_threshold"].item())

    print(f"[INFO] Đang nạp mô hình nhận diện khuôn mặt...")
    face_app = create_face_app()

    liveness_gate = None
    if liveness_enabled:
        print("[INFO] Đang kích hoạt Liveness Gate chống giả mạo (liveness_cam4k)...")
        liveness_gate = LivenessGate()

    print(f"[INFO] Mở Camera index: {args.camera}...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[WARN] CAP_DSHOW thất bại; thử backend mặc định cho camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[ERROR] Không thể mở camera {args.camera}. Vui lòng kiểm tra lại dây cắm hoặc thử --camera 0 / 1 / 2.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("=" * 70)
    print("   HỆ THỐNG NHẬN DIỆN KHUÔN MẶT REAL-TIME")
    print(f"   Camera: {args.camera} | Chống giả mạo: {'BẬT (BẢO VỆ)' if liveness_enabled else 'TẮT (NGUY HIỂM)'}")
    print("   Phím tắt:")
    print("     'q': Thoát chương trình")
    print("     'l': Bật/Tắt chế độ chống giả mạo Liveness")
    print("=" * 70)

    previous_time = 0.0
    while True:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Không đọc được frame từ camera.")
            break
        frame = cv2.flip(frame, 1)

        current_time = time.time()
        fps = 1 / (current_time - previous_time) if previous_time else 0
        previous_time = current_time

        faces = face_app.get(frame)
        all_bboxes = [face.bbox.tolist() for face in faces]
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            embedding = normalized_embedding(face)

            is_spoof = False
            liveness_info = None
            if liveness_enabled and liveness_gate is not None:
                liveness_info = liveness_gate.check_liveness(frame, (x1, y1, x2, y2), all_bboxes=all_bboxes)
                is_spoof = not liveness_info["is_real"]

            if embedding is None:
                name = "Người lạ"
                score = 0.0
                color = (0, 140, 255) # Cam
            else:
                similarities = prototypes @ embedding
                best_index = int(np.argmax(similarities))
                score = float(similarities[best_index])
                if score >= cosine_threshold:
                    name = prototype_labels[best_index]
                    color = (0, 230, 118) # Xanh lá
                else:
                    name = "Người lạ"
                    color = (0, 140, 255) # Cam

            # Nếu phát hiện giả mạo FAKE
            if liveness_enabled and is_spoof:
                color = (0, 0, 255) # Đỏ
                label = f"[FAKE {liveness_info['fake_score']*100:.0f}%] {name}"
            elif liveness_enabled:
                label = f"[REAL] {name} ({score:.2f})"
                if name != "Người lạ":
                    attendance.record(name, score, status="HỢP LỆ")
            else:
                label = f"{name} ({score:.2f})"
                if name != "Người lạ":
                    attendance.record(name, score, status="HỢP LỆ")

            # Vẽ khung nhận diện
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text_top = max(0, y1 - 28)
            cv2.rectangle(frame, (x1, text_top), (x1 + len(label) * 12 + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 6, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Header thông tin FPS và trạng thái Liveness
        status_text = "LIVENESS: ON (Protected)" if liveness_enabled else "LIVENESS: OFF (Vulnerable)"
        status_color = (0, 230, 118) if liveness_enabled else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, status_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

        cv2.imshow("FaceID & Anti-Spoofing Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            liveness_enabled = not liveness_enabled
            if liveness_enabled and liveness_gate is None:
                liveness_gate = LivenessGate()
            print(f"[INFO] Đã chuyển Liveness sang: {'BẬT' if liveness_enabled else 'TẮT'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
