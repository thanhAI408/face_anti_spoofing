"""Tạo sẵn hồ sơ khuôn mặt nhân viên mẫu từ ảnh demo_real.jpg để hệ thống sẵn sàng test ngay lập tức."""
import os
import sys
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
from preprocessing import EMBEDDINGS_FILE, PROTOTYPES_FILE, build_prototypes, COSINE_THRESHOLD

def init_demo_employee():
    face_app = create_face_app(allowed_modules=["detection", "recognition"])
    
    # Đọc ảnh demo_real
    real_img_path = BASE_DIR.parent / "demo_real.jpg"
    if not real_img_path.is_file():
        print(f"[WARN] Không tìm thấy {real_img_path}")
        return

    img = cv2.imread(str(real_img_path))
    faces = face_app.get(img)
    if not faces:
        print("[WARN] Không nhận diện được khuôn mặt trong ảnh demo")
        return

    emb = normalized_embedding(faces[0])
    if emb is None:
        print("[WARN] Không trích xuất được embedding")
        return

    # Tạo hồ sơ nhân viên mẫu: Nguyễn Văn A
    demo_name = "Nguyễn Văn A"
    embeddings = np.array([emb], dtype=np.float32)
    labels = np.array([demo_name], dtype=str)
    video_ids = np.array(["demo/sample_1"], dtype=str)

    prototype_labels, prototypes = build_prototypes(embeddings, labels, video_ids)

    emb_path = BASE_DIR / EMBEDDINGS_FILE
    proto_path = BASE_DIR / PROTOTYPES_FILE

    np.savez_compressed(
        str(emb_path),
        signature=np.asarray("demo_init"),
        embeddings=embeddings,
        labels=labels,
        video_ids=video_ids,
    )
    np.savez_compressed(
        str(proto_path),
        labels=prototype_labels,
        prototypes=prototypes,
        cosine_threshold=np.asarray(COSINE_THRESHOLD, dtype=np.float32),
    )
    print(f"[THÀNH CÔNG] Đã khởi tạo nhân viên mẫu: '{demo_name}'")
    print(f"  -> File lưu: {proto_path.name}")

if __name__ == "__main__":
    init_demo_employee()
