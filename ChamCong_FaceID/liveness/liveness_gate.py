"""LivenessGate: Module kiểm định chống giả mạo khuôn mặt chuẩn xác cho Chấm công FaceID.
Sử dụng ĐÚNG mô hình liveness_cam4k.tflite và thuật toán quyết định FastBinaryDecision
hoàn toàn đồng bộ 100% với file chạy chay_test_camera.bat (run.py).
"""
import os
import threading
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np

# Nạp interpreter tối ưu
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter


def _iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    r, d = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, r - x) * max(0, d - y)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class Track:
    def __init__(self, track_id: int, bbox: tuple):
        self.id = track_id
        self.bbox = bbox
        self.samples = deque(maxlen=60)


class FastBinaryDecision:
    """Làm mượt quyết định thời gian thực chuẩn như pipeline.py của Face_Anti_spoofing."""
    def __init__(self, window_seconds=0.45, max_gap=0.8):
        self.window_seconds = window_seconds
        self.max_gap = max_gap
        self.tracks = []
        self.next_id = 1

    def update(self, bbox, probs, now):
        old = self.tracks
        pairs = sorted(
            ((_iou(bbox, t.bbox), t) for t in old),
            key=lambda x: x[0],
            reverse=True,
        )
        assigned = None
        for overlap, t in pairs:
            if overlap >= 0.25:
                assigned = t
                break
        if assigned is None:
            assigned = Track(self.next_id, bbox)
            self.next_id += 1
            self.tracks.append(assigned)
        assigned.bbox = bbox

        # Dọn dẹp track cũ
        self.tracks = [t for t in self.tracks if now - (t.samples[-1][0] if t.samples else now) < 2.0]

        if assigned.samples and now - assigned.samples[-1][0] > self.max_gap:
            assigned.samples.clear()

        assigned.samples.append((now, np.asarray(probs, dtype=np.float32)))

        while len(assigned.samples) > 1 and now - assigned.samples[0][0] > self.window_seconds:
            assigned.samples.popleft()

        values = np.stack([v for _, v in assigned.samples])
        weights = np.linspace(0.5, 1.0, len(values), dtype=np.float32)[:, None]
        mean = (values * weights).sum(axis=0) / weights.sum()

        winner = int(mean.argmax())
        return winner, mean


class LivenessGate:
    def __init__(self, model_path=None, real_threshold=0.5):
        base_dir = Path(__file__).resolve().parent
        if model_path is None:
            model_path = base_dir / "liveness_cam4k.tflite"
            if not model_path.is_file():
                model_path = base_dir.parents[1] / "web_models" / "liveness_cam4k.tflite"

        self.model_path = str(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình liveness: {self.model_path}")

        self.real_threshold = real_threshold
        self.lock = threading.RLock()
        self.temporal = FastBinaryDecision(window_seconds=0.45, max_gap=0.8)

        # 1. Khởi tạo mô hình liveness_cam4k.tflite (Chuẩn SC-M200)
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

        # 2. Bộ tinh chỉnh YuNet (đồng bộ 100% tỷ lệ cắt khuôn mặt với run.py)
        yunet_path = base_dir / "face_detection_yunet_2023mar.onnx"
        self.yunet = None
        if yunet_path.is_file():
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), 0.5, 0.3, 5000)
            except Exception:
                self.yunet = None

        # Khởi động (Warm up)
        with self.lock:
            dummy = np.zeros((1, 32, 32, 3), dtype=np.float32)
            self.interpreter.set_tensor(self.input_index, dummy)
            self.interpreter.invoke()
            _ = self.interpreter.get_tensor(self.output_index)

    def _extract_aligned_crop(self, frame_bgr, bbox):
        """Trích xuất khuôn mặt chuẩn xác bằng YuNet hoặc bbox gốc."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
        bw, bh = x2 - x1, y2 - y1

        if bw < 10 or bh < 10:
            return None

        # Nếu có YuNet, tinh chỉnh vùng crop để khớp 100% với lúc train model
        if self.yunet is not None:
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            roi = frame_bgr[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                self.yunet.setInputSize((roi.shape[1], roi.shape[0]))
                _, y_faces = self.yunet.detect(roi)
                if y_faces is not None and len(y_faces) > 0:
                    fx, fy, fw, fh = y_faces[0][:4]
                    cx1, cy1 = max(0, int(rx1 + fx)), max(0, int(ry1 + fy))
                    cx2, cy2 = min(w, int(cx1 + fw)), min(h, int(cy1 + fh))
                    if cx2 > cx1 and cy2 > cy1:
                        return frame_bgr[cy1:cy2, cx1:cx2]

        return frame_bgr[y1:y2, x1:x2]

    def predict_crop(self, crop_bgr):
        """Dự đoán trực tiếp từ ảnh cắt khuôn mặt BGR (kích thước bất kỳ)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return 1.0, 0.0

        # Tiền xử lý khớp với pipeline huấn luyện: BGR, 32x32, chuẩn hóa [0, 1]
        face_32 = cv2.resize(crop_bgr, (32, 32)).astype(np.float32) / 255.0
        face_batch = np.expand_dims(face_32, axis=0)

        with self.lock:
            self.interpreter.set_tensor(self.input_index, face_batch)
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(self.output_index)[0]

        # Nhãn: Index 0 = fake, Index 1 = real
        p_fake = float(probs[0])
        p_real = float(probs[1])
        return p_fake, p_real

    def check_liveness(self, frame_bgr, bbox, all_bboxes=None, now=None):
        """Kiểm tra liveness cho khuôn mặt nằm trong bbox = (x1, y1, x2, y2)."""
        if now is None:
            now = time.monotonic()

        with self.lock:
            crop = self._extract_aligned_crop(frame_bgr, bbox)
            if crop is None or crop.size == 0:
                return {
                    "is_real": False,
                    "confidence": 1.0,
                    "fake_score": 1.0,
                    "real_score": 0.0,
                    "label": "fake",
                    "reason": "Khuôn mặt quá nhỏ hoặc nằm ngoài khung hình",
                }

            # 1. Suy luận qua mô hình liveness_cam4k
            p_fake, p_real = self.predict_crop(crop)

            # 2. Làm mượt chuỗi quyết định bằng FastBinaryDecision (y hệt run.py)
            status, mean = self.temporal.update(bbox, [p_fake, p_real], now)

            smooth_fake = float(mean[0])
            smooth_real = float(mean[1])

            # Đánh giá Real/Fake theo argmax và ngưỡng
            is_real = bool(status == 1 and smooth_real >= self.real_threshold)
            confidence = smooth_real if is_real else smooth_fake
            label = "real" if is_real else "fake"
            reason = "Người thật (Hợp lệ)" if is_real else "Phát hiện giả mạo (Ảnh / Màn hình)"

            return {
                "is_real": is_real,
                "confidence": round(confidence, 4),
                "fake_score": round(smooth_fake, 4),
                "real_score": round(smooth_real, 4),
                "label": label,
                "reason": reason,
            }
