"""Face detection, input-quality checks and short-lived prediction tracks."""
import os
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
warnings.filterwarnings('ignore')

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent


def preprocess(face):
    # Match the existing training pipeline: BGR, 32x32, [0,1].
    return cv2.resize(face, (32, 32)).astype(np.float32) / 255.0


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    r, d = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, r - x) * max(0, d - y)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class FaceDetector:
    def __init__(self, path=BASE / 'face_detector/face_detection_yunet_2023mar.onnx', confidence=0.62):
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing YuNet model: {path}')
        self.net = cv2.FaceDetectorYN.create(str(path), '', (320, 320), confidence, 0.3, 5000)

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = min(1.0, 480 / max(h, w))
        small = cv2.resize(frame, (round(w * scale), round(h * scale)))
        self.net.setInputSize((small.shape[1], small.shape[0]))
        _, rows = self.net.detect(small)
        faces = []
        if rows is None:
            return faces
        for row in rows:
            x, y, fw, fh = row[:4] / scale
            x1, y1, x2, y2 = max(0, int(x)), max(0, int(y)), min(w, int(x + fw)), min(h, int(y + fh))
            if x2 <= x1 or y2 <= y1:
                continue
            landmarks = row[4:14].reshape(5, 2) / scale
            faces.append({
                'box': (x1, y1, x2, y2),
                'detector_score': float(row[14]),
                'landmarks': landmarks,
                'clipped': bool(x1 > x + 2 or y1 > y + 2 or x2 < x + fw - 2 or y2 < y + fh - 2),
            })
        return faces


def quality(frame, face):
    x, y, r, b = face['box']
    if min(r - x, b - y) < 65:
        return 'Move closer'
    if face['clipped']:
        return 'Center your full face'
    points = face['landmarks']
    if not np.all((points[:, 0] >= x) & (points[:, 0] <= r) & (points[:, 1] >= y) & (points[:, 1] <= b)):
        return 'Face camera directly'
    eyes = points[:2]
    mouth = points[3:]
    if np.linalg.norm(eyes[0] - eyes[1]) < (r - x) * 0.12 or mouth[:, 1].mean() <= eyes[:, 1].mean():
        return 'Face camera directly'
    gray = cv2.cvtColor(frame[y:b, x:r], cv2.COLOR_BGR2GRAY)
    if gray.mean() < 35:
        return 'Need more light'
    if np.mean(gray > 245) > 0.45:
        return 'Reduce glare'
    if cv2.Laplacian(cv2.resize(gray, (128, 128)), cv2.CV_32F).var() < 12:
        return 'Hold still / improve focus'
    return None


def _compute_optical_metrics(crop, frame_shape):
    h, w = crop.shape[:2]
    mx, my = max(1, int(w * 0.12)), max(1, int(h * 0.12))
    core = crop[my:max(my + 2, h - my), mx:max(mx + 2, w - mx)]
    if core.size == 0:
        core = crop
    g64 = cv2.resize(cv2.cvtColor(core, cv2.COLOR_BGR2GRAY), (64, 64))
    lap = float(cv2.Laplacian(g64, cv2.CV_32F).var())
    ycrcb = cv2.cvtColor(core, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    y_ch = ycrcb[:, :, 0].astype(np.float32)
    y_mean = float(y_ch.mean())
    y_std = float(y_ch.std())
    y_cv = y_std / max(y_mean, 1.0)
    p10_y = float(np.percentile(y_ch, 10))
    cr_std = float(ycrcb[:, :, 1].std())
    s_mean = float(hsv[:, :, 1].mean())
    area_ratio = float(w * h) / max(float(frame_shape[0] * frame_shape[1]), 1.0)
    return {
        'lap': lap,
        'y_std': y_std,
        'y_cv': y_cv,
        'p10_y': p10_y,
        'cr_std': cr_std,
        's_mean': s_mean,
        'area_ratio': area_ratio,
    }


@dataclass
class Track:
    id: int
    box: tuple
    samples: deque = field(default_factory=lambda: deque(maxlen=180))


class TemporalDecision:
    """Geometric continuity only, not identity recognition. Drop missing tracks immediately."""
    def __init__(self, threshold=0.8, min_samples=6, min_seconds=0.4, max_gap=0.5, window_seconds=1.5):
        self.threshold, self.min_samples, self.min_seconds = threshold, min_samples, min_seconds
        self.max_gap, self.window_seconds = max_gap, window_seconds
        self.tracks = []
        self.next_id = 1

    def associate(self, boxes):
        old = self.tracks
        pairs = sorted(((iou(box, t.box), i, j) for i, box in enumerate(boxes) for j, t in enumerate(old)), reverse=True)
        assigned, used = {}, set()
        for overlap, i, j in pairs:
            if overlap < 0.3:
                break
            if i not in assigned and j not in used:
                assigned[i] = old[j]
                used.add(j)
        tracks = []
        for i, box in enumerate(boxes):
            track = assigned.get(i)
            if track is None:
                track = Track(self.next_id, box)
                self.next_id += 1
            track.box = box
            tracks.append(track)
        self.tracks = tracks
        return tracks

    def update(self, track, probs, now):
        if probs is None:
            track.samples.clear()
            return 'adjust', None
        if track.samples and now - track.samples[-1][0] > self.max_gap:
            track.samples.clear()
        track.samples.append((now, np.asarray(probs)))
        while track.samples and now - track.samples[0][0] > self.window_seconds:
            track.samples.popleft()
        if len(track.samples) < self.min_samples or now - track.samples[0][0] < self.min_seconds:
            return 'checking', None
        values = np.stack([v for _, v in track.samples])
        mean = values.mean(axis=0)
        winner = int(mean.argmax())
        if mean[winner] < self.threshold or (values.argmax(axis=1) == winner).mean() < 0.8 or probs[winner] < self.threshold:
            return 'uncertain', mean
        return winner, mean


class FastBinaryDecision:
    """Instant real/fake decision without yellow states (checking/uncertain/adjust)."""
    def __init__(self, window_seconds=0.45, max_gap=0.6, real_threshold=None):
        self.window_seconds = window_seconds
        self.max_gap = max_gap
        # None = plain argmax (original behavior). A float makes REAL stricter.
        self.real_threshold = real_threshold
        self.tracks = []
        self.next_id = 1

    def associate(self, boxes):
        old = self.tracks
        pairs = sorted(((iou(box, t.box), i, j) for i, box in enumerate(boxes) for j, t in enumerate(old)), reverse=True)
        assigned, used = {}, set()
        for overlap, i, j in pairs:
            if overlap < 0.25:
                break
            if i not in assigned and j not in used:
                assigned[i] = old[j]
                used.add(j)
        tracks = []
        for i, box in enumerate(boxes):
            track = assigned.get(i)
            if track is None:
                track = Track(self.next_id, box)
                self.next_id += 1
            track.box = box
            tracks.append(track)
        self.tracks = tracks
        return tracks

    def update(self, track, probs, now):
        if probs is None:
            probs = [0.5, 0.5]
        if track.samples and now - track.samples[-1][0] > self.max_gap:
            track.samples.clear()
        track.samples.append((now, np.asarray(probs, dtype=np.float32)))
        while len(track.samples) > 1 and now - track.samples[0][0] > self.window_seconds:
            track.samples.popleft()
        values = np.stack([v for _, v in track.samples])
        weights = np.linspace(0.5, 1.0, len(values), dtype=np.float32)[:, None]
        mean = (values * weights).sum(axis=0) / weights.sum()
        if self.real_threshold is None:
            return int(mean.argmax()), mean
        # Stricter REAL: require high confidence (>= real_threshold)
        if float(mean[1]) >= self.real_threshold:
            return 1, mean
        if float(mean[0]) < 0.85:
            fake_conf = min(0.99, max(float(mean[0]), 0.85 + 0.20 * (self.real_threshold - float(mean[1]))))
            mean = np.array([fake_conf, 1.0 - fake_conf], dtype=np.float32)
        return 0, mean


class Pipeline:
    def __init__(self, model_path=BASE / 'liveness_cam4k.h5', labels_path=BASE / 'le_cam4k.pickle',
                 detector_path=BASE / 'face_detector/face_detection_yunet_2023mar.onnx',
                 confidence=0.62, threshold=0.8, binary_only=True, fusion=False):
        import json
        self.detector = FaceDetector(detector_path, confidence)
        self.binary_only = binary_only
        # fusion=False: single full-face crop + plain argmax (the version validated at epoch 14/18).
        # fusion=True : experimental 3-crop fusion + optical print-attack rules (not default).
        self.fusion = fusion

        model_path_obj = Path(model_path)
        tflite_candidate = BASE / 'web_models' / f'{model_path_obj.stem}.tflite'
        if str(model_path).endswith('.tflite'):
            from inference_runtime import LiteModel
            self.model = LiteModel(model_path)
            self.labels = json.loads(Path(labels_path).read_text(encoding='utf-8'))
        elif tflite_candidate.is_file():
            import pickle
            from inference_runtime import LiteModel
            self.model = LiteModel(tflite_candidate)
            with open(labels_path, 'rb') as f:
                self.labels = [str(x).lower() for x in pickle.load(f).classes_]
        else:
            import pickle
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path, compile=False)
            with open(labels_path, 'rb') as f:
                self.labels = [str(x).lower() for x in pickle.load(f).classes_]
            if tuple(self.model.input_shape[1:]) != (32, 32, 3) or self.model.output_shape[-1] != 2:
                raise ValueError('Expected 32x32 BGR model and fake/real label encoder')

        if set(self.labels) != {'fake', 'real'}:
            raise ValueError('Expected 32x32 BGR model and fake/real label encoder')
        self.temporal = FastBinaryDecision() if binary_only else TemporalDecision(threshold)

    def process(self, frame, now, temporal=None):
        temporal = self.temporal if temporal is None else temporal
        faces = self.detector.detect(frame)
        tracks = temporal.associate([f['box'] for f in faces])
        if not faces:
            return faces

        if not self.binary_only or not self.fusion:
            valid, batch = [], []
            for i, face in enumerate(faces):
                if self.binary_only:
                    face['quality'] = None
                else:
                    face['quality'] = quality(frame, face)
                if face['quality'] is None:
                    x, y, r, b = face['box']
                    valid.append(i)
                    batch.append(preprocess(frame[y:b, x:r]))
            predictions = {}
            if batch:
                output = self.model(np.stack(batch), training=False)
                if hasattr(output, 'numpy'):
                    output = output.numpy()
                predictions = dict(zip(valid, output))
            for i, (face, track) in enumerate(zip(faces, tracks)):
                status, mean = temporal.update(track, predictions.get(i), now)
                face['track_id'] = track.id
                face['status'] = self.labels[status] if isinstance(status, int) else status
                face['scores'] = None if mean is None else {label: float(mean[j]) for j, label in enumerate(self.labels)}
            return faces

        # Binary-only mode with Multi-Region Crop Fusion + Optical Print-Attack Shield
        batch = []
        metrics_list = []
        for face in faces:
            face['quality'] = None
            x, y, r, b = face['box']
            bw, bh = r - x, b - y
            c_full = frame[y:b, x:r]
            ix1, ix2 = x + int(bw * 0.10), r - int(bw * 0.10)
            iy1, iy2 = y + int(bh * 0.17), b - int(bh * 0.08)
            c_inner = frame[iy1:iy2, ix1:ix2] if (ix2 - ix1 > 12 and iy2 - iy1 > 12) else c_full
            ux1, ux2 = x + int(bw * 0.08), r - int(bw * 0.08)
            uy1, uy2 = y + int(bh * 0.05), y + int(bh * 0.52)
            c_upper = frame[uy1:uy2, ux1:ux2] if (ux2 - ux1 > 12 and uy2 - uy1 > 12) else c_full

            batch.extend([preprocess(c_full), preprocess(c_inner), preprocess(c_upper)])
            metrics_list.append(_compute_optical_metrics(c_full, frame.shape))

        output = self.model(np.stack(batch), training=False)
        if hasattr(output, 'numpy'):
            output = output.numpy()

        max_area = max(m['area_ratio'] for m in metrics_list)
        max_lap = max(m['lap'] for m in metrics_list)

        for i, (face, track, m) in enumerate(zip(faces, tracks, metrics_list)):
            p_full = output[3 * i]
            p_inner = output[3 * i + 1]
            p_upper = output[3 * i + 2]

            # 1. Optical Print-Attack & Multi-Face Photo Card Shield (Priority #1: Never predict FAKE as REAL)
            is_flat_print = (
                (m['y_cv'] < 0.255 and m['p10_y'] > 60.0 and m['s_mean'] < 82.0 and m['lap'] < 135.0)
                or (m['p10_y'] > 66.0 and m['y_std'] < 24.5 and m['s_mean'] < 78.0)
            )
            is_secondary_card = (
                len(faces) > 1
                and m['area_ratio'] < 0.65 * max_area
                and (
                    p_full[0] > 0.30
                    or p_upper[0] > 0.30
                    or m['y_cv'] < 0.275
                    or m['p10_y'] > 56.0
                    or m['s_mean'] < 84.0
                    or m['lap'] < 0.60 * max_lap
                )
            )

            # Optical rules only veto when the neural model is not already highly confident it is a live face
            model_doubts = float(p_full[1]) < 0.90
            if (is_flat_print or is_secondary_card) and model_doubts:
                real_p = min(float(p_full[1]), float(p_upper[1]), 0.04)
            # 2. Partial Phone/Object Occlusion on Real 3D Face (Fixes phone on forehead / mouth)
            elif m['area_ratio'] >= 0.045 and m['p10_y'] < 52.0 and m['y_cv'] > 0.28 and m['cr_std'] < 12.2:
                if p_inner[1] > 0.85 and p_full[1] > 0.35:
                    real_p = 0.25 * float(p_full[1]) + 0.75 * float(p_inner[1])
                elif p_upper[1] > 0.94 and m['y_cv'] > 0.36 and m['p10_y'] < 38.0 and m['area_ratio'] > 0.065:
                    real_p = 0.85 * float(p_upper[1]) + 0.15 * max(float(p_full[1]), float(p_inner[1]))
                else:
                    real_p = 0.60 * float(p_full[1]) + 0.40 * float(p_inner[1])
            # 3. Default conservative anti-spoof fusion
            else:
                real_p = 0.55 * float(p_full[1]) + 0.25 * float(p_inner[1]) + 0.20 * float(p_upper[1])

            real_p = float(np.clip(real_p, 0.001, 0.999))
            combined_probs = np.array([1.0 - real_p, real_p], dtype=np.float32)

            status, mean = temporal.update(track, combined_probs, now)
            face['track_id'] = track.id
            face['status'] = self.labels[status] if isinstance(status, int) else status
            face['scores'] = None if mean is None else {label: float(mean[j]) for j, label in enumerate(self.labels)}
        return faces
