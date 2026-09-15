"""Face detection, input-quality checks and short-lived prediction tracks."""
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
    intersection = max(0, r-x) * max(0, d-y)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class FaceDetector:
    def __init__(self, path=BASE/'face_detector/face_detection_yunet_2023mar.onnx', confidence=0.85):
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing YuNet model: {path}')
        self.net = cv2.FaceDetectorYN.create(str(path), '', (320, 320), confidence, 0.3, 5000)

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = min(1.0, 640/max(h, w))
        small = cv2.resize(frame, (round(w*scale), round(h*scale)))
        self.net.setInputSize((small.shape[1], small.shape[0]))
        _, rows = self.net.detect(small)
        faces = []
        if rows is None:
            return faces
        for row in rows:
            x, y, fw, fh = row[:4]/scale
            x1, y1, x2, y2 = max(0, int(x)), max(0, int(y)), min(w, int(x+fw)), min(h, int(y+fh))
            if x2 <= x1 or y2 <= y1:
                continue
            landmarks = row[4:14].reshape(5, 2)/scale
            faces.append({'box': (x1,y1,x2,y2), 'detector_score': float(row[14]),
                          'landmarks': landmarks, 'clipped': bool(x1 > x+2 or y1 > y+2 or x2 < x+fw-2 or y2 < y+fh-2)})
        return faces


def quality(frame, face):
    x,y,r,b = face['box']
    if min(r-x,b-y) < 65:
        return 'Move closer'
    if face['clipped']:
        return 'Center your full face'
    points = face['landmarks']
    if not np.all((points[:,0] >= x) & (points[:,0] <= r) & (points[:,1] >= y) & (points[:,1] <= b)):
        return 'Face camera directly'
    eyes = points[:2]
    mouth = points[3:]
    if np.linalg.norm(eyes[0]-eyes[1]) < (r-x)*0.12 or mouth[:,1].mean() <= eyes[:,1].mean():
        return 'Face camera directly'
    gray = cv2.cvtColor(frame[y:b,x:r],cv2.COLOR_BGR2GRAY)
    if gray.mean() < 35:
        return 'Need more light'
    if np.mean(gray > 245) > 0.45:
        return 'Reduce glare'
    if cv2.Laplacian(cv2.resize(gray,(128,128)), cv2.CV_32F).var() < 12:
        return 'Hold still / improve focus'
    return None


@dataclass
class Track:
    id: int
    box: tuple
    samples: deque = field(default_factory=lambda: deque(maxlen=180))


class TemporalDecision:
    """Geometric continuity only, not identity recognition. Drop missing tracks immediately."""
    def __init__(self, threshold=0.8, min_samples=6, min_seconds=0.4):
        self.threshold, self.min_samples, self.min_seconds = threshold, min_samples, min_seconds
        self.tracks = []
        self.next_id = 1

    def associate(self, boxes):
        old = self.tracks
        pairs = sorted(((iou(box, t.box), i, j) for i,box in enumerate(boxes) for j,t in enumerate(old)), reverse=True)
        assigned, used = {}, set()
        for overlap,i,j in pairs:
            if overlap < 0.4:
                break
            if i not in assigned and j not in used:
                assigned[i] = old[j]
                used.add(j)
        tracks = []
        for i,box in enumerate(boxes):
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
        if track.samples and now-track.samples[-1][0] > 0.5:
            track.samples.clear()
        track.samples.append((now,np.asarray(probs)))
        while track.samples and now-track.samples[0][0] > 1.5:
            track.samples.popleft()
        if len(track.samples) < self.min_samples or now-track.samples[0][0] < self.min_seconds:
            return 'checking', None
        values = np.stack([v for _,v in track.samples])
        mean = values.mean(axis=0)
        winner = int(mean.argmax())
        # Do not carry a previous result through an opposing current prediction.
        if mean[winner] < self.threshold or (values.argmax(axis=1)==winner).mean() < 0.8 or probs[winner] < self.threshold:
            return 'uncertain', mean
        return winner, mean


class Pipeline:
    def __init__(self, model_path=BASE/'liveness.h5', labels_path=BASE/'le.pickle', detector_path=BASE/'face_detector/face_detection_yunet_2023mar.onnx', confidence=0.85, threshold=0.8):
        import pickle
        from tensorflow.keras.models import load_model
        self.detector = FaceDetector(detector_path,confidence)
        self.model = load_model(model_path,compile=False)
        with open(labels_path,'rb') as f:
            self.labels = [str(x).lower() for x in pickle.load(f).classes_]
        if set(self.labels) != {'fake','real'} or tuple(self.model.input_shape[1:]) != (32,32,3) or self.model.output_shape[-1] != 2:
            raise ValueError('Expected 32x32 BGR model and fake/real label encoder')
        self.temporal = TemporalDecision(threshold)

    def process(self, frame, now):
        faces = self.detector.detect(frame)
        tracks = self.temporal.associate([f['box'] for f in faces])
        valid, batch = [], []
        for i,face in enumerate(faces):
            face['quality'] = quality(frame,face)
            if face['quality'] is None:
                x,y,r,b = face['box']
                valid.append(i)
                batch.append(preprocess(frame[y:b,x:r]))
        predictions = {}
        if batch:
            output = self.model(np.stack(batch),training=False).numpy()
            predictions = dict(zip(valid,output))
        for i,(face,track) in enumerate(zip(faces,tracks)):
            status, mean = self.temporal.update(track,predictions.get(i),now)
            face['track_id'] = track.id
            face['status'] = self.labels[status] if isinstance(status,int) else status
            face['scores'] = None if mean is None else {label:float(mean[j]) for j,label in enumerate(self.labels)}
        return faces
