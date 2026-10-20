"""Run the camera or a reproducible headless video check."""
import os
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
warnings.filterwarnings('ignore')

import argparse
import json
from pathlib import Path
import threading
import time
import cv2
from pipeline import BASE, Pipeline

if hasattr(cv2, 'setLogLevel'):
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass

CAMERA_LABELS = {
    0: 'Cam 0 (USB SC-M200)',
    1: 'Cam 1 (Laptop Webcam)',
    2: 'Cam 2 (Auxiliary)',
    3: 'Cam 3 (Auxiliary)',
}


def _is_valid_camera_frame(frame):
    if frame is None or frame.size == 0:
        return False
    # Reject virtual camera placeholder screens (e.g. DroidCam solid sky-blue screen with std == 0)
    if frame.std(axis=(0, 1)).max() < 2.0:
        return False
    return True


def _open_raw_camera(index=None, quiet=False):
    # Prioritize requested camera index, then scan fallback indices (skipping virtual cameras like DroidCam)
    if index is None or index == 0:
        candidates = [0, 1, 2, 3]
    elif index == 1:
        candidates = [1, 2, 3, 0]
    else:
        candidates = [index] + [i for i in (0, 1, 2, 3) if i != index]
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if os.name == 'nt' else [cv2.CAP_ANY]
    for idx in candidates:
        for backend in backends:
            for use_720p in (True, False):
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                if use_720p:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                good = 0
                last_frame = None
                for _ in range(20):
                    ok, frame = cap.read()
                    if ok and _is_valid_camera_frame(frame):
                        good += 1
                        last_frame = frame
                        if good >= 2:
                            h, w = last_frame.shape[:2]
                            logical_idx = 1 if (index == 1 and idx != 0) else idx
                            if not quiet:
                                cam_name = CAMERA_LABELS.get(logical_idx, f'Camera {idx}')
                                if index == 1 and logical_idx != 1:
                                    print('[INFO] Khong tim thay camera laptop that (Cam 1 chi la camera ao DroidCam man hinh xanh). Dung camera USB SC-M200.', flush=True)
                                print(f'[INFO] Active camera: {cam_name} (device #{idx}, {w}x{h}, {cap.getBackendName()})', flush=True)
                            return cap, logical_idx, idx, last_frame
                cap.release()
    raise RuntimeError('No usable camera. Check USB connection or select --camera INDEX.')


class LiveCameraRuntime:
    """Decoupled capture + AI inference threads for zero-stutter real-time display and auto USB recovery."""
    def __init__(self, pipe, cam_index=None):
        self.pipe = pipe
        self.requested_idx = cam_index
        self.cap, self.active_idx, self.hw_idx, first_frame = _open_raw_camera(cam_index)
        self.lock = threading.Lock()
        self.pipe_lock = threading.Lock()
        self.latest_frame = self._prepare_frame(first_frame)
        self.frame_id = 1
        self.ai_frame_id = 0
        self.latest_faces = []
        self.stopped = False

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.ai_thread = threading.Thread(target=self._ai_loop, daemon=True)
        self.capture_thread.start()
        self.ai_thread.start()

    @staticmethod
    def _prepare_frame(frame):
        frame = cv2.flip(frame, 1)
        if frame.shape[1] > 960:
            frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
        return frame

    def _capture_loop(self):
        fail_count = 0
        while not self.stopped:
            ok, raw = self.cap.read()
            if ok and raw is not None and raw.size > 0:
                fail_count = 0
                prepared = self._prepare_frame(raw)
                with self.lock:
                    self.latest_frame = prepared
                    self.frame_id += 1
            else:
                fail_count += 1
                # Tolerate brief USB cable movement glitches without disconnecting;
                # only re-open if hardware stream genuinely drops for >1.2 seconds.
                if fail_count >= 50:
                    try:
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        time.sleep(0.25)
                        new_cap, new_idx, new_hw_idx, first_frame = _open_raw_camera(self.active_idx, quiet=True)
                        prepared = self._prepare_frame(first_frame)
                        with self.lock:
                            self.cap = new_cap
                            self.active_idx = new_idx
                            self.hw_idx = new_hw_idx
                            self.latest_frame = prepared
                            self.frame_id += 1
                        fail_count = 0
                    except Exception:
                        time.sleep(0.25)
                else:
                    time.sleep(0.015)

    def _ai_loop(self):
        last_processed = 0
        while not self.stopped:
            with self.lock:
                fid = self.frame_id
                frame = self.latest_frame
            if frame is None or fid == last_processed:
                time.sleep(0.003)
                continue
            last_processed = fid
            with self.pipe_lock:
                faces = self.pipe.process(frame, time.monotonic())
            with self.lock:
                self.latest_faces = faces
                self.ai_frame_id = fid

    def get_state(self):
        with self.lock:
            return self.latest_frame, list(self.latest_faces), self.frame_id, self.active_idx

    def switch_camera(self, target_idx):
        if target_idx == self.active_idx:
            return
        print(f'[INFO] Switching to camera {target_idx}...', flush=True)
        try:
            new_cap, new_idx, new_hw_idx, first_frame = _open_raw_camera(target_idx)
            prepared = self._prepare_frame(first_frame)
            with self.pipe_lock:
                self.pipe.temporal.tracks.clear()
            with self.lock:
                old_cap = self.cap
                self.cap = new_cap
                self.active_idx = new_idx
                self.hw_idx = new_hw_idx
                self.latest_frame = prepared
                self.latest_faces = []
                self.frame_id += 1
            try:
                old_cap.release()
            except Exception:
                pass
        except Exception as exc:
            print(f'[INFO] Could not switch to camera {target_idx}: {exc}', flush=True)

    def release(self):
        self.stopped = True
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=0.5)
        if self.ai_thread.is_alive():
            self.ai_thread.join(timeout=0.5)
        try:
            self.cap.release()
        except Exception:
            pass


def choose_camera_interactive():
    print('\n======================================================')
    print('   CHON CAMERA DE TEST FACE ANTI-SPOOFING (CAM4K)')
    print('======================================================')
    print('  [0] Camera USB SC-M200 (Mac dinh - khuyen dung)')
    print('  [1] Camera lien may (Laptop Built-in Webcam)')
    print('======================================================')
    choice = input('Nhap so camera (0 hoac 1, nhan Enter de chon 0): ').strip()
    if choice in ('1', '2', '3'):
        return int(choice)
    return 0


def render(frame, faces, fps, cam_idx=None):
    canvas = frame.copy()
    for face in faces:
        x, y, r, b = face['box']
        is_real = (face.get('status') == 'real')
        # Strict binary colors: Green for REAL, Red for FAKE (never yellow)
        color = (60, 210, 70) if is_real else (50, 50, 240)
        status_key = 'real' if is_real else 'fake'
        tag = 'REAL' if is_real else 'FAKE'
        score = face['scores'].get(status_key, 0.0) if face.get('scores') else 0.0
        label = f'{tag} | {score * 100:.1f}%'
        cv2.rectangle(canvas, (x, y), (r, b), color, 2)
        tx, ty = max(0, min(x, canvas.shape[1] - min(350, canvas.shape[1]))), max(24, y - 8)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .6, 2)
        cv2.rectangle(canvas, (tx, ty - th - 6), (min(canvas.shape[1], tx + tw + 8), ty + 4), (20, 20, 20), -1)
        cv2.putText(canvas, label, (tx + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2, cv2.LINE_AA)
    # Separate status bars: never paint over the image sent to either model.
    canvas = cv2.copyMakeBorder(canvas, 55, 32, 0, 0, cv2.BORDER_CONSTANT, value=(24, 27, 32))
    cam_tag = f" | {CAMERA_LABELS.get(cam_idx, f'Cam {cam_idx}')}" if cam_idx is not None else ''
    cv2.putText(canvas, f'FACE ANTI-SPOOFING (CAM4K)    {fps:.1f} FPS{cam_tag}', (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'No face detected' if not faces else 'Green: REAL | Red: FAKE', (12, 44), cv2.FONT_HERSHEY_SIMPLEX, .45, (180, 200, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Phim [0]: USB SC-M200 | Phim [1]: Cam Laptop | Phim [C]: Doi Cam | [Q / ESC]: Thoat', (10, canvas.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, .43, (190, 215, 190), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', help='Path to input video file (omit for live camera)')
    ap.add_argument('--camera', type=int, default=None, help='Camera index: 0 for USB SC-M200 (default), 1 for Laptop Webcam')
    ap.add_argument('--choose-camera', action='store_true', help='Show interactive menu to choose camera before starting')
    ap.add_argument('-m', '--model', default=str(BASE / 'liveness_cam4k.h5'))
    ap.add_argument('-l', '--le', default=str(BASE / 'le_cam4k.pickle'))
    ap.add_argument('-d', '--detector', default=str(BASE / 'face_detector/face_detection_yunet_2023mar.onnx'))
    ap.add_argument('-c', '--confidence', type=float, default=.62)
    ap.add_argument('--threshold', type=float, default=.8)
    ap.add_argument('--snapshot')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--report')
    args = ap.parse_args()
    if not 0 < args.confidence < 1 or not .5 < args.threshold <= 1 or args.max_frames < 0:
        ap.error('Invalid confidence, threshold or max-frames')

    cam_index = args.camera
    if args.choose_camera and not args.input and not args.headless:
        cam_index = choose_camera_interactive()

    print('[INFO] Loading Face Anti-Spoofing models (cam4k)...', flush=True)
    pipe = Pipeline(args.model, args.le, args.detector, args.confidence, args.threshold, binary_only=True)
    cap = None
    live_runtime = None
    window = 'Face Anti-Spoofing | [0]: USB SC-M200 | [1]: Laptop | [Q/ESC]: Quit'
    records = []
    try:
        if args.input:
            cap = cv2.VideoCapture(args.input)
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open {args.input}')
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            source_fps = source_fps if 0 < source_fps < 240 else 30
        else:
            live_runtime = LiveCameraRuntime(pipe, cam_index)
            source_fps = 30

        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 960, 620)
            if not args.input:
                print('[INFO] Cua so camera da mo! Nhan phim [0] (USB SC-M200), [1] (Cam Laptop), [C] (Doi Cam), hoac [Q] de thoat.', flush=True)

        count = 0
        fps = 30.0
        last_fid = -1
        last_tick = time.perf_counter()

        while True:
            if args.input:
                start = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame.shape[1] > 960:
                    frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
                now = count / source_fps
                faces = pipe.process(frame, now)
                elapsed = time.perf_counter() - start
                fps = .9 * fps + .1 / max(elapsed, 1e-6) if count else 1 / max(elapsed, 1e-6)
                active_cam_idx = None
            else:
                frame, faces, fid, active_cam_idx = live_runtime.get_state()
                if frame is None:
                    time.sleep(0.005)
                    continue
                if fid == last_fid:
                    # Wait briefly while still polling keyboard events
                    if not args.headless:
                        key = cv2.waitKey(2) & 255
                        if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                            break
                        if key in (ord('0'), ord('1'), ord('c'), ord('C')):
                            target_idx = (1 - (active_cam_idx or 0)) if key in (ord('c'), ord('C')) else (0 if key == ord('0') else 1)
                            live_runtime.switch_camera(target_idx)
                    else:
                        time.sleep(0.003)
                    continue
                last_fid = fid
                now_perf = time.perf_counter()
                elapsed = now_perf - last_tick
                last_tick = now_perf
                fps = .88 * fps + .12 / max(elapsed, 1e-6) if count else 30.0

            canvas = render(frame, faces, fps, active_cam_idx)
            count += 1
            if args.report:
                records.append({'frame': count, 'faces': [{k: v for k, v in face.items() if k != 'landmarks'} for face in faces]})
            if args.snapshot and count == 31:
                Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(args.snapshot, canvas):
                    raise IOError('Failed to save snapshot')
            if count == 1 or count % 100 == 0:
                print(f'[INFO] frames={count} faces={len(faces)} statuses={[f["status"] for f in faces]} fps={fps:.1f}', flush=True)
            if not args.headless:
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 255
                if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if not args.input and key in (ord('0'), ord('1'), ord('c'), ord('C')):
                    target_idx = (1 - (active_cam_idx or 0)) if key in (ord('c'), ord('C')) else (0 if key == ord('0') else 1)
                    live_runtime.switch_camera(target_idx)
            if args.max_frames and count >= args.max_frames:
                break
        if count == 0:
            raise RuntimeError('Input contained no readable frames')
    finally:
        if live_runtime is not None:
            live_runtime.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
