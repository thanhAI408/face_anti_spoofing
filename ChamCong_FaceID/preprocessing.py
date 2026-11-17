import json
import os

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "video_data")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "face_embeddings.npz")
PROTOTYPES_FILE = os.path.join(BASE_DIR, "face_prototypes.npz")
MAX_SAMPLES_PER_VIDEO = 120
MIN_DET_SCORE = 0.7
MIN_FACE_SIDE = 80
MIN_BLUR_VARIANCE = 40.0
MAX_POSE_ANGLE = 35.0
# Hiệu chỉnh bằng calibrate_threshold.py: người lạ hết bị nhận nhầm từ 0.50; cộng 0.05
# dự phòng vì frame realtime không qua bộ lọc chất lượng như frame dùng để hiệu chỉnh.
COSINE_THRESHOLD = 0.55
RANDOM_SEED = 42
CACHE_VERSION = 2
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def infer_person_name(video_name):
    stem = os.path.splitext(video_name)[0]
    stem = stem.replace("-", "_")
    tokens = [token for token in stem.split("_") if token.strip()]
    if not tokens:
        return "unknown_person"

    cleaned = []
    for token in tokens:
        if token.lower() in {"real", "cam", "video", "clip", "face"}:
            continue
        if token.isdigit() and len(token) in {2, 3, 4, 6, 8, 12}:
            continue
        cleaned.append(token)

    # Giữ toàn bộ họ tên (vd. nguyen_van_a) để hai người cùng họ không bị gộp nhãn.
    if cleaned:
        return "_".join(cleaned)
    return "_".join(tokens)


def organize_root_videos():
    root_level_videos = []
    for filename in sorted(os.listdir(VIDEO_DIR)):
        if os.path.isdir(os.path.join(VIDEO_DIR, filename)):
            continue
        if filename.lower().endswith(VIDEO_EXTENSIONS):
            root_level_videos.append(filename)

    if not root_level_videos:
        return

    print(f"[INFO] Phát hiện {len(root_level_videos)} video ở thư mục gốc; đang chuẩn hóa về cấu trúc video_data/<tên người>/.")
    for filename in root_level_videos:
        person_name = infer_person_name(filename)
        target_dir = os.path.join(VIDEO_DIR, person_name)
        os.makedirs(target_dir, exist_ok=True)
        src = os.path.join(VIDEO_DIR, filename)
        dst = os.path.join(target_dir, filename)

        if os.path.abspath(src) == os.path.abspath(dst):
            continue

        if os.path.exists(dst):
            base, ext = os.path.splitext(filename)
            index = 1
            while os.path.exists(os.path.join(target_dir, f"{base}_{index}{ext}")):
                index += 1
            dst = os.path.join(target_dir, f"{base}_{index}{ext}")

        os.replace(src, dst)
        print(f"[INFO] Đã chuyển video: {filename} -> {os.path.relpath(dst, VIDEO_DIR)}")


def discover_videos():
    organize_root_videos()
    videos = []
    for root, _, filenames in os.walk(VIDEO_DIR):
        relative_dir = os.path.relpath(root, VIDEO_DIR)
        for filename in filenames:
            if not filename.lower().endswith(VIDEO_EXTENSIONS):
                continue
            video_path = os.path.join(root, filename)
            if relative_dir == ".":
                continue

            person_name = relative_dir.split(os.sep)[0]
            relative_video = os.path.relpath(video_path, VIDEO_DIR)
            videos.append((video_path, person_name, relative_video))

    if not videos:
        raise RuntimeError(
            f"Không tìm thấy video hợp lệ trong {VIDEO_DIR}/<tên người>/. "
            "Hãy đặt video vào thư mục người tương ứng."
        )
    return sorted(videos, key=lambda item: item[2])


def cache_signature(videos):
    file_signatures = []
    for video_path, _, relative_video in videos:
        stat = os.stat(video_path)
        file_signatures.append((relative_video, stat.st_size, stat.st_mtime_ns))

    settings = {
        "version": CACHE_VERSION,
        "max_samples_per_video": MAX_SAMPLES_PER_VIDEO,
        "min_det_score": MIN_DET_SCORE,
        "min_face_side": MIN_FACE_SIDE,
        "min_blur_variance": MIN_BLUR_VARIANCE,
        "max_pose_angle": MAX_POSE_ANGLE,
    }
    return json.dumps({"files": file_signatures, "settings": settings}, ensure_ascii=False)


def load_embedding_cache(signature):
    if not os.path.exists(EMBEDDINGS_FILE):
        return None

    try:
        with np.load(EMBEDDINGS_FILE, allow_pickle=False) as cache:
            if str(cache["signature"].item()) != signature:
                print("[INFO] Cache embedding đã cũ; sẽ trích xuất lại video.")
                return None
            embeddings = cache["embeddings"].astype(np.float32)
            labels = cache["labels"].astype(str)
            video_ids = cache["video_ids"].astype(str)
    except (OSError, KeyError, ValueError) as exc:
        print(f"[WARN] Không đọc được cache embedding: {exc}")
        return None

    if len(embeddings) != len(labels) or len(labels) != len(video_ids):
        print("[WARN] Cache embedding không nhất quán; sẽ trích xuất lại video.")
        return None
    print(f"[INFO] Đã nạp {len(embeddings)} embeddings từ cache.")
    return embeddings, labels, video_ids


def face_passes_quality(face, frame):
    if face.det_score < MIN_DET_SCORE:
        return False, "low_score"

    x1, y1, x2, y2 = np.rint(face.bbox).astype(int)
    height, width = frame.shape[:2]
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if min(x2 - x1, y2 - y1) < MIN_FACE_SIDE:
        return False, "small"

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return False, "small"
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray_face, cv2.CV_64F).var() < MIN_BLUR_VARIANCE:
        return False, "blurry"

    pose = getattr(face, "pose", None)
    if pose is not None:
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose.size >= 2 and np.all(np.isfinite(pose[:2])):
            if np.max(np.abs(pose[:2])) > MAX_POSE_ANGLE:
                return False, "pose"
    return True, "ok"


def extract_embeddings(videos):
    # Import trễ để relabel.py dùng lại được các hàm ở đây mà không cần cài InsightFace.
    from face_runtime import create_face_app, normalized_embedding

    app = create_face_app()
    embeddings = []
    labels = []
    video_ids = []

    for video_path, person_name, relative_video in videos:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Không mở được video: {video_path}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            print(f"[ERROR] Không đọc được số frame của video: {video_path}")
            continue

        sample_indices = np.linspace(
            0,
            frame_count - 1,
            num=min(MAX_SAMPLES_PER_VIDEO, frame_count),
            dtype=int,
        )
        sample_set = set(sample_indices.tolist())
        stats = {"no_face": 0, "multiple_faces": 0, "low_score": 0, "small": 0, "blurry": 0, "pose": 0}
        accepted = 0
        print(f"[INFO] Đang xử lý video: {relative_video} (Label: {person_name})")

        for frame_idx in range(frame_count):
            if not cap.grab():
                break
            if frame_idx not in sample_set:
                continue

            success, frame = cap.retrieve()
            if not success:
                continue
            faces = app.get(frame)
            if not faces:
                stats["no_face"] += 1
                continue
            if len(faces) != 1:
                stats["multiple_faces"] += 1
                continue

            face = faces[0]
            passes_quality, reason = face_passes_quality(face, frame)
            if not passes_quality:
                stats[reason] += 1
                continue

            embedding = normalized_embedding(face)
            if embedding is None:
                continue
            embeddings.append(embedding)
            labels.append(person_name)
            video_ids.append(relative_video)
            accepted += 1

        cap.release()
        print(f" -> Embeddings: {accepted}; thống kê loại: {stats}")

    if not embeddings:
        raise RuntimeError("Không trích xuất được embedding đạt chất lượng từ các video.")
    return (
        np.asarray(embeddings, dtype=np.float32),
        np.asarray(labels, dtype=str),
        np.asarray(video_ids, dtype=str),
    )


def save_embedding_cache(signature, embeddings, labels, video_ids):
    np.savez_compressed(
        EMBEDDINGS_FILE,
        signature=np.asarray(signature),
        embeddings=embeddings,
        labels=labels,
        video_ids=video_ids,
    )
    print(f"[INFO] Đã lưu embedding cache: {EMBEDDINGS_FILE}")


def build_prototypes(embeddings, labels, video_ids):
    video_means = {}
    for label in np.unique(labels):
        label_mask = labels == label
        for video_id in np.unique(video_ids[label_mask]):
            samples = embeddings[label_mask & (video_ids == video_id)]
            mean_embedding = samples.mean(axis=0)
            video_means.setdefault(label, []).append(mean_embedding / np.linalg.norm(mean_embedding))

    prototype_labels = sorted(video_means)
    prototypes = []
    for label in prototype_labels:
        mean_embedding = np.mean(video_means[label], axis=0)
        prototypes.append(mean_embedding / np.linalg.norm(mean_embedding))
    return np.asarray(prototype_labels, dtype=str), np.asarray(prototypes, dtype=np.float32)


def evaluate_leave_one_video_out(embeddings, labels, video_ids):
    videos_by_label = {
        label: np.unique(video_ids[labels == label])
        for label in np.unique(labels)
    }
    for label, videos in sorted(videos_by_label.items()):
        if len(videos) < 2:
            print(f"[WARN] {label}: chỉ có một video, bỏ qua khi đánh giá.")

    results = []
    for video_id in sorted(np.unique(video_ids)):
        holdout_mask = video_ids == video_id
        holdout_labels = np.unique(labels[holdout_mask])
        if len(holdout_labels) != 1:
            print(f"[WARN] Bỏ qua video có nhiều nhãn: {video_id}")
            continue

        true_label = holdout_labels[0]
        if len(videos_by_label[true_label]) < 2:
            continue

        train_mask = ~holdout_mask
        if len(np.unique(labels[train_mask])) < 2:
            continue

        prototype_labels, prototypes = build_prototypes(
            embeddings[train_mask], labels[train_mask], video_ids[train_mask]
        )
        video_embedding = embeddings[holdout_mask].mean(axis=0)
        norm = np.linalg.norm(video_embedding)
        if not np.isfinite(norm) or norm == 0:
            continue
        video_embedding = video_embedding / norm

        similarities = prototypes @ video_embedding
        best_index = int(np.argmax(similarities))
        results.append((true_label, prototype_labels[best_index], float(similarities[best_index])))

    if not results:
        print("[WARN] Chưa đủ video độc lập để đánh giá leave-one-video-out.")
        return

    true_labels = np.asarray([result[0] for result in results], dtype=str)
    predictions = np.asarray([result[1] for result in results], dtype=str)
    scores = np.asarray([result[2] for result in results], dtype=np.float32)
    accuracy = np.mean(predictions == true_labels)
    per_label_accuracy = [
        np.mean(predictions[true_labels == label] == label)
        for label in np.unique(true_labels)
    ]
    macro_accuracy = np.mean(per_label_accuracy)
    accepted = scores >= COSINE_THRESHOLD
    coverage = np.mean(accepted)
    accepted_accuracy = (
        np.mean(predictions[accepted] == true_labels[accepted])
        if np.any(accepted)
        else 0.0
    )
    print(f"[EVAL] Leave-one-video-out: {len(results)} video; accuracy={accuracy:.3f}; macro={macro_accuracy:.3f}")
    print(
        f"[EVAL] Coverage tại cosine >= {COSINE_THRESHOLD:.2f}: {coverage:.3f}; "
        f"accuracy trên video được nhận: {accepted_accuracy:.3f}"
    )


def main():
    videos = discover_videos()
    if not videos:
        raise RuntimeError(f"Không tìm thấy video trong cấu trúc {VIDEO_DIR}/<tên người>/.")

    signature = cache_signature(videos)
    cached = load_embedding_cache(signature)
    if cached is None:
        embeddings, labels, video_ids = extract_embeddings(videos)
        save_embedding_cache(signature, embeddings, labels, video_ids)
    else:
        embeddings, labels, video_ids = cached

    missing_people = sorted({person_name for _, person_name, _ in videos} - set(labels))
    if missing_people:
        raise RuntimeError(
            "Không trích xuất được embedding đạt chất lượng cho: "
            + ", ".join(missing_people)
            + ". Xem thống kê chất lượng theo từng video."
        )

    if len(np.unique(labels)) < 2:
        raise RuntimeError("Cần ít nhất hai người có embedding để tạo prototype nhận diện.")

    evaluate_leave_one_video_out(embeddings, labels, video_ids)
    prototype_labels, prototypes = build_prototypes(embeddings, labels, video_ids)

    np.savez_compressed(
        PROTOTYPES_FILE,
        labels=prototype_labels,
        prototypes=prototypes,
        cosine_threshold=np.asarray(COSINE_THRESHOLD, dtype=np.float32),
    )
    print(f"[HOÀN TẤT] Đã lưu prototype cosine: {PROTOTYPES_FILE}")
    print("[INFO] Dùng video người lạ riêng để hiệu chỉnh COSINE_THRESHOLD cho open-set.")


if __name__ == "__main__":
    main()