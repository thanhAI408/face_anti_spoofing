"""Sửa nhãn trong face_embeddings.npz và dựng lại prototype mà không cần GPU.

Xem mỗi video giống ai nhất:
    python relabel.py
Đổi nhãn (theo nhãn cũ hoặc theo đường dẫn video) rồi lưu lại:
    python relabel.py --map 1=bang --map real/real.mp4=thang --save
"""
import argparse

import numpy as np

from preprocessing import (
    COSINE_THRESHOLD,
    EMBEDDINGS_FILE,
    PROTOTYPES_FILE,
    build_prototypes,
    evaluate_leave_one_video_out,
)


def load_embeddings():
    with np.load(EMBEDDINGS_FILE, allow_pickle=False) as cache:
        return (
            str(cache["signature"].item()),
            cache["embeddings"].astype(np.float32),
            cache["labels"].astype(str),
            cache["video_ids"].astype(str),
        )


def parse_mappings(items):
    mappings = []
    for item in items:
        source, sep, target = item.partition("=")
        if not sep or not source.strip() or not target.strip():
            raise SystemExit(f"--map không hợp lệ: {item!r}; cần dạng <nhãn cũ hoặc video>=<tên mới>.")
        mappings.append((source.strip(), target.strip()))
    return mappings


def apply_mappings(labels, video_ids, mappings):
    labels = labels.copy()
    for source, target in mappings:
        # Khóa có "/" là đường dẫn video (vd. real/real.mp4), ngược lại là nhãn cũ.
        mask = video_ids == source if "/" in source else labels == source
        if not np.any(mask):
            raise SystemExit(f"Không tìm thấy nhãn hoặc video: {source}")
        labels[mask] = target
        print(f"[INFO] {source} -> {target} ({int(mask.sum())} embeddings)")
    return labels


def print_video_report(embeddings, labels, video_ids):
    print(f"\n{'Video':40s} {'Nhãn':12s} {'Số mẫu':>6s}  Giống nhất (cosine, không tính chính video đó)")
    for video_id in sorted(np.unique(video_ids)):
        mask = video_ids == video_id
        label = labels[mask][0]
        video_embedding = embeddings[mask].mean(axis=0)
        video_embedding /= np.linalg.norm(video_embedding)

        others = ~mask
        if len(np.unique(labels[others])) == 0:
            continue
        prototype_labels, prototypes = build_prototypes(
            embeddings[others], labels[others], video_ids[others]
        )
        similarities = prototypes @ video_embedding
        order = np.argsort(similarities)[::-1][:3]
        ranking = ", ".join(f"{prototype_labels[i]} {similarities[i]:.2f}" for i in order)
        print(f"{video_id:40s} {label:12s} {int(mask.sum()):6d}  {ranking}")
    print(f"\n[INFO] Ngưỡng hiện tại: cosine >= {COSINE_THRESHOLD:.2f} mới được coi là cùng người.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", action="append", default=[], help="<nhãn cũ hoặc video>=<tên mới>, dùng nhiều lần")
    parser.add_argument("--save", action="store_true", help="ghi đè face_embeddings.npz và face_prototypes.npz")
    args = parser.parse_args()

    signature, embeddings, labels, video_ids = load_embeddings()
    labels = apply_mappings(labels, video_ids, parse_mappings(args.map))
    print_video_report(embeddings, labels, video_ids)

    if len(np.unique(labels)) < 2:
        raise SystemExit("Cần ít nhất hai người để dựng prototype.")
    evaluate_leave_one_video_out(embeddings, labels, video_ids)

    if not args.save:
        print("[INFO] Chưa lưu gì; thêm --save để ghi kết quả.")
        return

    np.savez_compressed(
        EMBEDDINGS_FILE,
        signature=np.asarray(signature),
        embeddings=embeddings,
        labels=labels,
        video_ids=video_ids,
    )
    prototype_labels, prototypes = build_prototypes(embeddings, labels, video_ids)
    np.savez_compressed(
        PROTOTYPES_FILE,
        labels=prototype_labels,
        prototypes=prototypes,
        cosine_threshold=np.asarray(COSINE_THRESHOLD, dtype=np.float32),
    )
    print(f"[HOÀN TẤT] Đã lưu {EMBEDDINGS_FILE} và {PROTOTYPES_FILE} với nhãn: {', '.join(prototype_labels)}")


if __name__ == "__main__":
    main()
