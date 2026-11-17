"""Đánh giá nhận diện theo từng frame và chọn COSINE_THRESHOLD từ face_embeddings.npz.

- Người quen: leave-one-video-out. Mỗi video được nhận diện bằng prototype dựng từ các video
  còn lại (kể cả video khác của cùng người); mỗi frame là một lần nhận diện, đúng như realtime.
- Người lạ: leave-one-person-out. Bỏ hẳn một người khỏi prototype rồi đưa frame của người đó
  vào; mọi frame có điểm >= ngưỡng là một lần nhận nhầm người lạ thành người quen.

    python calibrate_threshold.py --max-far 0.01
"""
import argparse

import numpy as np

from preprocessing import COSINE_THRESHOLD, EMBEDDINGS_FILE, build_prototypes


def best_matches(embeddings, labels, video_ids, train_mask, test_mask):
    prototype_labels, prototypes = build_prototypes(
        embeddings[train_mask], labels[train_mask], video_ids[train_mask]
    )
    similarities = embeddings[test_mask] @ prototypes.T
    best = np.argmax(similarities, axis=1)
    return prototype_labels[best], similarities[np.arange(len(best)), best]


def collect_scores(embeddings, labels, video_ids):
    known = {"person": [], "correct": [], "score": []}
    for video_id in np.unique(video_ids):
        test_mask = video_ids == video_id
        person = labels[test_mask][0]
        train_mask = ~test_mask
        if person not in labels[train_mask]:
            continue  # người chỉ có một video: không có prototype nào để so
        predicted, scores = best_matches(embeddings, labels, video_ids, train_mask, test_mask)
        known["person"].extend([person] * len(scores))
        known["correct"].extend(predicted == person)
        known["score"].extend(scores)

    unknown = {"person": [], "score": []}
    for person in np.unique(labels):
        test_mask = labels == person
        _, scores = best_matches(embeddings, labels, video_ids, ~test_mask, test_mask)
        unknown["person"].extend([person] * len(scores))
        unknown["score"].extend(scores)

    return (
        {key: np.asarray(value) for key, value in known.items()},
        {key: np.asarray(value) for key, value in unknown.items()},
    )


def rates_at(threshold, known, unknown):
    accepted = known["score"] >= threshold
    unknown_accepted = unknown["score"] >= threshold
    people = np.unique(unknown["person"])
    return {
        # Người quen được nhận đúng tên.
        "tar": np.mean(accepted & known["correct"]),
        # Người quen bị gọi nhầm sang tên người khác.
        "mis": np.mean(accepted & ~known["correct"]),
        # Người lạ bị nhận thành người quen; macro để người có nhiều frame không lấn át.
        "far": np.mean(unknown_accepted),
        "far_macro": np.mean([np.mean(unknown_accepted[unknown["person"] == p]) for p in people]),
        "far_worst": max(np.mean(unknown_accepted[unknown["person"] == p]) for p in people),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-far", type=float, default=0.01, help="tỉ lệ người lạ bị nhận nhầm tối đa (macro)")
    args = parser.parse_args()

    with np.load(EMBEDDINGS_FILE, allow_pickle=False) as cache:
        embeddings = cache["embeddings"].astype(np.float32)
        labels = cache["labels"].astype(str)
        video_ids = cache["video_ids"].astype(str)

    known, unknown = collect_scores(embeddings, labels, video_ids)
    print(
        f"[INFO] Người quen: {len(known['score'])} frame từ {len(np.unique(known['person']))} người; "
        f"người lạ: {len(unknown['score'])} frame từ {len(np.unique(unknown['person']))} người."
    )
    print(f"[INFO] Top-1 đúng (chưa xét ngưỡng): {np.mean(known['correct']):.3f}")

    print(f"\n{'Ngưỡng':>6s} {'Nhận đúng':>9s} {'Nhầm tên':>8s} {'Lạ->quen':>8s} {'macro':>6s} {'tệ nhất':>7s}")
    thresholds = np.round(np.arange(0.30, 0.81, 0.05), 2)
    for threshold in thresholds:
        r = rates_at(threshold, known, unknown)
        marker = "  <- hiện tại" if np.isclose(threshold, COSINE_THRESHOLD) else ""
        print(
            f"{threshold:6.2f} {r['tar']:9.3f} {r['mis']:8.3f} {r['far']:8.3f} "
            f"{r['far_macro']:6.3f} {r['far_worst']:7.3f}{marker}"
        )

    fine = np.round(np.arange(0.30, 0.90, 0.01), 2)
    passing = [t for t in fine if rates_at(t, known, unknown)["far_macro"] <= args.max_far]
    if not passing:
        print(f"\n[WARN] Không ngưỡng nào giữ tỉ lệ người lạ bị nhận nhầm <= {args.max_far:.3f}.")
        return
    recommended = passing[0]
    r = rates_at(recommended, known, unknown)
    print(
        f"\n[KẾT QUẢ] Ngưỡng nhỏ nhất giữ người lạ bị nhận nhầm <= {args.max_far:.1%} (macro): {recommended:.2f}\n"
        f"          Nhận đúng {r['tar']:.1%} frame người quen, nhầm tên {r['mis']:.1%}, "
        f"người lạ bị nhận nhầm {r['far_macro']:.1%} (tệ nhất {r['far_worst']:.1%})."
    )

    print("\n[INFO] Theo từng người tại ngưỡng đề xuất (nhận đúng khi là người quen / bị nhận nhầm khi là người lạ):")
    for person in np.unique(unknown["person"]):
        known_mask = known["person"] == person
        known_rate = (
            f"{np.mean((known['score'][known_mask] >= recommended) & known['correct'][known_mask]):.3f}"
            if np.any(known_mask)
            else "  -  "
        )
        unknown_rate = np.mean(unknown["score"][unknown["person"] == person] >= recommended)
        print(f"  {person:10s} {known_rate:>6s} / {unknown_rate:.3f}")


if __name__ == "__main__":
    main()
