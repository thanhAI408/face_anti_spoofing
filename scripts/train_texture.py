from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from face_antispoofing.dataset import assert_no_subject_leakage  # noqa: E402
from face_antispoofing.evaluation.metrics import (  # noqa: E402
    aggregate_video_scores,
    compute_binary_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3-Small texture PAD and export ONNX"
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/models/texture_mobilenetv3.onnx")
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "label", "subject_id", "source_video", "split"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Manifest is empty or missing columns: {sorted(missing)}")
    assert_no_subject_leakage(rows)
    return rows


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ManifestDataset:
    """Top-level dataset so Windows DataLoader workers can pickle it."""

    def __init__(self, samples: list[dict[str, str]], data_root: Path, transform) -> None:
        self.samples = samples
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.samples[index]
        with Image.open(self.data_root / row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError:
        print('Missing training dependencies. Run: pip install -e ".[train]"', file=sys.stderr)
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest_path = args.manifest.resolve()
    data_root = manifest_path.parent
    rows = _read_manifest(manifest_path)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("Both train and val splits must contain samples")
    for name, subset in (("train", train_rows), ("val", val_rows)):
        labels = {int(row["label"]) for row in subset}
        if labels != {0, 1}:
            raise ValueError(f"Split '{name}' must contain both spoof and live; found {labels}")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.78, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(240),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(
        ManifestDataset(train_rows, data_root, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ManifestDataset(val_rows, data_root, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    weights = None if args.no_pretrained else MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_counts = [sum(int(row["label"]) == label for row in train_rows) for label in (0, 1)]
    if min(class_counts) == 0:
        raise ValueError("Training split must contain both spoof (0) and live (1)")
    total = sum(class_counts)
    class_weights = torch.tensor(
        [total / (2 * count) for count in class_counts], dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_state = None
    best_acer = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * len(labels)
        scheduler.step()

        model.eval()
        labels_all: list[int] = []
        scores_all: list[float] = []
        with torch.inference_mode():
            for images, labels in val_loader:
                probabilities = torch.softmax(model(images.to(device)), dim=1)[:, 1]
                labels_all.extend(labels.tolist())
                scores_all.extend(probabilities.cpu().tolist())
        frame_metrics = compute_binary_metrics(labels_all, scores_all, threshold=0.5)
        scored_rows = [
            {**row, "label": int(row["label"]), "score": score}
            for row, score in zip(val_rows, scores_all, strict=True)
        ]
        video_rows = aggregate_video_scores(scored_rows)
        video_metrics = compute_binary_metrics(
            [item["label"] for item in video_rows],
            [item["score"] for item in video_rows],
            threshold=0.5,
        )
        entry = {
            "epoch": epoch,
            "train_loss": loss_total / len(train_rows),
            "validation_frame": frame_metrics,
            "validation_video": video_metrics,
        }
        history.append(entry)
        print(
            f"epoch={epoch:03d} loss={entry['train_loss']:.4f} "
            f"video_val_ACER={video_metrics['acer']:.4f} "
            f"video_val_AUC={video_metrics['roc_auc']:.4f}"
        )
        if video_metrics["acer"] < best_acer:
            best_acer = float(video_metrics["acer"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval().cpu()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".pt")
    torch.save(
        {
            "state_dict": best_state,
            "class_order": ["spoof", "live"],
            "input_size": 224,
            "seed": args.seed,
        },
        checkpoint_path,
    )
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    report_path = output_path.with_suffix(".training.json")
    report = {
        "created_at_unix": int(time.time()),
        "manifest": str(manifest_path),
        "manifest_sha256": _hash(manifest_path),
        "model": "MobileNetV3-Small",
        "onnx_sha256": _hash(output_path),
        "class_order": ["spoof", "live"],
        "device": str(device),
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "best_validation_video_acer": best_acer,
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved ONNX model: {output_path}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved training report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
