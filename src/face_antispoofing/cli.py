from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

from .config import AppConfig, FusionConfig, TextureConfig, load_config
from .domain import FrameResult, Verdict

COLORS = {
    Verdict.LIVE: (0, 190, 0),
    Verdict.SPOOF: (0, 0, 230),
    Verdict.RETRY: (0, 170, 255),
    Verdict.UNCERTAIN: (0, 170, 255),
    Verdict.VERIFYING: (255, 170, 0),
    Verdict.NO_FACE: (170, 170, 170),
    Verdict.ERROR: (0, 0, 230),
}


def _project_root() -> Path:
    return Path.cwd()


def _override_config(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    camera = dataclasses.replace(
        config.camera,
        index=config.camera.index if args.camera is None else args.camera,
    )
    texture = config.texture
    if args.model:
        texture = dataclasses.replace(texture, model_path=args.model)
    fusion = config.fusion
    challenge = config.challenge
    if args.challenge_only:
        texture = TextureConfig(
            model_path="__disabled__",
            input_size=texture.input_size,
            run_every_n_frames=texture.run_every_n_frames,
        )
        fusion = FusionConfig(
            texture_weight=0.0,
            challenge_weight=0.75,
            motion_weight=0.25,
            min_live_evidence_weight=0.70,
            min_spoof_evidence_weight=0.70,
            live_threshold=0.70,
            spoof_threshold=0.25,
            hard_spoof_texture_threshold=0.0,
            hard_spoof_frames=fusion.hard_spoof_frames,
            smoothing_window=fusion.smoothing_window,
            min_stable_frames=fusion.min_stable_frames,
            require_challenge_for_live=True,
        )
    return dataclasses.replace(
        config,
        camera=camera,
        texture=texture,
        challenge=challenge,
        fusion=fusion,
    )


def _draw_overlay(frame, result: FrameResult, fps: float) -> None:
    import cv2

    color = COLORS[result.verdict]
    if result.box:
        box = result.box
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)
    score = "--" if result.live_score is None else f"{result.live_score:.3f}"
    lines = [
        f"{result.verdict.value}  live={score}",
        f"reason={result.reason}",
        f"fps={fps:.1f}  latency={result.latency_ms:.1f}ms",
    ]
    if result.challenge:
        lines.append(
            f"challenge {result.challenge.completed_steps}/{result.challenge.total_steps}: "
            f"{result.challenge.prompt_ascii}"
        )
    signal_text = "  ".join(
        f"{name}={'--' if signal.score is None else f'{signal.score:.2f}'}"
        for name, signal in result.signals.items()
    )
    if signal_text:
        lines.append(signal_text)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 28 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color if index == 0 else (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        "Q: quit | R: new challenge",
        (12, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def run_demo(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError:
        print("Missing runtime dependencies. Run: pip install -e .", file=sys.stderr)
        return 2

    config = _override_config(load_config(args.config), args)
    from .pipeline import AntiSpoofPipeline

    pipeline = AntiSpoofPipeline(config, _project_root())
    if not pipeline.texture_model_available and not args.challenge_only:
        print(
            "[WARNING] Texture ONNX model is missing. The default policy will not approve LIVE "
            "without enough evidence. Train the model or use --challenge-only for a limited demo."
        )
    camera = cv2.VideoCapture(config.camera.index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    camera.set(cv2.CAP_PROP_FPS, config.camera.fps)
    if not camera.isOpened():
        pipeline.close()
        print(f"Cannot open camera index {config.camera.index}", file=sys.stderr)
        return 2

    previous = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped returning frames", file=sys.stderr)
                return 2
            if config.camera.mirror:
                frame = cv2.flip(frame, 1)
            result = pipeline.process(frame)
            current = time.perf_counter()
            instantaneous = 1.0 / max(1e-6, current - previous)
            fps = instantaneous if fps == 0.0 else 0.90 * fps + 0.10 * instantaneous
            previous = current
            _draw_overlay(frame, result, fps)
            cv2.imshow("RGB Face Anti-Spoofing", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                return 0
            if key == ord("r"):
                pipeline.reset()
    finally:
        pipeline.close()
        camera.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-signal RGB face anti-spoofing")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run realtime webcam verification")
    demo.add_argument("--config", default="config/default.toml")
    demo.add_argument("--model", help="Override the ONNX texture model path")
    demo.add_argument("--camera", type=int, help="Override the webcam index")
    demo.add_argument(
        "--challenge-only",
        action="store_true",
        help="Demo landmarks without a trained texture model; weaker security",
    )
    demo.set_defaults(func=run_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
