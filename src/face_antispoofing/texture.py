from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .domain import Signal


class OpenCVDNNTextureModel:
    """CPU-friendly ONNX adapter. Class order must be [spoof, live]."""

    def __init__(self, model_path: str | Path, input_size: int = 224) -> None:
        self._path = Path(model_path)
        self._input_size = input_size
        self._net = None
        self._cv2 = None
        if self._path.is_file():
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("opencv-python is required for ONNX inference") from exc
            self._cv2 = cv2
            self._net = cv2.dnn.readNetFromONNX(str(self._path))

    @property
    def available(self) -> bool:
        return self._net is not None

    def predict(self, face: np.ndarray) -> Signal:
        if self._net is None or self._cv2 is None:
            return Signal("texture", None, 0.0, "model_missing", {"path": str(self._path)})
        started = time.perf_counter()
        rgb = self._cv2.cvtColor(face, self._cv2.COLOR_BGR2RGB)
        resized = self._cv2.resize(rgb, (self._input_size, self._input_size))
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        blob = np.transpose(tensor, (2, 0, 1))[None, ...]
        self._net.setInput(blob)
        raw = np.asarray(self._net.forward()).reshape(-1)
        if raw.size != 2:
            return Signal(
                "texture",
                None,
                0.0,
                "invalid_model_output",
                {"shape": list(raw.shape)},
            )
        logits = raw - np.max(raw)
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return Signal(
            "texture",
            float(probabilities[1]),
            1.0,
            "onnx_texture_probability",
            {"latency_ms": latency_ms},
        )
