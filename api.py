"""
Минимальный HTTP API для предсказания направления взгляда.
Эндпоинты:
  GET  /health
  POST /predict { "image_path": "..."} или { "image_b64": "..." }
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

from features import extract_eye_features, gaze_to_zone

BASE = Path(__file__).parent
MODEL_NAMES = ["face_landmarker_v2_with_blendshapes.task", "face_landmarker.task"]


def _get_model_path() -> Path | None:
    """MediaPipe не поддерживает кириллицу в путях — используем копию в корне."""
    target = BASE / "face_landmarker.task"
    if target.exists():
        return target
    for name in MODEL_NAMES:
        src = BASE / "шаблон" / name
        if src.exists():
            import shutil

            shutil.copy(src, target)
            return target
    return None


def _load_models() -> Dict[str, Any]:
    """Загружает модели и детектор; бросает исключение при отсутствии файлов."""
    model_dir = BASE if (BASE / "gaze_model_h.pkl").exists() else BASE / "шаблон"
    for fname in ["gaze_model_h.pkl", "gaze_model_v.pkl", "gaze_scaler.pkl"]:
        if not (model_dir / fname).exists():
            raise FileNotFoundError(f"Файл {fname} не найден. Сначала запустите train.py")

    model_h = joblib.load(model_dir / "gaze_model_h.pkl")
    model_v = joblib.load(model_dir / "gaze_model_v.pkl")
    scaler = joblib.load(model_dir / "gaze_scaler.pkl")

    model_path = _get_model_path()
    if not model_path:
        raise FileNotFoundError("face_landmarker.task не найден в папке проекта")

    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(opts)

    return {"model_h": model_h, "model_v": model_v, "scaler": scaler, "detector": detector}


class GazeAPIHandler(BaseHTTPRequestHandler):
    """Обработчик HTTP запросов. Контекст хранится в server.context."""

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send_json({"error": "Not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        try:
            img = _load_image(data)
        except Exception as exc:  # noqa: BLE001 - возвращаем текст ошибки клиенту
            self._send_json({"error": str(exc)}, status=400)
            return

        ctx = self.server.context  # type: ignore[attr-defined]
        rgb = np.ascontiguousarray(np.array(img))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = ctx["detector"].detect(mp_img)

        if not res.face_landmarks:
            self._send_json({"error": "Face not detected"}, status=422)
            return

        feat = extract_eye_features(res.face_landmarks[0])
        if feat is None:
            self._send_json({"error": "Invalid landmarks"}, status=422)
            return

        X = np.array([feat["features"]])
        Xs = ctx["scaler"].transform(X)
        h = float(ctx["model_h"].predict(Xs)[0])
        v = float(ctx["model_v"].predict(Xs)[0])
        zone = gaze_to_zone(h, v)

        self._send_json({"h_deg": h, "v_deg": v, "zone": zone})


def _load_image(data: Dict[str, Any]) -> Image.Image:
    """Загружает изображение из пути (внутри проекта) или base64."""
    if "image_path" in data:
        path = Path(data["image_path"]).resolve()
        base_res = BASE.resolve()
        # Запрет path traversal: файл должен быть внутри директории проекта
        if not path.is_file() or base_res not in path.parents:
            raise ValueError("Допустимы только пути к файлам внутри директории проекта")
        return Image.open(path).convert("RGB")

    if "image_b64" in data:
        raw = base64.b64decode(data["image_b64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")

    raise ValueError("Нужен image_path или image_b64")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    context = _load_models()
    server = ThreadingHTTPServer((host, port), GazeAPIHandler)
    server.context = context  # type: ignore[attr-defined]
    print(f"API запущен: http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP API для распознавания взгляда")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
