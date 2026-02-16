"""
GUI-клиент для отслеживания взгляда через HTTP API.
Захватывает кадры с вебкамеры, отправляет их на api.py (POST /predict),
получает h_deg, v_deg, zone и отображает с сглаживанием.
Запуск: сначала запустите API (python api.py --host 127.0.0.1 --port 8000),
затем: python app_api_client.py
"""
import base64
import io
import json
import sys
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

DIRECTION_RU = {
    "center": "Прямо",
    "left": "Влево",
    "right": "Вправо",
    "up": "Вверх",
    "down": "Вниз",
    "left-up": "Лево-верх",
    "left-down": "Лево-низ",
    "right-up": "Право-верх",
    "right-down": "Право-низ",
}
CAP_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else 0
DEFAULT_CAM = 1
SMOOTH_ALPHA = 0.3


def list_cameras(max_check=6):
    out = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i, CAP_BACKEND)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                out.append(i)
    return out


def predict_via_api(base_url: str, image_rgb: np.ndarray, jpeg_quality: int = 85) -> dict | None:
    """
    Отправляет кадр на POST /predict в виде JPEG base64.
    Возвращает {"h_deg", "v_deg", "zone"} или None при ошибке.
    """
    pil = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    url = base_url.rstrip("/") + "/predict"
    body = json.dumps({"image_b64": b64}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "error" in data:
                return None
            return {
                "h_deg": float(data["h_deg"]),
                "v_deg": float(data["v_deg"]),
                "zone": data.get("zone", "center"),
            }
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError):
        return None


class GazeAPIClientApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Eye Gaze Tracker (API Client)")

        self.cap = None
        self.camera_running = False
        self.prediction_running = False
        self.request_pending = False
        self.h_smooth = 0.0
        self.v_smooth = 0.0
        self.initialized = False
        self.api_result = None  # (h, v, zone) from worker
        self.api_result_ready = False
        self.api_error = None  # last error message for status

        self._build_ui()

    def _build_ui(self):
        ctrl = ttk.Frame(self.root, padding=5)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="API:").pack(side=tk.LEFT, padx=(0, 5))
        self.api_var = tk.StringVar(value="http://127.0.0.1:8000")
        self.api_entry = ttk.Entry(ctrl, textvariable=self.api_var, width=24)
        self.api_entry.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(ctrl, text="Камера:").pack(side=tk.LEFT, padx=(0, 5))
        cams = list_cameras()
        self.cam_var = tk.StringVar(value=str(DEFAULT_CAM))
        self.cam_combo = ttk.Combobox(
            ctrl, textvariable=self.cam_var, width=6, values=cams or [0, 1, 2]
        )
        self.cam_combo.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(ctrl, text="Запустить камеру", command=self._toggle_camera).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(ctrl, text="Запустить предсказание", command=self._toggle_prediction).pack(
            side=tk.LEFT, padx=2
        )

        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=5)
        self.direction_label = tk.Label(
            self.root, text="Направление: ---", font=("Arial", 18)
        )
        self.direction_label.pack()
        self.coords_label = tk.Label(
            self.root, text="H: ---°  V: ---°", font=("Consolas", 11), fg="#555"
        )
        self.coords_label.pack()
        self.coords_canvas = tk.Canvas(
            self.root, width=160, height=160, bg="#f5f5f5", highlightthickness=1
        )
        self.coords_canvas.pack(pady=5)
        self._draw_coords_grid()
        self.status_label = tk.Label(
            self.root,
            text="Введите URL API, выберите камеру и нажмите «Запустить камеру»",
            fg="gray",
        )
        self.status_label.pack(pady=5)

    def _draw_coords_grid(self):
        c = self.coords_canvas
        c.delete("all")
        w, h = 160, 160
        cx, cy = w // 2, h // 2
        step = 50
        c.create_line(0, cy, w, cy, fill="#ccc", width=1)
        c.create_line(cx, 0, cx, h, fill="#ccc", width=1)
        for i in range(-1, 2):
            for j in range(-1, 2):
                x1 = cx + i * step - step // 2
                y1 = cy + j * step - step // 2
                x2, y2 = x1 + step, y1 + step
                c.create_rectangle(x1, y1, x2, y2, outline="#ddd", fill="")
        c.create_text(cx, 12, text="V↑", font=("Arial", 9), fill="#888")
        c.create_text(w - 8, cy, text="H→", font=("Arial", 9), fill="#888")

    def _update_coords_display(self, h_deg: float, v_deg: float) -> None:
        c = self.coords_canvas
        c.delete("point")
        w, h = 160, 160
        cx, cy = w // 2, h // 2
        scale = 2.0
        x = cx + h_deg * scale
        y = cy - v_deg * scale
        r = 6
        c.create_oval(
            x - r, y - r, x + r, y + r,
            fill="#2196F3", outline="#1565C0", tags="point"
        )

    def _toggle_camera(self):
        if self.camera_running:
            self.camera_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.status_label.config(text="Камера остановлена")
        else:
            try:
                cid = int(self.cam_var.get())
            except ValueError:
                self.status_label.config(text="Неверный индекс камеры")
                return
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(cid, CAP_BACKEND)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(cid, 0)
            if not self.cap.isOpened():
                self.status_label.config(text=f"Не удалось открыть камеру {cid}")
                return
            self.camera_running = True
            self.status_label.config(text=f"Камера {cid} активна")
        self._update()

    def _toggle_prediction(self):
        self.prediction_running = not self.prediction_running
        self.api_error = None
        if not self.prediction_running:
            self.initialized = False
            self.coords_label.config(text="H: ---°  V: ---°")
            self.coords_canvas.delete("point")
        self.status_label.config(
            text=f"Предсказание: {'вкл (через API)' if self.prediction_running else 'выкл'}"
        )

    def _on_api_result(self, result: dict | None):
        """Вызывается из worker-потока; сохраняем результат для main thread."""
        self.request_pending = False
        if result is None:
            self.api_error = "Ошибка API или лицо не найдено"
            return
        self.api_result = (result["h_deg"], result["v_deg"], result["zone"])
        self.api_result_ready = True

    def _send_frame_async(self, frame_rgb: np.ndarray):
        base_url = self.api_var.get().strip()
        if not base_url:
            self.status_label.config(text="Укажите URL API")
            return

        def worker():
            res = predict_via_api(base_url, frame_rgb)
            self.root.after(0, lambda: self._on_api_result(res))

        self.request_pending = True
        self.api_error = None
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _update(self):
        if not self.camera_running or not self.cap or not self.cap.isOpened():
            self.root.after(500, self._update)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._update)
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        direction_text = "---"

        if self.prediction_running and not self.request_pending:
            self._send_frame_async(rgb.copy())

        if self.api_result_ready and self.api_result is not None:
            self.api_result_ready = False
            h_pred, v_pred, zone = self.api_result
            if not self.initialized:
                self.h_smooth, self.v_smooth = h_pred, v_pred
                self.initialized = True
            else:
                self.h_smooth = SMOOTH_ALPHA * h_pred + (1 - SMOOTH_ALPHA) * self.h_smooth
                self.v_smooth = SMOOTH_ALPHA * v_pred + (1 - SMOOTH_ALPHA) * self.v_smooth
            direction_text = DIRECTION_RU.get(zone, zone)
            self.coords_label.config(text=f"H: {self.h_smooth:+.1f}°  V: {self.v_smooth:+.1f}°")
            self._update_coords_display(self.h_smooth, self.v_smooth)

        if self.api_error:
            self.status_label.config(text=self.api_error, fg="red")
        elif self.prediction_running:
            self.status_label.config(text="Предсказание: вкл (через API)", fg="gray")

        img = Image.fromarray(rgb).resize((640, 480))
        self.video_label.imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.video_label.imgtk)
        self.direction_label.config(text=f"Направление: {direction_text}")

        self.root.after(30, self._update)

    def run(self):
        self.root.mainloop()
        if self.cap:
            self.cap.release()


def main():
    app = GazeAPIClientApp()
    app.run()


if __name__ == "__main__":
    main()
