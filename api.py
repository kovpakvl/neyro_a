# Минимальный HTTP API для предсказания взгляда. # Описание файла
import argparse  # Аргументы CLI
import base64  # Base64 кодирование
import io  # Буфер для изображений
import json  # JSON формат
from http.server import BaseHTTPRequestHandler  # HTTP обработчик
from http.server import ThreadingHTTPServer  # HTTP сервер
from pathlib import Path  # Пути
# 
import joblib  # Сохранение моделей
import mediapipe as mp  # MediaPipe
import numpy as np  # Математика
from mediapipe.tasks import python  # Опции MediaPipe
from mediapipe.tasks.python import vision  # Задачи vision
from PIL import Image  # Работа с изображениями
# 
from features import extract_eye_features, gaze_to_zone  # Признаки и зоны
# 
BASE = Path(__file__).parent  # База проекта
# 
# Возвращает путь к модели детектора. 
def _get_model_path() -> Path:  # Путь модели
    target = BASE / "face_landmarker.task"  # Целевой файл
    if target.exists():  # Если есть файл
        return target  # Возврат пути
    src = BASE / "шаблон" / "face_landmarker.task"  # Источник
    if src.exists():  # Если есть копия
        import shutil  # Копирование файла
# 
        shutil.copy(src, target)  # Копируем модель
    return target  # Возврат пути
# 
# Загружает модели регрессии и детектор лица. 
def _load_models() -> dict:  # Загрузка моделей
    model_dir = BASE / "out" / "models"  # Папка моделей
    model_h = joblib.load(model_dir / "model_h.pkl")  # Модель H
    model_v = joblib.load(model_dir / "model_v.pkl")  # Модель V
    scaler = joblib.load(model_dir / "scaler.pkl")  # Скалер
# 
    model_path = _get_model_path()  # Путь к модели детектора
    base_opts = python.BaseOptions(  # Базовые опции
        model_asset_path=str(model_path),  # Путь модели
    )
    opts = vision.FaceLandmarkerOptions(  # Опции детектора
        base_options=base_opts,  # База
        num_faces=1,  # Одно лицо
    )
    detector = vision.FaceLandmarker.create_from_options(opts)  # Детектор
# 
    return {  # Контекст сервера
        "model_h": model_h,  # Модель H
        "model_v": model_v,  # Модель V
        "scaler": scaler,  # Скалер
        "detector": detector,  # Детектор
    }
# 
# Обработчик HTTP запросов для API. 
class GazeAPIHandler(BaseHTTPRequestHandler):  # Класс хендлера
    # Отправляет JSON ответ. 
    def _send_json(self, payload: dict, status: int = 200) -> None:  # Ответ
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")  # Тело
        self.send_response(status)  # Код ответа
        content_type = "application/json; charset=utf-8"  # Тип
        self.send_header("Content-Type", content_type)  # Тип
        self.send_header("Content-Length", str(len(body)))  # Длина
        self.end_headers()  # Конец заголовков
        self.wfile.write(body)  # Запись ответа
# 
    def do_GET(self) -> None:  # noqa: N802 # GET запрос
        if self.path == "/health":  # Проверка пути
            self._send_json({"status": "ok"})  # Ответ OK
            return  # Выход
        self._send_json({"error": "Not found"}, status=404)  # 404
# 
    def do_POST(self) -> None:  # noqa: N802 # POST запрос
        if self.path != "/predict":  # Проверка пути
            self._send_json({"error": "Not found"}, status=404)  # 404
            return  # Выход
# 
        length = int(self.headers.get("Content-Length", "0"))  # Длина тела
        raw = self.rfile.read(length) if length > 0 else b""  # Данные
        data = json.loads(raw.decode("utf-8"))  # Парсинг JSON
        img = _load_image(data)  # Картинка
# 
        ctx = self.server.context  # type: ignore[attr-defined] # Контекст
        rgb = np.ascontiguousarray(np.array(img))  # RGB массив
        mp_img = mp.Image(  # MP Image
            image_format=mp.ImageFormat.SRGB,  # Формат
            data=rgb,  # Данные
        )
        res = ctx["detector"].detect(mp_img)  # Детект лица
# 
        if not res.face_landmarks:  # Нет лендмарков
            err = {"error": "Face not detected"}  # Текст ошибки
            self._send_json(err, status=422)  # Ошибка
            return  # Выход
# 
        feat = extract_eye_features(res.face_landmarks[0])  # Признаки
        if feat is None:  # Невалидные признаки
            err = {"error": "Invalid landmarks"}  # Текст ошибки
            self._send_json(err, status=422)  # Ошибка
            return  # Выход
# 
        X = np.array([feat["features"]])  # Вектор признаков
        Xs = ctx["scaler"].transform(X)  # Масштабирование
        h = float(ctx["model_h"].predict(Xs)[0])  # Прогноз H
        v = float(ctx["model_v"].predict(Xs)[0])  # Прогноз V
        zone = gaze_to_zone(h, v)  # Зона взгляда
# 
        self._send_json({"h_deg": h, "v_deg": v, "zone": zone})  # Ответ
# 
# Загружает изображение из base64. 
def _load_image(data: dict) -> Image.Image:  # Декодирование
    raw = base64.b64decode(data["image_b64"])  # Base64 -> bytes
    return Image.open(io.BytesIO(raw)).convert("RGB")  # PIL -> RGB
# 
# Запускает HTTP сервер API. 
def run(host: str = "127.0.0.1", port: int = 8000) -> None:  # Запуск
    context = _load_models()  # Модели и детектор
    server = ThreadingHTTPServer((host, port), GazeAPIHandler)  # Сервер
    server.context = context  # type: ignore[attr-defined] # Контекст
    print(f"API запущен: http://{host}:{port}")  # Сообщение
    server.serve_forever()  # Основной цикл
# 
# Разбор аргументов и запуск. 
def main() -> None:  # CLI точка входа
    parser = argparse.ArgumentParser(  # Парсер
        description="HTTP API для распознавания взгляда",  # Описание
    )
    parser.add_argument("--host", default="127.0.0.1")  # Хост
    parser.add_argument("--port", type=int, default=8000)  # Порт
    args = parser.parse_args()  # Аргументы
    run(args.host, args.port)  # Запуск
# 
if __name__ == "__main__":  # Запуск как скрипта
    main()  # Вызов main
