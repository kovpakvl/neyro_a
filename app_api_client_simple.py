# Простое GUI-приложение для демонстрации API взгляда. # Описание
import base64  # Base64 кодирование
import io  # Буфер для изображения
import json  # JSON формат
import sys  # Платформа
import urllib.request  # HTTP запросы
#
import cv2  # OpenCV
import numpy as np  # Численные операции
from PIL import Image, ImageTk  # Работа с изображениями
import tkinter as tk  # Tkinter GUI
from tkinter import ttk  # Тематические виджеты
#
DIRECTION_RU = {  # Словарь зон
    "center": "Прямо",  # Центр
    "left": "Влево",  # Влево
    "right": "Вправо",  # Вправо
    "up": "Вверх",  # Вверх
    "down": "Вниз",  # Вниз
    "left-up": "Лево-верх",  # Лево-верх
    "left-down": "Лево-низ",  # Лево-низ
    "right-up": "Право-верх",  # Право-верх
    "right-down": "Право-низ",  # Право-низ
}  # Конец словаря
#
CAP_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else 0  # Бэкенд
#
#
# Отправляет кадр в API и возвращает результат. 
def predict_via_api(base_url: str, image_rgb: np.ndarray) -> dict:  # Запрос
    pil = Image.fromarray(image_rgb)  # PIL изображение
    buf = io.BytesIO()  # Буфер
    pil.save(buf, format="JPEG", quality=85)  # JPEG в буфер
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")  # Base64
    url = base_url.rstrip("/") + "/predict"  # URL запроса
    body = json.dumps({"image_b64": b64}).encode("utf-8")  # JSON тело
    headers = {  # Заголовки
        "Content-Type": "application/json; charset=utf-8",  # Тип
    }
    req = urllib.request.Request(  # HTTP запрос
        url,  # URL
        data=body,  # Данные
        method="POST",  # Метод
        headers=headers,  # Заголовки
    )
    with urllib.request.urlopen(req, timeout=5) as resp:  # Ответ
        data = json.loads(resp.read().decode("utf-8"))  # JSON ответ
        return {  # Результат
            "h_deg": float(data["h_deg"]),  # Угол H
            "v_deg": float(data["v_deg"]),  # Угол V
            "zone": data.get("zone", "center"),  # Зона
        }
#
#
# Простое GUI приложение клиента. 
class SimpleGazeApp:  # Класс приложения
    def __init__(self):  # Инициализация
        self.root = tk.Tk()  # Главное окно
        title = "Eye Gaze Tracker (Simple API Client)"  # Заголовок
        self.root.title(title)  # Установка заголовка
#
        self.cap = None  # Объект камеры
        self.running = False  # Флаг камеры
        self.predicting = False  # Флаг предсказаний
#
        self._build_ui()  # Сборка интерфейса
#
    # Создает элементы интерфейса. 
    def _build_ui(self):  # UI построение
        ctrl = ttk.Frame(self.root, padding=5)  # Панель
        ctrl.pack(fill=tk.X)  # Растяжение по X
#
        ttk.Label(ctrl, text="API:").pack(  # Метка API
            side=tk.LEFT,  # Слева
            padx=(0, 5),  # Отступ
        )
        self.api_var = tk.StringVar(  # Переменная URL
            value="http://127.0.0.1:8000",  # URL
        )
        ttk.Entry(  # Поле ввода
            ctrl,  # Родитель
            textvariable=self.api_var,  # Переменная
            width=24,  # Ширина
        ).pack(  # Размещение
            side=tk.LEFT,  # Слева
            padx=(0, 10),  # Отступ
        )
#
        ttk.Label(ctrl, text="Камера:").pack(  # Метка камеры
            side=tk.LEFT,  # Слева
            padx=(0, 5),  # Отступ
        )
        self.cam_var = tk.StringVar(value="0")  # Индекс камеры
        ttk.Combobox(  # Выпадающий список
            ctrl,  # Родитель
            textvariable=self.cam_var,  # Переменная
            width=6,  # Ширина
            values=[0, 1, 2],  # Список
        ).pack(  # Размещение
            side=tk.LEFT,  # Слева
            padx=(0, 5),  # Отступ
        )
#
        ttk.Button(  # Кнопка камеры
            ctrl,  # Родитель
            text="Камера",  # Текст
            command=self._toggle_camera,  # Обработчик
        ).pack(  # Размещение
            side=tk.LEFT,  # Слева
            padx=2,  # Отступ
        )
        ttk.Button(  # Кнопка предсказания
            ctrl,  # Родитель
            text="Предсказание",  # Текст
            command=self._toggle_predict,  # Обработчик
        ).pack(  # Размещение
            side=tk.LEFT,  # Слева
            padx=2,  # Отступ
        )
#
        self.video_label = tk.Label(self.root)  # Виджет видео
        self.video_label.pack(pady=5)  # Отступ
#
        self.direction_label = tk.Label(  # Метка направления
            self.root,  # Родитель
            text="Направление: ---",  # Текст
            font=("Arial", 18),  # Шрифт
        )
        self.direction_label.pack()  # Размещение
#
        self.coords_label = tk.Label(  # Метка координат
            self.root,  # Родитель
            text="H: ---°  V: ---°",  # Текст
            font=("Consolas", 11),  # Шрифт
        )
        self.coords_label.pack()  # Размещение
#
        self.canvas = tk.Canvas(  # Канвас
            self.root,  # Родитель
            width=160,  # Ширина
            height=160,  # Высота
            bg="#f5f5f5",  # Фон
        )
        self.canvas.pack(pady=5)  # Отступ
        self._draw_grid()  # Сетка
#
    # Рисует координатную сетку. 
    def _draw_grid(self):  # Сетка
        c = self.canvas  # Канвас
        c.delete("all")  # Очистка
        w, h = 160, 160  # Размеры
        cx, cy = w // 2, h // 2  # Центр
        c.create_line(0, cy, w, cy, fill="#ccc")  # Горизонталь
        c.create_line(cx, 0, cx, h, fill="#ccc")  # Вертикаль
        c.create_text(cx, 12, text="V↑", font=("Arial", 9), fill="#888")  # V
        text_h = "H→"  # Подпись H
        c.create_text(  # Текст H
            w - 8,  # X
            cy,  # Y
            text=text_h,  # Текст
            font=("Arial", 9),  # Шрифт
            fill="#888",  # Цвет
        )
#
    # Рисует точку направления взгляда. 
    def _draw_point(self, h_deg: float, v_deg: float):  # Точка
        c = self.canvas  # Канвас
        c.delete("point")  # Удаление точки
        w, h = 160, 160  # Размеры
        cx, cy = w // 2, h // 2  # Центр
        scale = 2.0  # Масштаб
        x = cx + h_deg * scale  # X
        y = cy - v_deg * scale  # Y
        r = 6  # Радиус
        c.create_oval(  # Овал
            x - r,  # X1
            y - r,  # Y1
            x + r,  # X2
            y + r,  # Y2
            fill="#2196F3",  # Цвет
            tags="point",  # Тег
        )
#
    # Запускает или останавливает камеру. 
    def _toggle_camera(self):  # Камера
        if self.running:  # Если работает
            self.running = False  # Остановить
            if self.cap:  # Если есть объект
                self.cap.release()  # Освобождение
                self.cap = None  # Сброс
        else:  # Иначе включить
            cid = int(self.cam_var.get())  # Индекс
            self.cap = cv2.VideoCapture(cid, CAP_BACKEND)  # Открыть
            if not self.cap.isOpened():  # Проверка
                self.cap = cv2.VideoCapture(cid, 0)  # Бэкенд 0
            self.running = True  # Флаг
        self._update()  # Запуск цикла
#
    # Запускает или останавливает предсказание. 
    def _toggle_predict(self):  # Предсказание
        self.predicting = not self.predicting  # Флаг
#
    # Цикл обновления кадров. 
    def _update(self):  # Обновление
        cam_ok = (  # Камера
            self.running  # Флаг
            and self.cap  # Объект
            and self.cap.isOpened()  # Статус
        )
        if not cam_ok:  # Нет камеры
            self.root.after(500, self._update)  # Повтор
            return  # Выход
#
        ret, frame = self.cap.read()  # Кадр
        if not ret:  # Если не прочитан
            self.root.after(30, self._update)  # Повтор
            return  # Выход
#
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR->RGB
        direction_text = "---"  # Текст направления
#
        if self.predicting:  # Если включено
            res = predict_via_api(self.api_var.get().strip(), rgb)  # Запрос
            h_pred = res["h_deg"]  # H
            v_pred = res["v_deg"]  # V
            zone = res["zone"]  # Зона
            direction_text = DIRECTION_RU.get(zone, zone)  # Текст зоны
            msg = f"H: {h_pred:+.1f}°  V: {v_pred:+.1f}°"  # Текст
            self.coords_label.config(text=msg)  # Обновление
            self._draw_point(h_pred, v_pred)  # Точка
#
        img = Image.fromarray(rgb).resize((640, 480))  # PIL кадр
        self.video_label.imgtk = ImageTk.PhotoImage(image=img)  # Tk image
        self.video_label.config(image=self.video_label.imgtk)  # Показ
        msg = f"Направление: {direction_text}"  # Текст
        self.direction_label.config(text=msg)  # Направление
#
        self.root.after(30, self._update)  # Следующий цикл
#
    # Запускает GUI цикл. 
    def run(self):  # Запуск
        self.root.mainloop()  # Главный цикл
        if self.cap:  # Если камера была
            self.cap.release()  # Освобождение
#
#
# Запускает приложение. 
def main():  # Точка входа
    app = SimpleGazeApp()  # Создание приложения
    app.run()  # Запуск
#
#
if __name__ == "__main__":  # Запуск как скрипта
    main()  # Вызов main

