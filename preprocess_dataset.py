import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Импортируем функции извлечения признаков и преобразования углов взгляда в зону (из вашего файла features.py).
from features import extract_eye_features, gaze_to_zone

# Ширина, до которой приводим все изображения перед детектом (для унификации размеров).
RESIZE_W = 1280
# Список колонок признаков f0..f10 (всего 11 признаков).
FEATURE_COLS = [f"f{i}" for i in range(11)]
# Список числовых колонок: признаки + углы H, V, P.
NUM_COLS = FEATURE_COLS + ["H", "V", "P"]


# Функция: парсит имя файла Columbia и извлекает углы P/V/H из шаблона имени.
def parse_columbia(name: str):
    # Пытаемся сопоставить имя файла с ожидаемым шаблоном и извлечь 3 группы: P, V, H.
    m = re.match(r"^\d+_\d+m_(-?\d+)P_(-?\d+)V_(-?\d+)H\.jpg$", name, re.I)
    # Если имя файла не соответствует шаблону — возвращаем None (метки не найдены).
    if not m:
        return None
    # Преобразуем извлечённые группы в int: pitch (P), vertical (V), horizontal (H).
    p, v, h = int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Возвращаем кортеж меток (P, V, H).
    return p, v, h


# Функция: создаёт и возвращает детектор FaceLandmarker по пути к модели.
def make_detector(model_path: str):
    # Создаём базовые опции MediaPipe, указывая путь к файлу модели.
    base = python.BaseOptions(model_asset_path=model_path)
    # Создаём опции FaceLandmarker: передаём base_options и ограничиваемся одним лицом.
    opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    # Создаём экземпляр FaceLandmarker по заданным опциям и возвращаем его.
    return vision.FaceLandmarker.create_from_options(opts)


# Функция: проходит по датасету, извлекает метки из имён и признаки из лица, собирает DataFrame строк.
def collect_rows(dataset_root: str, model_path: str):
    # Приводим корень датасета к объекту Path.
    root = Path(dataset_root)
    # Находим все папки-персоны (директории с цифровыми именами) и сортируем их.
    persons = sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()])
    # Создаём детектор по указанному пути к модели.
    det = make_detector(model_path)

    # Список, куда будем складывать строки (каждая строка = один файл/пример).
    rows = []
    # Счётчик пропущенных/невалидных файлов.
    skipped = 0

    # Идём по каждой папке персоны.
    for d in persons:
        # ID персоны = имя папки.
        pid = d.name
        # Идём по всем jpg в папке персоны.
        for img_path in d.glob("*.jpg"):
            # Парсим метки (P, V, H) из имени файла.
            lab = parse_columbia(img_path.name)
            # Если метки не распарсились — считаем файл пропущенным и идём дальше.
            if not lab:
                skipped += 1
                continue
            # Распаковываем метки.
            p, v, h = lab

            # Считываем изображение с диска через OpenCV.
            img = cv2.imread(str(img_path))
            # Если OpenCV не смог прочитать изображение — пропускаем.
            if img is None:
                skipped += 1
                continue

            # Берём высоту и ширину исходного изображения.
            hh, ww = img.shape[:2]
            # Считаем коэффициент масштабирования к RESIZE_W (защита от деления на 0 через max(1, ww)).
            scale = RESIZE_W / max(1, ww)
            # Ресайзим изображение до фиксированной ширины, сохраняя пропорции по высоте.
            img = cv2.resize(img, (RESIZE_W, int(hh * scale)), interpolation=cv2.INTER_LANCZOS4)
            # Конвертируем BGR (OpenCV) -> RGB и делаем массив contiguous для MediaPipe.
            rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Оборачиваем RGB-массив в объект mp.Image с указанием формата SRGB.
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Запускаем детектор по изображению и получаем результат.
            res = det.detect(mp_img)
            # Если лицо/лендмарки не найдены — пропускаем пример.
            if not res.face_landmarks:
                skipped += 1
                continue

            # Извлекаем признаки глаз из лендмарков первого (и единственного) лица, передавая head_pose=p.
            feats = extract_eye_features(res.face_landmarks[0], head_pose=p)
            # Если извлечение признаков не удалось — пропускаем.
            if feats is None:
                skipped += 1
                continue

            # Формируем базовую часть строки (метаданные + метки + зона взгляда).
            row = {
                # Записываем ID персоны.
                "person_id": pid,
                # Записываем имя файла.
                "filename": img_path.name,
                # Записываем горизонтальный угол взгляда.
                "H": h,
                # Записываем вертикальный угол взгляда.
                "V": v,
                # Записываем pitch/позу головы.
                "P": p,
                # Переводим (H,V) в категориальную зону взгляда.
                "zone": gaze_to_zone(h, v),
            }
            # Заполняем признаки f0..f10 из feats["features"].
            for i, val in enumerate(feats["features"]):
                # Кладём конкретный признак в соответствующую колонку.
                row[f"f{i}"] = val
            # Добавляем готовую строку в общий список.
            rows.append(row)

        # Печатаем прогресс по персоне (что папка обработана).
        print(f"{pid}: ok")

    # Превращаем список словарей в DataFrame.
    df = pd.DataFrame(rows)
    # Печатаем количество строк и сколько примеров пропущено.
    print(f"rows={len(df)}, skipped={skipped}")
    # Возвращаем DataFrame.
    return df


# Функция: удаляет выбросы по IQR для указанных колонок (по умолчанию H и V).
def remove_outliers_iqr(df: pd.DataFrame, cols=("H", "V")):
    # Создаём маску True для всех строк (будем постепенно сужать).
    mask = pd.Series(True, index=df.index)
    # Проходим по каждой колонке, по которой фильтруем выбросы.
    for c in cols:
        # Считаем 25-й перцентиль (Q1).
        q1 = df[c].quantile(0.25)
        # Считаем 75-й перцентиль (Q3).
        q3 = df[c].quantile(0.75)
        # IQR = Q3 - Q1.
        iqr = q3 - q1
        # Нижняя граница = Q1 - 1.5*IQR.
        lo = q1 - 1.5 * iqr
        # Верхняя граница = Q3 + 1.5*IQR.
        hi = q3 + 1.5 * iqr
        # Обновляем маску: оставляем только строки, где значение в пределах [lo, hi].
        mask &= df[c].between(lo, hi)
    # Возвращаем отфильтрованный DataFrame.
    return df[mask]


# Функция: чистит DataFrame — удаляет NaN/дубликаты и выбросы по H/V.
def clean_df(df: pd.DataFrame):
    # Определяем ключевые колонки, которые должны быть заполнены (id, углы, и 11 признаков).
    key_cols = ["person_id", "H", "V", "P"] + FEATURE_COLS
    # Удаляем строки, где есть NaN хотя бы в одной ключевой колонке.
    df = df.dropna(subset=key_cols)
    # Удаляем полные дубликаты строк.
    df = df.drop_duplicates()
    # Запоминаем количество строк до удаления выбросов.
    before = len(df)
    # Удаляем выбросы по IQR для H и V.
    df = remove_outliers_iqr(df, ("H", "V"))
    # Печатаем, сколько строк удалили как выбросы.
    print(f"outliers removed: {before - len(df)}")
    # Возвращаем очищенный DataFrame.
    return df


# Функция: строит и сохраняет корреляционную матрицу и scatter(H,V) в указанную папку.
def analysis_plots(df: pd.DataFrame, out_dir: str):
    # Приводим out_dir к Path.
    out = Path(out_dir)
    # Создаём папку (и родителей), если её нет.
    out.mkdir(parents=True, exist_ok=True)

    # Берём копию данных, где нет NaN в числовых колонках, чтобы можно было считать корреляции.
    d = df.dropna(subset=NUM_COLS).copy()
    # Если после фильтрации данных нет — печатаем сообщение и выходим.
    if d.empty:
        print("Нет данных для анализа")
        return

    # Считаем корреляционную матрицу для числовых колонок.
    corr = d[NUM_COLS].corr()
    # Создаём фигуру указанного размера под heatmap корреляций.
    plt.figure(figsize=(10, 8))
    # Рисуем матрицу как изображение, задавая цветовую карту и диапазон [-1, 1].
    plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    # Добавляем цветовую шкалу справа.
    plt.colorbar(shrink=0.8)
    # Подписываем ось X названиями колонок, поворачиваем для читаемости.
    plt.xticks(range(len(NUM_COLS)), NUM_COLS, rotation=45, ha="right")
    # Подписываем ось Y названиями колонок.
    plt.yticks(range(len(NUM_COLS)), NUM_COLS)
    # Уплотняем layout, чтобы подписи не обрезались.
    plt.tight_layout()
    # Сохраняем картинку корреляций в файл corr.png.
    plt.savefig(out / "corr.png", dpi=120)
    # Закрываем фигуру, чтобы не копить память.
    plt.close()

    # Печатаем, куда сохранили анализ.
    print(f"analysis saved: {out}")


# Функция: точка входа — выбирает папку датасета/модель, собирает данные, чистит, сохраняет CSV и графики.
def main():
    # Определяем базовую папку как папку, где лежит этот скрипт.
    base = Path(__file__).parent
    # Пытаемся использовать датасет в папке "Columbia Gaze Data Set" рядом со скриптом.
    dataset_root = str(base / "Columbia Gaze Data Set")
    # Если такой папки нет — пробуем альтернативное имя "Columbia Gaze Resized".
    if not Path(dataset_root).exists():
        dataset_root = str(base / "Columbia Gaze Resized")

    # Определяем путь к модели face_landmarker.task рядом со скриптами.
    model_path = str(base / "face_landmarker.task")
    # Если файла модели нет — падаем с понятной ошибкой.
    if not Path(model_path).exists():
        raise FileNotFoundError("face_landmarker.task положи рядом со скриптами")

    # Папка вывода — out рядом со скриптом.
    out_dir = base / "out"
    # Создаём папку out, если её нет.
    out_dir.mkdir(exist_ok=True)

    # Собираем DataFrame из датасета (чтение изображений, детект, признаки, метки).
    df = collect_rows(dataset_root, model_path)
    # Если DataFrame пустой — печатаем и выходим.
    if df.empty:
        print("empty df")
        return

    # Чистим DataFrame (NaN/дубликаты/выбросы).
    df = clean_df(df)
    # Путь для сохранения CSV.
    csv_path = out_dir / "gaze_dataset.csv"
    # Сохраняем DataFrame в CSV без индекса.
    df.to_csv(csv_path, index=False)
    # Печатаем путь и количество строк.
    print(f"saved: {csv_path} ({len(df)})")

    # Строим и сохраняем графики анализа в out/analysis.
    analysis_plots(df, str(out_dir / "analysis"))

    # Печатаем распределение зон взгляда.
    print("zones:", df["zone"].value_counts().to_dict())
    # Печатаем финальное сообщение.
    print("done")


# Стандартная проверка: если файл запущен как скрипт — вызываем main().
if __name__ == "__main__":
    # Запускаем основную функцию.
    main()
