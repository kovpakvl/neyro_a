# Инвариантные признаки для оценки направления взгляда.
# Координаты зрачка относительно глаза — устойчивы к масштабу и сдвигу камеры.

import numpy as np

# MediaPipe Face Mesh: индексы точек глаз
# Индексы landmark-точек для левого глаза: верх, низ, левый край, правый край, зрачок.
LEFT_EYE = {'top': 159, 'bottom': 145, 'left': 33, 'right': 133, 'pupil': 468}
# Индексы landmark-точек для правого глаза: верх, низ, левый край, правый край, зрачок.
RIGHT_EYE = {'top': 386, 'bottom': 374, 'left': 362, 'right': 263, 'pupil': 473}

# Точки для оценки наклона головы (нос, подбородок, виски)
# Индекс точки кончика носа (используется как ориентир смещения по X).
NOSE_TIP = 1
# Индекс точки подбородка (здесь задан, но в функции оценки yaw не используется напрямую).
CHIN = 152
# Индекс левого "виска" (левая граница лица).
LEFT_TEMPLE = 234
# Индекс правого "виска" (правая граница лица).
RIGHT_TEMPLE = 454


# Функция: оценивает поворот головы по горизонтали (yaw) в градусах по landmarks.
def _estimate_head_pose(landmarks):
    """
    Оценка угла поворота головы (yaw) в градусах из landmarks.
    Нужна при инференсе, когда нет метки P из датасета.
    """
    # Если landmarks отсутствуют или их меньше, чем нужно для доступа к ключевым индексам — возвращаем 0.
    if landmarks is None or len(landmarks) < 455:
        return 0.0
    # Берём landmark носа по индексу NOSE_TIP.
    nose = landmarks[NOSE_TIP]
    # Берём landmark левого виска по индексу LEFT_TEMPLE.
    left_t = landmarks[LEFT_TEMPLE]
    # Берём landmark правого виска по индексу RIGHT_TEMPLE.
    right_t = landmarks[RIGHT_TEMPLE]
    # Вычисляем "ширину лица" как разницу координат x между правым и левым виском.
    face_w = abs(right_t.x - left_t.x)
    # Если ширина лица слишком мала (почти 0) — возвращаем 0, чтобы избежать деления на 0.
    if face_w < 1e-6:
        return 0.0
    # Находим центр лица по X как среднее между x левого и правого виска.
    center_x = (left_t.x + right_t.x) / 2
    # Считаем нормализованное смещение носа от центра (в долях половины ширины лица).
    offset = (nose.x - center_x) / (face_w / 2 + 1e-9)
    # Преобразуем смещение в градусы (масштабируем на 15) и ограничиваем диапазоном [-15; 15].
    return np.clip(offset * 15.0, -15.0, 15.0)


# Функция: извлекает признаки одного глаза (положение зрачка + геометрия глаза) по landmarks и индексам eye_points.
def _extract_eye_features(landmarks, eye_points):
    """Признаки одного глаза: x_rel, y_rel зрачка + aspect, width, height."""
    # Берём landmark левой точки глаза (край) по индексу eye_points['left'].
    left_pt = landmarks[eye_points['left']]
    # Берём landmark правой точки глаза (край) по индексу eye_points['right'].
    right_pt = landmarks[eye_points['right']]
    # Берём landmark верхней точки глаза по индексу eye_points['top'].
    top_pt = landmarks[eye_points['top']]
    # Берём landmark нижней точки глаза по индексу eye_points['bottom'].
    bottom_pt = landmarks[eye_points['bottom']]
    # Берём landmark зрачка по индексу eye_points['pupil'].
    pupil = landmarks[eye_points['pupil']]

    # Считаем вектор от левого края глаза к правому по x и y.
    dx, dy = right_pt.x - left_pt.x, right_pt.y - left_pt.y
    # Длина этого вектора = "ширина глаза" (евклидова).
    eye_width = np.hypot(dx, dy)
    # Если ширина глаза слишком мала — возвращаем None (признаки посчитать нормально нельзя).
    if eye_width < 1e-6:
        return None

    # Находим центр глаза по X как среднее между левым и правым краем.
    cx = (left_pt.x + right_pt.x) / 2
    # Находим центр глаза по Y как среднее между левым и правым краем.
    cy = (left_pt.y + right_pt.y) / 2
    # Оцениваем "высоту глаза" как среднее расстояние от центра до верхней и нижней точек.
    eye_height = (
        # Расстояние от верхней точки до центра.
        np.hypot(top_pt.x - cx, top_pt.y - cy)
        # Плюс расстояние от нижней точки до центра.
        + np.hypot(bottom_pt.x - cx, bottom_pt.y - cy)
    ) / 2  # Делим на 2, чтобы получить среднее.

    # Вектор от левого края глаза к зрачку по x и y.
    pupil_dx, pupil_dy = pupil.x - left_pt.x, pupil.y - left_pt.y
    # x_rel = проекция вектора (левый край -> зрачок) на ось глаза (левый край -> правый край), нормированная.
    x_rel = (pupil_dx * dx + pupil_dy * dy) / (dx * dx + dy * dy + 1e-9)

    # Вектор от верхней точки глаза к нижней по x и y (вертикальная ось глаза).
    tx, ty = bottom_pt.x - top_pt.x, bottom_pt.y - top_pt.y
    # Вектор от верхней точки глаза к зрачку по x и y.
    pupil_tx, pupil_ty = pupil.x - top_pt.x, pupil.y - top_pt.y
    # y_rel = проекция (верх -> зрачок) на (верх -> низ), нормированная.
    y_rel = (pupil_tx * tx + pupil_ty * ty) / (tx * tx + ty * ty + 1e-9)

    # aspect = отношение высоты глаза к ширине (показатель "открытости" глаза).
    aspect = eye_height / (eye_width + 1e-9)
    # Возвращаем список признаков: положение зрачка (x_rel, y_rel) и геометрию (aspect, width, height).
    return [x_rel, y_rel, aspect, eye_width, eye_height]


# Функция: формирует общий вектор признаков для обоих глаз + head_pose (нормализованный).
def extract_eye_features(landmarks, head_pose=None):
    """
    Вектор признаков для обоих глаз.
    head_pose: градусы (-30..30) или None — тогда оценивается из landmarks.
    """
    # Если landmarks отсутствуют или их меньше, чем нужно для индексов зрачков — возвращаем None.
    if landmarks is None or len(landmarks) < 474:
        return None

    # Извлекаем признаки левого глаза.
    left_f = _extract_eye_features(landmarks, LEFT_EYE)
    # Извлекаем признаки правого глаза.
    right_f = _extract_eye_features(landmarks, RIGHT_EYE)
    # Если хотя бы один глаз не удалось обработать — возвращаем None.
    if left_f is None or right_f is None:
        return None

    # Если head_pose не задан — оцениваем yaw головы по landmarks.
    if head_pose is None:
        head_pose = _estimate_head_pose(landmarks)
    # Нормализуем head_pose: делим на 30 и ограничиваем в диапазон [-1; 1].
    hp = np.clip(head_pose / 30.0, -1.0, 1.0)

    # Склеиваем признаки: 5 левого + 5 правого + 1 head_pose => 11 признаков.
    features = left_f + right_f + [hp]
    # Возвращаем словарь с отдельными блоками и финальным вектором.
    return {'left': left_f, 'right': right_f, 'head_pose': hp, 'features': features}


# Функция: переводит углы (H, V) в категориальную "зону взгляда" из 9 вариантов.
def gaze_to_zone(h_deg, v_deg, center_th=5.0, diag_th=4.0):
    """
    Углы H, V (градусы) -> 9 зон: center, left, right, up, down,
    left-up, left-down, right-up, right-down.
    """
    # Если оба угла близки к 0 (в пределах порога центра) — это центр.
    if abs(h_deg) < center_th and abs(v_deg) < center_th:
        return 'center'
    # Диагонали: обе компоненты должны быть достаточно значимы.
    if abs(h_deg) >= diag_th and abs(v_deg) >= diag_th:
        # Определяем горизонтальную сторону по знаку H.
        h_side = 'left' if h_deg < 0 else 'right'
        # Определяем вертикальную сторону по знаку V (положительное = вверх).
        v_side = 'up' if v_deg > 0 else 'down'
        # Возвращаем комбинированную диагональную зону.
        return f'{h_side}-{v_side}'
    # Только горизонталь: если |H| >= |V|, доминирует горизонтальное направление.
    if abs(h_deg) >= abs(v_deg):
        return 'left' if h_deg < 0 else 'right'
    # Только вертикаль: иначе доминирует вертикальное направление.
    return 'up' if v_deg > 0 else 'down'
