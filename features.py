"""
Инвариантные признаки для оценки направления взгляда.
Координаты зрачка относительно глаза — устойчивы к масштабу и сдвигу камеры.
"""
import numpy as np

# MediaPipe Face Mesh: индексы точек глаз
LEFT_EYE = {'top': 159, 'bottom': 145, 'left': 33, 'right': 133, 'pupil': 468}
RIGHT_EYE = {'top': 386, 'bottom': 374, 'left': 362, 'right': 263, 'pupil': 473}

# Точки для оценки наклона головы (нос, подбородок, виски)
NOSE_TIP = 1
CHIN = 152
LEFT_TEMPLE = 234
RIGHT_TEMPLE = 454


def _estimate_head_pose(landmarks):
    """
    Оценка угла поворота головы (yaw) в градусах из landmarks.
    Нужна при инференсе, когда нет метки P из датасета.
    """
    if landmarks is None or len(landmarks) < 455:
        return 0.0
    nose = landmarks[NOSE_TIP]
    left_t = landmarks[LEFT_TEMPLE]
    right_t = landmarks[RIGHT_TEMPLE]
    face_w = abs(right_t.x - left_t.x)
    if face_w < 1e-6:
        return 0.0
    center_x = (left_t.x + right_t.x) / 2
    offset = (nose.x - center_x) / (face_w / 2 + 1e-9)
    return np.clip(offset * 15.0, -15.0, 15.0)


def _extract_eye_features(landmarks, eye_points):
    """Признаки одного глаза: x_rel, y_rel зрачка + aspect, width, height."""
    left_pt = landmarks[eye_points['left']]
    right_pt = landmarks[eye_points['right']]
    top_pt = landmarks[eye_points['top']]
    bottom_pt = landmarks[eye_points['bottom']]
    pupil = landmarks[eye_points['pupil']]

    dx, dy = right_pt.x - left_pt.x, right_pt.y - left_pt.y
    eye_width = np.hypot(dx, dy)
    if eye_width < 1e-6:
        return None

    cx = (left_pt.x + right_pt.x) / 2
    cy = (left_pt.y + right_pt.y) / 2
    eye_height = (
        np.hypot(top_pt.x - cx, top_pt.y - cy)
        + np.hypot(bottom_pt.x - cx, bottom_pt.y - cy)
    ) / 2

    pupil_dx, pupil_dy = pupil.x - left_pt.x, pupil.y - left_pt.y
    x_rel = (pupil_dx * dx + pupil_dy * dy) / (dx * dx + dy * dy + 1e-9)

    tx, ty = bottom_pt.x - top_pt.x, bottom_pt.y - top_pt.y
    pupil_tx, pupil_ty = pupil.x - top_pt.x, pupil.y - top_pt.y
    y_rel = (pupil_tx * tx + pupil_ty * ty) / (tx * tx + ty * ty + 1e-9)

    aspect = eye_height / (eye_width + 1e-9)
    return [x_rel, y_rel, aspect, eye_width, eye_height]


def extract_eye_features(landmarks, head_pose=None):
    """
    Вектор признаков для обоих глаз.
    head_pose: градусы (-30..30) или None — тогда оценивается из landmarks.
    """
    if landmarks is None or len(landmarks) < 474:
        return None

    left_f = _extract_eye_features(landmarks, LEFT_EYE)
    right_f = _extract_eye_features(landmarks, RIGHT_EYE)
    if left_f is None or right_f is None:
        return None

    if head_pose is None:
        head_pose = _estimate_head_pose(landmarks)
    hp = np.clip(head_pose / 30.0, -1.0, 1.0)

    features = left_f + right_f + [hp]
    return {'left': left_f, 'right': right_f, 'head_pose': hp, 'features': features}


def gaze_to_zone(h_deg, v_deg, center_th=5.0, diag_th=4.0):
    """
    Углы H, V (градусы) -> 9 зон: center, left, right, up, down,
    left-up, left-down, right-up, right-down.
    """
    if abs(h_deg) < center_th and abs(v_deg) < center_th:
        return 'center'
    # Диагонали: обе компоненты значимы
    if abs(h_deg) >= diag_th and abs(v_deg) >= diag_th:
        h_side = 'left' if h_deg < 0 else 'right'
        v_side = 'up' if v_deg > 0 else 'down'
        return f'{h_side}-{v_side}'
    # Только горизонталь
    if abs(h_deg) >= abs(v_deg):
        return 'left' if h_deg < 0 else 'right'
    # Только вертикаль
    return 'up' if v_deg > 0 else 'down'
