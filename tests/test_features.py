# tests/test_features.py
from features import gaze_to_zone, extract_eye_features


def test_gaze_to_zone_center():
    # H=0, V=0 должны попасть в центр
    assert gaze_to_zone(0.0, 0.0) == "center"
    # чуть внутри порога тоже центр
    assert gaze_to_zone(4.9, -4.9) == "center"


def test_gaze_to_zone_diagonal():
    # Диагональ: H<0 (left), V>0 (up)
    assert gaze_to_zone(-10.0, 10.0) == "left-up"
    # Диагональ: H>0 (right), V<0 (down)
    assert gaze_to_zone(10.0, -10.0) == "right-down"


def test_extract_eye_features_bad_landmarks():
    # Если landmarks None — функция должна вернуть None (защита)
    assert extract_eye_features(None) is None

    # Если landmarks слишком мало (len < 474) — тоже None
    assert extract_eye_features([object()] * 10) is None

