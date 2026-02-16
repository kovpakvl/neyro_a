"""
Обучение модели оценки взгляда.
Регрессия H, V по инвариантным признакам. GroupKFold по person_id.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

FEATURE_COLS = [f'f{i}' for i in range(11)]
TARGET_H, TARGET_V = 'H', 'V'
N_FOLDS = 5
TUNE_FOLDS = 3
TUNE_ITER = 8
CONTROL_N = 10
CONTROL_TOL = 7.0  # допускаемая ошибка в градусах

PARAM_SPACE = {
    'n_estimators': [80, 120, 160],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': [1.0, 'sqrt', 0.5],
}


def load_data(csv_paths, undersample_center=0.5, oversample_down_up=2, random_state=42):
    """
    Загружает один или несколько CSV. undersample center, oversample down/up.
    random_state — для воспроизводимости выборки.
    """
    dfs = []
    for p in csv_paths:
        if Path(p).exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        return None, None, None, None, None

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_H, TARGET_V])
    df['person_id'] = df['person_id'].astype(str)  # иначе GroupKFold падает при смеси типов

    rng = np.random.default_rng(random_state)
    if undersample_center < 1.0 and 'zone' in df.columns:
        center = df['zone'] == 'center'
        n = center.sum()
        keep = int(n * undersample_center)
        if n > keep:
            drop_idx = rng.choice(df.index[center].tolist(), size=n - keep, replace=False)
            df = df.drop(drop_idx)

    if oversample_down_up > 1 and 'zone' in df.columns:
        du = df[df['zone'].astype(str).str.contains('down|up', regex=True, na=False)]
        if len(du) > 0:
            extra = du.sample(n=len(du) * (oversample_down_up - 1), replace=True, random_state=random_state)
            df = pd.concat([df, extra], ignore_index=True)

    X = df[FEATURE_COLS].values
    y_h, y_v = df[TARGET_H].values, df[TARGET_V].values
    groups = df['person_id'].values
    return X, y_h, y_v, groups, df


def _tune_model(X, y, groups, seed=42):
    """Быстрый подбор гиперпараметров по MAE."""
    gkf = GroupKFold(n_splits=TUNE_FOLDS)
    base = RandomForestRegressor(random_state=seed)
    search = RandomizedSearchCV(
        base, PARAM_SPACE, n_iter=TUNE_ITER, scoring='neg_mean_absolute_error',
        cv=gkf, n_jobs=-1, random_state=seed
    )
    search.fit(X, y, groups=groups)
    return search.best_params_, -search.best_score_


def _cv_metrics(X, y_h, y_v, groups, params_h, params_v, n_folds=N_FOLDS):
    """MAE для train/val, используется для графиков качества."""
    gkf = GroupKFold(n_splits=n_folds)
    h_tr, h_val, v_tr, v_val = [], [], [], []
    for fold, (tr, val) in enumerate(gkf.split(X, y_h, groups)):
        m_h = RandomForestRegressor(random_state=42, **params_h)
        m_v = RandomForestRegressor(random_state=42, **params_v)
        m_h.fit(X[tr], y_h[tr])
        m_v.fit(X[tr], y_v[tr])
        h_tr.append(mean_absolute_error(y_h[tr], m_h.predict(X[tr])))
        v_tr.append(mean_absolute_error(y_v[tr], m_v.predict(X[tr])))
        h_val.append(mean_absolute_error(y_h[val], m_h.predict(X[val])))
        v_val.append(mean_absolute_error(y_v[val], m_v.predict(X[val])))
        print(f"  Fold {fold+1}: MAE H={h_val[-1]:.2f}° V={v_val[-1]:.2f}°")
    return h_tr, h_val, v_tr, v_val


def _plot_cv(out_dir, h_tr, h_val, v_tr, v_val):
    """Сохраняет график качества train/val."""
    x = np.arange(1, len(h_tr) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(x, h_tr, 'o-', label='H train')
    plt.plot(x, h_val, 'o-', label='H val')
    plt.plot(x, v_tr, 'o-', label='V train')
    plt.plot(x, v_val, 'o-', label='V val')
    plt.xlabel('Fold')
    plt.ylabel('MAE (degrees)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = Path(out_dir) / 'quality_cv.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"График качества сохранен: {out_path}")


def _control_tests(model_h, model_v, X, y_h, y_v, n=CONTROL_N, tol=CONTROL_TOL):
    """Контрольные тесты: n фиксированных примеров, успех если |err| <= tol."""
    rnd = np.random.RandomState(42)
    idx = rnd.choice(len(X), size=min(n, len(X)), replace=False)
    pred_h = model_h.predict(X[idx])
    pred_v = model_v.predict(X[idx])
    ok = (np.abs(pred_h - y_h[idx]) <= tol) & (np.abs(pred_v - y_v[idx]) <= tol)
    print(f"Контрольные тесты: {ok.sum()}/{len(idx)} (tol={tol}°)")


def train(csv_paths=None, output_dir='.', n_folds=N_FOLDS,
          undersample_center=0.5, oversample_down_up=2):
    """
    Обучает регрессоры H и V. Сохраняет scaler, model_h, model_v.
    """
    base = Path(__file__).parent
    default_paths = [
        base / 'gaze_dataset.csv',
        base / 'шаблон' / 'gaze_dataset.csv',
    ]
    paths = csv_paths or [str(p) for p in default_paths]

    result = load_data(paths, undersample_center, oversample_down_up)
    if result[0] is None:
        print("Нет CSV. Запустите preprocess_dataset.py")
        return

    X, y_h, y_v, groups, df = result
    np.random.seed(42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Подбор гиперпараметров (H)...")
    params_h, score_h = _tune_model(X_scaled, y_h, groups)
    print(f"  Лучшее MAE (H): {score_h:.2f}°, params={params_h}")
    print("Подбор гиперпараметров (V)...")
    params_v, score_v = _tune_model(X_scaled, y_v, groups)
    print(f"  Лучшее MAE (V): {score_v:.2f}°, params={params_v}")

    h_tr, h_val, v_tr, v_val = _cv_metrics(X_scaled, y_h, y_v, groups, params_h, params_v, n_folds)
    print(f"Среднее MAE: H={np.mean(h_val):.2f}° V={np.mean(v_val):.2f}°")

    # Вес для down/up
    w = np.ones(len(y_v))
    if 'zone' in df.columns:
        w[df['zone'].astype(str).str.contains('down|up', regex=True, na=False)] = 1.5

    model_h = RandomForestRegressor(random_state=42, **params_h)
    model_v = RandomForestRegressor(random_state=42, **params_v)
    model_h.fit(X_scaled, y_h, sample_weight=w)
    model_v.fit(X_scaled, y_v, sample_weight=w)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _plot_cv(out, h_tr, h_val, v_tr, v_val)
    _control_tests(model_h, model_v, X_scaled, y_h, y_v)
    joblib.dump(scaler, out / 'gaze_scaler.pkl')
    joblib.dump(model_h, out / 'gaze_model_h.pkl')
    joblib.dump(model_v, out / 'gaze_model_v.pkl')
    print(f"Модели сохранены в {out}")


if __name__ == "__main__":
    train()
