# Обучение модели взгляда. # Описание модуля
from pathlib import Path  # Пути файловой системы
# 
import joblib  # Сериализация моделей
import matplotlib.pyplot as plt  # Визуализация метрик
import numpy as np  # Численные операции
import pandas as pd  # Табличные данные
from sklearn.metrics import mean_absolute_error  # Метрика ошибки
from sklearn.model_selection import GroupKFold  # Кросс-валидация
from sklearn.neural_network import MLPRegressor  # Нейросетевой регрессор
from sklearn.preprocessing import StandardScaler  # Масштабирование
# 
FEATURE_COLS = [f"f{i}" for i in range(11)]  # Список признаков
# 
# Загружает CSV и готовит массивы для обучения. 
def load_csv(path: str):  # Чтение CSV
    df = pd.read_csv(path)  # Загружаем файл
    cols = FEATURE_COLS + ["H", "V", "person_id"]  # Колонки
    df = df.dropna(subset=cols)  # Удаляем NaN
    X = df[FEATURE_COLS].values  # Матрица признаков
    y_h = df["H"].values  # Цель H
    y_v = df["V"].values  # Цель V
    g = df["person_id"].astype(str).values  # Группы
    return X, y_h, y_v, g  # Возвращаем данные
# 
# Считает метрики и сохраняет график по фолдам. 
def cv_plot(Xs, y_h, y_v, g, out_path: str):  # CV и график
    gkf = GroupKFold(n_splits=5)  # 5 фолдов
    h_tr, h_va, v_tr, v_va = [], [], [], []  # Списки MAE
# 
    for tr, va in gkf.split(Xs, y_h, g):  # Итерация фолдов
        mh = MLPRegressor(  # Модель H
            hidden_layer_sizes=(64, 32),  # Слои
            alpha=1e-4,  # Регуляризация
            max_iter=300,  # Итерации
            random_state=42,  # Сид
            early_stopping=True,  # Ранняя остановка
            n_iter_no_change=10,  # Терпение
        )
        mv = MLPRegressor(  # Модель V
            hidden_layer_sizes=(64, 32),  # Слои
            alpha=1e-4,  # Регуляризация
            max_iter=300,  # Итерации
            random_state=42,  # Сид
            early_stopping=True,  # Ранняя остановка
            n_iter_no_change=10,  # Терпение
        )
# 
        mh.fit(Xs[tr], y_h[tr])  # Обучение H
        mv.fit(Xs[tr], y_v[tr])  # Обучение V
# 
        pred_h_tr = mh.predict(Xs[tr])  # Прогноз H train
        pred_v_tr = mv.predict(Xs[tr])  # Прогноз V train
        pred_h_va = mh.predict(Xs[va])  # Прогноз H val
        pred_v_va = mv.predict(Xs[va])  # Прогноз V val
        h_tr.append(mean_absolute_error(y_h[tr], pred_h_tr))  # MAE H tr
        v_tr.append(mean_absolute_error(y_v[tr], pred_v_tr))  # MAE V tr
        h_va.append(mean_absolute_error(y_h[va], pred_h_va))  # MAE H va
        v_va.append(mean_absolute_error(y_v[va], pred_v_va))  # MAE V va
# 
    x = np.arange(1, len(h_tr) + 1)  # Ось X
    plt.figure(figsize=(7, 4))  # Размер фигуры
    plt.plot(x, h_tr, "o-", label="H train")  # Линия H train
    plt.plot(x, h_va, "o-", label="H val")  # Линия H val
    plt.plot(x, v_tr, "o-", label="V train")  # Линия V train
    plt.plot(x, v_va, "o-", label="V val")  # Линия V val
    plt.xlabel("fold")  # Подпись оси X
    plt.ylabel("MAE (deg)")  # Подпись оси Y
    plt.grid(True, alpha=0.3)  # Сетка
    plt.legend()  # Легенда
    plt.tight_layout()  # Компоновка
    plt.savefig(out_path, dpi=120)  # Сохранение
    plt.close()  # Закрытие фигуры
# 
    return float(np.mean(h_va)), float(np.mean(v_va))  # Средние MAE
# 
# Подбирает параметры по минимальному среднему MAE. 
def tune_params(Xs, y_h, y_v, g):  # Подбор параметров
    grid = [  # Мини-сетка
        {"hidden_layer_sizes": (64,), "alpha": 1e-4},  # Вариант 1
        {"hidden_layer_sizes": (64, 32), "alpha": 1e-4},  # Вариант 2
        {"hidden_layer_sizes": (64, 32), "alpha": 1e-3},  # Вариант 3
    ]
    gkf = GroupKFold(n_splits=3)  # 3 фолда
    best = None  # Лучшие параметры
    best_score = float("inf")  # Лучший скор
    for params in grid:  # Перебор сетки
        fold_scores = []  # Список скорингов
        for tr, va in gkf.split(Xs, y_h, g):  # Фолды
            mh = MLPRegressor(  # Модель H
                random_state=42,  # Сид
                max_iter=300,  # Итерации
                early_stopping=True,  # Ранняя остановка
                n_iter_no_change=10,  # Терпение
                **params,  # Параметры
            )
            mv = MLPRegressor(  # Модель V
                random_state=42,  # Сид
                max_iter=300,  # Итерации
                early_stopping=True,  # Ранняя остановка
                n_iter_no_change=10,  # Терпение
                **params,  # Параметры
            )
            mh.fit(Xs[tr], y_h[tr])  # Обучение H
            mv.fit(Xs[tr], y_v[tr])  # Обучение V
            mae_h = mean_absolute_error(  # MAE H
                y_h[va],  # Истина H
                mh.predict(Xs[va]),  # Прогноз H
            )
            mae_v = mean_absolute_error(  # MAE V
                y_v[va],  # Истина V
                mv.predict(Xs[va]),  # Прогноз V
            )
            fold_scores.append(mae_h + mae_v)  # Сумма MAE
        score = float(np.mean(fold_scores))  # Средний скор
        if score < best_score:  # Проверка лучшего
            best_score = score  # Обновляем скор
            best = params  # Обновляем параметры
    msg = f"best params: {best}, score={best_score:.3f}"  # Сообщение
    print(msg)  # Печать
    return best  # Возврат параметров
# 
# Обучает модели и сохраняет артефакты. 
def train_and_save(csv_path: str, out_dir: str):  # Обучение
    out = Path(out_dir)  # Папка вывода
    out.mkdir(parents=True, exist_ok=True)  # Создание папки
# 
    X, y_h, y_v, g = load_csv(csv_path)  # Загрузка данных
    scaler = StandardScaler()  # Скалер
    Xs = scaler.fit_transform(X)  # Масштабирование
# 
    best_params = tune_params(Xs, y_h, y_v, g)  # Подбор
    mae_h, mae_v = cv_plot(  # CV-график
        Xs,  # Признаки
        y_h,  # Цель H
        y_v,  # Цель V
        g,  # Группы
        str(out / "quality_cv.png"),  # Путь
    )
    msg = f"CV MAE: H={mae_h:.2f}  V={mae_v:.2f}"  # Сообщение
    print(msg)  # Печать
# 
    mh = MLPRegressor(  # Финальная H
        max_iter=300,  # Итерации
        random_state=42,  # Сид
        early_stopping=True,  # Ранняя остановка
        n_iter_no_change=10,  # Терпение
        **best_params,  # Лучшие параметры
    )
    mv = MLPRegressor(  # Финальная V
        max_iter=300,  # Итерации
        random_state=42,  # Сид
        early_stopping=True,  # Ранняя остановка
        n_iter_no_change=10,  # Терпение
        **best_params,  # Лучшие параметры
    )
    mh.fit(Xs, y_h)  # Обучение H
    mv.fit(Xs, y_v)  # Обучение V
# 
    joblib.dump(scaler, out / "scaler.pkl")  # Сохраняем скалер
    joblib.dump(mh, out / "model_h.pkl")  # Сохраняем модель H
    joblib.dump(mv, out / "model_v.pkl")  # Сохраняем модель V
    print(f"saved models: {out}")  # Сообщение
# 
# Точка входа скрипта обучения. 
def main():  # Запуск обучения
    base = Path(__file__).parent  # База проекта
    csv_path = base / "out" / "gaze_dataset.csv"  # Путь CSV
    if not csv_path.exists():  # Проверка файла
        msg = (  # Сообщение
            "Нет данных. Сначала запустите "  # Часть 1
            "preprocess_dataset.py"  # Часть 2
        )
        print(msg)  # Печать
        return  # Выход
    train_and_save(str(csv_path), str(base / "out" / "models"))  # Запуск
# 
if __name__ == "__main__":  # Запуск как скрипта
    main()  # Вызов main
