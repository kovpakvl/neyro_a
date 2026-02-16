import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import extract_eye_features, gaze_to_zone

RESIZE_W = 1280
FEATURE_COLS = [f"f{i}" for i in range(11)]
NUM_COLS = FEATURE_COLS + ["H", "V", "P"]


def parse_columbia(name: str):
    m = re.match(r"^\d+_\d+m_(-?\d+)P_(-?\d+)V_(-?\d+)H\.jpg$", name, re.I)
    if not m:
        return None
    p, v, h = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return p, v, h


def make_detector(model_path: str):
    base = python.BaseOptions(model_asset_path=model_path)
    opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


def collect_rows(dataset_root: str, model_path: str):
    root = Path(dataset_root)
    persons = sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()])
    det = make_detector(model_path)

    rows = []
    skipped = 0

    for d in persons:
        pid = d.name
        for img_path in d.glob("*.jpg"):
            lab = parse_columbia(img_path.name)
            if not lab:
                skipped += 1
                continue
            p, v, h = lab

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            hh, ww = img.shape[:2]
            scale = RESIZE_W / max(1, ww)
            img = cv2.resize(img, (RESIZE_W, int(hh * scale)), interpolation=cv2.INTER_LANCZOS4)
            rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = det.detect(mp_img)
            if not res.face_landmarks:
                skipped += 1
                continue

            feats = extract_eye_features(res.face_landmarks[0], head_pose=p)
            if feats is None:
                skipped += 1
                continue

            row = {
                "person_id": pid,
                "filename": img_path.name,
                "H": h,
                "V": v,
                "P": p,
                "zone": gaze_to_zone(h, v),
            }
            for i, val in enumerate(feats["features"]):
                row[f"f{i}"] = val
            rows.append(row)

        print(f"{pid}: ok")

    df = pd.DataFrame(rows)
    print(f"rows={len(df)}, skipped={skipped}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols=("H", "V")):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        mask &= df[c].between(lo, hi)
    return df[mask]


def clean_df(df: pd.DataFrame):
    key_cols = ["person_id", "H", "V", "P"] + FEATURE_COLS
    df = df.dropna(subset=key_cols)
    df = df.drop_duplicates()
    before = len(df)
    df = remove_outliers_iqr(df, ("H", "V"))
    print(f"outliers removed: {before - len(df)}")
    return df


def analysis_plots(df: pd.DataFrame, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = df.dropna(subset=NUM_COLS).copy()
    if d.empty:
        print("no data for analysis")
        return

    corr = d[NUM_COLS].corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(shrink=0.8)
    plt.xticks(range(len(NUM_COLS)), NUM_COLS, rotation=45, ha="right")
    plt.yticks(range(len(NUM_COLS)), NUM_COLS)
    plt.tight_layout()
    plt.savefig(out / "corr.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(d["H"], d["V"], s=8, alpha=0.35)
    plt.xlabel("H")
    plt.ylabel("V")
    plt.tight_layout()
    plt.savefig(out / "scatter_H_V.png", dpi=120)
    plt.close()

    print(f"analysis saved: {out}")


def main():
    base = Path(__file__).parent
    dataset_root = str(base / "Columbia Gaze Data Set")
    if not Path(dataset_root).exists():
        dataset_root = str(base / "Columbia Gaze Resized")

    model_path = str(base / "face_landmarker.task")
    if not Path(model_path).exists():
        raise FileNotFoundError("face_landmarker.task положи рядом со скриптами")

    out_dir = base / "out"
    out_dir.mkdir(exist_ok=True)

    df = collect_rows(dataset_root, model_path)
    if df.empty:
        print("empty df")
        return

    df = clean_df(df)
    csv_path = out_dir / "gaze_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved: {csv_path} ({len(df)})")

    analysis_plots(df, str(out_dir / "analysis"))

    print("zones:", df["zone"].value_counts().to_dict())
    print("done")


if __name__ == "__main__":
    main()
