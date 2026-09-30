"""生成教学用合成数据，不代表任何真实大学生调查。"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
ROOT = Path(__file__).resolve().parent


def generate_demo_data(n: int = 500, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(3.5, 1.3, n), 0.2, 8)
    video = np.clip(rng.normal(2.5, 1.15, n), 0.1, 7)
    sleep = np.clip(rng.normal(7.1, 0.85, n), 4.5, 9.5)
    exercise = np.clip(rng.normal(32, 18, n), 0, 100)
    exam = np.clip(56 + 4.8 * study - 0.7 * video + 0.8 * (sleep - 7)
                   + rng.normal(0, 11, n), 0, 100)
    satisfaction = np.clip(3.3 + 0.42 * (sleep - 7) + rng.normal(0, 0.8, n), 1, 5)
    return pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
        "gender": rng.choice(["女", "男"], n),
        "grade": rng.choice(["大一", "大二", "大三", "大四"], n),
        "short_video_hours": video.round(2),
        "study_hours": study.round(2),
        "sleep_hours": sleep.round(2),
        "exercise_minutes": exercise.round(1),
        "exam_score": exam.round(1),
        "satisfaction": satisfaction.round(2),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "demo_students.csv")
    args = parser.parse_args()
    frame = generate_demo_data(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Generated {len(frame)} synthetic students: {args.output}")


if __name__ == "__main__":
    main()
