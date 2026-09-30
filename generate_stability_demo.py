"""合成演示数据，仅用于展示稳定性分析功能。原 V1 Demo 不变。"""
from pathlib import Path
import numpy as np
import pandas as pd

STABILITY_DEMO_SEED = 0


def generate_stability_demo() -> pd.DataFrame:
    # 固定参数经真实计算选定，用于教学，不代表从真实数据中筛选显著结果。
    rng = np.random.default_rng(STABILITY_DEMO_SEED)
    x = np.append(rng.normal(size=79), 6.0)
    y = np.append(rng.normal(size=79), 4.8)
    return pd.DataFrame({"student_id": [f"SYN{i:03d}" for i in range(1, 81)],
                         "synthetic_x": x, "synthetic_y": y})


if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "data" / "stability_demo.csv"
    path.parent.mkdir(exist_ok=True)
    generate_stability_demo().to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Synthetic stability demo saved: {path}")
