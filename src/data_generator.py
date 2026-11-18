from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

def generate_demo_datasets(output_dir="data/raw", n_samples=1000, random_state=42):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    Xc, yc = make_classification(
        n_samples=n_samples, n_features=6, n_informative=4,
        n_redundant=0, weights=[0.55, 0.45], random_state=random_state
    )
    cls = pd.DataFrame(Xc, columns=[f"numeric_{i}" for i in range(1, 7)])
    cls["category"] = rng.choice(["A", "B", "C"], n_samples)
    cls["target"] = yc
    cls.loc[rng.choice(n_samples, 35, replace=False), "numeric_2"] = np.nan
    cls.loc[rng.choice(n_samples, 25, replace=False), "category"] = None
    cls.to_csv(out / "classification.csv", index=False)

    Xr, yr = make_regression(
        n_samples=n_samples, n_features=6, n_informative=5,
        noise=18.0, random_state=random_state
    )
    reg = pd.DataFrame(Xr, columns=[f"numeric_{i}" for i in range(1, 7)])
    reg["category"] = rng.choice(["urban", "suburban", "rural"], n_samples)
    reg["target"] = yr
    reg.loc[rng.choice(n_samples, 30, replace=False), "numeric_3"] = np.nan
    reg.loc[rng.choice(n_samples, 20, replace=False), "category"] = None
    reg.to_csv(out / "regression.csv", index=False)
    return out / "classification.csv", out / "regression.csv"

if __name__ == "__main__":
    a, b = generate_demo_datasets()
    print(f"Created {a} and {b}")
