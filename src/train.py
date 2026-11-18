import argparse, json, os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from src.preprocessing import build_preprocessor, feature_names
from src.feature_engineering import add_generic_features
from src.data_versioning import snapshot_dataset

def load_config(path):
    return json.loads(Path(path).read_text())

def train(config_path="config/config.json"):
    cfg = load_config(config_path)
    task = cfg["task"].lower()
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")
    df = pd.read_csv(cfg["dataset_path"])
    target = cfg["target"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    df = df.dropna(subset=[target]).copy()
    y = df.pop(target)
    X = add_generic_features(df)
    if X.empty:
        raise ValueError("Dataset has no feature columns.")

    stratify = y if task == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(cfg.get("test_size", 0.2)),
        random_state=int(cfg.get("random_state", 42)), stratify=stratify
    )

    preprocessor = build_preprocessor(X_train)
    if task == "classification":
        models = {
            "logistic_regression": LogisticRegression(max_iter=1500),
            "random_forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
        }
    else:
        models = {
            "ridge": Ridge(),
            "random_forest": RandomForestRegressor(n_estimators=250, random_state=42)
        }

    Path("artifacts").mkdir(exist_ok=True)
    Path("features").mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    snapshot_dataset(cfg["dataset_path"], notes=f"Training input for {task}") 

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(cfg.get("experiment_name", "universal-mlops"))
    results = []
    best = None
    best_score = None
    metric = cfg.get("primary_metric", "f1" if task == "classification" else "rmse").lower()

    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        if task == "classification":
            scores = {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0))
            }
            selection_score = scores.get(metric, scores["f1"])
            better = best_score is None or selection_score > best_score
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            scores = {
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": rmse,
                "r2": float(r2_score(y_test, pred))
            }
            selection_score = scores.get(metric, scores["rmse"])
            better = best_score is None or (
                selection_score < best_score if metric in {"mae", "rmse"} else selection_score > best_score
            )

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("task", task)
            mlflow.log_param("model_name", name)
            mlflow.log_param("dataset", cfg["dataset_path"])
            mlflow.log_metrics(scores)
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )
            results.append({"model": name, **scores, "run_id": run.info.run_id})
        if better:
            best, best_score = pipe, selection_score
            best_name = name
            best_scores = scores
            best_run_id = run.info.run_id

    # Persist exact serving pipeline and feature-store outputs.
    joblib.dump(best, "artifacts/model.joblib")
    metadata = {
        "task": task, "target": target, "model_name": best_name,
        "metrics": best_scores, "feature_columns": X.columns.tolist(),
        "class_labels": [str(v) for v in sorted(pd.Series(y).unique().tolist(), key=str)] if task == "classification" else None
    }
    Path("artifacts/model_metadata.json").write_text(json.dumps(metadata, indent=2))
    transformed_train = best.named_steps["preprocessor"].transform(X_train)
    transformed_test = best.named_steps["preprocessor"].transform(X_test)
    names = feature_names(best.named_steps["preprocessor"])
    pd.DataFrame(transformed_train.toarray() if hasattr(transformed_train, "toarray") else transformed_train,
                 columns=names or None).to_csv("features/train_features.csv", index=False)
    pd.DataFrame(transformed_test.toarray() if hasattr(transformed_test, "toarray") else transformed_test,
                 columns=names or None).to_csv("features/test_features.csv", index=False)
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv("reports/model_comparison.csv", index=False)

    registered_name = cfg.get("registered_model_name", "UniversalMLOpsModel")
    model_uri = f"runs:/{best_run_id}/model"
    try:
        registered = mlflow.register_model(model_uri, registered_name)
        metadata["registered_model_name"] = registered_name
        metadata["registered_model_version"] = registered.version
    except Exception as exc:
        metadata["registry_note"] = f"Registration was not completed: {exc}"
    Path("artifacts/model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"best_model": best_name, "metrics": best_scores, "results": results}, indent=2))
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.json")
    args = parser.parse_args()
    train(args.config)
