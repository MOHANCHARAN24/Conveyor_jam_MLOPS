import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.feature_engineering import add_conveyor_features
from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    feature_names,
    validate_sensor_frame,
)


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train(config_path: str | Path = "config/config.json") -> dict:
    cfg = load_config(config_path)
    dataset_paths = cfg["dataset_paths"]
    datasets = []
    for dataset_path in dataset_paths:
        dataset = pd.read_csv(dataset_path)
        validate_sensor_frame(dataset, require_target=True)
        datasets.append(dataset)
    raw = pd.concat(datasets, ignore_index=True)

    engineered = add_conveyor_features(raw)
    Path("features").mkdir(exist_ok=True)
    feature_store_path = Path("features/conveyor_feature_store.csv")
    engineered.to_csv(feature_store_path, index=False)
    engineered = pd.read_csv(feature_store_path)
    X = engineered[FEATURE_COLUMNS]
    y = engineered[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg.get("test_size", 0.2)),
        random_state=int(cfg.get("random_state", 42)),
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=42,
        ),
    }

    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment(cfg.get("experiment_name", "conveyor-jam-prediction"))

    results = []
    confusion_matrices = {}
    best = None
    best_f1 = -1.0
    for name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        scores = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        }
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
        confusion_matrices[name] = {
            "labels": ["Normal Operation", "Jam Risk"],
            "matrix": matrix,
        }

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("model_name", name)
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_param("dataset_versions", ",".join(dataset_paths))
            mlflow.log_metrics(scores)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )
            results.append(
                {
                    "model": name,
                    **scores,
                    "run_id": run.info.run_id,
                }
            )
            if scores["f1"] > best_f1:
                best = pipeline
                best_f1 = scores["f1"]
                best_name = name
                best_scores = scores
                best_run_id = run.info.run_id

    if best is None:
        raise RuntimeError("No candidate model was trained.")

    joblib.dump(best, "artifacts/model.joblib")
    metadata = {
        "task": "classification",
        "target": TARGET_COLUMN,
        "model_name": best_name,
        "metrics": best_scores,
        "feature_columns": FEATURE_COLUMNS,
        "dataset_paths": dataset_paths,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
    }
    transformed_train = best.named_steps["preprocessor"].transform(X_train)
    transformed_test = best.named_steps["preprocessor"].transform(X_test)
    names = feature_names(best.named_steps["preprocessor"])
    train_array = (
        transformed_train.toarray()
        if hasattr(transformed_train, "toarray")
        else transformed_train
    )
    test_array = (
        transformed_test.toarray()
        if hasattr(transformed_test, "toarray")
        else transformed_test
    )
    pd.DataFrame(train_array, columns=names or None).to_csv(
        "features/train_features.csv", index=False
    )
    pd.DataFrame(test_array, columns=names or None).to_csv(
        "features/test_features.csv", index=False
    )
    pd.DataFrame(results).to_csv("reports/model_comparison.csv", index=False)
    Path("reports/confusion_matrices.json").write_text(
        json.dumps(confusion_matrices, indent=2), encoding="utf-8"
    )

    registered_name = cfg.get("registered_model_name", "ConveyorJamRiskModel")
    registered = mlflow.register_model(
        f"runs:/{best_run_id}/model",
        registered_name,
    )
    metadata["registered_model_name"] = registered_name
    metadata["registered_model_version"] = registered.version
    Path("artifacts/model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    result = {
        "best_model": best_name,
        "registered_model_name": registered_name,
        "registered_model_version": registered.version,
        "metrics": best_scores,
        "results": results,
        "confusion_matrices": confusion_matrices,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.json")
    args = parser.parse_args()
    train(args.config)
