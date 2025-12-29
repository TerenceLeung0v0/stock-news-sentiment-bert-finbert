from typing import Any
from project_types import ModelKey, EvalSplit
from artifacts_utils import (
    load_label_map_for_model, load_tokenization_config,
    load_split_metrics, load_split_evaluation, load_split_prediction,
    load_training_log_history, load_best_model_info,
)

import pandas as pd
import numpy as np

_FULL_METRIC_KEYS = (
    "eval_loss", "eval_accuracy",
    "eval_macro_precision", "eval_macro_recall", "eval_macro_f1",
    "eval_weighted_precision", "eval_weighted_recall", "eval_weighted_f1",
    "eval_runtime", "eval_samples_per_second", "eval_steps_per_second",
    "epoch"
)

def load_model_artifacts(model_key: ModelKey) -> dict[str, Any]:
    """
    Load all persisted artifacts for a trained model for analysis and comparison.

    Note:
    - Metrics conceptually support ["train", "val", "test"] (MetricSplit),
      while evaluation and prediction artifacts only exist for ["val", "test"] (EvalSplit).
    - For analysis efficiency and code simplicity, I intentionally iterate over
      ["val", "test"] once and load all split-dependent artifacts together.
    - This avoids duplicated loops without affecting correctness or clarity
      in downstream analysis notebooks.
    """
    eval_splits: list[EvalSplit] = ["val", "test"]
    label_order, label_to_id, id_to_label = load_label_map_for_model(model_key)
    
    artifacts = {
        "model_key": model_key,
        "label_map": {
            "label_order": label_order,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
        },
        "tokenization_config": {},
        "metrics": {},
        "evaluation": {},
        "predictions": {},
        "training_log": None,
        "best_model_info": None,
    }
    
    artifacts["tokenization_config"] = load_tokenization_config()

    for split in eval_splits:
        artifacts["metrics"][split] = load_split_metrics(model_key, split)
        artifacts["evaluation"][split] = load_split_evaluation(model_key, split)
        artifacts["predictions"][split] = load_split_prediction(model_key, split)
    
    artifacts["training_log"] = load_training_log_history(model_key)
    artifacts["best_model_info"] = load_best_model_info(model_key)
    
    return artifacts

def build_metrics_comparison_table(
    model_artifacts_map: dict[ModelKey, dict[str, Any]],
    split: EvalSplit,
    metric_keys: list[str] | None=None
) -> pd.DataFrame:
    """
    Build a comparison table across models for a given split.
    - model_artifacts_map: output of load_model_artifacts() for each model
    - split: "val" or "test"
    - metric_keys: optional subset of metrics to include, otherwise, default metric_keys would be loaded
    """
    rows = []

    if metric_keys is None:
        exclude = {"eval_runtime", "eval_samples_per_second", "eval_steps_per_second"}
        metric_keys = [key for key in _FULL_METRIC_KEYS if key not in exclude]
    
    for model_key, artifacts in model_artifacts_map.items():
        split_metrics = artifacts.get("metrics", {}).get(split, {})
        row = {"model": model_key, "split": split}

        for k in metric_keys:
            row[k] = split_metrics.get(k, None)
            
        rows.append(row)

    cols = ["model", "split"] + metric_keys
    
    return pd.DataFrame(rows, columns=cols)

def get_primary_score(
    artifacts: dict[str, Any],
    split: EvalSplit
) -> float:
    split_metrics = artifacts.get("metrics", {}).get(split, {})
    
    for key in ["eval_macro_f1", "macro_f1"]:
        if key in split_metrics:
            return float(split_metrics[key])
    
    raise KeyError(f"macro_f1 key not found in metrics json for split={split}")

def get_split_scores(artifacts: dict[str, Any]) -> dict[str, Any]:
    val_metric = artifacts["metrics"]["val"]
    test_metric = artifacts["metrics"]["test"]

    scores = {
        "val": {
            "macro_f1": float(val_metric.get("eval_macro_f1") or val_metric.get("macro_f1")),
            "weighted_f1": float(val_metric.get("eval_weighted_f1") or val_metric.get("weighted_f1"))
        },
        "test": {
            "macro_f1": float(test_metric.get("eval_macro_f1") or test_metric.get("macro_f1")),
            "weighted_f1": float(test_metric.get("eval_weighted_f1") or test_metric.get("weighted_f1"))            
        }   
    }

    return scores

def get_confusion_matrix(
    artifacts: dict[str, Any],
    split: EvalSplit
) -> np.ndarray:
    try:
        cm_list = artifacts["evaluation"][split]["data"]["confusion_matrix"]
    except KeyError as e:
        raise KeyError(
            f"Confusion matrix not found for model={artifacts.get('model_key')}, split={split}"
        ) from e
    
    return np.array(cm_list)

def top_confusions(
    df_pred: pd.DataFrame,
    top_k: int=10
) -> pd.DataFrame:
    required = {"is_correct", "y_true_label", "y_pred_label"}
    missing = required - set(df_pred.columns)
    
    if missing:
        raise KeyError(f"Required columns are missing from df_pred: {sorted(missing)}")
    
    df_err = df_pred[~df_pred["is_correct"]].copy()
    
    return (df_err.groupby(["y_true_label", "y_pred_label"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(top_k)
                .reset_index(drop=True))

def classification_report_to_df(
    artifacts: dict,
    split: EvalSplit
) -> pd.DataFrame:
    rows = []
    
    # Standard keys returned for each label in classification report
    expected_metrics = {"precision", "recall", "f1-score", "support"}
    cr = artifacts["evaluation"][split]["metrics"]["classification_report"]
    
    for key, value in cr.items():
        if isinstance(value, dict) and expected_metrics.issubset(value.keys()):
            rows.append({
                "label": key,
                "precision": value["precision"],
                "recall": value["recall"],
                "f1": value["f1-score"],
                "support": value["support"],
            })
    
    return pd.DataFrame(rows)
