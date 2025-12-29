from typing import Any, Sequence
from pathlib import Path
from project_types import ModelKey, MetricSplit, EvalSplit
from config import (
    HF_MODELS,
    RESULTS_DIR, PREPROCESSING_DIR, TUNING_MODELS_DIR, BEST_MODELS_DIR, RESULTS_COMPARISON_DIR,
    LABEL_MAP_NAME,
    TRAIN_METRICS_NAME, VALIDATION_METRICS_NAME, TEST_METRICS_NAME,
    TRAINING_LOG_HISTORY_NAME,
    BEST_MODEL_INFO_NAME,
    TOKENIZATION_CONFIG_NAME,
    TOKEN_LENGTH_STATS_NAME,
    TRUNCATION_RATES_NAME, TRUNCATION_SHORTLIST_NAME,
    VALIDATION_EVALUATION_NAME, TEST_EVALUATION_NAME,
    VALIDATION_PREDICTION_NAME, TEST_PREDICTION_NAME,
    BEST_MODEL_INFO_NAME,
    BEST_MODEL_CONFIG_NAME, BEST_MODEL_SAFE_TENSORS_NAME, BEST_MODEL_SPECIAL_TOKEN_MAP_NAME,
    BEST_MODEL_TOKENIZER_NAME, BEST_MODEL_TOKENIZER_CONFIG_NAME, BEST_MODEL_TRAINING_ARGS_NAME,
    BEST_MODEL_VOCAB_NAME,
    FINAL_MODEL_SELECTION_NAME
)

from utils import is_file_non_empty

import pandas as pd
import json

# -----------------------------------------------------------------------------
# Constants & mappings
# -----------------------------------------------------------------------------
_SPLIT_TO_FILENAME: dict[MetricSplit, str] = {
    "train": TRAIN_METRICS_NAME,
    "val": VALIDATION_METRICS_NAME,
    "test": TEST_METRICS_NAME
}

# Evaluation artifacts (val / test only)
_EVAL_SPLIT_TO_FILENAME: dict[EvalSplit, str] = {
    "val": VALIDATION_EVALUATION_NAME,
    "test": TEST_EVALUATION_NAME
}

# Prediction artifacts (val / test only)
_PRED_SPLIT_TO_FILENAME: dict[EvalSplit, str] = {
    "val": VALIDATION_PREDICTION_NAME,
    "test": TEST_PREDICTION_NAME
}

# -----------------------------------------------------------------------------
# Internal helpers (private)
# -----------------------------------------------------------------------------
def _with_threshold_suffix(
    path: Path,
    threshold: float
) -> Path:
    threshold_str = f"{threshold:.1f}".replace(".", "_")
    
    return path.with_name(f"{path.stem}_{threshold_str}{path.suffix}")

def _validate_model_key(model_key: ModelKey) -> ModelKey:
    if model_key not in HF_MODELS:
        valid_keys = list(HF_MODELS.keys())
        
        raise ValueError(f"model_key must be one of {valid_keys}: Got {model_key}")
    
    return model_key

def _path_serializer(obj: Any) -> str:
    """
    Path is not serializable in JSON: Path -> str
    """
    if isinstance(obj, Path):
        return str(obj)
    
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# -----------------------------------------------------------------------------
# Path builders (model-level / comparison-level)
# -----------------------------------------------------------------------------
def get_tuning_model_dir(model_key: ModelKey) -> Path:
    return TUNING_MODELS_DIR / _validate_model_key(model_key)

def get_best_model_dir(model_key: ModelKey) -> Path:
    return BEST_MODELS_DIR /  _validate_model_key(model_key)

def get_label_map_path(model_key: ModelKey) -> Path:
    return PREPROCESSING_DIR / _validate_model_key(model_key) / LABEL_MAP_NAME

def get_split_metrics_path(
    model_key: ModelKey,
    split_type: MetricSplit
) -> Path:    
    return RESULTS_DIR / _validate_model_key(model_key) / _SPLIT_TO_FILENAME[split_type]

def get_training_log_history_path(model_key: ModelKey) -> Path:
    return RESULTS_DIR /  _validate_model_key(model_key) / TRAINING_LOG_HISTORY_NAME

def get_best_model_info_path(model_key: ModelKey) -> Path:
    return BEST_MODELS_DIR /  _validate_model_key(model_key) / BEST_MODEL_INFO_NAME

def get_best_model_artifact_paths(model_key: ModelKey) -> list[Path]:
    base_path = get_best_model_dir(model_key)
    
    names = (
        BEST_MODEL_CONFIG_NAME,
        BEST_MODEL_SAFE_TENSORS_NAME,
        BEST_MODEL_SPECIAL_TOKEN_MAP_NAME,
        BEST_MODEL_TOKENIZER_NAME,
        BEST_MODEL_TOKENIZER_CONFIG_NAME,
        BEST_MODEL_TRAINING_ARGS_NAME,
        BEST_MODEL_VOCAB_NAME
    )
    
    return [(base_path / name) for name in names]

def get_split_evaluation_path(
    model_key: ModelKey,
    split_type: EvalSplit
) -> Path:
    return RESULTS_DIR / _validate_model_key(model_key) / _EVAL_SPLIT_TO_FILENAME[split_type]

def get_split_prediction_path(
    model_key: ModelKey,
    split_type: EvalSplit    
) -> Path:
    return RESULTS_DIR / _validate_model_key(model_key) / _PRED_SPLIT_TO_FILENAME[split_type]

def get_result_comparison_dir() -> Path:
    return RESULTS_COMPARISON_DIR

def get_tokenization_config_path() -> Path:
    return PREPROCESSING_DIR / TOKENIZATION_CONFIG_NAME

def get_token_length_stats_path() -> Path:
    return RESULTS_COMPARISON_DIR / TOKEN_LENGTH_STATS_NAME

def get_truncation_rates_path() -> Path:
    return RESULTS_COMPARISON_DIR / TRUNCATION_RATES_NAME

def get_truncation_shortlist_base_path() -> Path:
    return RESULTS_COMPARISON_DIR / TRUNCATION_SHORTLIST_NAME

def get_truncation_shortlist_path() -> Path:
    base_dir = get_result_comparison_dir()
    base_stem = Path(TRUNCATION_SHORTLIST_NAME).stem
    matching_files = list(base_dir.glob(f"{base_stem}*.csv"))

    if not matching_files:
        raise FileNotFoundError(f"No truncation shortlist found in {base_dir}")
    
    matching_files.sort()
    
    return matching_files[-1]

def get_final_model_selection_path() -> Path:
    return RESULTS_COMPARISON_DIR / FINAL_MODEL_SELECTION_NAME

# -----------------------------------------------------------------------------
# Generic JSON/CSV IO helpers
# -----------------------------------------------------------------------------
def save_json(
    payload: dict[str, Any],
    path: Path,
    overwrite: bool=True
) -> Path:
    """
    Save a dict payload to a JSON file under artifacts
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}. Set overwrite=True to overwrite.")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False, default=_path_serializer)
        
    print(f"Saved to {path}")
    
    return path

def load_json(path: Path) -> dict[str, Any]:
    """
    Load a dict payload from a JSON file under artifacts
    """    
    if not is_file_non_empty(path): 
        raise RuntimeError(f"Missing or empty JSON file: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    return payload

def save_csv(
    name: str,
    df: pd.DataFrame,
    path: Path,
    overwrite: bool=True,
    index: bool=False
) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}. Set overwrite=True to overwrite.")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=index)
    print(f"{name} saved to: {path}")
    
    return path

def load_csv(path: Path) -> pd.DataFrame :
    """
    Load dataFrame from a CSV file
    """    
    if not is_file_non_empty(path): 
        raise RuntimeError(f"Missing or empty CSV file: {path}")
    
    return pd.read_csv(path)

# -----------------------------------------------------------------------------
# Tokenization artifacts (preprocessing-level)
# -----------------------------------------------------------------------------
def save_tokenization_config(
    payload: dict[str, Any],
    overwrite: bool=True
) -> Path:
    """
    Save tokenization configuration
    """
    allowed_padding = {"max_length", "longest"}
    required = {"max_length", "padding", "truncation"}
    missing = required - set(payload.keys())
    
    if missing:
        raise ValueError(f"Required keys are missing from the payload: {sorted(missing)}")
    
    if not isinstance(payload["max_length"], int):
        raise TypeError("max_length must be int")

    if payload["max_length"] <= 0:
        raise ValueError("max_length must be positive")

    if payload["padding"] not in allowed_padding:
        raise ValueError(f"Invalid padding: {payload['padding']}")

    if not isinstance(payload["truncation"], bool):
        raise TypeError("truncation must be boolean")
    
    return save_json(
        payload=payload,
        path=get_tokenization_config_path(),
        overwrite=overwrite
    )

def load_tokenization_config() -> dict[str, Any]:
    payload = load_json(get_tokenization_config_path())
    
    required = {"max_length", "padding", "truncation"}
    missing = required - set(payload.keys())
    
    if missing:
        raise ValueError(f"Required keys are missing from the payload: {sorted(missing)}")
    
    return payload

# -----------------------------------------------------------------------------
# Label mapping artifacts (model-level)
# -----------------------------------------------------------------------------
def save_label_map(
    model_key: ModelKey,
    label_order: Sequence[str],
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    overwrite: bool=True
) -> Path:
    """
    Save label map
    Prerequisite:
    - label_order
    - label_to_id
    - id_to_label
    Make sure the inputs are built from build_label_mappings()
    """
    expected_label_ids = list(range(len(label_order)))
    
    if list(label_to_id.keys()) != list(label_order):
        raise ValueError("label_to_id keys must match label_order exactly")
    
    if [label_to_id[label] for label in label_order] != expected_label_ids:
        raise ValueError("label_to_id must be in label_order order")
    
    if [id_to_label[id] for id in expected_label_ids] != list(label_order):
        raise ValueError("id_to_label must invert label_to_id exactly")
    
    payload = {
        "label_order": list(label_order),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label
    }
    
    return save_json(
        payload=payload,
        path=get_label_map_path(model_key),
        overwrite=overwrite
    )

def load_label_map(path: Path) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    Low-level loader: loads a label_map.json from an explicit path.
    """
    payload = load_json(path)
    
    if "label_order" not in payload or "label_to_id" not in payload or "id_to_label" not in payload:
        raise ValueError("Invalid: label_order, label_to_id or id_to_label are missing")
    
    label_order = payload["label_order"]
    label_to_id = payload["label_to_id"]
    id_to_label_raw = payload["id_to_label"]    # Integers are converted into string in json
    
    id_to_label = {int(k): v for k, v in id_to_label_raw.items()}
    
    return label_order, label_to_id, id_to_label

def load_label_map_for_model(model_key: ModelKey) -> tuple[list[str], dict[str, int], dict[int, str]]:
    return load_label_map(get_label_map_path(model_key))

# -----------------------------------------------------------------------------
# Training & evaluation artifacts (model-level)
# -----------------------------------------------------------------------------
def save_split_metrics(
    model_key: ModelKey,
    metrics: dict[str, Any],
    split_type: MetricSplit,
    overwrite: bool=True
) -> Path:
    """
    Save train, val or test metrics
    """
    return save_json(
        payload=metrics,
        path=get_split_metrics_path(model_key, split_type),
        overwrite=overwrite
    )

def load_split_metrics(
    model_key: ModelKey,
    split_type: MetricSplit
) -> dict[str, Any]:
    return load_json(get_split_metrics_path(model_key, split_type))

def save_training_log_history(
    model_key: ModelKey,
    df_train_log: pd.DataFrame,
    overwrite: bool=True,
    index: bool=False
) -> Path:
    """
    Save training log history
    """
    return save_csv(
        name="Training log history",
        df=df_train_log,
        path=get_training_log_history_path(model_key),
        overwrite=overwrite,
        index=index
    )

def load_training_log_history(model_key: ModelKey) -> pd.DataFrame:
    return load_csv(get_training_log_history_path(model_key))

def save_best_model_info(
    model_key: ModelKey,
    payload: dict[str, Any],
    overwrite: bool=True
) -> Path:
    """
    Save best model information
    """
    required = {"best_model_checkpoint", "best_metric", "metric_for_best_model", "model_key", "global_step"}
    missing = required - set(payload.keys())
    
    if missing:
        raise ValueError(f"Required keys are missing from the payload: {sorted(missing)}")
    
    if model_key != payload["model_key"]:
        raise ValueError(f"model_key is not matched: Input = {model_key}, payload = {payload['model_key']} ")

    if not isinstance(payload["best_metric"], float):
        raise TypeError("best_metric must be float")
    
    if not isinstance(payload["global_step"], int):
        raise TypeError("global_step must be int")    
    
    return save_json(
        payload=payload,
        path=get_best_model_info_path(model_key),
        overwrite=overwrite
    )    

def load_best_model_info(model_key: ModelKey) -> dict[str, Any]:
    payload = load_json(get_best_model_info_path(model_key))
    
    required = {"best_model_checkpoint", "best_metric", "metric_for_best_model", "global_step", "model_key"}
    missing = required - set(payload.keys())
    
    if missing:
        raise ValueError(f"Required keys are missing from the payload: {sorted(missing)}")
    
    if model_key != payload["model_key"]:
        raise ValueError(f"model_key is not matched: Input = {model_key}, payload = {payload['model_key']} ")
    
    return payload

def save_split_evaluation(
    model_key: ModelKey,
    split_type: EvalSplit,
    evaluation: dict[str, Any],
    overwrite: bool=True
) -> Path:
    """
    Save split evalutation
    """
    return save_json(
        payload=evaluation,
        path=get_split_evaluation_path(model_key, split_type),
        overwrite=overwrite
    )    

def load_split_evaluation(
    model_key: ModelKey,
    split_type: EvalSplit 
) -> dict[str, Any]:
    return load_json(get_split_evaluation_path(model_key, split_type))

def save_split_prediction(
    model_key: ModelKey,
    split_type: EvalSplit,
    df_pred: pd.DataFrame,
    overwrite: bool=True,
    index: bool=False 
) -> Path:
    return save_csv(
        name=f"{split_type.capitalize()} prediction",
        df=df_pred,
        path=get_split_prediction_path(model_key, split_type),
        overwrite=overwrite,
        index=index
    )

def load_split_prediction(
    model_key: ModelKey,
    split_type: EvalSplit 
) -> pd.DataFrame:
    return load_csv(get_split_prediction_path(model_key, split_type))

# -----------------------------------------------------------------------------
# Tokenization comparison artifacts (comparison-level)
# -----------------------------------------------------------------------------
def save_tokenization_analysis(
    df_stats: pd.DataFrame,
    df_trunc: pd.DataFrame,
    overwrite: bool=True,
    index: bool=False
) -> tuple[Path, Path]:
    """
    Save tokenization analysis:
    - token length statistics
    - truncation rate across max_length candidates
    """
    token_length_stats_path = save_csv(
        name="Token length statistics",
        df=df_stats,
        path=get_token_length_stats_path(),
        overwrite=overwrite,
        index=index
    )
    
    truncation_rates_path = save_csv(
        name="Truncation rates",
        df=df_trunc,
        path=get_truncation_rates_path(),
        overwrite=overwrite,
        index=index
    )
    
    return token_length_stats_path, truncation_rates_path
    
def save_truncation_shortlist(
    df_trunc_shortlist: pd.DataFrame,
    threshold: float,
    overwrite: bool=True,
    index: bool=False
) -> Path:
    """
    Save truncation shortlist
    """
    file_path = _with_threshold_suffix(get_truncation_shortlist_base_path(), threshold)
    
    return save_csv(
        name="Truncation shortlist",
        df=df_trunc_shortlist,
        path=file_path,
        overwrite=overwrite,
        index=index
    )

def save_final_model_selection(
    payload: dict[str, Any],
    overwrite: bool=True    
) -> Path:
    """
    Save final model selection
    """
    return save_json(
        payload=payload,
        path=get_final_model_selection_path(),
        overwrite=overwrite
    )
    
def load_final_model_selection() -> dict[str, Any]:
    return load_json(get_final_model_selection_path())
