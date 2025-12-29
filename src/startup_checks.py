from typing import Iterable
from pathlib import Path
from config import REQUIRED_DIRS, BERT_KEY, FINBERT_KEY
from project_types import ModelKey, MetricSplit, EvalSplit
from utils import is_file_non_empty
from artifacts_utils import (
    get_tokenization_config_path, get_token_length_stats_path, get_truncation_rates_path, get_truncation_shortlist_path,
    get_label_map_path, get_split_metrics_path, get_split_evaluation_path, get_split_prediction_path,
    get_training_log_history_path, get_best_model_info_path, get_best_model_artifact_paths,
    get_final_model_selection_path
)

def ensure_required_dirs(dirs: Iterable[str | Path]) -> None:
    """
    Ensure required directories exist:
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def ensure_project_dirs() -> None:
    """
    Call this function at the beginning of each Notebook
    - ensure that all required directories exist
    """
    ensure_required_dirs(REQUIRED_DIRS)

def check_required_data(path: Path) -> None:
    if not is_file_non_empty(path):
        raise RuntimeError(f"Required data file is missing or empty: {path}")

def check_required_comparison_artifacts() -> list[str]:
    missings = []
    
    token_length_stats_path = get_token_length_stats_path()
    truncation_rates_path = get_truncation_rates_path()
    truncation_shortlist_path = get_truncation_shortlist_path()
    
    if not is_file_non_empty(token_length_stats_path):
        missings.append(f"{token_length_stats_path} is missing or empty")

    if not is_file_non_empty(truncation_rates_path):
        missings.append(f"{truncation_rates_path} is missing or empty")

    if not is_file_non_empty(truncation_shortlist_path):
        missings.append(f"{truncation_shortlist_path} is missing or empty")
        
    return missings

def check_model_key_artifacts(model_key: ModelKey) -> list[str]:
    missings = []
    metric_splits: list[MetricSplit] = ["val", "test"]
    eval_splits: list[EvalSplit] = ["val", "test"]
    
    # preprocssing
    label_map_path = get_label_map_path(model_key)
    
    # results
    split_metrics_paths = [get_split_metrics_path(model_key, metric_split) for metric_split in metric_splits]
    split_evaluation_paths = [get_split_evaluation_path(model_key, eval_split) for eval_split in eval_splits]
    split_prediction_paths = [get_split_prediction_path(model_key, eval_split) for eval_split in eval_splits]
    training_log_history_path = get_training_log_history_path(model_key)
    
    # best
    best_model_info_path = get_best_model_info_path(model_key)
    best_model_artifact_paths = get_best_model_artifact_paths(model_key)
    
    if not is_file_non_empty(label_map_path):
        missings.append(f"{label_map_path} is missing or empty")

    for split_metrics_path in split_metrics_paths:
        if not is_file_non_empty(split_metrics_path):
            missings.append(f"{split_metrics_path} is missing or empty") 
        
    for split_evaluation_path in split_evaluation_paths:
        if not is_file_non_empty(split_evaluation_path):
            missings.append(f"{split_evaluation_path} is missing or empty") 

    for split_prediction_path in split_prediction_paths:
        if not is_file_non_empty(split_prediction_path):
            missings.append(f"{split_prediction_path} is missing or empty")

    if not is_file_non_empty(training_log_history_path):
        missings.append(f"{training_log_history_path} is missing or empty")
        
    if not is_file_non_empty(best_model_info_path):
        missings.append(f"{best_model_info_path} is missing or empty")

    for best_model_artifact_path in best_model_artifact_paths:
        if not is_file_non_empty(best_model_artifact_path):
            missings.append(f"{best_model_artifact_path} is missing or empty")
            
    return missings

def check_required_artifacts(
    require_tokenization_config=True,
    require_comparison_artifacts=True,
    require_model_key_artifacts=True,
) -> None:
    """
    Validate all required artifacts before running notebook
    - require_tokenization_config: Whether "tokenization_config.json" must exist
    - require_comparison_artifacts: Whether comparison artifacts must exist
    - require_model_key_artifacts: Whether model dependent artifacts must exist
    Remark:
    - Inference typically requires either model file or weights file (depending on loading strategy)
    """
    missings = []

    token_cfg_path = get_tokenization_config_path()
    
    if require_tokenization_config and not is_file_non_empty(token_cfg_path):
        missings.append(f"{token_cfg_path} is missing or empty")

    if require_comparison_artifacts:
        missings.extend(check_required_comparison_artifacts())
    
    if require_model_key_artifacts:
        missings.extend(check_model_key_artifacts(BERT_KEY))
        missings.extend(check_model_key_artifacts(FINBERT_KEY))
    
    if missings:
        print("Some required artifacts are missing.")
        
        for missing in missings:
            print(missing)
        
        raise RuntimeError("Required artifacts are missing. Please generate them in prior notebooks")
    else:
        print("All required artifacts are found. Notebook is ready")

def check_required_final_selection() -> None:
    path = get_final_model_selection_path()
    check_required_data(path)
    
    print(f"Final model selection artifact found: {path}")
