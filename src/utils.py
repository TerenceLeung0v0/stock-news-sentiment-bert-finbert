from typing import Sequence, Callable, Any
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import numpy as np
import pandas as pd

def to_list(data: Any) -> list:
    if isinstance(data, list):
        return data
    
    if hasattr(data, "tolist"):
        return data.tolist()
    
    return list(data)

def is_file_non_empty(path: str | Path) -> bool:
    """
    Check if input path
    - exists
    - is a file
    - is non-empty
    """
    p = Path(path)
    
    return p.exists() and p.is_file() and p.stat().st_size > 0

def token_length_summary(
    tokenizer_name: str,
    token_lens: list[int]
) -> dict:
    token_lens_arr = np.array(token_lens)
    
    summary = {
        "tokenizer": tokenizer_name,
        "sample_size": int(token_lens_arr.size),
        "mean": float(token_lens_arr.mean()),
        "p50": int(np.percentile(token_lens_arr, 50)),
        "p90": int(np.percentile(token_lens_arr, 90)),
        "p95": int(np.percentile(token_lens_arr, 95)),
        "p99": int(np.percentile(token_lens_arr, 99)),
        "max": int(token_lens_arr.max()),        
    }
    
    return summary

def truncation_rate(
    token_lens: list[int],
    max_len: int
) -> float:
    """
    Calculate truncation rate (%):
    proportion of samples with token length > max_len
    """
    token_lens_arr = np.array(token_lens)
    
    return float((token_lens_arr > max_len).mean() * 100)

def generate_truncation_candidates(
    start: int,
    end: int,
) -> list[int]:
    """
    Generate max_length candidates for Transformer tokenization
    Rules for GPU-friendly alignment:
    - Power of 2
    or
    - Multiples of 32
    """
    if start <= 0 or end <= 0:
        raise ValueError("start and end must be positive integers")
    
    if start > end:
        raise ValueError("start must be less or equal to end")
    
    candidates = []
    
    for value in range(start, end + 1):
        is_power_of_2 = (value & (value - 1)) == 0
        is_multiple_of_32 = (value % 32) == 0

        if is_power_of_2 or is_multiple_of_32:
            candidates.append(value)
            
    return candidates

def quadratic_scaling_ratio(
    token_len_a: int,
    token_len_b: int
) -> float:
    """
    Calculates the relative increase in O(L^2) attention cost
    - +ve = Cost increase
    - -ve = Cost decrease
    """
    if token_len_a <= 0 or token_len_b <= 0:
        raise ValueError("Input token_len must be positive integer")
    
    return (token_len_b**2) / (token_len_a**2) - 1

def compare_computational_cost_info(
    name: str,
    candidate: list[int],
    token_len_from: int,
    token_len_to: int,
    total_sample_size: int
) -> None:
    cost_ratio = quadratic_scaling_ratio(token_len_from, token_len_to)
    cost_sign = "increased" if cost_ratio > 0 else "decreased"
      
    tr_from = truncation_rate(candidate, token_len_from)
    tr_to = truncation_rate(candidate, token_len_to)
    tr_diff = tr_to - tr_from
    tr_diff_sign = ["increased", "Dropped"] if tr_diff > 0 else ["decreased", "Saved"]

    samples_delta = total_sample_size * abs(tr_diff) / 100
    
    print(f"\nModel: {name}")
    print(f"Max length: {token_len_from} -> {token_len_to}")
    print(f"Computational cost (O(L^2) proxy) {cost_sign}: {abs(cost_ratio * 100):.2f}%")
    print(f"Truncation rate {tr_diff_sign[0]}: {abs(tr_diff):.2f}%")
    print(f"Total sample size: {total_sample_size}")
    print(f"{tr_diff_sign[1]} samples: {samples_delta:.1f}")    

def build_label_mappings(labels: Sequence[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    Build label mappings from a sequence of labels
    - label_order: Preserve input order
    - label_to_id: dict of {label: id}
    - id_to_label: dict of {id: label}
    """
    label_order = pd.unique(labels).tolist()
    label_to_id = {label: idx for idx, label in enumerate(label_order)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    return label_order, label_to_id, id_to_label

def encode_labels(
    labels: pd.Series,
    label_to_id: dict[str, int]
) -> pd.Series:
    return labels.map(label_to_id).astype(int)

def decode_labels(
    ids: Sequence[int],
    id_to_label: dict[int, str]
) -> list[str]:
    return [id_to_label[id] for id in ids]

def check_unseen_labels(
    labels: pd.Series,
    label_to_id: dict[str, int],
    split_name: str
) -> None:
    """
    Raise error if unseen labels are found in a split.
    - If unseen is found, need to check label normalization or dataset consistency
    
    Note:
    This function assumes labels have already been cleaned and normalized during dataset preparation.
    """
    unseen = set(labels.unique()) - set(label_to_id.keys())
    
    if unseen:
        raise ValueError(f"Unseen labels are found in {split_name}: {sorted(unseen)}")

def build_tokenize_batch_fn(
    tokenizer,
    max_length: int,
    padding: str,
    truncation: bool,
    return_tensors=None
) -> Callable[[Sequence[str]], Any]:
    def tokenize_batch(texts: Sequence[str]):
        return tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    return tokenize_batch

def compute_classification_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    
    # Macro (class-balanced)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    # Weighted (distribution-aware)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )
    
    metrics = {
        "accuracy": accuracy,
        
        # Priminary metrics for model selection
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        
        # Deployment monitoring
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }
    
    return metrics
