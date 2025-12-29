from dataclasses import dataclass
from transformers import Trainer, PreTrainedTokenizerBase, PreTrainedModel
from datasets import Dataset
from utils import to_list, decode_labels

import numpy as np
import pandas as pd
import torch

@dataclass(frozen=True)
class InferenceContext:
    device: torch.device
    max_length: int
    id2label: dict[int, str]

def build_prediction_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id_to_label: dict[int, str]
) -> pd.DataFrame:
    y_true_label = decode_labels(to_list(y_true), id_to_label)
    y_pred_label = decode_labels(to_list(y_pred), id_to_label)
    
    return pd.DataFrame({
        "y_true": y_true,
        "y_true_label": y_true_label,
        "y_pred": y_pred,
        "y_pred_label": y_pred_label,
        "is_correct": (y_true == y_pred)
    })

def predict_labels(
    trainer: Trainer,
    ds_test: Dataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For evaluation only
    """
    pred_out = trainer.predict(test_dataset=ds_test)
    
    logits = pred_out.predictions
    y_true = pred_out.label_ids
    y_pred = np.argmax(logits, axis=-1)
    
    return logits, y_true, y_pred

def validate_model_id2label(model: PreTrainedModel) -> dict[int, str]:
    raw_id2label = model.config.id2label
    
    if not raw_id2label:
        raise ValueError("model.config.id2label is empty")

    try:
        id2label = {int(k): v for k, v in raw_id2label.items()}    # Normalize the key to int
    except Exception as e:
        raise ValueError(f"Invalid id2label keys: {list(raw_id2label.keys())}") from e

    num_labels = int(model.config.num_labels)
    expected = set(range(num_labels))
    actual = set(id2label.keys())
    
    if actual != expected:
        raise ValueError(f"id2label keys mismatch: missing={expected-actual}, extra={actual-expected}")

    return id2label

def validate_model_device(
    model: PreTrainedModel,
    device: torch.device
) -> None:
    model_device = next(model.parameters()).device

    if model_device.type != device.type:
        raise ValueError(f"Model on {model_device}, expected {device}. Did you call model.to(device)?")

    # only check index when cuda:x is specified
    if device.type == "cuda" and device.index is not None:
        if model_device.index != device.index:
            raise ValueError(f"Model on cuda:{model_device.index}, expected cuda:{device.index}")

def prepare_inference_context(
    model: PreTrainedModel,
    device: torch.device,
    max_length: int,
) -> InferenceContext:
    """
    Prepare inference context
    """
    model.to(device)
    model.eval()

    validate_model_device(model, device)
    id2label = validate_model_id2label(model)

    return InferenceContext(
        device=device,
        max_length=max_length,
        id2label=id2label,
    )

def predict(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    ctx: InferenceContext
) -> pd.DataFrame:
    """
    For inference:
    Pre-requisites before executing this function:
    - ctx is created by prepare_inference_context
    """
    if model.training:
        model.eval()
    
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=ctx.max_length,
        return_tensors="pt"
    )    
    enc = {key: value.to(ctx.device) for key, value in enc.items()}     # Move inputs to the same device of the model

    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1)   # (batch, num_labels)
        pred_ids = probs.argmax(dim=-1)     # (batch, )
    
    pred_ids_cpu = pred_ids.detach().cpu().tolist()  # Ensure CPU list for downstream use
    probs_cpu = probs.detach().cpu().numpy()
    
    if probs_cpu.shape[1] != len(ctx.id2label):
        raise ValueError(f"num_labels mismatch: probs={probs_cpu.shape[1]} id2label={len(ctx.id2label)}")

    pred_labels = [ctx.id2label[id] for id in pred_ids_cpu]    
    prob_cols = [f"prob_{ctx.id2label[i]}" for i in range(len(ctx.id2label))]

    df = pd.DataFrame({
        "text": texts,
        "pred_id": pred_ids_cpu,
        "pred_label": pred_labels,
    })
    
    
    df_probs = pd.DataFrame(probs_cpu, columns=prob_cols)

    return pd.concat([df, df_probs], axis=1)
