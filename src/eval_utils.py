from typing import Sequence, Any
from project_types import NormalType
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_order: Sequence[str],
    normalize: NormalType=None,
) -> np.ndarray:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    label_ids = list(range(len(label_order)))
    
    return confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=label_ids,
        normalize=normalize
    )

def plot_confusion_matrix(
    cm: np.ndarray,
    label_order: Sequence[str],
    title: str="Confusion Matrix",
) -> None:
    tick_labels = [str(label) for label in label_order]
    fmt = "d" if cm.dtype.kind in {"i", "u"} else ".2f"
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()    

def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_order: Sequence[str],
    zero_division: int=0,
    output_dict: bool=False
) -> str | dict[str, Any]:
    """
    This function returns 2 types of report:
    - output_dict=False -> str
    - output_dict=True -> dict[str, Any]
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)    
    
    label_ids = list(range(len(label_order)))
    target_names = [str(label) for label in label_order]     # Ensure target_names are string    

    return classification_report(
        y_true_arr,
        y_pred_arr,
        labels=label_ids,
        target_names=target_names,
        zero_division=zero_division,
        output_dict=output_dict,
    )

def evaluate_and_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_order: Sequence[str],
    cm_normalize: NormalType=None,
    show_confusion_matrix: bool=True,
    plot_cm: bool=True,
    title: str="Confusion Matrix",
    show_classification_report: bool=True,
) -> dict[str, Any]:
    """
    Evaluate classification results. Optionally prints report and shows Confusion Matrix.
    Returns a dict suitable for saving as JSON.
    """
    cm = compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        label_order=label_order,
        normalize=cm_normalize
    )
    
    cr = compute_classification_report(
        y_true,
        y_pred,
        label_order=label_order,
        zero_division=0,
        output_dict=True,
    )
    
    if show_classification_report:
        print(compute_classification_report(
            y_true,
            y_pred,
            label_order=label_order,
            zero_division=0,
        ))
    
    if show_confusion_matrix:
        if plot_cm:
            plot_confusion_matrix(
                cm,
                label_order=label_order,
                title=title,
            )
        else:
            print(cm)
    
    evaluation = {
        "metadata": {
            "label_order": [str(label) for label in label_order],
            "confusion_matrix_normalize": cm_normalize,
        },
        "metrics": {
            "classification_report": cr,
        },
        "data": {
            "confusion_matrix": cm.tolist()
        }
    }
    
    return evaluation
