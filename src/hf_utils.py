from typing import Callable, Sequence, Any, TypeAlias, Mapping
from datasets import Dataset

import pandas as pd

Batch: TypeAlias = Mapping[str, Sequence[Any]]
MapFn: TypeAlias = Callable[[Batch], dict[str, Any]]

def create_hf_datasets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    hf_map_fn: Callable[[dict[str, Any]], dict[str, Any]],
    text_col: str="text",
    encoded_label_col: str="label_encoded",
    remove_text_col: bool=True,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Convert split dataFrame into Dataset format for Hugging Face.
    """
    ds_train = Dataset.from_pandas(df_train[[text_col, encoded_label_col]], preserve_index=False)
    ds_val = Dataset.from_pandas(df_val[[text_col, encoded_label_col]], preserve_index=False)
    ds_test = Dataset.from_pandas(df_test[[text_col, encoded_label_col]], preserve_index=False)
    
    remove_cols = [text_col] if remove_text_col else []
    
    ds_train = ds_train.map(hf_map_fn, batched=True, remove_columns=remove_cols)
    ds_val = ds_val.map(hf_map_fn, batched=True, remove_columns=remove_cols)
    ds_test = ds_test.map(hf_map_fn, batched=True, remove_columns=remove_cols)
    
    return ds_train, ds_val, ds_test

def build_hf_map_fn(
    tokenize_batch: Callable[[Sequence[str]], Any],
    text_col: str="text",
    encoded_label_col: str="label_encoded",
    output_label_key: str="labels"
) -> MapFn:
    """
    For datasets.map(batched=True) only
    """
    def hf_tokenize_batch(batch: Batch) -> dict[str, Any]:
        enc = tokenize_batch(batch[text_col])
        enc[output_label_key] = batch[encoded_label_col]
        
        return enc
    
    return hf_tokenize_batch
