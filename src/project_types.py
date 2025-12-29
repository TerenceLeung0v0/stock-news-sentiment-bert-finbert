from typing import Literal

# Synchronize with HF_MODELS keys: BERT_KEY, FINBERT_KEY
ModelKey = Literal["bert-base", "finbert"]

MetricSplit = Literal["train", "val", "test"]
EvalSplit = Literal["val", "test"]

NormalType = Literal["true", "pred", "all"] | None
