from transformers import AutoModelForSequenceClassification, PreTrainedModel

def build_sequence_classifier(
    model_id: str,
    num_labels: int,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str]
) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label=id_to_label
    )
