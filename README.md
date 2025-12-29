# Financial News Sentiment Classification (BERT vs FinBERT)

## Overview

This project extends a financial news sentiment classification task using **pretrained Transformer models**.
The objective is to evaluate whether a **domain-adapted Transformer (FinBERT)** provides measurable performance gains over a **general-purpose Transformer (BERT)** under a controlled and reproducible experimental setup.

The project emphasizes **model comparison, selection discipline, and inference reproducibility**.

## Dataset

Given short financial news sentences, classify the sentiment into:

- **Negative**
- **Neutral**
- **Positive**

This is a **multi-class text classification** problem with class imbalance.

- **Source**: Financial PhraseBank (Kaggle)
- **Samples**: ~5,800 financial news sentences
- **Labels**: negative / neutral / positive

The raw dataset is cleaned and normalized before modeling.

## Project structure
```text
stock-news-sentiment-tensorflow/
├── notebooks/
├── src/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── preprocessing/
│	│	├── finbert/
│	│	├── bert-base/
│   ├── models/
│   │   ├── best/
│	│	│	├── finbert/
│	│	│	├── bert-base/
│   └── results/
│		├── finbert/
│		├── comparison/
│		├── bert-base/
├── README.md
├── requirements.txt
└── .gitignore
```

## Methodology

The project follows a structured Transformer-based workflow:

1. **Data Preprocessing**
   Cleaning, normalization, and label standardization of financial news text.

2. **Tokenization Analysis**
   Analysis of token length distribution to select a consistent maximum sequence length shared across models.

3. **Model Training**
   - BERT (bert-base-uncased) as a general-domain baseline
   - FinBERT (ProsusAI/finbert) as a finance-domain pretrained model
   - Both models are fine-tuned using identical splits and training configurations.

4. **Model Evaluation**
   Models are evaluated using:
   - Macro F1-score
   - Confusion matrix and per-class metrics

5. **Model Selection**
   The final model is selected using validation Macro F1-score only, with the test set reserved strictly for reporting.

6. **Final Inference**
   The selected model is loaded from persisted artifacts and used for inference on unseen examples.

## Evaluation Protocol

A fixed train / validation / test split is used across all experiments.

Validation metrics are used exclusively for model selection.

The test set is evaluated once for final reporting.

Macro F1-score is used as the primary selection metric to account for class imbalance.

## Final Result & Model Comparison



## Interpretation

Validation results indicate that FinBERT provides modest but consistent gains over BERT in Macro F1-score, suggesting that domain-adapted pretraining captures financial sentiment nuances more effectively.

However, performance improvements remain bounded by dataset size and label ambiguity, consistent with observations from Project 1.

Weighted metrics remain stable across models, indicating that improvements are primarily driven by minority class handling rather than overall accuracy gains.

## Requirements

- Python >= 3.10
- Pytorch
- Transformers (Hugging Face)
- Dataset

## Installation

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## How to Run

Execute notebooks in the following order:

1. `01_data_overview_and_pre.ipynb`
2. `02_tokenizer_analysis_max_length_selection.ipynb`
3. `03_bert_baseline_training_and_evaluation.ipynb`
4. `04_finbert_training_and_evaluation.ipynb`
5. `05_model_comparison_and_selection.ipynb`
6. `06_final_model_inference.ipynb`

## Reproducibility Notes

The final selected model is persisted along with
  - Tokenization configuration
  - Label mappings
  - Selection criteria and scores

Inference is performed using the same preprocessing artifacts to avoid configuration drift.

The authoritative model selection record is stored in:
**artifacts/results/comparison/final_model_selection.json**

## Takeaways

- Domain-adapted Transformers (FinBERT) provide measurable but bounded improvements on small financial datasets
- Macro F1-score is essential for fair comparison under class imbalance
- Validation-based model selection prevents test leakage
- Persisted artifacts enable clean and reproducible inference workflow
- Increased model complexity does not guarantee large gains when data is limited

## Future Improvements

- Apply parameter-efficient fine-tuning (LoRA / adapters)
- Explore data augmentation for minority classes
- Deploy inference via REST API

## Author

Terence Leung
