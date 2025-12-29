from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = BASE_PATH / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

DATA_OUTPUT_DIRS = [
    DATA_DIR,
    DATA_PROCESSED_DIR
]

# Dataset
TEXT_COL = "text"
LABEL_COL = "label"
DATASET_FILENAME = "financial_phrasebank.csv"   # Raw dataset filename
DATASET_NAME = Path(DATASET_FILENAME).stem
DATASET_EXT = Path(DATASET_FILENAME).suffix
DATASET_CLEAN_FILENAME = f"{DATASET_NAME}_clean{DATASET_EXT}"  # Cleaned dataset filename
RAW_FILE = DATA_RAW_DIR / DATASET_FILENAME
CLEAN_FILE = DATA_PROCESSED_DIR /  DATASET_CLEAN_FILENAME

# Artifacts directories
BERT = "bert-base"
FINBERT = "finbert"

ARTIFACTS_DIR = BASE_PATH / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESSING_DIR = ARTIFACTS_DIR / "preprocessing"
RESULTS_DIR = ARTIFACTS_DIR / "results"
RESULTS_BERT_DIR = RESULTS_DIR / BERT
RESULTS_FINBERT_DIR = RESULTS_DIR / FINBERT
RESULTS_COMPARISON_DIR = RESULTS_DIR / "comparison"
TUNING_MODELS_DIR = MODELS_DIR / "tuning"
BEST_MODELS_DIR = MODELS_DIR / "best"

ARTIFACTS_DIRS = [
    ARTIFACTS_DIR,
    MODELS_DIR,
    PREPROCESSING_DIR,
    RESULTS_DIR,
    RESULTS_COMPARISON_DIR,
    TUNING_MODELS_DIR,
    BEST_MODELS_DIR,
]

# Artifacts file names
TOKENIZATION_CONFIG_NAME = "tokenization_config.json"
LABEL_MAP_NAME = "label_map.json"
TOKEN_LENGTH_STATS_NAME =  "token_length_stats.csv"
TRUNCATION_RATES_NAME =  "truncation_rates.csv"
TRUNCATION_SHORTLIST_NAME = "truncation_shortlist_threshold.csv"
TRAINING_LOG_HISTORY_NAME = "training_log_history.csv"
TRAIN_METRICS_NAME = "train_metrics.json"
VALIDATION_METRICS_NAME = "val_metrics.json"
TEST_METRICS_NAME = "test_metrics.json"
VALIDATION_EVALUATION_NAME = "val_evaluation.json"
TEST_EVALUATION_NAME = "test_evaluation.json"
VALIDATION_PREDICTION_NAME = "val_prediction.csv"
TEST_PREDICTION_NAME = "test_prediction.csv"
FINAL_MODEL_SELECTION_NAME = "final_model_selection.json"

BEST_MODEL_INFO_NAME = "best_model_info.json"
BEST_MODEL_CONFIG_NAME = "config.json"
BEST_MODEL_SAFE_TENSORS_NAME = "model.safetensors"
BEST_MODEL_SPECIAL_TOKEN_MAP_NAME = "special_tokens_map.json"
BEST_MODEL_TOKENIZER_NAME = "tokenizer.json"
BEST_MODEL_TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
BEST_MODEL_TRAINING_ARGS_NAME = "training_args.bin"
BEST_MODEL_VOCAB_NAME = "vocab.txt"

# Artifacts file paths
TOKENIZATION_CONFIG_PATH = PREPROCESSING_DIR / TOKENIZATION_CONFIG_NAME

TOKEN_LENGTH_STATS_PATH = RESULTS_COMPARISON_DIR / TOKEN_LENGTH_STATS_NAME
TRUNCATION_RATES_PATH = RESULTS_COMPARISON_DIR / TRUNCATION_RATES_NAME
TRUNCATION_SHORTLIST_PATH = RESULTS_COMPARISON_DIR / TRUNCATION_SHORTLIST_NAME

# Required directories exist before any execution
REQUIRED_DIRS = DATA_OUTPUT_DIRS + ARTIFACTS_DIRS

# HuggingFace models mapping
# Any key changes: Required to synchronize in src/types.py
BERT_KEY = BERT
FINBERT_KEY = FINBERT

HF_MODELS = {
    BERT_KEY: {
        "model_id": "bert-base-uncased",
        "description": "General-domain BERT base (uncased)"
    },
    FINBERT_KEY: {
        "model_id": "ProsusAI/finbert",
        "description": "Financial-domain BERT fine-tuned on sentiment"        
    }
}

RANDOM_STATE = 42


