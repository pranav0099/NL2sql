"""
NL2SQL Configuration Module
Centralized configuration for the entire project

Author: Pranav
Date: 2026-04-02
"""

import os
from pathlib import Path
import torch

# ============================================================================
# PROJECT ROOT AND MAIN DIRECTORIES
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATABASE = ROOT / "database"
LOGS = ROOT / "logs"
CONFIG = ROOT / "config"

# ============================================================================
# DATA SUBDIRECTORIES
# ============================================================================

RAW_DATA = DATA / "raw"
PROCESSED_DATA = DATA / "processed"
WIKISQL_DATA = DATA / "wikisql"
SPIDER_DATA = DATA / "spider"

# ============================================================================
# DATABASE PATHS
# ============================================================================

SAMPLE_DB = DATABASE / "sample.db"
SCHEMA_FILE = DATABASE / "schema.json"

# ============================================================================
# MODEL PATHS
# ============================================================================

VOCAB_FILE = MODELS / "vocab.json"
TFIDF_MODEL = MODELS / "tfidf_vectorizer.joblib"
TRANSFORMER_MODEL = MODELS / "transformer_classifier.joblib"
PATTERN_DB = MODELS / "patterns.json"

# Phase 3: ML Classifier
ML_MODEL_PATH = MODELS / "saved" / "intent_classifier.pkl"
VECTORIZER_PATH = MODELS / "saved" / "tfidf_vectorizer.pkl"

# ============================================================================
# LOG FILES
# ============================================================================

MAIN_LOG = LOGS / "nl2sql.log"
TRAIN_LOG = LOGS / "training.log"
INFERENCE_LOG = LOGS / "inference.log"

# ============================================================================
# ML HYPERPARAMETERS
# ============================================================================

# TF-IDF Settings
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 2
TFIDF_CONFIDENCE_THRESHOLD = 0.70

# ============================================================================
# DEEP LEARNING TRANSFORMER HYPERPARAMETERS
# ============================================================================

EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 3  # Used for both encoder and decoder
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
FFN_DIM = 512
DROPOUT = 0.1

# Training hyperparameters (Phase 4)
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 30
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 5  # Note: config also has EARLY_STOPPING_PATIENCE (legacy)

# Sequence length limits
MAX_SRC_LEN = 128
MAX_TGT_LEN = 128

# Paths for DL model (Phase 4)
DL_MODEL_PATH = "models/saved/transformer_nl2sql.pt"
VOCAB_PATH = "data/processed/vocab.json"


# ============================================================================
# VOCABULARY CONFIGURATION
# ============================================================================

VOCAB_SIZE = 16000    # Target vocabulary size (both NL and SQL)
MIN_FREQ = 1          # Minimum token frequency for inclusion (lower = larger vocab)

# ============================================================================
# SPECIAL TOKENS
# ============================================================================

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Token indices ( conventionally 0,1,2,3 for special tokens)
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ============================================================================
# INTENT LABELS
# ============================================================================

INTENTS = [
    "SELECT",                      # Simple SELECT * FROM table
    "SELECT_WHERE",               # SELECT with WHERE clause
    "SELECT_AGGREGATE",           # SELECT with COUNT, SUM, AVG, MAX, MIN
    "SELECT_ORDER",               # SELECT with ORDER BY
    "SELECT_JOIN",                # SELECT with JOIN
    "SELECT_GROUP",               # SELECT with GROUP BY
    "SELECT_LIMIT",               # SELECT with LIMIT
    "COMPLEX"                     # Multiple clauses (WHERE + ORDER + LIMIT etc.)
]

INTENT2IDX = {intent: idx for idx, intent in enumerate(INTENTS)}
IDX2INTENT = {idx: intent for intent, idx in INTENT2IDX.items()}

# ============================================================================
# NLP PIPELINE CONFIGURATION
# ============================================================================

# spaCy model name
SPACY_MODEL = "en_core_web_sm"

# Schema matching thresholds
SCHEMA_SIMILARITY_THRESHOLD = 0.3  # Minimum cosine similarity for table/column matches
FUZZY_THRESHOLD = 70               # Minimum fuzzy match score (0-100)

# General confidence threshold
CONFIDENCE_THRESHOLD = 0.70        # Minimum confidence for predictions

# Conversation context settings
MAX_CONTEXT_TURNS = 5               # Maximum number of turns to keep in context window

# ============================================================================
# SQL KEYWORDS (used in tokenization)
# ============================================================================

SQL_KEYWORDS = {
    # DML
    "SELECT", "FROM", "WHERE", "JOIN", "INNER JOIN", "LEFT JOIN",
    "RIGHT JOIN", "OUTER JOIN", "ON", "AND", "OR", "NOT",
    # Aggregation
    "COUNT", "SUM", "AVG", "MAX", "MIN",
    # Grouping & Ordering
    "GROUP BY", "HAVING", "ORDER BY", "ASC", "DESC",
    # Limits & Sets
    "LIMIT", "OFFSET", "DISTINCT", "ALL",
    # Comparison
    "=", ">", "<", ">=", "<=", "!=", "<>",
    # Misc
    "AS", "IN", "LIKE", "IS", "NULL", "TRUE", "FALSE",
    "CASE", "WHEN", "THEN", "ELSE", "END", "BETWEEN"
}

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DIRECTORIES TO CREATE
# ============================================================================

DIRS_TO_CREATE = [
    DATA,
    RAW_DATA,
    PROCESSED_DATA,
    WIKISQL_DATA,
    SPIDER_DATA,
    MODELS,
    DATABASE,
    LOGS,
    CONFIG
]


def create_dirs():
    """
    Create all project directories if they don't exist.
    Called on import to ensure folder structure exists.
    """
    for directory in DIRS_TO_CREATE:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Ensured directory exists: {directory}")


# ============================================================================
# INITIALIZATION
# ============================================================================

# Create directories on module import
create_dirs()

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config():
    """
    Validate configuration settings and print summary.
    Called by setup.py to verify installation.
    """
    print("\n" + "="*60)
    print("NL2SQL CONFIGURATION VALIDATION")
    print("="*60)
    print(f"Root Directory: {ROOT}")
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if DEVICE.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"\nNumber of Intents: {len(INTENTS)}")
    print(f"Intents: {', '.join(INTENTS)}")
    print(f"\nSpecial Tokens: {SPECIAL_TOKENS}")
    print(f"\nTransformer Config:")
    print(f"  Embed Dim: {EMBED_DIM}")
    print(f"  Num Heads: {NUM_HEADS}")
    print(f"  Num Layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"\nTF-IDF Config:")
    print(f"  Max Features: {TFIDF_MAX_FEATURES}")
    print(f"  N-Gram Range: {TFIDF_NGRAM_RANGE}")
    print(f"  Min DF: {TFIDF_MIN_DF}")
    print(f"  Confidence Threshold: {TFIDF_CONFIDENCE_THRESHOLD}")
    print("="*60 + "\n")


if __name__ == "__main__":
    validate_config()
