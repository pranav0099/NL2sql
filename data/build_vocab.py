"""
Vocabulary Builder Module
Builds vocabulary from processed WikiSQL and Spider datasets
with separate NL and SQL tokenization.

Author: Pranav
Date: 2026-04-02

Features:
  - Separate NL and SQL vocabularies
  - Tokenization preserving SQL keywords and operators
  - Minimum frequency filtering
  - Special tokens: <pad>, <sos>, <eos>, <unk>
  - Enrich records with token IDs
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

# Add project root to sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    PROCESSED_DATA, VOCAB_FILE,
    WIKISQL_DATA, SPIDER_DATA,
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    SQL_KEYWORDS, SPECIAL_TOKENS,
    INTENTS, INTENT2IDX
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# TOKENIZATION FUNCTIONS
# ============================================================================

def tokenize_nl(text: str) -> List[str]:
    """
    Tokenize natural language question.

    Strategy:
      - Lowercase
      - Split on whitespace and punctuation
      - Keep contractions together
      - Remove excessive punctuation

    Args:
        text: Input question string

    Returns:
        List of tokens
    """
    # Convert to lowercase
    text = text.lower()

    # Replace common punctuation with spaces
    # Keep basic symbols like @, $, % as they might be meaningful
    text = re.sub(r'[^\w\s@$%]', ' ', text)

    # Split on whitespace
    tokens = text.split()

    # Remove empty tokens
    tokens = [t for t in tokens if t.strip()]

    return tokens


def tokenize_sql(sql: str) -> List[str]:
    """
    Tokenize SQL query preserving keywords and operators.

    Strategy:
      - Preserve SQL keywords (uppercase or lowercase)
      - Split multi-character operators (>=, <=, !=, <>, =)
      - Split parentheses and commas as separate tokens
      - Keep string literals intact

    Args:
        sql: SQL query string

    Returns:
        List of tokens
    """
    # Remove extra whitespace
    sql = ' '.join(sql.split())

    # Pattern to match:
    # 1. String literals (single quotes)
    # 2. Operators (>=, <=, !=, <>, =, >, <)
    # 3. Parentheses and commas
    # 4. Identifiers and keywords (split by whitespace)

    tokens = []
    i = 0
    n = len(sql)

    while i < n:
        # Skip whitespace
        if sql[i].isspace():
            i += 1
            continue

        # Check for string literal
        if sql[i] == "'":
            # Find closing quote
            j = i + 1
            while j < n and sql[j] != "'":
                j += 1
            if j < n:
                tokens.append(sql[i:j+1])  # Include quotes
                i = j + 1
            else:
                tokens.append(sql[i:])
                break
            continue

        # Check for multi-char operators (>=, <=, !=, <>)
        if i + 1 < n and sql[i:i+2] in ['>=', '<=', '!=', '<>']:
            tokens.append(sql[i:i+2])
            i += 2
            continue

        # Check for single-char operators (=, >, <)
        if sql[i] in ['=', '>', '<']:
            tokens.append(sql[i])
            i += 1
            continue

        # Check for parentheses and commas
        if sql[i] in ['(', ')', ',']:
            tokens.append(sql[i])
            i += 1
            continue

        # Regular identifier/keyword - read until next delimiter
        j = i
        while j < n and not sql[j].isspace() and sql[j] not in ['(', ')', ',', '=', '>', '<', '!', "'"]:
            j += 1

        token = sql[i:j]
        if token:
            tokens.append(token)
            i = j
        else:
            # Safety: if no token was consumed (e.g. standalone '!' or other
            # unrecognized character), skip it to prevent infinite loop
            tokens.append(sql[i])
            i += 1

    # Normalize: lowercase keywords but preserve case for identifiers?
    # Strategy: Keep as-is for now, will normalize in vocab building
    normalized = []
    for token in tokens:
        # Check if it's a SQL keyword (case-insensitive)
        token_upper = token.upper()
        if token_upper in SQL_KEYWORDS:
            normalized.append(token_upper)
        elif token.startswith("'") and token.endswith("'"):
            # String literal - keep as is
            normalized.append(token)
        else:
            # Identifier - lowercase for consistency
            normalized.append(token.lower())

    return normalized


# ============================================================================
# VOCABULARY CLASS
# ============================================================================

class Vocabulary:
    """
    Vocabulary class for token to index mapping.

    Attributes:
        tok2idx: Dictionary mapping token to index
        idx2tok: List mapping index to token
        counter: Counter of token frequencies
        min_freq: Minimum frequency for inclusion
    """

    def __init__(self, min_freq: int = 2):
        """
        Initialize vocabulary.

        Args:
            min_freq: Minimum frequency for token inclusion
        """
        self.tok2idx: Dict[str, int] = {}
        self.idx2tok: List[str] = []
        self.counter: Counter = Counter()
        self.min_freq = min_freq
        self._special_tokens_added = False

    def add_token(self, token: str) -> None:
        """Add token to counter."""
        self.counter[token] += 1

    def add_tokens(self, tokens: List[str]) -> None:
        """Add multiple tokens."""
        self.counter.update(tokens)

    def build(self, force_include: Set[str] = None) -> None:
        """
        Build vocabulary from counter.

        Args:
            force_include: Set of tokens to always include (e.g., SQL keywords)
        """
        if force_include is None:
            force_include = set()

        # Start with special tokens
        self.tok2idx = {}
        self.idx2tok = []

        # Add special tokens first
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.tok2idx[token] = idx
            self.idx2tok.append(token)

        # Add force-included tokens (SQL keywords)
        for token in sorted(force_include):
            if token not in self.tok2idx:
                self.tok2idx[token] = len(self.tok2idx)
                self.idx2tok.append(token)

        # Add tokens that meet minimum frequency
        for token, count in self.counter.most_common():
            if count >= self.min_freq and token not in self.tok2idx:
                self.tok2idx[token] = len(self.tok2idx)
                self.idx2tok.append(token)

        logger.info(f"Built vocabulary with {len(self.tok2idx)} tokens")
        logger.info(f"  Special tokens: {len(SPECIAL_TOKENS)}")
        logger.info(f"  SQL keywords: {len(force_include)}")
        logger.info(f"  Regular tokens: {len(self.tok2idx) - len(SPECIAL_TOKENS) - len(force_include)}")

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Encode list of tokens to indices.

        Args:
            tokens: List of tokens

        Returns:
            List of token indices
        """
        return [self.tok2idx.get(token, self.tok2idx[UNK_TOKEN]) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """
        Decode list of indices to tokens.

        Args:
            indices: List of token indices

        Returns:
            List of tokens
        """
        return [self.idx2tok[idx] for idx in indices if 0 <= idx < len(self.idx2tok)]

    def to_dict(self) -> Dict[str, Any]:
        """Export vocabulary to dictionary."""
        return {
            "tok2idx": self.tok2idx.copy(),
            "idx2tok": self.idx2tok.copy(),
            "min_freq": self.min_freq,
            "num_tokens": len(self.tok2idx)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vocabulary':
        """Load vocabulary from dictionary."""
        vocab = cls(min_freq=data.get("min_freq", 2))
        vocab.tok2idx = data["tok2idx"]
        vocab.idx2tok = data["idx2tok"]
        return vocab

    def __len__(self) -> int:
        return len(self.tok2idx)

    def __contains__(self, token: str) -> bool:
        return token in self.tok2idx


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load JSON data from file.

    Args:
        file_path: Path to JSON file

    Returns:
        List of records
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Handle dictionary format (like HuggingFace datasets)
            return [data]
        elif isinstance(data, list):
            return data
        else:
            logger.error(f"Unexpected data type from {file_path}: {type(data)}")
            return []
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return []


def load_all_splits() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load all splits from WikiSQL and Spider datasets.

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    logger.info("Loading dataset splits...")

    train_data = []
    val_data = []
    test_data = []

    # Load WikiSQL
    wikisql_train = load_json_file(WIKISQL_DATA / "train.json")
    wikisql_val = load_json_file(WIKISQL_DATA / "validation.json")
    wikisql_test = load_json_file(WIKISQL_DATA / "test.json")

    logger.info(f"WikiSQL: train={len(wikisql_train)}, val={len(wikisql_val)}, test={len(wikisql_test)}")

    train_data.extend(wikisql_train)
    val_data.extend(wikisql_val)
    test_data.extend(wikisql_test)

    # Load Spider
    spider_train = load_json_file(SPIDER_DATA / "train.json")
    spider_val = load_json_file(SPIDER_DATA / "validation.json")
    spider_test = load_json_file(SPIDER_DATA / "test.json")

    logger.info(f"Spider: train={len(spider_train)}, val={len(spider_val)}, test={len(spider_test)}")

    train_data.extend(spider_train)
    val_data.extend(spider_val)
    test_data.extend(spider_test)

    logger.info(f"Total: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    return train_data, val_data, test_data


def enrich_with_tokens(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add tokenized and encoded fields to each record.

    Args:
        data: List of dataset records

    Returns:
        Enriched records with tokens and ids
    """
    enriched = []

    for record in data:
        # Tokenize NL
        nl_tokens = tokenize_nl(record["question"])

        # Tokenize SQL
        sql_tokens = tokenize_sql(record["query"])

        # Create enriched record
        enriched_record = record.copy()
        enriched_record["nl_tokens"] = nl_tokens
        enriched_record["sql_tokens"] = sql_tokens
        # Will add ids after vocab is built
        enriched.append(enriched_record)

    return enriched


def compute_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute dataset statistics.

    Args:
        data: Dataset records

    Returns:
        Statistics dictionary
    """
    stats = {
        "total_records": len(data),
        "intent_distribution": {},
        "nl_lengths": [],
        "sql_lengths": [],
        "total_nl_tokens": 0,
        "total_sql_tokens": 0,
        "unique_nl_tokens": set(),
        "unique_sql_tokens": set()
    }

    for record in data:
        # Intent
        intent = record.get("intent", "UNKNOWN")
        stats["intent_distribution"][intent] = stats["intent_distribution"].get(intent, 0) + 1

        # Token lengths
        nl_tokens = record.get("nl_tokens", [])
        sql_tokens = record.get("sql_tokens", [])

        stats["nl_lengths"].append(len(nl_tokens))
        stats["sql_lengths"].append(len(sql_tokens))
        stats["total_nl_tokens"] += len(nl_tokens)
        stats["total_sql_tokens"] += len(sql_tokens)
        stats["unique_nl_tokens"].update(nl_tokens)
        stats["unique_sql_tokens"].update(sql_tokens)

    # Compute summary stats
    import numpy as np
    stats["unique_nl_tokens"] = len(stats["unique_nl_tokens"])
    stats["unique_sql_tokens"] = len(stats["unique_sql_tokens"])
    stats["avg_nl_length"] = float(np.mean(stats["nl_lengths"]))
    stats["avg_sql_length"] = float(np.mean(stats["sql_lengths"]))
    stats["max_nl_length"] = int(np.max(stats["nl_lengths"]))
    stats["max_sql_length"] = int(np.max(stats["sql_lengths"]))
    stats["min_nl_length"] = int(np.min(stats["nl_lengths"]))
    stats["min_sql_length"] = int(np.min(stats["sql_lengths"]))
    stats["median_nl_length"] = float(np.median(stats["nl_lengths"]))
    stats["median_sql_length"] = float(np.median(stats["sql_lengths"]))

    return stats


def print_statistics(stats: Dict[str, Any], title: str) -> None:
    """
    Print formatted statistics.

    Args:
        stats: Statistics dictionary
        title: Title for printing
    """
    logger.info("\n" + "="*60)
    logger.info(title)
    logger.info("="*60)
    logger.info(f"Total records: {stats['total_records']}")
    logger.info(f"Total NL tokens: {stats['total_nl_tokens']}")
    logger.info(f"Total SQL tokens: {stats['total_sql_tokens']}")
    logger.info(f"Unique NL tokens: {stats['unique_nl_tokens']}")
    logger.info(f"Unique SQL tokens: {stats['unique_sql_tokens']}")
    logger.info(f"\nNL sequence length: min={stats['min_nl_length']}, "
                f"max={stats['max_nl_length']}, avg={stats['avg_nl_length']:.1f}, "
                f"median={stats['median_nl_length']:.1f}")
    logger.info(f"SQL sequence length: min={stats['min_sql_length']}, "
                f"max={stats['max_sql_length']}, avg={stats['avg_sql_length']:.1f}, "
                f"median={stats['median_sql_length']:.1f}")

    logger.info("\nIntent Distribution:")
    for intent, count in sorted(stats["intent_distribution"].items(), key=lambda x: x[1], reverse=True):
        pct = (count / stats["total_records"]) * 100
        logger.info(f"  {intent:20s}: {count:5d} ({pct:5.1f}%)")

    logger.info("="*60 + "\n")


# ============================================================================
# MAIN BUILD FUNCTION
# ============================================================================

def build_vocabulary() -> None:
    """
    Main function: Build vocabularies and enrich datasets.
    """
    logger.info("="*80)
    logger.info("BUILDING VOCABULARY")
    logger.info("="*80)

    # Load all splits
    train_data, val_data, test_data = load_all_splits()

    if not train_data:
        raise ValueError("No training data loaded! Cannot build vocabulary.")

    # Enrich with tokens
    logger.info("Tokenizing datasets...")
    train_enriched = enrich_with_tokens(train_data)
    val_enriched = enrich_with_tokens(val_data)
    test_enriched = enrich_with_tokens(test_data)

    # Build NL vocabulary
    logger.info("\nBuilding NL vocabulary...")
    nl_vocab = Vocabulary(min_freq=2)
    for record in train_enriched:
        nl_vocab.add_tokens(record["nl_tokens"])

    # Build SQL vocabulary
    logger.info("Building SQL vocabulary...")
    sql_vocab = Vocabulary(min_freq=2)
    for record in train_enriched:
        sql_vocab.add_tokens(record["sql_tokens"])

    # Add all SQL keywords to SQL vocab (even if not seen)
    logger.info(f"Adding {len(SQL_KEYWORDS)} SQL keywords to SQL vocabulary...")
    sql_vocab.build(force_include=SQL_KEYWORDS)

    # Build NL vocab (no force-include)
    nl_vocab.build()

    # Compute statistics
    logger.info("\nComputing statistics...")
    train_stats = compute_statistics(train_enriched)
    print_statistics(train_stats, "TRAINING SET STATISTICS")

    # Encode sequences with vocab
    logger.info("Encoding sequences...")
    for record in train_enriched + val_enriched + test_enriched:
        record["nl_ids"] = nl_vocab.encode(record["nl_tokens"])
        record["sql_ids"] = sql_vocab.encode(record["sql_tokens"])
        record["intent_idx"] = INTENT2IDX.get(record.get("intent", "SELECT"), 0)

    # Save enriched datasets
    logger.info("\nSaving enriched datasets...")
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train.json": train_enriched,
        "val.json": val_enriched,
        "test.json": test_enriched
    }

    for filename, data in datasets.items():
        output_path = PROCESSED_DATA / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved {len(data)} records to {output_path}")

    # Save vocabulary
    vocab_data = {
        "nl_vocab": nl_vocab.to_dict(),
        "sql_vocab": sql_vocab.to_dict(),
        "special_tokens": SPECIAL_TOKENS,
        "intents": INTENTS,
        "intent2idx": INTENT2IDX,
        "statistics": {
            "train": train_stats
        }
    }

    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved vocabulary to {VOCAB_FILE}")
    logger.info(f"NL vocabulary size: {len(nl_vocab)}")
    logger.info(f"SQL vocabulary size: {len(sql_vocab)}")

    # Print sample
    logger.info("\nSample Encoding:")
    sample = train_enriched[0]
    logger.info(f"  Question: {sample['question']}")
    logger.info(f"  NL tokens: {sample['nl_tokens'][:10]}")
    logger.info(f"  NL ids: {sample['nl_ids'][:10]}")
    logger.info(f"  SQL: {sample['query'][:80]}...")
    logger.info(f"  SQL tokens: {sample['sql_tokens'][:10]}")
    logger.info(f"  SQL ids: {sample['sql_ids'][:10]}")
    logger.info(f"  Intent: {sample['intent']} ({sample['intent_idx']})")

    logger.info("="*80)
    logger.info("VOCABULARY BUILD COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    try:
        PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
        build_vocabulary()
    except Exception as e:
        logger.error(f"Vocabulary build failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
