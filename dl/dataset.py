"""
NL2SQL — Phase 4: Deep Learning Dataset and DataLoader
Run: python dl/dataset.py

PyTorch Dataset and DataLoader for the Transformer.

**COMPLETE - Step 3 done**
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# ── Project root on sys.path ────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    MAX_SRC_LEN,
    MAX_TGT_LEN,
    BATCH_SIZE,
    DEVICE,
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# VOCABULARY LOADER
# ============================================================================

def load_vocab(vocab_path: str) -> Tuple[Dict, Dict]:
    """
    Load vocabulary from vocab.json and return normalised dicts.

    Supports two on-disk formats for ``idx2tok``:
      1. A JSON list  –  index is the token id.
      2. A JSON dict  –  keys may be strings ("0", "1", …) or ints.

    Both are normalised to ``dict[int, str]`` so downstream code can do
    ``idx2tok[token_id]`` without worrying about key types.

    Args:
        vocab_path: Path to the vocab.json file produced by setup.py.

    Returns:
        Tuple of ``(nl_vocab, sql_vocab)`` where each dict has keys:
            - ``tok2idx`` (dict[str, int])
            - ``idx2tok`` (dict[int, str])
            - ``size``    (int)

    Raises:
        FileNotFoundError: If *vocab_path* does not exist.
    """
    vocab_path = Path(vocab_path)

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    logger.info(f"Loading vocabulary from {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as fh:
        vocab_data = json.load(fh)

    nl_vocab = vocab_data["nl_vocab"]
    sql_vocab = vocab_data["sql_vocab"]

    # ── Normalise idx2tok to dict[int, str] ──────────────────────────────
    for vocab in (nl_vocab, sql_vocab):
        raw = vocab["idx2tok"]
        if isinstance(raw, list):
            vocab["idx2tok"] = {i: tok for i, tok in enumerate(raw)}
        else:
            vocab["idx2tok"] = {int(k): v for k, v in raw.items()}

        # Ensure 'size' key is present and derived from tok2idx
        if "size" not in vocab:
            vocab["size"] = len(vocab["tok2idx"])

    logger.info(f"NL vocabulary size : {nl_vocab['size']}")
    logger.info(f"SQL vocabulary size: {sql_vocab['size']}")

    return nl_vocab, sql_vocab


# ============================================================================
# DATASET CLASS
# ============================================================================

class NL2SQLDataset(Dataset):
    """
    PyTorch Dataset wrapping the preprocessed WikiSQL JSON records.

    Each sample contains:
        * ``src``  – padded source token-id tensor  (natural-language question)
        * ``tgt``  – padded target token-id tensor  (SQL query with SOS / EOS)
        * ``src_len`` / ``tgt_len`` – actual (unpadded) lengths
        * ``question`` / ``query`` / ``intent`` – original strings for logging
    """

    def __init__(
        self,
        data_path: str,
        nl_vocab: Dict,
        sql_vocab: Dict,
        max_src_len: int = MAX_SRC_LEN,
        max_tgt_len: int = MAX_TGT_LEN,
    ):
        """
        Initialise dataset.

        Args:
            data_path:   Path to a JSON file (train.json / val.json / test.json).
            nl_vocab:    Natural-language vocabulary dict (from ``load_vocab``).
            sql_vocab:   SQL vocabulary dict (from ``load_vocab``).
            max_src_len: Maximum source sequence length (truncated beyond this).
            max_tgt_len: Maximum target sequence length (including SOS and EOS).
        """
        self.data_path = Path(data_path)
        self.nl_vocab = nl_vocab
        self.sql_vocab = sql_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Special-token indices
        self.pad_idx = nl_vocab["tok2idx"][PAD_TOKEN]   # 0
        self.sos_idx = sql_vocab["tok2idx"][SOS_TOKEN]   # 1
        self.eos_idx = sql_vocab["tok2idx"][EOS_TOKEN]   # 2
        self.unk_idx = nl_vocab["tok2idx"][UNK_TOKEN]    # 3

        # Load and filter records
        self.records = self._load_and_filter()

        logger.info(f"Dataset initialised from {self.data_path}")
        logger.info(f"  Valid records : {len(self.records)}")
        logger.info(f"  Max src len  : {self.max_src_len}")
        logger.info(f"  Max tgt len  : {self.max_tgt_len}")

    # ------------------------------------------------------------------ IO
    def _load_and_filter(self) -> List[Dict[str, Any]]:
        """
        Load JSON records and discard any with empty ``nl_ids`` or ``sql_ids``.

        Returns:
            List of valid record dictionaries.

        Raises:
            FileNotFoundError: If *data_path* does not exist.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as fh:
            raw_records = json.load(fh)

        total_loaded = len(raw_records)
        valid: List[Dict[str, Any]] = []
        filtered = 0

        for rec in raw_records:
            nl_ids = rec.get("nl_ids", [])
            sql_ids = rec.get("sql_ids", [])

            if not nl_ids or not sql_ids:
                filtered += 1
                continue

            # Ensure int lists (JSON may deserialise as float on some systems)
            rec["nl_ids"] = [int(x) for x in nl_ids]
            rec["sql_ids"] = [int(x) for x in sql_ids]
            valid.append(rec)

        logger.info(f"Loaded {total_loaded} records, filtered {filtered} empty")

        return valid

    # --------------------------------------------------------- Dunder API
    def __len__(self) -> int:
        """Return number of valid records in this dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a single training sample.

        Args:
            idx: Index into ``self.records``.

        Returns:
            Dictionary with keys ``src``, ``tgt``, ``src_len``, ``tgt_len``,
            ``question``, ``query``, ``intent``.
        """
        record = self.records[idx]

        nl_ids = record["nl_ids"]
        sql_ids = record["sql_ids"]

        src = self._encode_nl(nl_ids)
        tgt = self._encode_sql(sql_ids)

        # Actual (unpadded) lengths
        src_len = min(len(nl_ids), self.max_src_len)
        tgt_len = min(len(sql_ids) + 2, self.max_tgt_len)  # +2 for SOS + EOS

        return {
            "src": src,
            "tgt": tgt,
            "src_len": src_len,
            "tgt_len": tgt_len,
            "question": record.get("question", ""),
            "query": record.get("query", ""),
            "intent": record.get("intent", "UNKNOWN"),
        }

    # --------------------------------------------------------- Encoding
    def _encode_nl(self, nl_ids: List[int]) -> torch.LongTensor:
        """
        Convert NL token-id list to a padded tensor of length ``max_src_len``.

        Truncates if longer, pads with ``pad_idx`` if shorter.

        Args:
            nl_ids: Raw token ids from the JSON record.

        Returns:
            ``torch.LongTensor`` of shape ``(max_src_len,)``.
        """
        ids = nl_ids[: self.max_src_len]
        padding = [self.pad_idx] * (self.max_src_len - len(ids))
        return torch.LongTensor(ids + padding)

    def _encode_sql(self, sql_ids: List[int]) -> torch.LongTensor:
        """
        Prepend SOS, append EOS to SQL token ids, then pad to ``max_tgt_len``.

        If the resulting sequence exceeds ``max_tgt_len`` it is truncated
        but EOS is always preserved at the end.

        Args:
            sql_ids: Raw SQL token ids from the JSON record.

        Returns:
            ``torch.LongTensor`` of shape ``(max_tgt_len,)``.
        """
        full = [self.sos_idx] + sql_ids + [self.eos_idx]

        # Truncate but keep EOS at the boundary
        if len(full) > self.max_tgt_len:
            full = full[: self.max_tgt_len - 1] + [self.eos_idx]

        padding = [self.pad_idx] * (self.max_tgt_len - len(full))
        return torch.LongTensor(full + padding)


# ============================================================================
# COLLATE FUNCTION
# ============================================================================

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for :class:`DataLoader`.

    All tensors are already fixed-length from the dataset, so we simply
    stack them into batch-first tensors and collect the metadata lists.

    Args:
        batch: List of samples returned by ``NL2SQLDataset.__getitem__``.

    Returns:
        Dictionary with batched tensors and metadata::

            {
                "src":       LongTensor  (batch_size, max_src_len),
                "tgt":       LongTensor  (batch_size, max_tgt_len),
                "src_len":   LongTensor  (batch_size,),
                "tgt_len":   LongTensor  (batch_size,),
                "questions": list[str],
                "queries":   list[str],
                "intents":   list[str],
            }
    """
    src = torch.stack([item["src"] for item in batch])
    tgt = torch.stack([item["tgt"] for item in batch])
    src_len = torch.LongTensor([item["src_len"] for item in batch])
    tgt_len = torch.LongTensor([item["tgt_len"] for item in batch])
    questions = [item["question"] for item in batch]
    queries = [item["query"] for item in batch]
    intents = [item["intent"] for item in batch]

    return {
        "src": src,
        "tgt": tgt,
        "src_len": src_len,
        "tgt_len": tgt_len,
        "questions": questions,
        "queries": queries,
        "intents": intents,
    }


# ============================================================================
# DATALOADER FACTORY
# ============================================================================

def get_dataloaders(
    train_path: str,
    val_path: str,
    nl_vocab: Dict,
    sql_vocab: Dict,
    batch_size: int = BATCH_SIZE,
    max_src_len: int = MAX_SRC_LEN,
    max_tgt_len: int = MAX_TGT_LEN,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation :class:`DataLoader` instances.

    Args:
        train_path:  Path to training JSON.
        val_path:    Path to validation JSON.
        nl_vocab:    NL vocabulary dict.
        sql_vocab:   SQL vocabulary dict.
        batch_size:  Batch size for both loaders.
        max_src_len: Maximum source length.
        max_tgt_len: Maximum target length.
        num_workers: Number of DataLoader worker processes.

    Returns:
        Tuple of ``(train_loader, val_loader)``.
    """
    logger.info("Creating DataLoaders …")

    train_dataset = NL2SQLDataset(
        train_path, nl_vocab, sql_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )

    val_dataset = NL2SQLDataset(
        val_path, nl_vocab, sql_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )

    use_pin = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin,
    )

    logger.info(f"Train loader : {len(train_loader)} batches  ({len(train_dataset)} samples)")
    logger.info(f"Val   loader : {len(val_loader)} batches  ({len(val_dataset)} samples)")

    return train_loader, val_loader


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NL2SQL PHASE 4 — DATASET TEST")
    print("=" * 70)

    try:
        nl_vocab, sql_vocab = load_vocab("data/processed/vocab.json")

        train_loader, val_loader = get_dataloaders(
            "data/processed/train.json",
            "data/processed/val.json",
            nl_vocab,
            sql_vocab,
            batch_size=4,
        )

        batch = next(iter(train_loader))

        print(f"\nsrc shape   : {batch['src'].shape}")
        print(f"tgt shape   : {batch['tgt'].shape}")
        print(f"src_len     : {batch['src_len']}")
        print(f"tgt_len     : {batch['tgt_len']}")
        print(f"Question[0] : {batch['questions'][0]}")
        print(f"Query[0]    : {batch['queries'][0]}")

        print("\nDataset test PASSED")

    except Exception as exc:
        logger.error(f"Dataset test failed: {exc}", exc_info=True)
        print(f"\n[FAIL] Dataset test failed: {exc}")
        sys.exit(1)
