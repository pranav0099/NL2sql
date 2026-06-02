"""
NL2SQL — Phase 4: Deep Learning Transformer Model
Run: python dl/model.py

PyTorch Transformer encoder-decoder for SQL generation.

**COMPLETE - Step 4 done**
"""

import sys
from pathlib import Path
import math
from typing import Dict, Any

import torch
import torch.nn as nn

# ── Project root on sys.path ────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    DEVICE,
    EMBED_DIM,
    NUM_HEADS,
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    FFN_DIM,
    DROPOUT,
    MAX_SRC_LEN,
    MAX_TGT_LEN,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Registered as a buffer so that it moves to the correct device
    automatically but is *not* treated as a learnable parameter.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 512):
        """
        Args:
            embed_dim: Embedding / model dimension.
            dropout:   Dropout applied after adding positional encoding.
            max_len:   Maximum sequence length supported.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the positional table once
        pe = torch.zeros(max_len, embed_dim)                         # (max_len, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape → (1, max_len, D)  so we can broadcast over the batch dim
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns:
            Same shape with positional information added and dropout applied.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class NL2SQLTransformer(nn.Module):
    """
    Transformer encoder-decoder for NL → SQL generation.

    Architecture mirrors the standard *Attention Is All You Need* design
    with separate source / target embeddings and a linear output projection.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_encoder_layers: int = NUM_ENCODER_LAYERS,
        num_decoder_layers: int = NUM_DECODER_LAYERS,
        ffn_dim: int = FFN_DIM,
        dropout: float = DROPOUT,
        max_src_len: int = MAX_SRC_LEN,
        max_tgt_len: int = MAX_TGT_LEN,
        pad_idx: int = 0,
    ):
        """
        Args:
            src_vocab_size:     Source (NL) vocabulary size.
            tgt_vocab_size:     Target (SQL) vocabulary size.
            embed_dim:          Embedding / model dimension  (d_model).
            num_heads:          Number of multi-head attention heads.
            num_encoder_layers: Number of encoder Transformer blocks.
            num_decoder_layers: Number of decoder Transformer blocks.
            ffn_dim:            Inner dimension of the feed-forward network.
            dropout:            Dropout probability.
            max_src_len:        Maximum source sequence length.
            max_tgt_len:        Maximum target sequence length.
            pad_idx:            Padding token index.
        """
        super().__init__()

        # Store for checkpointing / inference
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.pad_idx = pad_idx
        self.embed_scale = math.sqrt(embed_dim)

        # ── Embeddings ───────────────────────────────────────────────────
        self.src_embedding = nn.Embedding(
            src_vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size, embed_dim, padding_idx=pad_idx
        )

        # ── Positional encoding ──────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(
            embed_dim, dropout, max_len=max(max_src_len, max_tgt_len) + 16
        )

        # ── Core Transformer ─────────────────────────────────────────────
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )

        # ── Output head ──────────────────────────────────────────────────
        self.output_proj = nn.Linear(embed_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        # ── Weight initialisation ────────────────────────────────────────
        self._init_weights()

        logger.info("NL2SQLTransformer created:")
        logger.info(f"  src_vocab  = {src_vocab_size}")
        logger.info(f"  tgt_vocab  = {tgt_vocab_size}")
        logger.info(f"  embed_dim  = {embed_dim}")
        logger.info(f"  heads      = {num_heads}")
        logger.info(f"  enc layers = {num_encoder_layers}")
        logger.info(f"  dec layers = {num_decoder_layers}")
        logger.info(f"  ffn_dim    = {ffn_dim}")
        logger.info(f"  dropout    = {dropout}")
        logger.info(f"  params     = {self.count_parameters():,}")

    # ------------------------------------------------------------------ init
    def _init_weights(self) -> None:
        """
        Initialise learnable weights.

        * Linear layers → Xavier uniform for weights, zeros for biases.
        * Embedding layers → Normal(0, d_model^{-0.5}).
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.embed_dim ** -0.5)
                # Re-zero the padding vector
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)

    # ---------------------------------------------------------------- masks
    def _make_src_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Build a boolean padding mask for the source.

        Args:
            src: ``(batch, src_len)`` token-id tensor.

        Returns:
            ``BoolTensor (batch, src_len)`` — ``True`` where ``src == pad_idx``.
        """
        return src == self.pad_idx

    def _make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Build the causal (look-ahead) mask for the decoder.

        Args:
            tgt: ``(batch, tgt_len)`` token-id tensor.

        Returns:
            ``BoolTensor (tgt_len, tgt_len)`` — upper-triangular ``True``
            entries prevent the decoder from attending to future positions.
        """
        tgt_len = tgt.size(1)
        mask = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
            diagonal=1,
        )
        return mask

    # ------------------------------------------------------------ forward
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Full encoder-decoder forward pass (teacher-forced training).

        Args:
            src: Source token ids ``(batch, src_len)``.
            tgt: Target token ids ``(batch, tgt_len)`` — typically
                 ``tgt[:, :-1]`` during training.

        Returns:
            Logits ``(batch, tgt_len, tgt_vocab_size)``.
        """
        # Embed + scale + positional encoding
        src_emb = self.pos_encoding(self.src_embedding(src) * self.embed_scale)
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * self.embed_scale)

        # Masks
        src_key_padding_mask = self._make_src_key_padding_mask(src)
        tgt_mask = self._make_tgt_mask(tgt)

        # Transformer
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # Project to vocabulary
        logits = self.output_proj(output)
        return logits

    # ----------------------------------------------------------- inference
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encode source sequence only (used during inference).

        Args:
            src: ``(batch, src_len)`` token-id tensor.

        Returns:
            Encoder memory ``(batch, src_len, embed_dim)``.
        """
        src_emb = self.pos_encoding(self.src_embedding(src) * self.embed_scale)
        src_key_padding_mask = self._make_src_key_padding_mask(src)

        # Store for later use in decode_step
        self._cached_src_padding_mask = src_key_padding_mask

        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )
        return memory

    def decode_step(
        self, tgt: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Run one decoder step on the current target prefix + encoder memory.

        Args:
            tgt:    ``(batch, current_tgt_len)`` token ids generated so far.
            memory: ``(batch, src_len, embed_dim)`` encoder output.

        Returns:
            Logits ``(batch, current_tgt_len, tgt_vocab_size)``.
        """
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * self.embed_scale)
        tgt_mask = self._make_tgt_mask(tgt)

        # Use cached src padding mask if available
        memory_key_padding_mask = getattr(
            self, "_cached_src_padding_mask", None
        )

        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.output_proj(output)
        return logits

    # ------------------------------------------------------------ utils
    def count_parameters(self) -> int:
        """
        Return the total number of *trainable* parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(nl_vocab: Dict, sql_vocab: Dict) -> NL2SQLTransformer:
    """
    Instantiate an NL2SQLTransformer using config constants.

    Args:
        nl_vocab:  NL vocabulary dict (must contain ``"size"``).
        sql_vocab: SQL vocabulary dict (must contain ``"size"``).

    Returns:
        Model moved to ``DEVICE`` and ready for training.
    """
    model = NL2SQLTransformer(
        src_vocab_size=nl_vocab["size"],
        tgt_vocab_size=sql_vocab["size"],
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        max_src_len=MAX_SRC_LEN,
        max_tgt_len=MAX_TGT_LEN,
        pad_idx=0,
    )

    model = model.to(DEVICE)
    logger.info(f"Model moved to {DEVICE}  ({model.count_parameters():,} params)")
    return model


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    from dl.dataset import load_vocab as _lv  # noqa: avoid circular at module level

    print("=" * 70)
    print("NL2SQL PHASE 4 — TRANSFORMER MODEL TEST")
    print("=" * 70)

    try:
        nl_vocab, sql_vocab = _lv("data/processed/vocab.json")
        model = create_model(nl_vocab, sql_vocab)

        print(f"\nParameters : {model.count_parameters():,}")

        src = torch.randint(0, 100, (4, 20)).to(DEVICE)
        tgt = torch.randint(0, 100, (4, 15)).to(DEVICE)

        with torch.no_grad():
            logits = model(src, tgt)

        print(f"Input  src : {src.shape}")
        print(f"Input  tgt : {tgt.shape}")
        print(f"Output     : {logits.shape}")
        assert logits.shape == (4, 15, sql_vocab["size"]), (
            f"Shape mismatch: {logits.shape}"
        )

        # Quick encode / decode_step check
        with torch.no_grad():
            memory = model.encode(src)
            step = model.decode_step(tgt[:, :5], memory)
        print(f"Memory     : {memory.shape}")
        print(f"Step logits: {step.shape}")

        print("\nModel test PASSED")

    except Exception as exc:
        logger.error(f"Model test failed: {exc}", exc_info=True)
        print(f"\n[FAIL] Model test failed: {exc}")
        sys.exit(1)

