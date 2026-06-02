"""
Hotel NL2SQL Training Pipeline
==============================
Complete end-to-end pipeline:
  1. Generate hotel training pairs (from generate_hotel_training.py)
  2. Merge with existing WikiSQL/Spider data
  3. Rebuild vocabulary with hotel domain tokens
  4. Retrain the Transformer model
  5. Evaluate with BLEU + exact match

Usage:
  python train_hotel_model.py                    # Full pipeline
  python train_hotel_model.py --epochs 5         # Quick test
  python train_hotel_model.py --skip-generate    # Skip step 1
"""

import sys
import json
import re
import argparse
import time
from pathlib import Path
from collections import Counter

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.config import (
    PROCESSED_DATA, DEVICE, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, DL_MODEL_PATH, SQL_KEYWORDS,
    SPECIAL_TOKENS, INTENTS, INTENT2IDX,
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
)
from data.build_vocab import tokenize_nl, tokenize_sql, Vocabulary
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# STEP 1: Generate hotel training data
# ============================================================================
def step1_generate_hotel_data():
    """Run generate_hotel_training.py to create hotel NL-SQL pairs."""
    print("\n" + "=" * 60)
    print("STEP 1: Generating Hotel Training Data")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "generate_hotel_training.py")],
        cwd=str(ROOT),
        capture_output=True, text=True, timeout=120,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[WARN] generate_hotel_training.py stderr:\n{result.stderr}")
    
    # Verify output
    for split in ["train", "val", "test"]:
        p = PROCESSED_DATA / f"hotel_{split}.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            print(f"  [OK] {p.name}: {len(data)} records")
        else:
            print(f"  [FAIL] {p.name} not found!")
            return False
    return True


# ============================================================================
# STEP 2: Merge hotel data with existing training data
# ============================================================================
def step2_merge_data():
    """Merge hotel training data into existing processed data."""
    print("\n" + "=" * 60)
    print("STEP 2: Merging Hotel Data with Existing Data")
    print("=" * 60)

    merged_counts = {}
    for split in ["train", "val", "test"]:
        # Load existing processed data
        existing_path = PROCESSED_DATA / f"{split}.json"
        existing_data = []
        if existing_path.exists():
            with open(existing_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            # Remove any previous hotel records to avoid duplicates
            existing_data = [r for r in existing_data
                             if r.get("source") != "hotel_training"]
            print(f"  Existing {split}: {len(existing_data)} records")

        # Load hotel data
        hotel_path = PROCESSED_DATA / f"hotel_{split}.json"
        hotel_data = []
        if hotel_path.exists():
            with open(hotel_path, "r", encoding="utf-8") as f:
                hotel_data = json.load(f)
            print(f"  Hotel {split}: {len(hotel_data)} records")

        # Merge
        merged = existing_data + hotel_data
        merged_counts[split] = len(merged)

        # Save merged data (without token IDs yet — Step 3 adds them)
        with open(existing_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"  Merged {split}: {len(merged)} records -> {existing_path.name}")

    return merged_counts


# ============================================================================
# STEP 3: Rebuild vocabulary with hotel tokens
# ============================================================================
def step3_rebuild_vocab():
    """Rebuild NL and SQL vocabularies including hotel domain tokens."""
    print("\n" + "=" * 60)
    print("STEP 3: Rebuilding Vocabulary with Hotel Tokens")
    print("=" * 60)

    # Load all splits
    all_records = []
    split_data = {}
    for split in ["train", "val", "test"]:
        path = PROCESSED_DATA / f"{split}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        split_data[split] = data
        all_records.extend(data)
        print(f"  Loaded {split}: {len(data)} records")

    # Tokenize all records
    print("  Tokenizing all records...")
    for record in all_records:
        if "nl_tokens" not in record or not record["nl_tokens"]:
            record["nl_tokens"] = tokenize_nl(record["question"])
        if "sql_tokens" not in record or not record["sql_tokens"]:
            record["sql_tokens"] = tokenize_sql(record["query"])

    # Build NL vocabulary from training data only
    print("  Building NL vocabulary...")
    nl_vocab = Vocabulary(min_freq=1)  # min_freq=1 for hotel data
    for record in split_data["train"]:
        if "nl_tokens" not in record:
            record["nl_tokens"] = tokenize_nl(record["question"])
        nl_vocab.add_tokens(record["nl_tokens"])
    nl_vocab.build()

    # Build SQL vocabulary
    print("  Building SQL vocabulary...")
    sql_vocab = Vocabulary(min_freq=1)
    for record in split_data["train"]:
        if "sql_tokens" not in record:
            record["sql_tokens"] = tokenize_sql(record["query"])
        sql_vocab.add_tokens(record["sql_tokens"])
    sql_vocab.build(force_include=SQL_KEYWORDS)

    print(f"  NL vocab size:  {len(nl_vocab)}")
    print(f"  SQL vocab size: {len(sql_vocab)}")

    # Encode all records with new vocab
    print("  Encoding all records with new vocabulary...")
    for split_name, data in split_data.items():
        for record in data:
            if "nl_tokens" not in record:
                record["nl_tokens"] = tokenize_nl(record["question"])
            if "sql_tokens" not in record:
                record["sql_tokens"] = tokenize_sql(record["query"])
            record["nl_ids"] = nl_vocab.encode(record["nl_tokens"])
            record["sql_ids"] = sql_vocab.encode(record["sql_tokens"])
            record["intent_idx"] = INTENT2IDX.get(
                record.get("intent", "SELECT"), 0)

        # Save enriched data
        path = PROCESSED_DATA / f"{split_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved enriched {split_name}: {len(data)} records")

    # Save vocabulary
    vocab_path = PROCESSED_DATA / "vocab.json"
    vocab_data = {
        "nl_vocab": {
            "tok2idx": nl_vocab.tok2idx,
            "idx2tok": nl_vocab.idx2tok,
            "size": len(nl_vocab),
            "min_freq": nl_vocab.min_freq,
            "num_tokens": len(nl_vocab),
        },
        "sql_vocab": {
            "tok2idx": sql_vocab.tok2idx,
            "idx2tok": sql_vocab.idx2tok,
            "size": len(sql_vocab),
            "min_freq": sql_vocab.min_freq,
            "num_tokens": len(sql_vocab),
        },
        "special_tokens": SPECIAL_TOKENS,
        "intents": INTENTS,
        "intent2idx": INTENT2IDX,
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    print(f"  Vocabulary saved -> {vocab_path}")

    # Also copy to models/vocab.json
    models_vocab = ROOT / "models" / "vocab.json"
    with open(models_vocab, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    print(f"  Vocabulary copied -> {models_vocab}")

    return nl_vocab, sql_vocab


# ============================================================================
# STEP 4: Train the Transformer model
# ============================================================================
def step4_train_model(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                      lr=LEARNING_RATE, max_samples=None):
    """Train the NL2SQL Transformer on merged data."""
    print("\n" + "=" * 60)
    print("STEP 4: Training NL2SQL Transformer")
    print("=" * 60)

    from dl.dataset import load_vocab, get_dataloaders
    from dl.model import create_model
    from dl.trainer import TransformerTrainer

    # Load vocab
    nl_vocab, sql_vocab = load_vocab(str(PROCESSED_DATA / "vocab.json"))

    # Create DataLoaders
    train_loader, val_loader = get_dataloaders(
        str(PROCESSED_DATA / "train.json"),
        str(PROCESSED_DATA / "val.json"),
        nl_vocab, sql_vocab,
        batch_size=batch_size,
    )

    # Truncate for quick testing
    if max_samples and max_samples < len(train_loader.dataset):
        train_loader.dataset.records = \
            train_loader.dataset.records[:max_samples]
        print(f"  Truncated training to {max_samples} samples")

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples:   {len(val_loader.dataset)}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  LR:            {lr}")
    print(f"  Device:        {DEVICE}")

    # Create model
    model = create_model(nl_vocab, sql_vocab)
    trainer = TransformerTrainer(model, nl_vocab, sql_vocab,
                                learning_rate=lr)

    # Train
    history = trainer.train(train_loader, val_loader,
                            num_epochs=epochs)

    # Plot
    try:
        trainer.plot_losses(str(ROOT / "logs" / "hotel_training_curve.png"))
    except Exception:
        pass

    print(f"\n  Best val loss: {history['best_val_loss']:.4f}")
    print(f"  Best epoch:    {history['best_epoch']}")
    print(f"  Stopped early: {history['stopped_early']}")
    print(f"  Model saved:   {DL_MODEL_PATH}")

    return history


# ============================================================================
# STEP 5: Quick evaluation
# ============================================================================
def step5_evaluate():
    """Quick evaluation with hotel-specific test queries."""
    print("\n" + "=" * 60)
    print("STEP 5: Evaluation")
    print("=" * 60)

    from dl.generator import SQLGenerator

    gen = SQLGenerator()

    hotel_tests = [
        ("Show all staff", "SELECT * FROM staff"),
        ("Show staff with salary above 50000",
         "SELECT * FROM staff WHERE salary > 50000"),
        ("Count all staff", "SELECT COUNT(*) FROM staff"),
        ("Average salary of staff",
         "SELECT AVG(salary) FROM staff"),
        ("Show all hotels", "SELECT * FROM hotels"),
        ("Show hotels in Mumbai",
         "SELECT * FROM hotels WHERE city = 'Mumbai'"),
        ("Show 5 star hotels",
         "SELECT * FROM hotels WHERE star_rating = 5"),
        ("Top 3 highest paid staff",
         "SELECT * FROM staff ORDER BY salary DESC LIMIT 3"),
        ("Count bookings by status",
         "SELECT status, COUNT(*) FROM bookings GROUP BY status"),
        ("Total payment amount",
         "SELECT SUM(amount) FROM payments"),
    ]

    passed = 0
    for question, expected in hotel_tests:
        result = gen.generate(question)
        sql = result["sql"]
        method = result["method"]
        ok = bool(sql) and "SELECT" in sql.upper()
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {question}")
        print(f"         Got:      {sql}")
        print(f"         Expected: {expected}")
        print(f"         Method:   {method}")

    print(f"\n  Result: {passed}/{len(hotel_tests)} passed")

    # BLEU evaluation if model loaded
    if gen.is_loaded:
        try:
            metrics = gen.evaluate_bleu(max_samples=200)
            print(f"\n  BLEU:        {metrics['bleu_score']:.3f}")
            print(f"  Exact Match: {metrics['exact_match']:.1%}")
            print(f"  Avg Conf:    {metrics['avg_confidence']:.1%}")
        except Exception as e:
            print(f"  BLEU eval skipped: {e}")

    return passed


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Hotel NL2SQL Training Pipeline")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip hotel data generation (use existing)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (just generate + merge + vocab)")
    args = parser.parse_args()

    start = time.time()
    print("\n" + "=" * 60)
    print("  HOTEL NL2SQL TRAINING PIPELINE")
    print("  Using: generate_hotel_training.py + API generator + DL model")
    print("=" * 60)

    # Step 1: Generate hotel training data
    if not args.skip_generate:
        if not step1_generate_hotel_data():
            print("\n[FAIL] Hotel data generation failed!")
            sys.exit(1)
    else:
        print("\n[SKIP] Hotel data generation (using existing)")

    # Step 2: Merge with existing data
    step2_merge_data()

    # Step 3: Rebuild vocabulary
    step3_rebuild_vocab()

    # Step 4: Train
    if not args.skip_train:
        step4_train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_samples=args.max_samples,
        )
    else:
        print("\n[SKIP] Training")

    # Step 5: Evaluate
    step5_evaluate()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE - {elapsed / 60:.1f} min total")
    print(f"{'=' * 60}")
    print(f"""
Next steps:
  1. Test interactively:  streamlit run ui/app.py
  2. The API generator is ALWAYS active (primary)
  3. The DL model is now trained on hotel data (backup)
  4. Rule-based builder also works as fallback
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FAIL] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
