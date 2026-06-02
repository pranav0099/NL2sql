"""
Generate Synthetic NL2SQL Datasets - Main Entry Point

This script generates large-scale, high-quality synthetic datasets for the NL2SQL project.
It replaces the corrupted Spider and WikiSQL data with clean, validated samples.

Author: Pranav
Date: 2026-04-04

Usage:
    python generate_datasets.py --train 50000 --val 5000 --test 5000
    python generate_datasets.py --test-small  # Generate 100 samples each for quick testing
"""

import sys
import argparse
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import create_dirs
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic NL2SQL datasets")

    # Size arguments
    parser.add_argument("--train", type=int, default=50000, help="Training samples per dataset (default: 50000)")
    parser.add_argument("--val", type=int, default=5000, help="Validation samples per dataset (default: 5000)")
    parser.add_argument("--test", type=int, default=5000, help="Test samples per dataset (default: 5000)")

    # Mode
    parser.add_argument("--test-small", action="store_true", help="Generate small test dataset (100 samples)")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup of existing data")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step after generation")

    args = parser.parse_args()

    # Ensure directories
    create_dirs()

    try:
        from data.generator import generate_all_datasets

        if args.test_small:
            logger.info("Generating SMALL test datasets (100 samples each)...")
            train_size = val_size = test_size = 100
        else:
            train_size = args.train
            val_size = args.val
            test_size = args.test
            logger.info(f"Generating FULL datasets: train={train_size}, val={val_size}, test={test_size}")

        # Generate
        generate_all_datasets(
            wiki_train=train_size,
            wiki_val=val_size,
            wiki_test=test_size,
            spider_train=train_size,
            spider_val=val_size,
            spider_test=test_size,
            backup_existing=not args.no_backup
        )

        # Validate if not skipping
        if not args.skip_validation:
            logger.info("="*80)
            logger.info("RUNNING VALIDATION")
            logger.info("="*80)
            from data.validate_dataset import validate_all_datasets

            results = validate_all_datasets(verbose=False)

            # Check if all valid
            all_ok = all(s["valid_percent"] == 100.0 for s in results.values())
            if all_ok:
                logger.info("✓ All datasets passed validation")
            else:
                logger.warning("✗ Some datasets have validation errors - check reports")
                return 1

        logger.info("="*80)
        logger.info("DATASET GENERATION COMPLETE")
        logger.info("="*80)
        return 0

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)