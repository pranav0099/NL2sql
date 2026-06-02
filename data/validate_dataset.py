"""
Dataset Validation Module
Validates generated datasets for correctness, schema consistency, and quality.

Author: Pranav
Date: 2026-04-04
"""

import json
import sqlite3
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import Counter
from datetime import datetime

from config.config import INTENTS, INTENT2IDX
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_schema_consistency(
    record: Dict[str, Any],
    schema_tables: Set[str],
    table_to_columns: Dict[str, Set[str]]
) -> List[str]:
    """
    Check that query only uses tables and columns that exist in schema.

    Args:
        record: Single dataset record
        schema_tables: Set of valid table names
        table_to_columns: Dict mapping table name to set of column names

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    query = record["query"].lower()
    source = record.get("source", "unknown")

    # Extract table names from query
    tables_in_query = set()
    table_pattern = r'\b(?:from|join)\s+([a-z_]+)(?:\s+[a-z]+)?\b'
    matches = re.findall(table_pattern, query)
    tables_in_query.update(matches)

    # Check tables
    for table in tables_in_query:
        if table not in schema_tables:
            errors.append(f"Unknown table '{table}' in query")

    # Extract column references: table.column
    col_pattern = r'\b([a-z_]+)\.([a-z_]+)\b'
    col_matches = re.findall(col_pattern, query)
    for tbl, col in col_matches:
        if tbl in schema_tables and col not in table_to_columns.get(tbl, set()):
            errors.append(f"Unknown column '{tbl}.{col}'")

    return errors

def validate_intent(record: Dict[str, Any]) -> List[str]:
    """Check that record has a valid intent."""
    errors = []
    intent = record.get("intent", "")
    if intent not in INTENTS:
        errors.append(f"Invalid intent: {intent}")
    return errors

def validate_required_fields(record: Dict[str, Any], source: str) -> List[str]:
    """Check that all required fields are present."""
    errors = []
    required_fields = ["id", "question", "query", "intent", "source"]

    if source == "wikisql":
        required_fields.append("table_id")
    elif source == "spider":
        required_fields.append("db_id")
        required_fields.append("schema")

    for field in required_fields:
        if field not in record:
            errors.append(f"Missing required field: {field}")

    return errors

def validate_sqlite_syntax(query: str, db_path: str = "database/sample.db") -> List[str]:
    """
    Try to parse/execute query with SQLite to check syntax.
    Note: Might fail on actual data but should parse OK.
    """
    errors = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Use EXPLAIN to check syntax without executing
        cursor.execute(f"EXPLAIN {query}")
        conn.close()
    except sqlite3.Error as e:
        errors.append(f"SQLite error: {str(e)}")
        conn.close()
    except Exception as e:
        errors.append(f"Parse error: {str(e)}")

    return errors

def validate_record(record: Dict[str, Any], schema_info: Dict[str, Any], check_sqlite: bool = False) -> Dict[str, Any]:
    """
    Validate a single record.

    Returns:
        Dict with 'valid' (bool), 'errors' (list), 'warnings' (list)
    """
    result = {"valid": True, "errors": [], "warnings": []}
    source = record.get("source", "unknown")

    # Required fields
    field_errors = validate_required_fields(record, source)
    result["errors"].extend(field_errors)

    # Intent
    intent_errors = validate_intent(record)
    result["errors"].extend(intent_errors)

    # Schema consistency
    schema_errors = validate_schema_consistency(
        record,
        set(schema_info["table_names"]),
        {t: set(info["columns"]) for t, info in schema_info["tables"].items()}
    )
    result["errors"].extend(schema_errors)

    # SQLite syntax (optional, expensive)
    if check_sqlite and source == "spider":
        sqlite_errors = validate_sqlite_syntax(record["query"])
        result["errors"].extend(sqlite_errors)

    # Warnings
    if not record.get("question"):
        result["warnings"].append("Empty question")
    if len(record["query"]) > 500:
        result["warnings"].append("Very long query (>500 chars)")

    if result["errors"]:
        result["valid"] = False

    return result

# ============================================================================
# LOAD SCHEMA
# ============================================================================

def load_schema_for_validation(schema_path: str = "database/schema.json") -> Dict[str, Any]:
    """Load schema in simplified format for validation."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        full_schema = json.load(f)

    simplified = {
        "table_names": list(full_schema["tables"].keys()),
        "tables": {}
    }

    for table_name, table_info in full_schema["tables"].items():
        columns = [col["name"] for col in table_info["columns"]]
        simplified["tables"][table_name] = {"columns": columns}

    return simplified

# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def validate_dataset(
    filepath: str,
    schema_path: str = "database/schema.json",
    sample_limit: int = None,
    check_sqlite: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate an entire dataset file.

    Args:
        filepath: Path to dataset JSON file
        schema_path: Path to schema.json
        sample_limit: If set, only validate first N samples
        check_sqlite: Whether to run SQLite syntax check (slow)
        verbose: Print detailed results

    Returns:
        Summary dict with statistics
    """
    logger.info(f"Validating dataset: {filepath}")
    logger.info(f"Schema: {schema_path}")
    logger.info(f"Sample limit: {sample_limit}")
    logger.info(f"SQLite check: {check_sqlite}")

    # Load dataset
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if sample_limit:
        data = data[:sample_limit]
        logger.info(f"Validating first {len(data)} samples (limited)")
    else:
        logger.info(f"Validating all {len(data)} samples")

    # Load schema
    schema_info = load_schema_for_validation(schema_path)

    # Validate each record
    valid_count = 0
    invalid_count = 0
    error_counts = Counter()
    intent_dist = Counter()
    warnings_count = 0

    first_error_sample = None

    for i, record in enumerate(data):
        result = validate_record(record, schema_info, check_sqlite)

        if result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
            for err in result["errors"]:
                error_counts[err] += 1
            if first_error_sample is None:
                first_error_sample = {"index": i, "id": record.get("id"), "errors": result["errors"], "query": record.get("query", "")}

        intent_dist[record.get("intent", "UNKNOWN")] += 1
        warnings_count += len(result["warnings"])

        if verbose and i < 5:  # Log first 5
            logger.debug(f"  Sample {i}: {record.get('id')} - {'VALID' if result['valid'] else 'INVALID'}")

    # Summary
    total = len(data)
    valid_pct = (valid_count / total) * 100 if total > 0 else 0
    invalid_pct = (invalid_count / total) * 100 if total > 0 else 0

    summary = {
        "file": filepath,
        "total_samples": total,
        "valid_samples": valid_count,
        "invalid_samples": invalid_count,
        "valid_percent": valid_pct,
        "invalid_percent": invalid_pct,
        "intent_distribution": dict(intent_dist),
        "error_counts": dict(error_counts),
        "total_warnings": warnings_count,
        "top_errors": error_counts.most_common(5),
        "first_error_sample": first_error_sample
    }

    # Print report
    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)
    print(f"File:              {filepath}")
    print(f"Total samples:     {total}")
    print(f"Valid:             {valid_count} ({valid_pct:.2f}%)")
    print(f"Invalid:           {invalid_count} ({invalid_pct:.2f}%)")
    print(f"Warnings:          {warnings_count}")
    print("\n--- Intent Distribution ---")
    for intent, count in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {intent:20s}: {count:6d} ({pct:5.1f}%)")

    if error_counts:
        print("\n--- Top Errors ---")
        for err, count in error_counts.most_common(5):
            pct = (count / total) * 100
            print(f"  {count:4d}x ({pct:4.1f}%) {err}")
    else:
        print("\nNo errors found!")

    if first_error_sample:
        print("\n--- First Error Example ---")
        print(f"  Sample #{first_error_sample['index']} ({first_error_sample['id']})")
        print(f"  Query: {first_error_sample['query'][:150]}{'...' if len(first_error_sample['query']) > 150 else ''}")
        for err in first_error_sample['errors'][:5]:
            print(f"    - {err}")

    print("="*70 + "\n")

    return summary

def validate_all_datasets(
    base_dir: str = "data",
    datasets: List[str] = ["wikisql", "spider", "processed"],
    splits: List[str] = ["train", "validation", "test"],
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Validate all dataset files.

    Args:
        base_dir: Base data directory
        datasets: List of dataset names to validate
        splits: List of split names to validate
        **kwargs: Passed to validate_dataset()

    Returns:
        Dict mapping (dataset, split) to summary
    """
    base_path = Path(base_dir)
    results = {}

    for dataset in datasets:
        for split in splits:
            filepath = base_path / dataset / f"{split}.json"
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue

            summary = validate_dataset(str(filepath), **kwargs)
            results[f"{dataset}/{split}"] = summary

    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate NL2SQL datasets")
    parser.add_argument("file", nargs="?", help="Dataset file to validate (if not provided, validates all)")
    parser.add_argument("--schema", default="database/schema.json", help="Path to schema.json")
    parser.add_argument("--limit", type=int, help="Limit validation to first N samples")
    parser.add_argument("--sqlite", action="store_true", help="Also check SQLite syntax (slow)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    try:
        if args.file:
            # Validate single file
            validate_dataset(
                args.file,
                schema_path=args.schema,
                sample_limit=args.limit,
                check_sqlite=args.sqlite,
                verbose=not args.quiet
            )
        else:
            # Validate all datasets
            logger.info("Validating all datasets...")
            results = validate_all_datasets(
                schema_path=args.schema,
                sample_limit=args.limit,
                check_sqlite=args.sqlite,
                verbose=not args.quiet
            )

            # Summary table
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            print(f"{'Dataset/Split':<20} {'Total':<10} {'Valid':<10} {'Valid%':<10}")
            print("-"*70)
            for key, summary in sorted(results.items()):
                print(f"{key:<20} {summary['total_samples']:<10} {summary['valid_samples']:<10} {summary['valid_percent']:<10.1f}")
            print("="*70)

            # Overall status
            all_valid = all(s["valid_percent"] == 100.0 for s in results.values())
            if all_valid:
                logger.info("ALL DATASETS VALIDATED SUCCESSFULLY")
            else:
                logger.error("SOME DATASETS HAVE ERRORS - REVIEW ABOVE")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)