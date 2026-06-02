"""
Spider Dataset Download and Processing Module
Downloads and processes the Spider multi-table QA dataset,
with fallback to synthetic multi-table JOIN queries.

Author: Pranav
Date: 2026-04-02

Dataset: Spider (https://yale-lily.github.io/spider)
License: CC BY-SA 4.0
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from config.config import SPIDER_DATA, INTENTS
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# SYNTHETIC SPIDER DATA (Multi-table JOIN queries)
# ============================================================================

SYNTHETIC_SPIDER_TRAIN = [
    # Multi-table JOIN queries using our database schema

    {
        "question": "Show all customers with their orders",
        "query": """
            SELECT c.customer_id, c.first_name, c.last_name, c.email,
                   o.order_id, o.order_date, o.total_amount
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
        """,
        "db_id": "sample"
    },
    {
        "question": "List all orders with product details",
        "query": """
            SELECT o.order_id, o.order_date, p.product_name, p.category,
                   oi.quantity, oi.unit_price, oi.subtotal
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN products p ON oi.product_id = p.product_id
        """,
        "db_id": "sample"
    },
    {
        "question": "Show employees with their department names",
        "query": """
            SELECT e.emp_id, e.first_name, e.last_name, e.email,
                   d.dept_name, d.location, e.salary
            FROM employees e
            JOIN departments d ON e.dept_id = d.dept_id
        """,
        "db_id": "sample"
    },
    {
        "question": "Find customers who ordered Electronics products",
        "query": """
            SELECT DISTINCT c.customer_id, c.first_name, c.last_name, c.city
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN products p ON oi.product_id = p.product_id
            WHERE p.category = 'Electronics'
        """,
        "db_id": "sample"
    },
    {
        "question": "Show total spending by customers in Mumbai",
        "query": """
            SELECT c.customer_id, c.first_name, c.last_name,
                   SUM(o.total_amount) as total_spent
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            WHERE c.city = 'Mumbai'
            GROUP BY c.customer_id, c.first_name, c.last_name
            ORDER BY total_spent DESC
        """,
        "db_id": "sample"
    },
    {
        "question": "Calculate average order value by city",
        "query": """
            SELECT c.city, COUNT(*) as order_count,
                   AVG(o.total_amount) as avg_order_value
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.city
            ORDER BY avg_order_value DESC
        """,
        "db_id": "sample"
    },
    {
        "question": "Find top 10 customers by order count",
        "query": """
            SELECT c.customer_id, c.first_name, c.last_name, c.city,
                   COUNT(o.order_id) as order_count,
                   SUM(o.total_amount) as total_spent
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.customer_id, c.first_name, c.last_name, c.city
            ORDER BY order_count DESC
            LIMIT 10
        """,
        "db_id": "sample"
    },
    {
        "question": "Show product sales by category",
        "query": """
            SELECT p.category, COUNT(DISTINCT o.order_id) as order_count,
                   SUM(oi.quantity) as total_quantity,
                   SUM(oi.subtotal) as total_sales
            FROM products p
            JOIN order_items oi ON p.product_id = oi.product_id
            JOIN orders o ON oi.order_id = o.order_id
            GROUP BY p.category
            ORDER BY total_sales DESC
        """,
        "db_id": "sample"
    },
    {
        "question": "List departments with employee count and average salary",
        "query": """
            SELECT d.dept_name, d.location,
                   COUNT(e.emp_id) as emp_count,
                   AVG(e.salary) as avg_salary
            FROM departments d
            JOIN employees e ON d.dept_id = e.dept_id
            GROUP BY d.dept_name, d.location
            ORDER BY emp_count DESC
        """,
        "db_id": "sample"
    },
    {
        "question": "Find customers who ordered in 2024",
        "query": """
            SELECT DISTINCT c.customer_id, c.first_name, c.last_name, c.email
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            WHERE o.order_date LIKE '2024%'
        """,
        "db_id": "sample"
    },
    {
        "question": "Show monthly sales trends for 2024",
        "query": """
            SELECT month, COUNT(*) as order_count,
                   SUM(total_amount) as monthly_sales
            FROM orders
            WHERE year = 2024 OR order_date LIKE '2024%'
            GROUP BY month
            ORDER BY month
        """,
        "db_id": "sample"
    },
    {
        "question": "Find best selling product categories by revenue",
        "query": """
            SELECT p.category, SUM(oi.subtotal) as revenue,
                   COUNT(DISTINCT o.order_id) as orders
            FROM products p
            JOIN order_items oi ON p.product_id = oi.product_id
            JOIN orders o ON oi.order_id = o.order_id
            GROUP BY p.category
            ORDER BY revenue DESC
            LIMIT 5
        """,
        "db_id": "sample"
    },
    {
        "question": "Show employees with department and location",
        "query": """
            SELECT e.emp_id, e.first_name, e.last_name, e.email,
                   d.dept_name, d.location, e.salary, e.hire_date
            FROM employees e
            JOIN departments d ON e.dept_id = d.dept_id
            WHERE e.city = d.location
        """,
        "db_id": "sample"
    },
    {
        "question": "Calculate department budget utilization",
        "query": """
            SELECT d.dept_name, d.budget,
                   SUM(e.salary) as total_salary,
                   (SUM(e.salary) / d.budget) * 100 as utilization_pct
            FROM departments d
            JOIN employees e ON d.dept_id = e.dept_id
            GROUP BY d.dept_name, d.budget
        """,
        "db_id": "sample"
    },
    {
        "question": "Find repeat customers with more than 5 orders",
        "query": """
            SELECT c.customer_id, c.first_name, c.last_name, c.city,
                   COUNT(o.order_id) as order_count,
                   AVG(o.total_amount) as avg_order_value
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.customer_id, c.first_name, c.last_name, c.city
            HAVING COUNT(o.order_id) > 5
            ORDER BY order_count DESC
        """,
        "db_id": "sample"
    },

    # Additional complex queries
    {
        "question": "Show products that have never been ordered",
        "query": """
            SELECT p.product_id, p.product_name, p.category, p.price
            FROM products p
            LEFT JOIN order_items oi ON p.product_id = oi.product_id
            WHERE oi.order_id IS NULL
        """,
        "db_id": "sample"
    },
    {
        "question": "List customers with no orders",
        "query": """
            SELECT c.customer_id, c.first_name, c.last_name, c.city
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            WHERE o.order_id IS NULL
        """,
        "db_id": "sample"
    },
    {
        "question": "Show top performing departments by salary efficiency",
        "query": """
            SELECT d.dept_name, d.location,
                   COUNT(e.emp_id) as emp_count,
                   AVG(e.salary) as avg_salary,
                   SUM(e.salary) / d.budget as budget_ratio
            FROM departments d
            JOIN employees e ON d.dept_id = e.dept_id
            GROUP BY d.dept_name, d.location, d.budget
            ORDER BY budget_ratio DESC
        """,
        "db_id": "sample"
    },
    {
        "question": "Find customers with orders in multiple cities",
        "query": """
            SELECT c.customer_id, c.first_name, c.last_name,
                   COUNT(DISTINCT o.city) as cities_count
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.customer_id, c.first_name, c.last_name
            HAVING COUNT(DISTINCT o.city) > 1
        """,
        "db_id": "sample"
    },
    {
        "question": "Show average product rating by category with stock analysis",
        "query": """
            SELECT category,
                   AVG(rating) as avg_rating,
                   COUNT(*) as product_count,
                   SUM(stock_qty) as total_stock
            FROM products
            GROUP BY category
            HAVING COUNT(*) >= 3
            ORDER BY avg_rating DESC
        """,
        "db_id": "sample"
    }
]

SYNTHETIC_SPIDER_VAL = SYNTHETIC_SPIDER_TRAIN[:5]  # Use first 5 for validation
SYNTHETIC_SPIDER_TEST = SYNTHETIC_SPIDER_TRAIN[5:10]  # Use next 5 for test


# ============================================================================
# DATABASE SCHEMA FOR SPIDER FORMAT
# ============================================================================

SAMPLE_SCHEMA = {
    "sample": {
        "table_names_original": [
            "customers", "products", "orders", "order_items",
            "employees", "departments", "sales"
        ],
        "table_names": [
            "customers", "products", "orders", "order_items",
            "employees", "departments", "sales"
        ],
        "column_names": [
            ["customers", "customer_id"],
            ["customers", "first_name"],
            ["customers", "last_name"],
            ["customers", "email"],
            ["customers", "phone"],
            ["customers", "city"],
            ["customers", "state"],
            ["customers", "join_date"],
            ["customers", "total_orders"],
            ["customers", "total_spent"],
            ["products", "product_id"],
            ["products", "product_name"],
            ["products", "category"],
            ["products", "price"],
            ["products", "stock_qty"],
            ["products", "rating"],
            ["orders", "order_id"],
            ["orders", "customer_id"],
            ["orders", "order_date"],
            ["orders", "status"],
            ["orders", "total_amount"],
            ["orders", "city"],
            ["order_items", "item_id"],
            ["order_items", "order_id"],
            ["order_items", "product_id"],
            ["order_items", "quantity"],
            ["order_items", "unit_price"],
            ["order_items", "subtotal"],
            ["employees", "emp_id"],
            ["employees", "first_name"],
            ["employees", "last_name"],
            ["employees", "email"],
            ["employees", "dept_id"],
            ["employees", "salary"],
            ["employees", "hire_date"],
            ["employees", "city"],
            ["departments", "dept_id"],
            ["departments", "dept_name"],
            ["departments", "location"],
            ["departments", "budget"],
            ["sales", "sale_id"],
            ["sales", "customer_id"],
            ["sales", "month"],
            ["sales", "year"],
            ["sales", "sales_amount"],
            ["sales", "city"],
            ["sales", "category"]
        ],
        "column_types": [
            "number", "text", "text", "text", "text", "text", "text", "time", "number", "number",
            "number", "text", "text", "number", "number", "number",
            "number", "number", "time", "text", "number", "text",
            "number", "number", "number", "number", "number", "number",
            "number", "text", "text", "text", "number", "number", "time", "text",
            "number", "text", "text", "number",
            "number", "number", "number", "number", "text", "text"
        ],
        "foreign_keys": [
            ["orders", "customer_id", "customers", "customer_id"],
            ["order_items", "order_id", "orders", "order_id"],
            ["order_items", "product_id", "products", "product_id"],
            ["employees", "dept_id", "departments", "dept_id"],
            ["sales", "customer_id", "customers", "customer_id"]
        ],
        "primary_keys": [
            ["customers", "customer_id"],
            ["products", "product_id"],
            ["orders", "order_id"],
            ["order_items", "item_id"],
            ["employees", "emp_id"],
            ["departments", "dept_id"],
            ["sales", "sale_id"]
        ]
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def classify_intent(sql: str) -> str:
    """
    Classify SQL query into intent category (same as WikiSQL).

    Args:
        sql: SQL query string

    Returns:
        Intent label string
    """
    sql_upper = sql.upper()

    has_where = "WHERE" in sql_upper
    has_group = "GROUP BY" in sql_upper
    has_order = "ORDER BY" in sql_upper
    has_limit = "LIMIT" in sql_upper
    has_join = "JOIN" in sql_upper
    has_aggregate = any(agg in sql_upper for agg in ["COUNT", "SUM", "AVG", "MAX", "MIN"])
    has_having = "HAVING" in sql_upper

    # Complex: multiple clauses
    clauses = sum([has_where, has_group, has_order, has_limit, has_join])
    if clauses >= 2 or (has_aggregate and has_group) or (has_group and has_having):
        return "COMPLEX"

    # Single intent
    if has_join:
        return "SELECT_JOIN"
    elif has_aggregate:
        if has_group:
            return "SELECT_GROUP"
        else:
            return "SELECT_AGGREGATE"
    elif has_order:
        return "SELECT_ORDER"
    elif has_limit:
        return "SELECT_LIMIT"
    elif has_where:
        return "SELECT_WHERE"
    else:
        return "SELECT"


def clean_question(text: str) -> str:
    """Normalize question text."""
    import re
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' ?.!')
    return text


def load_database_schema(db_id: str) -> Dict[str, Any]:
    """
    Load schema for a specific database.

    Args:
        db_id: Database identifier

    Returns:
        Schema dictionary
    """
    if db_id in SAMPLE_SCHEMA:
        return SAMPLE_SCHEMA[db_id]
    else:
        logger.warning(f"Schema not found for db_id '{db_id}'. Using 'sample' as default.")
        return SAMPLE_SCHEMA["sample"]


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def download_from_huggingface() -> Optional[Any]:
    """
    Attempt to download Spider from HuggingFace.
    Note: 'spider' dataset may not be available, try alternative names.

    Returns:
        Dataset if successful, None otherwise
    """
    if not HF_AVAILABLE:
        logger.warning("HuggingFace 'datasets' library not available")
        return None

    dataset_names = [
        "spider",
        "taskinjon/spider",
        "seq2seq/spider"
    ]

    for dataset_name in dataset_names:
        try:
            logger.info(f"Attempting to download '{dataset_name}' from HuggingFace...")
            dataset = load_dataset(dataset_name)
            logger.info(f"Successfully downloaded '{dataset_name}'")
            logger.info(f"Splits: {list(dataset.keys())}")
            return dataset
        except Exception as e:
            logger.debug(f"Failed to download '{dataset_name}': {e}")
            continue

    logger.warning("All HuggingFace download attempts failed")
    return None


def create_synthetic_dataset() -> Dict[str, List[Dict[str, Any]]]:
    """
    Create synthetic Spider dataset from predefined complex queries.

    Returns:
        Dictionary with 'train', 'validation', 'test' splits
    """
    logger.info("Creating synthetic Spider dataset...")

    # Combine train, val, test from synthetic data
    train_data = [{
        "question": rec["question"],
        "query": rec["query"].strip(),
        "db_id": rec["db_id"]
    } for rec in SYNTHETIC_SPIDER_TRAIN]

    val_data = [{
        "question": rec["question"],
        "query": rec["query"].strip(),
        "db_id": rec["db_id"]
    } for rec in SYNTHETIC_SPIDER_VAL]

    test_data = [{
        "question": rec["question"],
        "query": rec["query"].strip(),
        "db_id": rec["db_id"]
    } for rec in SYNTHETIC_SPIDER_TEST]

    logger.info(f"Split sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Enrich records
    train_enriched = enrich_records(train_data, "train")
    val_enriched = enrich_records(val_data, "validation")
    test_enriched = enrich_records(test_data, "test")

    return {
        "train": train_enriched,
        "validation": val_enriched,
        "test": test_enriched
    }


def enrich_records(
    records: List[Dict[str, Any]],
    split: str
) -> List[Dict[str, Any]]:
    """
    Enrich records with intent classification and cleaned questions.

    Args:
        records: Raw records
        split: Split name (train/validation/test)

    Returns:
        Enriched records
    """
    enriched = []

    for idx, record in enumerate(records):
        question = clean_question(record["question"])
        query = record["query"]
        db_id = record["db_id"]
        intent = classify_intent(query)

        # Get schema for this database
        schema = load_database_schema(db_id)

        enriched_record = {
            "id": f"spider_{split}_{idx:06d}",
            "question": question,
            "query": query,
            "db_id": db_id,
            "schema": schema,
            "intent": intent,
            "source": "spider",
            "raw_question": record["question"]
        }
        enriched.append(enriched_record)

    return enriched


def save_splits(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Save dataset splits to JSON files.

    Args:
        splits: Dictionary with 'train', 'validation', 'test' keys
    """
    SPIDER_DATA.mkdir(parents=True, exist_ok=True)

    for split, records in splits.items():
        output_file = SPIDER_DATA / f"{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(records)} records to {output_file}")

    # Print distribution
    print_intent_distribution(splits)


def print_intent_distribution(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print intent distribution for each split."""
    logger.info("\n" + "="*60)
    logger.info("SPIDER INTENT DISTRIBUTION")
    logger.info("="*60)

    for split, records in splits.items():
        intent_counts = {}
        for record in records:
            intent = record.get("intent", "UNKNOWN")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        logger.info(f"\n{split.upper()} ({len(records)} records):")
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(records)) * 100
            logger.info(f"  {intent:20s}: {count:4d} ({pct:5.1f}%)")

    logger.info("="*60 + "\n")


def verify_spider_dataset(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """Verify loaded dataset quality."""
    logger.info("\n" + "="*60)
    logger.info("SPIDER DATASET VERIFICATION")
    logger.info("="*60)

    total_records = sum(len(records) for records in splits.values())
    logger.info(f"Total records: {total_records}")

    # Check all required splits exist
    for split in ["train", "validation", "test"]:
        count = len(splits.get(split, []))
        status = "[OK]" if count > 0 else "[FAIL]"
        logger.info(f"  {split}: {count} records {status}")

    # Sample record
    if splits.get("train"):
        sample = splits["train"][0]
        logger.info("\nSample record:")
        logger.info(f"  ID: {sample['id']}")
        logger.info(f"  Question: {sample['question']}")
        logger.info(f"  Intent: {sample['intent']}")
        logger.info(f"  DB ID: {sample['db_id']}")
        logger.info(f"  SQL (truncated): {sample['query'][:100]}...")

    logger.info("="*60 + "\n")


def download_spider() -> None:
    """
    Main function: Generate (synthetic) Spider dataset.
    Uses the synthetic generator to produce large-scale, valid multi-table datasets.
    """
    logger.info("="*80)
    logger.info("GENERATING SPIDER DATASET (SYNTHETIC)")
    logger.info("="*80)

    try:
        # Import generator
        from data.generator import generate_spider_splits

        # Generate datasets
        splits = generate_spider_splits(
            train_size=50000,
            val_size=5000,
            test_size=5000,
            distribution="natural"
        )
    except Exception as e:
        logger.error(f"Failed to generate Spider: {e}")
        raise

    # Verify splits exist
    required_splits = ["train", "validation", "test"]
    for split in required_splits:
        if split not in splits or len(splits[split]) == 0:
            logger.error(f"Missing or empty split: {split}")
            raise ValueError(f"Dataset split '{split}' is missing or empty")

    # Save to files
    save_splits(splits)

    logger.info("Spider dataset ready!")
    logger.info(f"Files saved to: {SPIDER_DATA}")
    logger.info("="*80)


def process_huggingface_dataset(dataset: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process HuggingFace Spider dataset into our format.
    Handles various Spider dataset formats.

    Args:
        dataset: HuggingFace dataset

    Returns:
        Dictionary with 'train', 'validation', 'test' splits
    """
    logger.info("Processing HuggingFace Spider dataset...")

    result = {}

    # Map common split names
    split_mapping = {
        "train": "train",
        "validation": "validation",
        "val": "validation",
        "test": "test"
    }

    for hf_split, our_split in split_mapping.items():
        if hf_split in dataset:
            records = []
            for idx, row in enumerate(dataset[hf_split]):
                # Spider format varies by dataset source
                question = row.get("question", row.get("utterance", ""))
                query = row.get("query", row.get("sql", ""))
                db_id = row.get("db_id", "sample")

                if not question or not query:
                    logger.warning(f"Skipping record {idx} in {hf_split}: missing question or query")
                    continue

                enriched = enrich_records([{
                    "question": question,
                    "query": query,
                    "db_id": db_id
                }], our_split)[0]
                enriched["id"] = f"spider_{our_split}_{idx:06d}"

                records.append(enriched)

            result[our_split] = records
            logger.info(f"Processed {len(records)} records for {our_split}")

    if not result:
        raise ValueError("No valid splits found in HuggingFace dataset")

    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        SPIDER_DATA.mkdir(parents=True, exist_ok=True)
        download_spider()
    except Exception as e:
        logger.error(f"Failed to download Spider: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
