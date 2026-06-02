"""
WikiSQL Dataset Download and Processing Module
Downloads and processes the WikiSQL dataset from HuggingFace,
with fallback to synthetic data if download fails.

Author: Pranav
Date: 2026-04-02

Dataset: WikiSQL (https://github.com/salesforce/WikiSQL)
License: MIT
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
    from datasets import load_dataset, DatasetDict
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not available. Will use synthetic data only.")

from config.config import (
    WIKISQL_DATA, RAW_DATA, VOCAB_FILE,
    INTENTS, INTENT2IDX, SQL_KEYWORDS
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# SYNTHETIC FALLBACK DATA
# ============================================================================

SYNTHETIC_WIKISQL_TRAIN = [
    # SELECT - Simple queries
    {
        "question": "Show all customers",
        "query": "SELECT * FROM customers",
        "table_id": "customers"
    },
    {
        "question": "List all products",
        "query": "SELECT * FROM products",
        "table_id": "products"
    },
    {
        "question": "Display all orders",
        "query": "SELECT * FROM orders",
        "table_id": "orders"
    },
    {
        "question": "Show all employees",
        "query": "SELECT * FROM employees",
        "table_id": "employees"
    },
    {
        "question": "List all departments",
        "query": "SELECT * FROM departments",
        "table_id": "departments"
    },

    # SELECT_WHERE - With WHERE clause
    {
        "question": "Show customers from Mumbai",
        "query": "SELECT * FROM customers WHERE city = 'Mumbai'",
        "table_id": "customers"
    },
    {
        "question": "Find products that cost more than 1000",
        "query": "SELECT * FROM products WHERE price > 1000",
        "table_id": "products"
    },
    {
        "question": "List orders with status delivered",
        "query": "SELECT * FROM orders WHERE status = 'delivered'",
        "table_id": "orders"
    },
    {
        "question": "Show employees in Engineering department",
        "query": "SELECT * FROM employees WHERE dept_id = 1",
        "table_id": "employees"
    },
    {
        "question": "Find customers who joined after 2023",
        "query": "SELECT * FROM customers WHERE join_date > '2023-01-01'",
        "table_id": "customers"
    },
    {
        "question": "Show products with rating above 4",
        "query": "SELECT * FROM products WHERE rating > 4.0",
        "table_id": "products"
    },
    {
        "question": "List orders placed in Mumbai",
        "query": "SELECT * FROM orders WHERE city = 'Mumbai'",
        "table_id": "orders"
    },
    {
        "question": "Find customers from Delhi",
        "query": "SELECT * FROM customers WHERE city = 'Delhi'",
        "table_id": "customers"
    },
    {
        "question": "Show employees with salary greater than 500000",
        "query": "SELECT * FROM employees WHERE salary > 500000",
        "table_id": "employees"
    },

    # SELECT_AGGREGATE - With COUNT, SUM, AVG, MAX, MIN
    {
        "question": "How many customers are there?",
        "query": "SELECT COUNT(*) FROM customers",
        "table_id": "customers"
    },
    {
        "question": "What is the total number of orders?",
        "query": "SELECT COUNT(*) FROM orders",
        "table_id": "orders"
    },
    {
        "question": "What is the average price of products?",
        "query": "SELECT AVG(price) FROM products",
        "table_id": "products"
    },
    {
        "question": "What is the maximum salary?",
        "query": "SELECT MAX(salary) FROM employees",
        "table_id": "employees"
    },
    {
        "question": "What is the minimum price?",
        "query": "SELECT MIN(price) FROM products",
        "table_id": "products"
    },
    {
        "question": "What is the total amount spent?",
        "query": "SELECT SUM(total_amount) FROM orders",
        "table_id": "orders"
    },
    {
        "question": "How many products are in Electronics category?",
        "query": "SELECT COUNT(*) FROM products WHERE category = 'Electronics'",
        "table_id": "products"
    },
    {
        "question": "What is the average order value?",
        "query": "SELECT AVG(total_amount) FROM orders",
        "table_id": "orders"
    },

    # SELECT_ORDER - With ORDER BY
    {
        "question": "Show customers sorted by total spent",
        "query": "SELECT * FROM customers ORDER BY total_spent DESC",
        "table_id": "customers"
    },
    {
        "question": "List products sorted by price",
        "query": "SELECT * FROM products ORDER BY price ASC",
        "table_id": "products"
    },
    {
        "question": "Show orders by date",
        "query": "SELECT * FROM orders ORDER BY order_date DESC",
        "table_id": "orders"
    },
    {
        "question": "List employees sorted by salary",
        "query": "SELECT * FROM employees ORDER BY salary DESC",
        "table_id": "employees"
    },
    {
        "question": "Show products sorted by rating",
        "query": "SELECT * FROM products ORDER BY rating DESC",
        "table_id": "products"
    },

    # SELECT_JOIN - With JOIN
    {
        "question": "Show all orders with customer names",
        "query": "SELECT o.*, c.first_name, c.last_name FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
        "table_id": "orders"
    },
    {
        "question": "List order items with product names",
        "query": "SELECT oi.*, p.product_name FROM order_items oi JOIN products p ON oi.product_id = p.product_id",
        "table_id": "order_items"
    },
    {
        "question": "Show employees with department names",
        "query": "SELECT e.*, d.dept_name FROM employees e JOIN departments d ON e.dept_id = d.dept_id",
        "table_id": "employees"
    },
    {
        "question": "List orders with customer and city",
        "query": "SELECT o.order_id, c.first_name, c.city, o.total_amount FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
        "table_id": "orders"
    },

    # SELECT_GROUP - With GROUP BY
    {
        "question": "How many orders per customer?",
        "query": "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
        "table_id": "orders"
    },
    {
        "question": "Total sales by city",
        "query": "SELECT city, SUM(total_amount) FROM orders GROUP BY city",
        "table_id": "orders"
    },
    {
        "question": "Average product price by category",
        "query": "SELECT category, AVG(price) FROM products GROUP BY category",
        "table_id": "products"
    },
    {
        "question": "Number of employees per department",
        "query": "SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id",
        "table_id": "employees"
    },
    {
        "question": "Total orders by status",
        "query": "SELECT status, COUNT(*) FROM orders GROUP BY status",
        "table_id": "orders"
    },

    # SELECT_LIMIT - With LIMIT
    {
        "question": "Show first 10 customers",
        "query": "SELECT * FROM customers LIMIT 10",
        "table_id": "customers"
    },
    {
        "question": "List top 5 expensive products",
        "query": "SELECT * FROM products ORDER BY price DESC LIMIT 5",
        "table_id": "products"
    },
    {
        "question": "Show 5 most recent orders",
        "query": "SELECT * FROM orders ORDER BY order_date DESC LIMIT 5",
        "table_id": "orders"
    },
    {
        "question": "Show only one record",
        "query": "SELECT * FROM customers LIMIT 1",
        "table_id": "customers"
    },

    # COMPLEX - Multiple clauses
    {
        "question": "Show top 10 customers from Mumbai by total spent",
        "query": "SELECT * FROM customers WHERE city = 'Mumbai' ORDER BY total_spent DESC LIMIT 10",
        "table_id": "customers"
    },
    {
        "question": "Count delivered orders in Bangalore",
        "query": "SELECT COUNT(*) FROM orders WHERE status = 'delivered' AND city = 'Bangalore'",
        "table_id": "orders"
    },
    {
        "question": "Show Electronics products with rating above 4 ordered by price",
        "query": "SELECT * FROM products WHERE category = 'Electronics' AND rating > 4.0 ORDER BY price",
        "table_id": "products"
    },
    {
        "question": "Find customers with more than 10 orders",
        "query": "SELECT * FROM customers WHERE total_orders > 10 ORDER BY total_orders DESC",
        "table_id": "customers"
    },
    {
        "question": "Show orders with total amount greater than 10000",
        "query": "SELECT * FROM orders WHERE total_amount > 10000 ORDER BY total_amount DESC",
        "table_id": "orders"
    },
    {
        "question": "Count products and average price by category where stock > 100",
        "query": "SELECT category, COUNT(*), AVG(price) FROM products WHERE stock_qty > 100 GROUP BY category",
        "table_id": "products"
    },
    {
        "question": "Show employees in Engineering ordered by salary descending",
        "query": "SELECT * FROM employees WHERE dept_id = 1 ORDER BY salary DESC",
        "table_id": "employees"
    },
    {
        "question": "Find average order value for delivered orders",
        "query": "SELECT AVG(total_amount) FROM orders WHERE status = 'delivered'",
        "table_id": "orders"
    },
    {
        "question": "Show customers who spent more than 50000",
        "query": "SELECT * FROM customers WHERE total_spent > 50000 ORDER BY total_spent DESC",
        "table_id": "customers"
    },
    {
        "question": "Top 5 categories by total sales",
        "query": "SELECT category, SUM(sales_amount) FROM sales GROUP BY category ORDER BY SUM(sales_amount) DESC LIMIT 5",
        "table_id": "sales"
    }
]

# Schema definitions for each table
SCHEMAS = {
    "customers": {
        "table": "customers",
        "columns": [
            "customer_id", "first_name", "last_name", "email", "phone",
            "city", "state", "join_date", "total_orders", "total_spent"
        ],
        "types": ["INTEGER", "TEXT", "TEXT", "TEXT", "TEXT",
                  "TEXT", "TEXT", "DATE", "INTEGER", "REAL"]
    },
    "products": {
        "table": "products",
        "columns": [
            "product_id", "product_name", "category", "price", "stock_qty", "rating"
        ],
        "types": ["INTEGER", "TEXT", "TEXT", "REAL", "INTEGER", "REAL"]
    },
    "orders": {
        "table": "orders",
        "columns": [
            "order_id", "customer_id", "order_date", "status",
            "total_amount", "city"
        ],
        "types": ["INTEGER", "INTEGER", "DATE", "TEXT", "REAL", "TEXT"]
    },
    "order_items": {
        "table": "order_items",
        "columns": [
            "item_id", "order_id", "product_id", "quantity", "unit_price", "subtotal"
        ],
        "types": ["INTEGER", "INTEGER", "INTEGER", "INTEGER", "REAL", "REAL"]
    },
    "employees": {
        "table": "employees",
        "columns": [
            "emp_id", "first_name", "last_name", "email",
            "dept_id", "salary", "hire_date", "city"
        ],
        "types": ["INTEGER", "TEXT", "TEXT", "TEXT",
                  "INTEGER", "REAL", "DATE", "TEXT"]
    },
    "departments": {
        "table": "departments",
        "columns": [
            "dept_id", "dept_name", "location", "budget"
        ],
        "types": ["INTEGER", "TEXT", "TEXT", "REAL"]
    },
    "sales": {
        "table": "sales",
        "columns": [
            "sale_id", "customer_id", "month", "year", "sales_amount", "city", "category"
        ],
        "types": ["INTEGER", "INTEGER", "INTEGER", "INTEGER", "REAL", "TEXT", "TEXT"]
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def classify_intent(sql: str) -> str:
    """
    Classify SQL query into intent category.

    Args:
        sql: SQL query string

    Returns:
        Intent label string
    """
    sql_upper = sql.upper()

    # Check for complex queries first (multiple clauses)
    has_where = "WHERE" in sql_upper
    has_group = "GROUP BY" in sql_upper
    has_order = "ORDER BY" in sql_upper
    has_limit = "LIMIT" in sql_upper
    has_join = "JOIN" in sql_upper
    has_aggregate = any(agg in sql_upper for agg in ["COUNT", "SUM", "AVG", "MAX", "MIN"])

    # Complex: multiple clauses
    clauses = sum([has_where, has_group, has_order, has_limit, has_join])
    if clauses >= 2 or (has_aggregate and has_group):
        return "COMPLEX"

    # Single intent classification
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
    """
    Normalize natural language question.

    Args:
        text: Raw question text

    Returns:
        Cleaned question text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove trailing punctuation for consistency
    text = text.rstrip('?').rstrip('.').rstrip('!').strip()

    return text


def enrich_record(
    record: Dict[str, Any],
    schema: Dict[str, Any],
    source: str = "wikisql"
) -> Dict[str, Any]:
    """
    Enrich a raw record with additional fields.

    Args:
        record: Raw record with question, query, table_id
        schema: Schema information for the table
        source: Data source identifier

    Returns:
        Enriched record with all required fields
    """
    cleaned_question = clean_question(record["question"])
    intent = classify_intent(record["query"])

    return {
        "id": f"{source}_{len(record):06d}",  # Will be updated with proper ID
        "question": cleaned_question,
        "query": record["query"].strip(),
        "table_id": record["table_id"],
        "schema": schema,
        "intent": intent,
        "source": source,
        "raw_question": record["question"]  # Keep original for reference
    }


def generate_id(record_idx: int, split: str) -> str:
    """Generate unique ID for record."""
    return f"wikisql_{split}_{record_idx:06d}"


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def download_from_huggingface() -> Optional[DatasetDict]:
    """
    Attempt to download WikiSQL from HuggingFace.

    Returns:
        DatasetDict if successful, None otherwise
    """
    if not HF_AVAILABLE:
        logger.warning("HuggingFace 'datasets' library not available")
        return None

    try:
        logger.info("Attempting to download WikiSQL from HuggingFace...")
        dataset = load_dataset("salesforce-wikisql", trust_remote_code=True)
        logger.info(f"Successfully downloaded WikiSQL from HuggingFace")
        logger.info(f"Splits: {list(dataset.keys())}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to download from HuggingFace: {e}")
        return None


def create_synthetic_dataset() -> Dict[str, List[Dict[str, Any]]]:
    """
    Create synthetic dataset from predefined examples.

    Returns:
        Dictionary with 'train', 'validation', 'test' splits
    """
    logger.info("Creating synthetic WikiSQL dataset...")

    # Use synthetic data and shuffle
    data = SYNTHETIC_WIKISQL_TRAIN.copy()
    random.shuffle(data)

    # Split: 70% train, 15% validation, 15% test
    n_total = len(data)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    logger.info(f"Split sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Enrich all records
    for split, records in [("train", train_data), ("validation", val_data), ("test", test_data)]:
        enriched = []
        for idx, record in enumerate(records):
            table_id = record["table_id"]
            schema = SCHEMAS.get(table_id, SCHEMAS["customers"])
            enriched_record = enrich_record(record, schema, "wikisql")
            enriched_record["id"] = generate_id(idx, split if split != "validation" else "val")
            enriched.append(enriched_record)
        records.clear()
        records.extend(enriched)

    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }


def process_huggingface_dataset(dataset: DatasetDict) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process HuggingFace WikiSQL dataset into our format.

    Args:
        dataset: HuggingFace DatasetDict

    Returns:
        Dictionary with 'train', 'validation', 'test' splits
    """
    logger.info("Processing HuggingFace WikiSQL dataset...")

    result = {}

    for split in ["train", "validation", "test"]:
        if split not in dataset:
            logger.warning(f"Split '{split}' not found in dataset")
            continue

        records = []
        for idx, row in enumerate(dataset[split]):
            # WikiSQL format: question, sql, table_id
            question = row["question"]
            sql_dict = row["sql"]
            table_id = row["table_id"]

            # Convert SQL dict to string
            # WikiSQL sql format: {"sel": 0, "agg": 0, "conds": [[2, "Equal", "a"]]}
            sql_str = sql_to_string(sql_dict)

            # Get schema for table
            schema = get_wikisql_schema(table_id)  # Will need implementation

            enriched = enrich_record({
                "question": question,
                "query": sql_str,
                "table_id": table_id
            }, schema, "wikisql")
            enriched["id"] = generate_id(idx, split)

            records.append(enriched)

        result[split] = records
        logger.info(f"Processed {len(records)} records for {split} split")

    return result


def sql_to_string(sql_dict: Dict[str, Any]) -> str:
    """
    Convert WikiSQL dict format to SQL string.
    This is a simplified converter - will need table info.
    """
    # Simplified: In real implementation, need table schema to properly convert
    sel = sql_dict.get("sel", 0)
    agg = sql_dict.get("agg", 0)
    conds = sql_dict.get("conds", [])

    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    sel_agg = agg_ops[agg] if agg > 0 else ""
    sel_col = f"col{sel}"  # Placeholder

    if not conds:
        if sel_agg:
            return f"SELECT {sel_agg}({sel_col}) FROM table"
        else:
            return f"SELECT * FROM table"

    # Simplified WHERE clause
    where_parts = []
    for col_idx, op, val in conds:
        op_map = {"Equal": "=", "NotEqual": "!=", "GreaterThan": ">",
                  "LessThan": "<", "GreaterEqual": ">=", "LessEqual": "<="}
        op_str = op_map.get(op, "=")
        where_parts.append(f"col{col_idx} {op_str} '{val}'")

    where_clause = " AND ".join(where_parts)
    if sel_agg:
        return f"SELECT {sel_agg}({sel_col}) FROM table WHERE {where_clause}"
    else:
        return f"SELECT * FROM table WHERE {where_clause}"


def get_wikisql_schema(table_id: str) -> Dict[str, Any]:
    """
    Get schema for a WikiSQL table.
    In real implementation, would load from dataset.
    """
    # Simplified schema
    return {
        "table": table_id,
        "columns": ["col0", "col1", "col2", "col3", "col4"],
        "types": ["TEXT", "INTEGER", "REAL", "TEXT", "INTEGER"]
    }


def save_splits(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Save dataset splits to JSON files.

    Args:
        splits: Dictionary with 'train', 'validation', 'test' keys
    """
    WIKISQL_DATA.mkdir(parents=True, exist_ok=True)

    for split, records in splits.items():
        output_file = WIKISQL_DATA / f"{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(records)} records to {output_file}")

    # Save intent distribution
    print_intent_distribution(splits)


def print_intent_distribution(splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print intent distribution for each split."""
    logger.info("\n" + "="*60)
    logger.info("INTENT DISTRIBUTION")
    logger.info("="*60)

    for split, records in splits.items():
        intent_counts = {intent: 0 for intent in INTENTS}
        for record in records:
            intent = record.get("intent", "UNKNOWN")
            if intent in intent_counts:
                intent_counts[intent] += 1

        logger.info(f"\n{split.upper()} ({len(records)} records):")
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                pct = (count / len(records)) * 100
                logger.info(f"  {intent:20s}: {count:4d} ({pct:5.1f}%)")

    logger.info("="*60 + "\n")


def get_database_schema_from_db() -> Dict[str, Any]:
    """
    Load schema from database.
    Uses the sample database created earlier.
    """
    import sqlite3

    db_path = DATABASE / "sample.db"
    if not db_path.exists():
        logger.warning(f"Database not found at {db_path}. Using default schema.")
        return SCHEMAS["customers"]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    schema = {"tables": {}}

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = []
        types = []
        for row in cursor.fetchall():
            columns.append(row[1])  # name
            types.append(row[2])    # type

        schema["tables"][table] = {
            "columns": columns,
            "types": types
        }

    conn.close()
    return schema


def download_wikisql() -> None:
    """
    Main function: Generate (synthetic) WikiSQL dataset.
    Uses the synthetic generator to produce large-scale, valid datasets.
    """
    logger.info("="*80)
    logger.info("GENERATING WIKISQL DATASET (SYNTHETIC)")
    logger.info("="*80)

    try:
        # Import generator
        from data.generator import generate_wikisql_splits

        # Generate datasets
        splits = generate_wikisql_splits(
            train_size=50000,
            val_size=5000,
            test_size=5000,
            distribution="natural"
        )
    except Exception as e:
        logger.error(f"Failed to generate WikiSQL: {e}")
        raise

    # Verify splits exist
    required_splits = ["train", "validation", "test"]
    for split in required_splits:
        if split not in splits or len(splits[split]) == 0:
            logger.error(f"Missing or empty split: {split}")
            raise ValueError(f"Dataset split '{split}' is missing or empty")

    # Save to files
    save_splits(splits)

    logger.info("WikiSQL dataset ready!")
    logger.info(f"Files saved to: {WIKISQL_DATA}")
    logger.info("="*80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        # Ensure data directories exist
        WIKISQL_DATA.mkdir(parents=True, exist_ok=True)

        download_wikisql()
    except Exception as e:
        logger.error(f"Failed to download WikiSQL: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
