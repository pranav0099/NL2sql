"""
Synthetic NL2SQL Data Generator
Generates large-scale, schema-valid synthetic datasets for WikiSQL and Spider formats.

Author: Pranav
Date: 2026-04-04

Features:
  - Template-based generation with randomization
  - Full validation against schema
  - Natural intent distribution
  - Support for both single-table (WikiSQL) and multi-table (Spider) queries
  - Configurable dataset sizes
"""

import json
import random
import sqlite3
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter
from datetime import datetime, timedelta

from config.config import INTENTS, INTENT2IDX, SQL_KEYWORDS
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# SCHEMA LOADING
# ============================================================================

def load_schema(schema_path: str = "database/schema.json") -> Dict[str, Any]:
    """
    Load database schema from JSON file.

    Args:
        schema_path: Path to schema.json

    Returns:
        Schema dictionary with tables, columns, foreign_keys
    """
    with open(schema_path, 'r', encoding='utf-8') as f:
        full_schema = json.load(f)

    # Extract just the tables info
    tables = {}
    for table_name, table_info in full_schema["tables"].items():
        columns = [col["name"] for col in table_info["columns"]]
        column_types = [col["type"] for col in table_info["columns"]]
        foreign_keys = table_info.get("foreign_keys", [])

        tables[table_name] = {
            "columns": columns,
            "types": column_types,
            "foreign_keys": foreign_keys,
            "primary_key": [col["name"] for col in table_info["columns"] if col.get("pk")],
            "create_sql": table_info["create_sql"]
        }

    return {
        "tables": tables,
        "table_names": list(tables.keys())
    }

# Load schema on module import
SCHEMA = load_schema()
TABLE_NAMES = SCHEMA["table_names"]

# Define relationships for JOIN queries
RELATIONSHIPS = [
    ("orders", "customer_id", "customers", "customer_id"),
    ("order_items", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("employees", "dept_id", "departments", "dept_id"),
    ("sales", "customer_id", "customers", "customer_id")
]

# ============================================================================
# VALUE GENERATORS (for realistic data)
# ============================================================================

FIRST_NAMES = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
               "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
               "Thomas", "Sarah", "Charles", "Karen", "Emma", "Ava", "Oliver", "Sophia"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson"]

CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kolkata", "Jaipur",
          "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"]

STATES = ["Maharashtra", "California", "Texas", "Florida", "New York", "Illinois", "Pennsylvania"]

DEPARTMENTS = [
    ("Engineering", "Bangalore"),
    ("Sales", "Mumbai"),
    ("Marketing", "Delhi"),
    ("HR", "Pune"),
    ("Finance", "Chennai"),
    ("Support", "Hyderabad")
]

CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Toys",
              "Automotive", "Health", "Beauty", "Office Supplies"]

PRODUCT_NAMES = [
    "Laptop Pro 15", "Wireless Mouse", "Mechanical Keyboard", "Monitor 4K",
    "Smartphone X", "Tablet Plus", "Headphones Elite", "Speakers Max",
    "T-Shirt Classic", "Jeans Premium", "Jacket Winter", "Sneakers Sport",
    "Novel Bestseller", "Textbook College", "Biography Famous", "Cookbook Healthy",
    "Sofa Modern", "Table Dining", "Chair Office", "Lamp Desk",
    "Basketball Official", "Tennis Racket", "Running Shoes", "Yoga Mat"
]

STATUSES = ["pending", "processing", "shipped", "delivered", "cancelled"]

def random_date(start_year: int = 2020, end_year: int = 2024) -> str:
    """Generate random date in YYYY-MM-DD format."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    date = start + timedelta(days=random_days)
    return date.strftime("%Y-%m-%d")

def random_phone() -> str:
    """Generate random phone number."""
    return f"9{random.randint(10,99)}-{random.randint(10000000,99999999)}"

def random_email(first: str, last: str) -> str:
    """Generate email from name."""
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com"]
    return f"{first.lower()}.{last.lower()}@{random.choice(domains)}"

# ============================================================================
# TEMPLATE DEFINITIONS
# ============================================================================

# WikiSQL Templates (single-table, no joins)
WIKISQL_TEMPLATES = {
    "SELECT": [
        # Simple SELECT *
        ("show all {table}", "SELECT * FROM {table}"),
        ("list all {table}", "SELECT * FROM {table}"),
        ("display all {table}", "SELECT * FROM {table}"),
        ("give me all {table}", "SELECT * FROM {table}"),
        ("show {table}", "SELECT * FROM {table}"),
    ],
    "SELECT_WHERE": [
        ("show {table} from {city}", "SELECT * FROM {table} WHERE city = '{city}'"),
        ("find {table} in {city}", "SELECT * FROM {table} WHERE city = '{city}'"),
        ("list {table} located in {city}", "SELECT * FROM {table} WHERE city = '{city}'"),
        ("show {table} with {column} > {value}", "SELECT * FROM {table} WHERE {column} > {value}"),
        ("find {table} where {column} = '{text}'", "SELECT * FROM {table} WHERE {column} = '{text}'"),
        ("show {table} with {column} like '%{text}%'", "SELECT * FROM {table} WHERE {column} LIKE '%{text}%'"),
    ],
    "SELECT_AGGREGATE": [
        ("how many {table} are there", "SELECT COUNT(*) FROM {table}"),
        ("what is the total number of {table}", "SELECT COUNT(*) FROM {table}"),
        ("count all {table}", "SELECT COUNT(*) FROM {table}"),
        ("what is the average {column} of {table}", "SELECT AVG({column}) FROM {table}"),
        ("what is the maximum {column} in {table}", "SELECT MAX({column}) FROM {table}"),
        ("what is the minimum {column} of {table}", "SELECT MIN({column}) FROM {table}"),
        ("what is the total {column} for all {table}", "SELECT SUM({column}) FROM {table}"),
    ],
    "SELECT_ORDER": [
        ("show {table} sorted by {column}", "SELECT * FROM {table} ORDER BY {column} ASC"),
        ("list {table} ordered by {column} descending", "SELECT * FROM {table} ORDER BY {column} DESC"),
        ("show {table} by {column}", "SELECT * FROM {table} ORDER BY {column} DESC"),
        ("sort {table} with highest {column} first", "SELECT * FROM {table} ORDER BY {column} DESC"),
    ],
    "SELECT_LIMIT": [
        ("show first {limit} {table}", "SELECT * FROM {table} LIMIT {limit}"),
        ("list top {limit} {table}", "SELECT * FROM {table} LIMIT {limit}"),
        ("show only {limit} {table}", "SELECT * FROM {table} LIMIT {limit}"),
        ("give me {limit} {table}", "SELECT * FROM {table} LIMIT {limit}"),
    ],
    "SELECT_GROUP": [
        ("how many {table} per {column}", "SELECT {column}, COUNT(*) FROM {table} GROUP BY {column}"),
        ("count {table} by {column}", "SELECT {column}, COUNT(*) FROM {table} GROUP BY {column}"),
        ("total {agg_column} by {group_column} in {table}", "SELECT {group_column}, SUM({agg_column}) FROM {table} GROUP BY {group_column}"),
        ("average {agg_column} for each {group_column} in {table}", "SELECT {group_column}, AVG({agg_column}) FROM {table} GROUP BY {group_column}"),
    ],
    "COMPLEX": [
        ("show {table} with {column} > {value} ordered by {order_col} limit {limit}",
         "SELECT * FROM {table} WHERE {column} > {value} ORDER BY {order_col} DESC LIMIT {limit}"),
        ("find {table} in {city} with {column} > {value}",
         "SELECT * FROM {table} WHERE city = '{city}' AND {column} > {value}"),
        ("top {limit} {table} from {city} by {order_col}",
         "SELECT * FROM {table} WHERE city = '{city}' ORDER BY {order_col} DESC LIMIT {limit}"),
        ("count {table} where {column} > {value} group by {group_col}",
         "SELECT {group_col}, COUNT(*) FROM {table} WHERE {column} > {value} GROUP BY {group_col}"),
    ]
}

# Spider / Multi-table templates (with JOINs)
SPIDER_TEMPLATES = {
    "SELECT_JOIN": [
        ("show all {table1} with their {table2}",
         "SELECT {t1_cols}, {t2_cols} FROM {table1} JOIN {table2} ON {t1_fk} = {t2_pk}"),
        ("list {table1} along with {table2} details",
         "SELECT {t1_cols}, {t2_cols} FROM {table1} JOIN {table2} ON {t1_fk} = {t2_pk}"),
        ("combine {table1} and {table2}",
         "SELECT {t1_cols}, {t2_cols} FROM {table1} JOIN {table2} ON {t1_fk} = {t2_pk}"),
    ],
    "SELECT_WHERE": [
        ("show {table1} with {table2} where {condition}",
         "SELECT {t1_cols}, {t2_cols} FROM {table1} JOIN {table2} ON {t1_fk} = {t2_pk} WHERE {where_clause}"),
    ],
    "SELECT_AGGREGATE": [
        ("total {agg_col} of {table1} by {group_col}",
         "SELECT {group_col}, {agg_func}({agg_col}) FROM {table1} GROUP BY {group_col}"),
        ("average {agg_col} for each {group_col}",
         "SELECT {group_col}, AVG({agg_col}) FROM {table1} GROUP BY {group_col}"),
    ],
    "SELECT_ORDER": [
        ("show {table1} sorted by {order_col}",
         "SELECT {t1_cols} FROM {table1} ORDER BY {order_col} DESC"),
    ],
    "SELECT_GROUP": [
        ("count {table1} per {group_col}",
         "SELECT {group_col}, COUNT(*) FROM {table1} GROUP BY {group_col}"),
        ("sum of {agg_col} grouped by {group_col}",
         "SELECT {group_col}, SUM({agg_col}) FROM {table1} GROUP BY {group_col}"),
    ],
    "SELECT_LIMIT": [
        ("top {limit} {table1} by {order_col}",
         "SELECT * FROM {table1} ORDER BY {order_col} DESC LIMIT {limit}"),
    ],
    "COMPLEX": [
        ("show {table1} with {table2} where {where_clause} group by {group_col} having {having_clause}",
         "SELECT {t1_cols}, {t2_cols}, {agg_func}({agg_col}) FROM {table1} JOIN {table2} ON {t1_fk} = {t2_pk} WHERE {where_clause} GROUP BY {group_col} HAVING {having_clause}"),
        ("total {agg_col} by {group_col} for {table1} in {city}",
         "SELECT {group_col}, {agg_func}({agg_col}) FROM {table1} JOIN {table2} ON {t1_fk} = {t2_pk} WHERE {filter_col} = '{city}' GROUP BY {group_col}"),
        ("find {table1} with more than {value} {agg} group by {group_col}",
         "SELECT {group_col}, COUNT(*) FROM {table1} JOIN {table2} ON {t1_fk} = {t2_pk} GROUP BY {group_col} HAVING COUNT(*) > {value}"),
    ]
}

# Table-specific templates (single-table queries on specific tables)
TABLE_SPECIFIC_TEMPLATES = {
    "customers": {
        "SELECT": [
            ("show all customers", "SELECT * FROM customers"),
            ("list all customers", "SELECT * FROM customers"),
            ("give me the customer list", "SELECT * FROM customers"),
        ],
        "SELECT_WHERE": [
            ("show customers from {city}", "SELECT * FROM customers WHERE city = '{city}'"),
            ("find customers in {city}", "SELECT * FROM customers WHERE city = '{city}'"),
            ("show customers who joined after {date}", "SELECT * FROM customers WHERE join_date > '{date}'"),
            ("customers with more than {value} orders", "SELECT * FROM customers WHERE total_orders > {value}"),
            ("customers who spent more than {value}", "SELECT * FROM customers WHERE total_spent > {value}"),
        ],
        "SELECT_AGGREGATE": [
            ("how many customers total", "SELECT COUNT(*) FROM customers"),
            ("total number of customers", "SELECT COUNT(*) FROM customers"),
            ("average total spent by customers", "SELECT AVG(total_spent) FROM customers"),
            ("maximum spending by a customer", "SELECT MAX(total_spent) FROM customers"),
            ("total orders across all customers", "SELECT SUM(total_orders) FROM customers"),
        ],
        "SELECT_ORDER": [
            ("customers sorted by total spent", "SELECT * FROM customers ORDER BY total_spent DESC"),
            ("list customers by join date", "SELECT * FROM customers ORDER BY join_date DESC"),
            ("customers ordered by last name", "SELECT * FROM customers ORDER BY last_name ASC"),
        ],
        "SELECT_LIMIT": [
            ("top 10 customers by spending", "SELECT * FROM customers ORDER BY total_spent DESC LIMIT 10"),
            ("show 5 most recent customers", "SELECT * FROM customers ORDER BY join_date DESC LIMIT 5"),
        ],
        "COMPLEX": [
            ("show top {limit} customers from {city}",
             "SELECT * FROM customers WHERE city = '{city}' ORDER BY total_spent DESC LIMIT {limit}"),
            ("customers in {city} with more than {value} orders",
             "SELECT * FROM customers WHERE city = '{city}' AND total_orders > {value}"),
        ]
    },
    "products": {
        "SELECT": [("show all products", "SELECT * FROM products")],
        "SELECT_WHERE": [
            ("products in {category} category", "SELECT * FROM products WHERE category = '{category}'"),
            ("products with price > {value}", "SELECT * FROM products WHERE price > {value}"),
            ("products with rating above {rating}", "SELECT * FROM products WHERE rating > {rating}"),
            ("products with stock less than {value}", "SELECT * FROM products WHERE stock_qty < {value}"),
        ],
        "SELECT_AGGREGATE": [
            ("how many products", "SELECT COUNT(*) FROM products"),
            ("average price of products", "SELECT AVG(price) FROM products"),
            ("most expensive product price", "SELECT MAX(price) FROM products"),
            ("total inventory value", "SELECT SUM(price * stock_qty) FROM products"),
        ],
        "SELECT_ORDER": [
            ("products sorted by price", "SELECT * FROM products ORDER BY price DESC"),
            ("products by rating", "SELECT * FROM products ORDER BY rating DESC"),
        ],
        "SELECT_LIMIT": [
            ("top 5 expensive products", "SELECT * FROM products ORDER BY price DESC LIMIT 5"),
            ("show 10 products", "SELECT * FROM products LIMIT 10"),
        ],
        "COMPLEX": [
            ("show {category} products priced above {value}",
             "SELECT * FROM products WHERE category = '{category}' AND price > {value}"),
        ]
    },
    "orders": {
        "SELECT": [("show all orders", "SELECT * FROM orders")],
        "SELECT_WHERE": [
            ("orders in {city}", "SELECT * FROM orders WHERE city = '{city}'"),
            ("orders with status {status}", "SELECT * FROM orders WHERE status = '{status}'"),
            ("orders after {date}", "SELECT * FROM orders WHERE order_date > '{date}'"),
            ("orders over {value}", "SELECT * FROM orders WHERE total_amount > {value}"),
        ],
        "SELECT_AGGREGATE": [
            ("total number of orders", "SELECT COUNT(*) FROM orders"),
            ("average order value", "SELECT AVG(total_amount) FROM orders"),
            ("total revenue", "SELECT SUM(total_amount) FROM orders"),
        ],
        "SELECT_ORDER": [
            ("orders sorted by date", "SELECT * FROM orders ORDER BY order_date DESC"),
            ("orders by total amount", "SELECT * FROM orders ORDER BY total_amount DESC"),
        ],
        "SELECT_LIMIT": [
            ("recent 10 orders", "SELECT * FROM orders ORDER BY order_date DESC LIMIT 10"),
        ],
        "COMPLEX": [
            ("delivered orders in {city} over {value}",
             "SELECT * FROM orders WHERE status = 'delivered' AND city = '{city}' AND total_amount > {value}"),
        ]
    },
    "employees": {
        "SELECT": [("show all employees", "SELECT * FROM employees")],
        "SELECT_WHERE": [
            ("employees in {city}", "SELECT * FROM employees WHERE city = '{city}'"),
            ("employees with salary > {value}", "SELECT * FROM employees WHERE salary > {value}"),
            ("employees hired after {date}", "SELECT * FROM employees WHERE hire_date > '{date}'"),
        ],
        "SELECT_AGGREGATE": [
            ("total employees", "SELECT COUNT(*) FROM employees"),
            ("average salary", "SELECT AVG(salary) FROM employees"),
            ("highest salary", "SELECT MAX(salary) FROM employees"),
            ("total salary cost", "SELECT SUM(salary) FROM employees"),
        ],
        "SELECT_ORDER": [
            ("employees sorted by salary", "SELECT * FROM employees ORDER BY salary DESC"),
            ("employees by hire date", "SELECT * FROM employees ORDER BY hire_date DESC"),
        ],
        "COMPLEX": [
            ("employees in {dept} with salary > {value}",
             "SELECT * FROM employees WHERE dept_id = {dept_id} AND salary > {value}"),
        ]
    },
    "departments": {
        "SELECT": [("show all departments", "SELECT * FROM departments")],
        "SELECT_WHERE": [
            ("departments in {location}", "SELECT * FROM departments WHERE location = '{location}'"),
            ("departments with budget > {value}", "SELECT * FROM departments WHERE budget > {value}"),
        ],
        "SELECT_AGGREGATE": [
            ("total departments", "SELECT COUNT(*) FROM departments"),
            ("average budget", "SELECT AVG(budget) FROM departments"),
        ],
        "COMPLEX": []
    },
    "sales": {
        "SELECT": [("show all sales", "SELECT * FROM sales")],
        "SELECT_WHERE": [
            ("sales in {city}", "SELECT * FROM sales WHERE city = '{city}'"),
            ("sales in {year}", "SELECT * FROM sales WHERE year = {year}"),
            ("sales for {category}", "SELECT * FROM sales WHERE category = '{category}'"),
        ],
        "SELECT_AGGREGATE": [
            ("total sales amount", "SELECT SUM(sales_amount) FROM sales"),
            ("average sale amount", "SELECT AVG(sales_amount) FROM sales"),
            ("count of sales", "SELECT COUNT(*) FROM sales"),
        ],
        "SELECT_GROUP": [
            ("sales by category", "SELECT category, SUM(sales_amount) FROM sales GROUP BY category"),
            ("sales by month", "SELECT month, SUM(sales_amount) FROM sales GROUP BY month"),
        ],
        "COMPLEX": [
            ("sales in {year} by {city}",
             "SELECT city, SUM(sales_amount) FROM sales WHERE year = {year} GROUP BY city"),
        ]
    },
    "order_items": {
        "SELECT": [("show all order items", "SELECT * FROM order_items")],
        "SELECT_WHERE": [
            ("order items with quantity > {value}", "SELECT * FROM order_items WHERE quantity > {value}"),
            ("order items with subtotal > {value}", "SELECT * FROM order_items WHERE subtotal > {value}"),
        ],
        "SELECT_AGGREGATE": [
            ("total order items", "SELECT COUNT(*) FROM order_items"),
            ("total revenue from order items", "SELECT SUM(subtotal) FROM order_items"),
        ],
        "COMPLEX": []
    }
}

# ============================================================================
# DISTRIBUTION WEIGHTS (natural distribution)
# ============================================================================

INTENT_DISTRIBUTION = {
    "SELECT": 0.10,
    "SELECT_WHERE": 0.20,
    "SELECT_AGGREGATE": 0.15,
    "SELECT_ORDER": 0.10,
    "SELECT_JOIN": 0.20,  # Only for Spider (multi-table)
    "SELECT_GROUP": 0.15,
    "SELECT_LIMIT": 0.05,
    "COMPLEX": 0.05
}

# Table distribution (for single-table queries)
TABLE_DISTRIBUTION = {
    "customers": 0.20,
    "products": 0.20,
    "orders": 0.20,
    "employees": 0.15,
    "departments": 0.10,
    "sales": 0.10,
    "order_items": 0.05
}

# ============================================================================
# VALIDATION
# ============================================================================

class QueryValidator:
    """Validates generated SQL queries against the schema."""

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.tables = set(schema["table_names"])
        self.table_to_columns = {t: set(tbl["columns"]) for t, tbl in schema["tables"].items()}
        self.foreign_keys = []
        for tname, tbl in schema["tables"].items():
            for fk in tbl.get("foreign_keys", []):
                self.foreign_keys.append((tname, fk["from"], fk["table"], fk["to"]))

    def extract_tables_and_columns(self, sql: str) -> Tuple[Set[str], Set[Tuple[str, str]]]:
        """
        Extract table names and (table, column) pairs from SQL query.
        Simple regex-based extraction for valid SQLite syntax.
        """
        sql_lower = sql.lower()
        tables = set()
        columns = set()  # (table, column) pairs

        # Extract FROM clauses and JOIN clauses
        # Match: FROM table_name [AS alias] or JOIN table_name [AS alias]
        from_pattern = r'\b(?:from|join)\s+([a-z_]+)(?:\s+[a-z]+)?\b'
        matches = re.findall(from_pattern, sql_lower)
        tables.update(matches)

        # Extract column references: table.column or alias.column
        # Pattern: word.word where second word is not a keyword
        col_pattern = r'\b([a-z_]+)\.([a-z_]+)\b'
        col_matches = re.findall(col_pattern, sql_lower)
        for tbl, col in col_matches:
            if tbl in self.tables:
                columns.add((tbl, col))

        # Also check unqualified column names in SELECT (if only one table)
        if len(tables) == 1:
            table = list(tables)[0]
            # Find SELECT ... FROM pattern
            select_match = re.search(r'select\s+(.+?)\s+from', sql_lower, re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                # Extract words that are column names
                words = re.findall(r'\b([a-z_]+)\b', select_clause)
                for word in words:
                    if word != '*' and word in self.table_to_columns.get(table, set()):
                        columns.add((table, word))

        return tables, columns

    def validate(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL query.

        Returns:
            dict with keys: valid (bool), errors (list), tables (set), columns (set)
        """
        result = {
            "valid": True,
            "errors": [],
            "tables": set(),
            "columns": set()
        }

        # Extract tables and columns
        tables, columns = self.extract_tables_and_columns(sql)
        result["tables"] = tables
        result["columns"] = columns

        # Check tables exist
        for table in tables:
            if table not in self.tables:
                result["valid"] = False
                result["errors"].append(f"Unknown table: {table}")

        # Check columns exist in their tables
        for table, column in columns:
            if table not in self.tables:
                continue  # Already flagged
            if column not in self.table_to_columns.get(table, set()):
                result["valid"] = False
                result["errors"].append(f"Unknown column: {table}.{column}")

        # Additional checks
        # 1. Check that JOINs use valid foreign keys
        sql_lower = sql.lower()
        joins = re.findall(r'join\s+([a-z_]+)\s+on\s+([a-z_]+)\.([a-z_]+)\s*=\s*([a-z_]+)\.([a-z_]+)', sql_lower)
        for table2, t1, c1, t2, c2 in joins:
            if (t1, c1, t2, c2) not in self.foreign_keys and (t2, c2, t1, c1) not in self.foreign_keys:
                # Not a strict FK but might still be valid (many-to-many through junction table)
                # Allow if columns exist in their respective tables
                if c1 not in self.table_to_columns.get(t1, set()):
                    result["valid"] = False
                    result["errors"].append(f"JOIN column doesn't exist: {t1}.{c1}")
                if c2 not in self.table_to_columns.get(t2, set()):
                    result["valid"] = False
                    result["errors"].append(f"JOIN column doesn't exist: {t2}.{c2}")

        return result

# ============================================================================
# NLP UTILITIES
# ============================================================================

def classify_intent(sql: str) -> str:
    """Classify SQL query into intent category."""
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
    """Normalize natural language question."""
    text = text.lower()
    text = " ".join(text.split())
    text = text.rstrip('?').rstrip('.').rstrip('!').strip()
    return text

# ============================================================================
# TEMPLATE FILLING
# ============================================================================

class TemplateFiller:
    """Fills templates with random values to generate diverse queries."""

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.validator = QueryValidator(schema)

    def fill_single_table_template(self, table: str, intent: str, template: str, sql_template: str) -> Dict[str, Any]:
        """Generate a single-table WikiSQL example."""
        # Generate random values for placeholders
        format_dict = {}

        # Get table columns by type
        table_info = self.schema["tables"][table]
        numeric_cols = [c for i, c in enumerate(table_info["columns"]) if table_info["types"][i] in ["INTEGER", "REAL"]]
        text_cols = [c for i, c in enumerate(table_info["columns"]) if table_info["types"][i] == "TEXT"]
        date_cols = [c for i, c in enumerate(table_info["columns"]) if table_info["types"][i] == "DATE"]

        # Replace {table}
        format_dict["table"] = table

        # Replace {city}
        if "{city}" in sql_template:
            format_dict["city"] = random.choice(CITIES)

        # Replace {column}
        if "{column}" in sql_template:
            if numeric_cols:
                format_dict["column"] = random.choice(numeric_cols)
            else:
                format_dict["column"] = random.choice(table_info["columns"])

        # Replace {value} (numeric)
        if "{value}" in sql_template:
            format_dict["value"] = random.randint(1, 10000)

        # Replace {rating}
        if "{rating}" in sql_template:
            format_dict["rating"] = round(random.uniform(3.0, 5.0), 1)

        # Replace {text}
        if "{text}" in sql_template:
            format_dict["text"] = random.choice(["premium", "standard", "basic", "active", "inactive"])

        # Replace {date}
        if "{date}" in sql_template:
            format_dict["date"] = random_date()

        # Replace {limit}
        if "{limit}" in sql_template:
            format_dict["limit"] = random.choice([5, 10, 15, 20, 50])

        # Replace {order_col}
        if "{order_col}" in sql_template:
            sortable_cols = numeric_cols + date_cols
            if sortable_cols:
                format_dict["order_col"] = random.choice(sortable_cols)
            else:
                format_dict["order_col"] = random.choice(table_info["columns"])

        # Replace {group_col}
        if "{group_col}" in sql_template:
            groupable = [c for c in table_info["columns"] if c not in [table_info["primary_key"][0] if table_info["primary_key"] else "id"]]
            format_dict["group_col"] = random.choice(groupable) if groupable else table_info["columns"][1]

        # Also support {group_column} (alternate name)
        if "{group_column}" in sql_template:
            groupable = [c for c in table_info["columns"] if c not in [table_info["primary_key"][0] if table_info["primary_key"] else "id"]]
            format_dict["group_column"] = random.choice(groupable) if groupable else table_info["columns"][1]

        # Replace {agg_column} and {agg_func}
        if "{agg_column}" in sql_template:
            format_dict["agg_column"] = random.choice(numeric_cols) if numeric_cols else table_info["columns"][-1]
        if "{agg_func}" in sql_template:
            format_dict["agg_func"] = random.choice(["SUM", "AVG", "COUNT", "MAX", "MIN"])

        # Auto-fill missing placeholders based on name patterns
        # Find all placeholders in both templates that haven't been filled
        missing_keys = set(re.findall(r'\{(\w+)\}', sql_template)) | set(re.findall(r'\{(\w+)\}', template)) - set(format_dict.keys())

        for key in missing_keys:
            # Generic fillers based on key name
            if key == "city":
                format_dict[key] = random.choice(CITIES)
            elif key == "status":
                format_dict[key] = random.choice(STATUSES)
            elif key == "year":
                format_dict[key] = random.randint(2020, 2024)
            elif key == "dept_id":
                format_dict[key] = random.randint(1, len(DEPARTMENTS)-1)
            elif key == "dept":
                format_dict[key] = random.choice([d[0] for d in DEPARTMENTS])
            elif key == "value":
                format_dict[key] = random.randint(1, 10000)
            elif key == "limit":
                format_dict[key] = random.choice([5, 10, 15, 20, 50])
            elif key == "rating":
                format_dict[key] = round(random.uniform(3.0, 5.0), 1)
            elif key == "text":
                format_dict[key] = random.choice(["premium", "standard", "active", "inactive"])
            elif key == "date":
                format_dict[key] = random_date()
            elif key == "location":
                format_dict[key] = random.choice([d[1] for d in DEPARTMENTS])
            elif key == "category":
                format_dict[key] = random.choice(CATEGORIES)
            elif key == "order_col":
                sortable_cols = numeric_cols + date_cols
                format_dict[key] = random.choice(sortable_cols) if sortable_cols else random.choice(table_info["columns"])
            elif key == "group_col" or key == "group_column":
                pk = table_info.get("primary_key", [])
                groupable = [c for c in table_info["columns"] if not pk or c != pk[0]]
                format_dict[key] = random.choice(groupable) if groupable else random.choice(table_info["columns"])
            elif key == "agg_column" or key == "agg_col":
                format_dict[key] = random.choice(numeric_cols) if numeric_cols else random.choice(table_info["columns"])
            elif key == "agg_func":
                format_dict[key] = random.choice(["SUM", "AVG", "COUNT", "MAX", "MIN"])
            elif key == "column":
                # Choose appropriate column based on context
                if numeric_cols:
                    format_dict[key] = random.choice(numeric_cols)
                else:
                    format_dict[key] = random.choice(table_info["columns"])
            else:
                # Unknown placeholder - skip by leaving as is (will cause KeyError and be caught)
                pass

        # Fill SQL
        try:
            sql = sql_template.format(**format_dict)
        except KeyError as e:
            logger.warning(f"Missing key in template: {e}. Template: {sql_template[:100]}... Table: {table}")
            return None

        # Validate
        validation = self.validator.validate(sql)
        if not validation["valid"]:
            # Retry with different values or return None
            return None

        # Generate question
        question = template.format(**format_dict)

        # Randomize ID
        idx = random.randint(0, 999999)

        return {
            "id": f"wikisql_train_{idx:06d}",
            "question": clean_question(question),
            "query": sql,
            "table_id": table,
            "schema": {
                "table": table,
                "columns": table_info["columns"],
                "types": table_info["types"]
            },
            "intent": intent,
            "source": "wikisql",
            "raw_question": question
        }

    def fill_multi_table_template(self, intent: str, sql_template: str, question_template: str) -> Optional[Dict[str, Any]]:
        """Generate a multi-table Spider example."""
        format_dict = {}

        # Pick a relationship
        t1, t1_fk, t2, t2_fk = random.choice(RELATIONSHIPS)
        format_dict["table1"] = t1
        format_dict["table2"] = t2
        format_dict["t1_fk"] = t1_fk
        format_dict["t2_pk"] = t2_fk

        # Select some columns
        t1_cols = random.sample(self.schema["tables"][t1]["columns"], k=min(3, len(self.schema["tables"][t1]["columns"])))
        t2_cols = random.sample(self.schema["tables"][t2]["columns"], k=min(2, len(self.schema["tables"][t2]["columns"])))
        format_dict["t1_cols"] = ", ".join([f"{t1}.{c}" for c in t1_cols])
        format_dict["t2_cols"] = ", ".join([f"{t2}.{c}" for c in t2_cols])

        # Fill other placeholders
        numeric_cols_t1 = [c for i, c in enumerate(self.schema["tables"][t1]["columns"])
                          if self.schema["tables"][t1]["types"][i] in ["INTEGER", "REAL"]]
        numeric_cols_t2 = [c for i, c in enumerate(self.schema["tables"][t2]["columns"])
                          if self.schema["tables"][t2]["types"][i] in ["INTEGER", "REAL"]]

        if "{agg_col}" in sql_template:
            which_table = random.choice([t1, t2])
            cols = numeric_cols_t1 if which_table == t1 else numeric_cols_t2
            format_dict["agg_col"] = random.choice(cols) if cols else "id"
            format_dict["agg_func"] = random.choice(["SUM", "AVG", "COUNT", "MAX", "MIN"])

        if "{group_col}" in sql_template:
            t1_pk = self.schema["tables"][t1].get("primary_key", [])
            t2_pk = self.schema["tables"][t2].get("primary_key", [])
            groupable_t1 = [c for c in self.schema["tables"][t1]["columns"] if not t1_pk or c != t1_pk[0]]
            groupable_t2 = [c for c in self.schema["tables"][t2]["columns"] if not t2_pk or c != t2_pk[0]]
            all_groupable = groupable_t1 + [f"{t2}.{c}" for c in groupable_t2]
            format_dict["group_col"] = random.choice(all_groupable) if all_groupable else t1_cols[0]

        if "{where_clause}" in sql_template:
            where_table = random.choice([t1, t2])
            where_cols = self.schema["tables"][where_table]["columns"]
            text_cols = [c for i, c in enumerate(self.schema["tables"][where_table]["columns"])
                        if self.schema["tables"][where_table]["types"][i] == "TEXT"]
            if text_cols:
                col = random.choice(text_cols)
                val = random.choice(CITIES if col == "city" else CATEGORIES if col == "category" else ["active", "pending", "delivered"])
                format_dict["where_clause"] = f"{where_table}.{col} = '{val}'"
            else:
                format_dict["where_clause"] = f"{where_table}.id > {random.randint(1, 100)}"

        if "{having_clause}" in sql_template:
            format_dict["having_clause"] = f"COUNT(*) > {random.randint(1, 10)}"

        if "{filter_col}" in sql_template:
            filter_table = random.choice([t1, t2])
            text_cols = [c for i, c in enumerate(self.schema["tables"][filter_table]["columns"])
                        if self.schema["tables"][filter_table]["types"][i] == "TEXT"]
            format_dict["filter_col"] = random.choice(text_cols) if text_cols else "city"
            format_dict["city"] = random.choice(CITIES)
            format_dict["year"] = random.randint(2020, 2024)

        if "{limit}" in sql_template:
            format_dict["limit"] = random.choice([5, 10, 15, 20])

        if "{value}" in sql_template:
            format_dict["value"] = random.randint(5, 100)

        if "{dept_id}" in sql_template:
            format_dict["dept_id"] = random.randint(1, 5)
            format_dict["dept"] = random.choice(["Engineering", "Sales", "Marketing", "HR"])[0].upper() + random.choice(["Sales", "Engineering"])[1:]

        # Fill SQL
        try:
            sql = sql_template.format(**format_dict)
        except KeyError as e:
            logger.warning(f"Missing key in template: {e}")
            return None

        # Validate
        validation = self.validator.validate(sql)
        if not validation["valid"]:
            return None

        # Determine intent
        intent = classify_intent(sql)

        # Generate question
        try:
            question = question_template.format(**format_dict)
        except KeyError as e:
            logger.warning(f"Missing key in question template: {e}")
            return None

        idx = random.randint(0, 999999)
        return {
            "id": f"spider_train_{idx:06d}",
            "question": clean_question(question),
            "query": sql,
            "db_id": "sample",
            "schema": {
                "table_names_original": TABLE_NAMES,
                "table_names": TABLE_NAMES,
                "column_names": [[t, c] for t in TABLE_NAMES for c in SCHEMA["tables"][t]["columns"]],
                "column_types": [SCHEMA["tables"][t]["types"][i] for t in TABLE_NAMES for i, c in enumerate(SCHEMA["tables"][t]["columns"])],
                "foreign_keys": [[fk[0], fk[1], fk[2], fk[3]] for fk in self.validator.foreign_keys],
                "primary_keys": [[t, SCHEMA["tables"][t]["primary_key"][0]] for t in TABLE_NAMES if SCHEMA["tables"][t]["primary_key"]]
            },
            "intent": intent,
            "source": "spider",
            "raw_question": question
        }

# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_wikisql_single(intent: str, filler: TemplateFiller, table: str, template_pair: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    """Generate a single WikiSQL example."""
    question_tpl, sql_tpl = template_pair
    return filler.fill_single_table_template(table, intent, question_tpl, sql_tpl)

def generate_spider_single(intent: str, filler: TemplateFiller, template_pair: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    """Generate a single Spider example."""
    question_tpl, sql_tpl = template_pair
    return filler.fill_multi_table_template(intent, sql_tpl, question_tpl)

def generate_wikisql_splits(
    train_size: int = 50000,
    val_size: int = 5000,
    test_size: int = 5000,
    distribution: str = "natural"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate WikiSQL dataset splits.

    Args:
        train_size, val_size, test_size: Number of samples per split
        distribution: "natural" (weighted) or "equal"

    Returns:
        {"train": [...], "validation": [...], "test": [...]}
    """
    logger.info(f"Generating WikiSQL: train={train_size}, val={val_size}, test={test_size}")

    filler = TemplateFiller(SCHEMA)
    splits = {"train": [], "validation": [], "test": []}

    for split_name, target_size in [("train", train_size), ("validation", val_size), ("test", test_size)]:
        logger.info(f"Generating {split_name} split ({target_size} samples)...")
        attempts = 0
        max_attempts = target_size * 10  # Allow retries for validation failures
        generated = 0
        stats = Counter()

        while generated < target_size and attempts < max_attempts:
            attempts += 1

            # Select intent based on distribution
            if distribution == "natural":
                intent = random.choices(
                    list(INTENT_DISTRIBUTION.keys()),
                    weights=list(INTENT_DISTRIBUTION.values())
                )[0]
            else:
                intent = random.choice(list(INTENT_DISTRIBUTION.keys()))

            # For pure WikiSQL, skip SELECT_JOIN (it's multi-table)
            if intent == "SELECT_JOIN":
                continue

            # Select table
            table = random.choices(
                list(TABLE_DISTRIBUTION.keys()),
                weights=list(TABLE_DISTRIBUTION.values())
            )[0]

            # Get templates for this table and intent
            templates = TABLE_SPECIFIC_TEMPLATES.get(table, {}).get(intent, [])
            if not templates:
                # Fall back to generic templates
                templates = WIKISQL_TEMPLATES.get(intent, [])

            if not templates:
                continue

            # Select template
            template_pair = random.choice(templates)

            # Generate
            sample = generate_wikisql_single(intent, filler, table, template_pair)

            if sample:
                # Additional check: ensure query doesn't reference wrong table
                if table not in sample["query"]:
                    continue  # Template error, skip

                sample["id"] = sample["id"].replace("wikisql_train", f"wikisql_{'val' if split_name=='validation' else split_name}")
                splits[split_name].append(sample)
                generated += 1
                stats[intent] += 1

            if generated % 1000 == 0 and generated > 0:
                logger.info(f"  Generated {generated}/{target_size} (attempts: {attempts})")

        logger.info(f"  Complete: {generated} samples")
        logger.info(f"  Intent distribution: {dict(stats)}")
        logger.info(f"  Success rate: {generated/attempts*100:.2f}%")

        if generated < target_size:
            logger.warning(f"  Only generated {generated}/{target_size} after {attempts} attempts!")

    return splits

def generate_spider_splits(
    train_size: int = 50000,
    val_size: int = 5000,
    test_size: int = 5000,
    distribution: str = "natural"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate Spider (multi-table) dataset splits.

    Args:
        train_size, val_size, test_size: Number of samples per split
        distribution: "natural" or "equal"

    Returns:
        {"train": [...], "validation": [...], "test": [...]}
    """
    logger.info(f"Generating Spider: train={train_size}, val={val_size}, test={test_size}")

    filler = TemplateFiller(SCHEMA)
    splits = {"train": [], "validation": [], "test": []}

    # Spider template pool
    all_templates = []
    for intent, templates in SPIDER_TEMPLATES.items():
        for tpl in templates:
            all_templates.append((intent, tpl))

    for split_name, target_size in [("train", train_size), ("validation", val_size), ("test", test_size)]:
        logger.info(f"Generating {split_name} split ({target_size} samples)...")
        attempts = 0
        max_attempts = target_size * 20  # More retries for complex joins
        generated = 0
        stats = Counter()

        while generated < target_size and attempts < max_attempts:
            attempts += 1

            # Select intent based on distribution (skip SELECT_JOIN as it's already in templates)
            if distribution == "natural":
                # Weight distribution but exclude SELECT_JOIN (already in multi-table templates)
                weights = [INTENT_DISTRIBUTION[intent] for intent, _ in all_templates]
                total_weight = sum(weights)
                weights = [w/total_weight for w in weights]
                idx = random.choices(range(len(all_templates)), weights=weights)[0]
                intent, template_pair = all_templates[idx]
            else:
                intent, template_pair = random.choice(all_templates)

            # Generate
            sample = generate_spider_single(intent, filler, template_pair)

            if sample:
                sample["id"] = sample["id"].replace("spider_train", f"spider_{'val' if split_name=='validation' else split_name}")
                splits[split_name].append(sample)
                generated += 1
                stats[intent] += 1

            if generated % 1000 == 0 and generated > 0:
                logger.info(f"  Generated {generated}/{target_size} (attempts: {attempts})")

        logger.info(f"  Complete: {generated} samples")
        logger.info(f"  Intent distribution: {dict(stats)}")
        logger.info(f"  Success rate: {generated/attempts*100:.2f}%")

        if generated < target_size:
            logger.warning(f"  Only generated {generated}/{target_size} after {attempts} attempts!")

    return splits

def generate_all_datasets(
    base_dir: str = "data",
    wiki_train: int = 50000,
    wiki_val: int = 5000,
    wiki_test: int = 5000,
    spider_train: int = 50000,
    spider_val: int = 5000,
    spider_test: int = 5000,
    backup_existing: bool = True
) -> None:
    """
    Generate all dataset splits and save to disk.

    Args:
        base_dir: Base data directory
        wiki_*: Sizes for WikiSQL splits
        spider_*: Sizes for Spider splits
        backup_existing: If True, move old data to backup folder
    """
    base_path = Path(base_dir)

    if backup_existing:
        from datetime import datetime
        backup_dir = base_path / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        for subdir in ["wikisql", "spider", "processed"]:
            src = base_path / subdir
            if src.exists():
                dst = backup_dir / subdir
                import shutil
                shutil.copytree(src, dst)
                logger.info(f"Backed up {subdir} to {dst}")

    # Generate WikiSQL
    logger.info("="*60)
    logger.info("GENERATING WIKISQL DATASET")
    logger.info("="*60)
    wiki_splits = generate_wikisql_splits(wiki_train, wiki_val, wiki_test)

    for split_name, data in wiki_splits.items():
        output_file = base_path / "wikisql" / f"{split_name}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved WikiSQL {split_name}: {len(data)} samples to {output_file}")

    # Generate Spider
    logger.info("="*60)
    logger.info("GENERATING SPIDER DATASET")
    logger.info("="*60)
    spider_splits = generate_spider_splits(spider_train, spider_val, spider_test)

    for split_name, data in spider_splits.items():
        output_file = base_path / "spider" / f"{split_name}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved Spider {split_name}: {len(data)} samples to {output_file}")

    # Optionally create processed merge (both datasets combined)
    logger.info("="*60)
    logger.info("CREATING PROCESSED MERGE")
    logger.info("="*60)
    all_train = wiki_splits["train"] + spider_splits["train"]
    all_val = wiki_splits["validation"] + spider_splits["validation"]
    all_test = wiki_splits["test"] + spider_splits["test"]

    for split_name, data in [("train", all_train), ("val", all_val), ("test", all_test)]:
        output_file = base_path / "processed" / f"{split_name}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved Processed {split_name}: {len(data)} samples to {output_file}")

    logger.info("="*60)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"WikiSQL: {len(wiki_splits['train'])} train, {len(wiki_splits['validation'])} val, {len(wiki_splits['test'])} test")
    logger.info(f"Spider: {len(spider_splits['train'])} train, {len(spider_splits['validation'])} val, {len(spider_splits['test'])} test")
    logger.info(f"Processed (merged): {len(all_train)} train, {len(all_val)} val, {len(all_test)} test")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate synthetic NL2SQL datasets")
    parser.add_argument("--train-size", type=int, default=50000, help="Training samples per dataset")
    parser.add_argument("--val-size", type=int, default=5000, help="Validation samples per dataset")
    parser.add_argument("--test-size", type=int, default=5000, help="Test samples per dataset")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup of existing data")

    args = parser.parse_args()

    try:
        generate_all_datasets(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            backup_existing=not args.no_backup
        )
        logger.info("SUCCESS: All datasets generated")
        sys.exit(0)
    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)