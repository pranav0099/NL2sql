"""
Expand Synthetic NL2SQL Training Data
Generates 256 diverse training samples across 8 intents (32 each)
and merges with existing WikiSQL data.

Author: Pranav (Claude-generated)
Date: 2026-04-05
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger
from config.config import (
    WIKISQL_DATA, INTENTS,
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
)

logger = get_logger(__name__)

# ============================================================================
# SCHEMA DEFINITION
# ============================================================================

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
            "product_id", "product_name", "category", "price",
            "stock_qty", "rating"
        ],
        "types": ["INTEGER", "TEXT", "TEXT", "REAL",
                  "INTEGER", "REAL"]
    },
    "orders": {
        "table": "orders",
        "columns": [
            "order_id", "customer_id", "order_date", "status",
            "total_amount", "city"
        ],
        "types": ["INTEGER", "INTEGER", "DATE", "TEXT",
                  "REAL", "TEXT"]
    },
    "order_items": {
        "table": "order_items",
        "columns": [
            "item_id", "order_id", "product_id", "quantity",
            "unit_price", "subtotal"
        ],
        "types": ["INTEGER", "INTEGER", "INTEGER", "INTEGER",
                  "REAL", "REAL"]
    },
    "employees": {
        "table": "employees",
        "columns": [
            "emp_id", "first_name", "last_name", "email", "dept_id",
            "salary", "hire_date", "city"
        ],
        "types": ["INTEGER", "TEXT", "TEXT", "TEXT", "INTEGER",
                  "REAL", "DATE", "TEXT"]
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
            "sale_id", "customer_id", "month", "year",
            "sales_amount", "city", "category"
        ],
        "types": ["INTEGER", "INTEGER", "INTEGER", "INTEGER",
                  "REAL", "TEXT", "TEXT"]
    }
}

# Sample values for generating realistic queries
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
          "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat"]
CATEGORIES = ["Electronics", "Clothing", "Home & Kitchen", "Books",
              "Sports & Outdoors", "Beauty & Personal Care"]
ORDER_STATUSES = ["pending", "processing", "shipped", "delivered", "cancelled"]
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "HR", "Finance",
               "Operations", "Support", "R&D"]

# ============================================================================
# TOKENIZATION HELPERS (matching build_vocab.py)
# ============================================================================

def tokenize_nl(text: str) -> List[str]:
    """Tokenize natural language question."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s@$%]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t.strip()]

def tokenize_sql(sql: str) -> List[str]:
    """Tokenize SQL query preserving keywords and operators."""
    import re
    sql = ' '.join(sql.split())

    tokens = []
    i = 0
    n = len(sql)

    while i < n:
        if sql[i].isspace():
            i += 1
            continue

        if sql[i] == "'":
            j = i + 1
            while j < n and sql[j] != "'":
                j += 1
            if j < n:
                tokens.append(sql[i:j+1])
                i = j + 1
            else:
                tokens.append(sql[i:])
                break
            continue

        if i + 1 < n and sql[i:i+2] in ['>=', '<=', '!=', '<>']:
            tokens.append(sql[i:i+2])
            i += 2
            continue

        if sql[i] in ['=', '>', '<']:
            tokens.append(sql[i])
            i += 1
            continue

        if sql[i] in ['(', ')', ',']:
            tokens.append(sql[i])
            i += 1
            continue

        j = i
        while j < n and not sql[j].isspace() and sql[j] not in ['(', ')', ',', '=', '>', '<', '!', "'"]:
            j += 1

        token = sql[i:j]
        if token:
            tokens.append(token)
            i = j
        else:
            tokens.append(sql[i])
            i += 1

    # Normalize
    SQL_KEYWORDS = {
        "SELECT", "FROM", "WHERE", "JOIN", "INNER JOIN", "LEFT JOIN",
        "RIGHT JOIN", "OUTER JOIN", "ON", "AND", "OR", "NOT",
        "COUNT", "SUM", "AVG", "MAX", "MIN", "GROUP BY", "HAVING",
        "ORDER BY", "ASC", "DESC", "LIMIT", "OFFSET", "DISTINCT",
        "ALL", "=", ">", "<", ">=", "<=", "!=", "<>", "AS", "IN",
        "LIKE", "IS", "NULL", "TRUE", "FALSE", "CASE", "WHEN",
        "THEN", "ELSE", "END", "BETWEEN"
    }

    normalized = []
    for token in tokens:
        token_upper = token.upper()
        if token_upper in SQL_KEYWORDS:
            normalized.append(token_upper)
        elif token.startswith("'") and token.endswith("'"):
            normalized.append(token)
        else:
            normalized.append(token.lower())

    return normalized

# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_select_samples() -> List[Dict[str, Any]]:
    """Generate 32 SELECT (simple) samples."""
    samples = []
    templates = [
        ("Show all customers", "SELECT * FROM customers", "customers"),
        ("List all products", "SELECT * FROM products", "products"),
        ("Display all employees", "SELECT * FROM employees", "employees"),
        ("Get all orders", "SELECT * FROM orders", "orders"),
        ("Show all departments", "SELECT * FROM departments", "departments"),
        ("List every sale", "SELECT * FROM sales", "sales"),
        ("Fetch all order items", "SELECT * FROM order_items", "order_items"),
        ("Show all customers", "SELECT * FROM customers", "customers"),
        ("List all products", "SELECT * FROM products", "products"),
        ("Display every employee", "SELECT * FROM employees", "employees"),
        ("Get all orders", "SELECT * FROM orders", "orders"),
        ("Show all departments", "SELECT * FROM departments", "departments"),
        ("List all sales records", "SELECT * FROM sales", "sales"),
        ("Fetch every order item", "SELECT * FROM order_items", "order_items"),
        ("Show me all customers", "SELECT * FROM customers", "customers"),
        ("List every product", "SELECT * FROM products", "products"),
        ("Display all employees", "SELECT * FROM employees", "employees"),
        ("What customers exist?", "SELECT * FROM customers", "customers"),
        ("Which products are there?", "SELECT * FROM products", "products"),
        ("Show all orders", "SELECT * FROM orders", "orders"),
        ("List all departments", "SELECT * FROM departments", "departments"),
        ("Get all sales", "SELECT * FROM sales", "sales"),
        ("Show order items", "SELECT * FROM order_items", "order_items"),
        ("All customers please", "SELECT * FROM customers", "customers"),
        ("Every product", "SELECT * FROM products", "products"),
        ("All employees", "SELECT * FROM employees", "employees"),
        ("Entire orders table", "SELECT * FROM orders", "orders"),
        ("Complete departments", "SELECT * FROM departments", "departments"),
        ("Full sales data", "SELECT * FROM sales", "sales"),
        ("All order items", "SELECT * FROM order_items", "order_items"),
        ("Show me everything from customers", "SELECT * FROM customers", "customers"),
        ("List all table entries from products", "SELECT * FROM products", "products"),
    ]

    for i, (question, query, table) in enumerate(templates):
        schema = SCHEMAS[table]
        samples.append({
            "id": f"expanded_{i+1:06d}",
            "question": question,
            "query": query,
            "table_id": table,
            "schema": schema.copy(),
            "intent": "SELECT",
            "source": "synthetic_expanded",
            "raw_question": question,
            "nl_tokens": tokenize_nl(question),
            "sql_tokens": tokenize_sql(query)
        })

    return samples

def generate_select_where_samples() -> List[Dict[str, Any]]:
    """Generate 32 SELECT_WHERE samples."""
    samples = []
    counter = 0

    # Customers by city
    for city in CITIES[:8]:
        samples.append({
            "id": f"expanded_{32+counter+1:06d}",
            "question": f"Show customers from {city}",
            "query": f"SELECT * FROM customers WHERE city = '{city}'",
            "table_id": "customers",
            "schema": SCHEMAS["customers"].copy(),
            "intent": "SELECT_WHERE",
            "source": "synthetic_expanded",
            "raw_question": f"Show customers from {city}",
            "nl_tokens": tokenize_nl(f"Show customers from {city}"),
            "sql_tokens": tokenize_sql(f"SELECT * FROM customers WHERE city = '{city}'")
        })
        counter += 1

    # Orders by status
    for status in ORDER_STATUSES[:4]:
        samples.append({
            "id": f"expanded_{32+counter+1:06d}",
            "question": f"Find orders with status {status}",
            "query": f"SELECT * FROM orders WHERE status = '{status}'",
            "table_id": "orders",
            "schema": SCHEMAS["orders"].copy(),
            "intent": "SELECT_WHERE",
            "source": "synthetic_expanded",
            "raw_question": f"Find orders with status {status}",
            "nl_tokens": tokenize_nl(f"Find orders with status {status}"),
            "sql_tokens": tokenize_sql(f"SELECT * FROM orders WHERE status = '{status}'")
        })
        counter += 1

    # Products by category
    for cat in CATEGORIES[:4]:
        samples.append({
            "id": f"expanded_{32+counter+1:06d}",
            "question": f"List products in {cat} category",
            "query": f"SELECT * FROM products WHERE category = '{cat}'",
            "table_id": "products",
            "schema": SCHEMAS["products"].copy(),
            "intent": "SELECT_WHERE",
            "source": "synthetic_expanded",
            "raw_question": f"List products in {cat} category",
            "nl_tokens": tokenize_nl(f"List products in {cat} category"),
            "sql_tokens": tokenize_sql(f"SELECT * FROM products WHERE category = '{cat}'")
        })
        counter += 1

    # Employees by city
    for city in CITIES[8:12]:
        samples.append({
            "id": f"expanded_{32+counter+1:06d}",
            "question": f"Show employees in {city}",
            "query": f"SELECT * FROM employees WHERE city = '{city}'",
            "table_id": "employees",
            "schema": SCHEMAS["employees"].copy(),
            "intent": "SELECT_WHERE",
            "source": "synthetic_expanded",
            "raw_question": f"Show employees in {city}",
            "nl_tokens": tokenize_nl(f"Show employees in {city}"),
            "sql_tokens": tokenize_sql(f"SELECT * FROM employees WHERE city = '{city}'")
        })
        counter += 1

    # Salary filter
    samples.append({
        "id": f"expanded_{32+counter+1:06d}",
        "question": "Find employees with salary above 50000",
        "query": "SELECT * FROM employees WHERE salary > 50000",
        "table_id": "employees",
        "schema": SCHEMAS["employees"].copy(),
        "intent": "SELECT_WHERE",
        "source": "synthetic_expanded",
        "raw_question": "Find employees with salary above 50000",
        "nl_tokens": tokenize_nl("Find employees with salary above 50000"),
        "sql_tokens": tokenize_sql("SELECT * FROM employees WHERE salary > 50000")
    })
    counter += 1

    # Price filters
    samples.append({
        "id": f"expanded_{32+counter+1:06d}",
        "question": "List products with price greater than 1000",
        "query": "SELECT * FROM products WHERE price > 1000",
        "table_id": "products",
        "schema": SCHEMAS["products"].copy(),
        "intent": "SELECT_WHERE",
        "source": "synthetic_expanded",
        "raw_question": "List products with price greater than 1000",
        "nl_tokens": tokenize_nl("List products with price greater than 1000"),
        "sql_tokens": tokenize_sql("SELECT * FROM products WHERE price > 1000")
    })
    counter += 1

    # Rating filter
    samples.append({
        "id": f"expanded_{32+counter+1:06d}",
        "question": "Show products with rating above 4",
        "query": "SELECT * FROM products WHERE rating > 4.0",
        "table_id": "products",
        "schema": SCHEMAS["products"].copy(),
        "intent": "SELECT_WHERE",
        "source": "synthetic_expanded",
        "raw_question": "Show products with rating above 4",
        "nl_tokens": tokenize_nl("Show products with rating above 4"),
        "sql_tokens": tokenize_sql("SELECT * FROM products WHERE rating > 4.0")
    })
    counter += 1

    # Order amount
    samples.append({
        "id": f"expanded_{32+counter+1:06d}",
        "question": "Find orders with total amount greater than 10000",
        "query": "SELECT * FROM orders WHERE total_amount > 10000",
        "table_id": "orders",
        "schema": SCHEMAS["orders"].copy(),
        "intent": "SELECT_WHERE",
        "source": "synthetic_expanded",
        "raw_question": "Find orders with total amount greater than 10000",
        "nl_tokens": tokenize_nl("Find orders with total amount greater than 10000"),
        "sql_tokens": tokenize_sql("SELECT * FROM orders WHERE total_amount > 10000")
    })
    counter += 1

    # Date filter
    samples.append({
        "id": f"expanded_{32+counter+1:06d}",
        "question": "Show orders from 2024",
        "query": "SELECT * FROM orders WHERE order_date LIKE '2024%'",
        "table_id": "orders",
        "schema": SCHEMAS["orders"].copy(),
        "intent": "SELECT_WHERE",
        "source": "synthetic_expanded",
        "raw_question": "Show orders from 2024",
        "nl_tokens": tokenize_nl("Show orders from 2024"),
        "sql_tokens": tokenize_sql("SELECT * FROM orders WHERE order_date LIKE '2024%'")
    })
    counter += 1

    # Budget filter
    samples.append({
        "id": f"expanded_{32+counter+1:06d}",
        "question": "Find departments with budget above 2000000",
        "query": "SELECT * FROM departments WHERE budget > 2000000",
        "table_id": "departments",
        "schema": SCHEMAS["departments"].copy(),
        "intent": "SELECT_WHERE",
        "source": "synthetic_expanded",
        "raw_question": "Find departments with budget above 2000000",
        "nl_tokens": tokenize_nl("Find departments with budget above 2000000"),
        "sql_tokens": tokenize_sql("SELECT * FROM departments WHERE budget > 2000000")
    })
    counter += 1

    # More city variations
    for city in CITIES[12:16]:
        samples.append({
            "id": f"expanded_{32+counter+1:06d}",
            "question": f"Customers from {city}",
            "query": f"SELECT * FROM customers WHERE city = '{city}'",
            "table_id": "customers",
            "schema": SCHEMAS["customers"].copy(),
            "intent": "SELECT_WHERE",
            "source": "synthetic_expanded",
            "raw_question": f"Customers from {city}",
            "nl_tokens": tokenize_nl(f"Customers from {city}"),
            "sql_tokens": tokenize_sql(f"SELECT * FROM customers WHERE city = '{city}'")
        })
        counter += 1

    # Stock quantity filter
    samples.append({
        "id": f"expanded_{32+counter+1:06d}",
        "question": "Products with stock less than 100",
        "query": "SELECT * FROM products WHERE stock_qty < 100",
        "table_id": "products",
        "schema": SCHEMAS["products"].copy(),
        "intent": "SELECT_WHERE",
        "source": "synthetic_expanded",
        "raw_question": "Products with stock less than 100",
        "nl_tokens": tokenize_nl("Products with stock less than 100"),
        "sql_tokens": tokenize_sql("SELECT * FROM products WHERE stock_qty < 100")
    })
    counter += 1

    # Fill remaining to reach 32
    while len(samples) < 32:
        idx = len(samples) % 8
        if idx == 0:
            city = random.choice(CITIES)
            table = "customers"
            query = f"SELECT * FROM customers WHERE city = '{city}'"
            question = f"Show customers in {city}"
        elif idx == 1:
            cat = random.choice(CATEGORIES)
            table = "products"
            query = f"SELECT * FROM products WHERE category = '{cat}'"
            question = f"List {cat} products"
        elif idx == 2:
            status = random.choice(ORDER_STATUSES)
            table = "orders"
            query = f"SELECT * FROM orders WHERE status = '{status}'"
            question = f"Orders with status {status}"
        elif idx == 3:
            city = random.choice(CITIES)
            table = "employees"
            query = f"SELECT * FROM employees WHERE city = '{city}'"
            question = f"Employees based in {city}"
        elif idx == 4:
            table = "departments"
            query = "SELECT * FROM departments WHERE budget > 1000000"
            question = "Departments with large budget"
        elif idx == 5:
            table = "sales"
            city = random.choice(CITIES)
            query = f"SELECT * FROM sales WHERE city = '{city}'"
            question = f"Sales in {city}"
        elif idx == 6:
            table = "order_items"
            query = "SELECT * FROM order_items WHERE quantity > 5"
            question = "Order items with quantity over 5"
        else:
            table = "customers"
            query = "SELECT * FROM customers WHERE total_orders > 10"
            question = "Customers with many orders"

        sample = {
            "id": f"expanded_{32+counter+1:06d}",
            "question": question,
            "query": query,
            "table_id": table,
            "schema": SCHEMAS[table].copy(),
            "intent": "SELECT_WHERE",
            "source": "synthetic_expanded",
            "raw_question": question,
            "nl_tokens": tokenize_nl(question),
            "sql_tokens": tokenize_sql(query)
        }
        samples.append(sample)
        counter += 1

    return samples[:32]

def generate_select_aggregate_samples() -> List[Dict[str, Any]]:
    """Generate 32 SELECT_AGGREGATE samples."""
    samples = []
    counter = 0

    aggregates = [
        ("Count total customers", "SELECT COUNT(*) FROM customers", "customers"),
        ("Count all products", "SELECT COUNT(*) FROM products", "products"),
        ("Total number of orders", "SELECT COUNT(*) FROM orders", "orders"),
        ("How many employees?", "SELECT COUNT(*) FROM employees", "employees"),
        ("Number of departments", "SELECT COUNT(*) FROM departments", "departments"),
        ("Count all sales", "SELECT COUNT(*) FROM sales", "sales"),
        ("Count order items", "SELECT COUNT(*) FROM order_items", "order_items"),
        ("Average salary", "SELECT AVG(salary) FROM employees", "employees"),
        ("Mean product price", "SELECT AVG(price) FROM products", "products"),
        ("Average order total", "SELECT AVG(total_amount) FROM orders", "orders"),
        ("Mean rating", "SELECT AVG(rating) FROM products", "products"),
        ("Average budget", "SELECT AVG(budget) FROM departments", "departments"),
        ("Avg sales amount", "SELECT AVG(sales_amount) FROM sales", "sales"),
        ("Total spending", "SELECT SUM(total_spent) FROM customers", "customers"),
        ("Sum of all sales", "SELECT SUM(sales_amount) FROM sales", "sales"),
        ("Total revenue", "SELECT SUM(total_amount) FROM orders", "orders"),
        ("Sum of product prices", "SELECT SUM(price) FROM products", "products"),
        ("Total salary expense", "SELECT SUM(salary) FROM employees", "employees"),
        ("Total budget", "SELECT SUM(budget) FROM departments", "departments"),
        ("Maximum salary", "SELECT MAX(salary) FROM employees", "employees"),
        ("Highest price", "SELECT MAX(price) FROM products", "products"),
        ("Top order value", "SELECT MAX(total_amount) FROM orders", "orders"),
        ("Max rating", "SELECT MAX(rating) FROM products", "products"),
        ("Highest budget", "SELECT MAX(budget) FROM departments", "departments"),
        ("Largest sale", "SELECT MAX(sales_amount) FROM sales", "sales"),
        ("Minimum price", "SELECT MIN(price) FROM products", "products"),
        ("Lowest salary", "SELECT MIN(salary) FROM employees", "employees"),
        ("Smallest order", "SELECT MIN(total_amount) FROM orders", "orders"),
        ("Min rating", "SELECT MIN(rating) FROM products", "products"),
        ("Least budget", "SELECT MIN(budget) FROM departments", "departments"),
        ("Total stock quantity", "SELECT SUM(stock_qty) FROM products", "products"),
        ("Sum of all quantities in orders", "SELECT SUM(quantity) FROM order_items", "order_items"),
    ]

    for i, (question, query, table) in enumerate(aggregates):
        samples.append({
            "id": f"expanded_{64+i+1:06d}",
            "question": question,
            "query": query,
            "table_id": table,
            "schema": SCHEMAS[table].copy(),
            "intent": "SELECT_AGGREGATE",
            "source": "synthetic_expanded",
            "raw_question": question,
            "nl_tokens": tokenize_nl(question),
            "sql_tokens": tokenize_sql(query)
        })

    return samples

def generate_select_order_samples() -> List[Dict[str, Any]]:
    """Generate 32 SELECT_ORDER samples."""
    samples = []
    counter = 0

    order_templates = [
        # Top N by various columns
        ("Top 10 customers by total spent", "SELECT * FROM customers ORDER BY total_spent DESC LIMIT 10", "customers"),
        ("Top 5 products by price", "SELECT * FROM products ORDER BY price DESC LIMIT 5", "products"),
        ("Recent orders", "SELECT * FROM orders ORDER BY order_date DESC LIMIT 10", "orders"),
        ("Highest paid employees", "SELECT * FROM employees ORDER BY salary DESC LIMIT 10", "employees"),
        ("Best rated products", "SELECT * FROM products ORDER BY rating DESC LIMIT 10", "products"),
        ("Most recent orders", "SELECT * FROM orders ORDER BY order_date DESC LIMIT 20", "orders"),
        ("Top spending customers", "SELECT * FROM customers ORDER BY total_spent DESC LIMIT 15", "customers"),
        ("Cheapest products", "SELECT * FROM products ORDER BY price ASC LIMIT 10", "products"),
        ("Bottom orders by amount", "SELECT * FROM orders ORDER BY total_amount ASC LIMIT 10", "orders"),
        ("Lowest salary employees", "SELECT * FROM employees ORDER BY salary ASC LIMIT 10", "employees"),
        # Alphabetical
        ("Customers sorted by name", "SELECT * FROM customers ORDER BY first_name ASC", "customers"),
        ("Products sorted alphabetically", "SELECT * FROM products ORDER BY product_name ASC", "products"),
        ("Departments by location", "SELECT * FROM departments ORDER BY location ASC", "departments"),
        # By various metrics
        ("Orders sorted by date", "SELECT * FROM orders ORDER BY order_date DESC", "orders"),
        ("Products by stock quantity", "SELECT * FROM products ORDER BY stock_qty DESC", "products"),
        ("Employees by hire date", "SELECT * FROM employees ORDER BY hire_date DESC", "employees"),
        ("Sales by amount", "SELECT * FROM sales ORDER BY sales_amount DESC", "sales"),
        ("Orders by city", "SELECT * FROM orders ORDER BY city ASC", "orders"),
        ("Customers by join date", "SELECT * FROM customers ORDER BY join_date DESC", "customers"),
        ("Products by rating", "SELECT * FROM products ORDER BY rating DESC", "products"),
        ("Departments by budget", "SELECT * FROM departments ORDER BY budget DESC", "departments"),
        ("Employees by city", "SELECT * FROM employees ORDER BY city ASC", "employees"),
        ("Sales by year month", "SELECT * FROM sales ORDER BY year DESC, month DESC", "sales"),
        ("Order items by subtotal", "SELECT * FROM order_items ORDER BY subtotal DESC LIMIT 20", "order_items"),
        # First N
        ("First 5 customers", "SELECT * FROM customers LIMIT 5", "customers"),
        ("First 10 products", "SELECT * FROM products LIMIT 10", "products"),
        ("First 20 orders", "SELECT * FROM orders LIMIT 20", "orders"),
        ("First 8 employees", "SELECT * FROM employees LIMIT 8", "employees"),
        # Last N via ORDER BY DESC LIMIT
        ("Last 10 orders", "SELECT * FROM orders ORDER BY order_id DESC LIMIT 10", "orders"),
        ("Last 5 customers", "SELECT * FROM customers ORDER BY customer_id DESC LIMIT 5", "customers"),
        ("Most recent sales", "SELECT * FROM sales ORDER BY year DESC, month DESC LIMIT 15", "sales"),
        ("Top 50 order items", "SELECT * FROM order_items ORDER BY item_id DESC LIMIT 50", "order_items"),
    ]

    for i, (question, query, table) in enumerate(order_templates):
        samples.append({
            "id": f"expanded_{96+i+1:06d}",
            "question": question,
            "query": query,
            "table_id": table,
            "schema": SCHEMAS[table].copy(),
            "intent": "SELECT_ORDER",
            "source": "synthetic_expanded",
            "raw_question": question,
            "nl_tokens": tokenize_nl(question),
            "sql_tokens": tokenize_sql(query)
        })

    return samples

def generate_select_join_samples() -> List[Dict[str, Any]]:
    """Generate 32 SELECT_JOIN samples."""
    samples = []
    counter = 0

    join_templates = [
        # Customers with orders
        ("Customers with their orders", "SELECT c.first_name, c.last_name, o.order_id, o.total_amount FROM customers c JOIN orders o ON c.customer_id = o.customer_id", "customers"),
        ("Customer orders", "SELECT c.first_name, c.last_name, o.order_date FROM customers c JOIN orders o ON c.customer_id = o.customer_id", "customers"),
        ("Orders with customer names", "SELECT o.order_id, o.order_date, c.first_name, c.last_name FROM orders o JOIN customers c ON o.customer_id = c.customer_id", "orders"),
        ("Customer order details", "SELECT c.city, o.order_id, o.total_amount FROM customers c JOIN orders o ON c.customer_id = o.customer_id", "customers"),
        # Orders with order items and products
        ("Orders with product details", "SELECT o.order_id, p.product_name, oi.quantity FROM orders o JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id", "orders"),
        ("Order items with products", "SELECT oi.order_id, p.product_name, p.category, oi.quantity FROM order_items oi JOIN products p ON oi.product_id = p.product_id", "order_items"),
        ("Products with orders", "SELECT p.product_name, oi.order_id, oi.quantity FROM products p JOIN order_items oi ON p.product_id = oi.product_id", "products"),
        ("Order quantities", "SELECT o.order_id, p.product_name, oi.quantity FROM orders o JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id", "orders"),
        # Employees with departments
        ("Employees with department name", "SELECT e.first_name, e.last_name, d.dept_name FROM employees e JOIN departments d ON e.dept_id = d.dept_id", "employees"),
        ("Employees with department location", "SELECT e.first_name, e.last_name, d.location, d.budget FROM employees e JOIN departments d ON e.dept_id = d.dept_id", "employees"),
        ("Department employee count", "SELECT d.dept_name, COUNT(e.emp_id) as emp_count FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id", "departments"),
        ("Employees and their budgets", "SELECT e.first_name, e.last_name, d.dept_name, d.budget FROM employees e JOIN departments d ON e.dept_id = d.dept_id", "employees"),
        # Customers with sales
        ("Customer sales data", "SELECT c.first_name, c.last_name, s.sales_amount, s.month, s.year FROM customers c JOIN sales s ON c.customer_id = s.customer_id", "customers"),
        ("Sales with customer names", "SELECT s.sale_id, c.first_name, c.last_name, s.sales_amount, s.city FROM sales s JOIN customers c ON s.customer_id = c.customer_id", "sales"),
        ("Customer sales by city", "SELECT c.city, s.sales_amount FROM customers c JOIN sales s ON c.customer_id = s.customer_id", "customers"),
        ("Customer and sales details", "SELECT c.first_name, c.city, s.sales_amount, s.category FROM customers c JOIN sales s ON c.customer_id = s.customer_id", "customers"),
        # Products with categories from sales
        ("Sales with product categories", "SELECT s.sales_amount, s.category, s.city FROM sales s", "sales"),
        # Multi-table complex joins
        ("Full customer order details", "SELECT c.first_name, c.last_name, o.order_id, o.total_amount, p.product_name, oi.quantity FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id", "customers"),
        ("Complete order information", "SELECT o.order_id, c.first_name, c.last_name, p.product_name, oi.quantity, p.price FROM orders o JOIN customers c ON o.customer_id = c.customer_id JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id", "orders"),
        ("Employee department details", "SELECT e.first_name, e.last_name, e.salary, d.dept_name, d.location, d.budget FROM employees e JOIN departments d ON e.dept_id = d.dept_id", "employees"),
        # City-based joins
        ("Customers and orders from same city", "SELECT c.first_name, c.city, o.order_id FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE c.city = o.city", "customers"),
        ("Employees in department cities", "SELECT e.first_name, e.city, d.dept_name, d.location FROM employees e JOIN departments d ON e.dept_id = d.dept_id", "employees"),
        # More varied joins
        ("Products ordered", "SELECT DISTINCT p.product_name, p.category FROM products p JOIN order_items oi ON p.product_id = oi.product_id", "products"),
        ("Customers with orders total", "SELECT c.first_name, c.last_name, COUNT(o.order_id) as order_count FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id", "customers"),
        ("Order summary", "SELECT o.order_id, COUNT(oi.product_id) as item_count, SUM(oi.quantity) as total_qty FROM orders o JOIN order_items oi ON o.order_id = oi.order_id GROUP BY o.order_id", "orders"),
        ("Product sales revenue", "SELECT p.product_name, SUM(oi.subtotal) as revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.product_id", "products"),
        ("Department salary totals", "SELECT d.dept_name, SUM(e.salary) as total_salary, AVG(e.salary) as avg_salary FROM departments d JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id", "departments"),
        ("Customer spending by city", "SELECT c.city, COUNT(o.order_id) as order_count, SUM(o.total_amount) as total_spent FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.city", "customers"),
        ("Monthly sales by category", "SELECT s.category, s.month, s.year, SUM(s.sales_amount) as total FROM sales s GROUP BY s.category, s.month, s.year", "sales"),
        ("Employee performance", "SELECT e.first_name, e.last_name, d.dept_name, e.salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE e.salary > (SELECT AVG(salary) FROM employees WHERE dept_id = e.dept_id)", "employees"),
    ]

    for i, (question, query, table) in enumerate(join_templates):
        samples.append({
            "id": f"expanded_{128+i+1:06d}",
            "question": question,
            "query": query,
            "table_id": table,
            "schema": SCHEMAS[table].copy(),
            "intent": "SELECT_JOIN",
            "source": "synthetic_expanded",
            "raw_question": question,
            "nl_tokens": tokenize_nl(question),
            "sql_tokens": tokenize_sql(query)
        })

    return samples

def generate_select_group_samples() -> List[Dict[str, Any]]:
    """Generate 32 SELECT_GROUP samples."""
    samples = []
    counter = 0

    group_templates = [
        ("Count customers by city", "SELECT city, COUNT(*) FROM customers GROUP BY city", "customers"),
        ("Customers grouped by city", "SELECT city, COUNT(*) as count FROM customers GROUP BY city", "customers"),
        ("Number of customers per city", "SELECT city, COUNT(*) FROM customers GROUP BY city", "customers"),
        ("Customer count per city", "SELECT city, COUNT(*) as cnt FROM customers GROUP BY city", "customers"),
        ("Total orders by status", "SELECT status, COUNT(*) FROM orders GROUP BY status", "orders"),
        ("Orders grouped by status", "SELECT status, COUNT(*) as count FROM orders GROUP BY status", "orders"),
        ("Count orders by status", "SELECT status, COUNT(*) FROM orders GROUP BY status", "orders"),
        ("Order status distribution", "SELECT status, COUNT(*) as cnt FROM orders GROUP BY status", "orders"),
        ("Products by category", "SELECT category, COUNT(*) FROM products GROUP BY category", "products"),
        ("Category product count", "SELECT category, COUNT(*) as count FROM products GROUP BY category", "products"),
        ("Products per category", "SELECT category, COUNT(*) FROM products GROUP BY category", "products"),
        ("Product count by category", "SELECT category, COUNT(*) as cnt FROM products GROUP BY category", "products"),
        ("Sales by city", "SELECT city, SUM(sales_amount) FROM sales GROUP BY city", "sales"),
        ("Total sales per city", "SELECT city, SUM(sales_amount) as total FROM sales GROUP BY city", "sales"),
        ("Sales grouped by city", "SELECT city, SUM(sales_amount) FROM sales GROUP BY city", "sales"),
        ("Sales amount by city", "SELECT city, SUM(sales_amount) FROM sales GROUP BY city", "sales"),
        ("Average salary by department", "SELECT dept_id, AVG(salary) FROM employees GROUP BY dept_id", "employees"),
        ("Employees grouped by department", "SELECT dept_id, AVG(salary), COUNT(*) FROM employees GROUP BY dept_id", "employees"),
        ("Salary stats by department", "SELECT dept_id, AVG(salary) as avg_sal, MAX(salary) as max_sal FROM employees GROUP BY dept_id", "employees"),
        ("Department salary averages", "SELECT dept_id, AVG(salary) FROM employees GROUP BY dept_id", "employees"),
        ("Order count by month", "SELECT month, COUNT(*) FROM orders GROUP BY month", "orders"),
        ("Orders per month", "SELECT month, COUNT(*) as cnt FROM orders GROUP BY month", "orders"),
        ("Monthly order distribution", "SELECT month, COUNT(*) FROM orders GROUP BY month", "orders"),
        ("Sales by month", "SELECT month, SUM(sales_amount) FROM sales GROUP BY month", "sales"),
        ("Monthly sales total", "SELECT month, SUM(sales_amount) as total FROM sales GROUP BY month", "sales"),
        ("Revenue by month", "SELECT month, SUM(sales_amount) FROM sales GROUP BY month", "sales"),
        ("Sales by category", "SELECT category, SUM(sales_amount) FROM sales GROUP BY category", "sales"),
        ("Category sales total", "SELECT category, SUM(sales_amount) as revenue FROM sales GROUP BY category", "sales"),
        ("Orders by city", "SELECT city, COUNT(*) FROM orders GROUP BY city", "orders"),
        ("Order distribution by city", "SELECT city, COUNT(*) as cnt FROM orders GROUP BY city", "orders"),
        ("Order count per city", "SELECT city, COUNT(*) FROM orders GROUP BY city", "orders"),
        ("Product sales summary", "SELECT p.category, COUNT(p.product_id), AVG(p.price) FROM products p GROUP BY p.category", "products"),
    ]

    for i, (question, query, table) in enumerate(group_templates):
        samples.append({
            "id": f"expanded_{160+i+1:06d}",
            "question": question,
            "query": query,
            "table_id": table,
            "schema": SCHEMAS[table].copy(),
            "intent": "SELECT_GROUP",
            "source": "synthetic_expanded",
            "raw_question": question,
            "nl_tokens": tokenize_nl(question),
            "sql_tokens": tokenize_sql(query)
        })

    return samples

def generate_select_limit_samples() -> List[Dict[str, Any]]:
    """Generate 32 SELECT_LIMIT samples."""
    samples = []
    counter = 0

    limit_templates = [
        ("Show first 5 customers", "SELECT * FROM customers LIMIT 5", "customers"),
        ("First 10 products", "SELECT * FROM products LIMIT 10", "products"),
        ("Show 5 orders", "SELECT * FROM orders LIMIT 5", "orders"),
        ("Get 10 employees", "SELECT * FROM employees LIMIT 10", "employees"),
        ("Show 5 departments", "SELECT * FROM departments LIMIT 5", "departments"),
        ("First 20 sales", "SELECT * FROM sales LIMIT 20", "sales"),
        ("Show 10 order items", "SELECT * FROM order_items LIMIT 10", "order_items"),
        ("First 3 customers", "SELECT * FROM customers LIMIT 3", "customers"),
        ("Show 7 products", "SELECT * FROM products LIMIT 7", "products"),
        ("Get 15 orders", "SELECT * FROM orders LIMIT 15", "orders"),
        ("First 8 employees", "SELECT * FROM employees LIMIT 8", "employees"),
        ("Show 4 departments", "SELECT * FROM departments LIMIT 4", "departments"),
        ("First 25 sales", "SELECT * FROM sales LIMIT 25", "sales"),
        ("Show 12 order items", "SELECT * FROM order_items LIMIT 12", "order_items"),
        # Top N via ORDER BY DESC LIMIT
        ("Top 5 customers by spending", "SELECT * FROM customers ORDER BY total_spent DESC LIMIT 5", "customers"),
        ("Top 3 products by price", "SELECT * FROM products ORDER BY price DESC LIMIT 3", "products"),
        ("Top 10 orders by amount", "SELECT * FROM orders ORDER BY total_amount DESC LIMIT 10", "orders"),
        ("Top 5 employees by salary", "SELECT * FROM employees ORDER BY salary DESC LIMIT 5", "employees"),
        ("Top 3 products by rating", "SELECT * FROM products ORDER BY rating DESC LIMIT 3", "products"),
        ("Top 10 sales by amount", "SELECT * FROM sales ORDER BY sales_amount DESC LIMIT 10", "sales"),
        # Random sample indication
        ("Any 5 customers", "SELECT * FROM customers LIMIT 5", "customers"),
        ("Any 10 products", "SELECT * FROM products LIMIT 10", "products"),
        ("Any 8 orders", "SELECT * FROM orders LIMIT 8", "orders"),
        ("Random sample of 15 employees", "SELECT * FROM employees LIMIT 15", "employees"),
        ("Sample of 20 orders", "SELECT * FROM orders LIMIT 20", "orders"),
        # Quick preview
        ("Preview 5 products", "SELECT * FROM products LIMIT 5", "products"),
        ("Quick look at 10 customers", "SELECT * FROM customers LIMIT 10", "customers"),
        ("First 50 order items", "SELECT * FROM order_items LIMIT 50", "order_items"),
        ("Top 100 orders", "SELECT * FROM orders ORDER BY order_date DESC LIMIT 100", "orders"),
        ("First 3 departments", "SELECT * FROM departments LIMIT 3", "departments"),
        # Limited by specific need
        ("Show 3 most expensive products", "SELECT * FROM products ORDER BY price DESC LIMIT 3", "products"),
        ("Get 5 highest paid employees", "SELECT * FROM employees ORDER BY salary DESC LIMIT 5", "employees"),
    ]

    for i, (question, query, table) in enumerate(limit_templates):
        samples.append({
            "id": f"expanded_{192+i+1:06d}",
            "question": question,
            "query": query,
            "table_id": table,
            "schema": SCHEMAS[table].copy(),
            "intent": "SELECT_LIMIT",
            "source": "synthetic_expanded",
            "raw_question": question,
            "nl_tokens": tokenize_nl(question),
            "sql_tokens": tokenize_sql(query)
        })

    return samples

def generate_complex_samples() -> List[Dict[str, Any]]:
    """Generate 32 COMPLEX (subquery/nested) samples."""
    samples = []
    counter = 0

    complex_templates = [
        # Subqueries in WHERE
        ("Customers who spent more than average", "SELECT * FROM customers WHERE total_spent > (SELECT AVG(total_spent) FROM customers)", "customers"),
        ("Products priced above average", "SELECT * FROM products WHERE price > (SELECT AVG(price) FROM products)", "products"),
        ("Employees earning above department average", "SELECT * FROM employees e WHERE salary > (SELECT AVG(salary) FROM employees WHERE dept_id = e.dept_id)", "employees"),
        ("Orders with amount above average", "SELECT * FROM orders WHERE total_amount > (SELECT AVG(total_amount) FROM orders)", "orders"),
        ("Products with rating above average", "SELECT * FROM products WHERE rating > (SELECT AVG(rating) FROM products)", "products"),
        ("Departments with budget above average", "SELECT * FROM departments WHERE budget > (SELECT AVG(budget) FROM departments)", "departments"),
        # NOT IN subqueries
        ("Customers with no orders", "SELECT * FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders)", "customers"),
        ("Products never ordered", "SELECT * FROM products WHERE product_id NOT IN (SELECT product_id FROM order_items)", "products"),
        ("Employees in departments with no employees", "SELECT * FROM departments WHERE dept_id NOT IN (SELECT dept_id FROM employees)", "departments"),
        ("Cities with no sales", "SELECT DISTINCT city FROM customers WHERE city NOT IN (SELECT city FROM sales)", "customers"),
        # EXISTS subqueries
        ("Customers who placed orders", "SELECT * FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)", "customers"),
        ("Products that have been ordered", "SELECT * FROM products p WHERE EXISTS (SELECT 1 FROM order_items oi WHERE oi.product_id = p.product_id)", "products"),
        # Comparison with aggregates
        ("Customers with more than 10 orders", "SELECT * FROM customers WHERE total_orders > 10", "customers"),
        ("Employees with salary over 100000", "SELECT * FROM employees WHERE salary > 100000", "employees"),
        ("Products in Electronics category", "SELECT * FROM products WHERE category = 'Electronics'", "products"),
        # Multiple conditions
        ("Customers from Mumbai with high spending", "SELECT * FROM customers WHERE city = 'Mumbai' AND total_spent > 50000", "customers"),
        ("Products in Electronics with high rating", "SELECT * FROM products WHERE category = 'Electronics' AND rating > 4.0", "products"),
        ("Departments with budget over 1M in Mumbai", "SELECT * FROM departments WHERE budget > 1000000 AND location LIKE '%Mumbai%'", "departments"),
        # Combined GROUP BY HAVING
        ("Departments with more than 5 employees", "SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id HAVING COUNT(*) > 5", "employees"),
        ("Cities with over 20 customers", "SELECT city, COUNT(*) FROM customers GROUP BY city HAVING COUNT(*) > 20", "customers"),
        ("Product categories with average price over 10000", "SELECT category, AVG(price) FROM products GROUP BY category HAVING AVG(price) > 10000", "products"),
        ("Status with more than 50 orders", "SELECT status, COUNT(*) FROM orders GROUP BY status HAVING COUNT(*) > 50", "orders"),
        # Complex multi-clause
        ("Top 5 customers in Mumbai by spending", "SELECT * FROM customers WHERE city = 'Mumbai' ORDER BY total_spent DESC LIMIT 5", "customers"),
        ("Recent high-value orders", "SELECT * FROM orders WHERE total_amount > 5000 ORDER BY order_date DESC LIMIT 10", "orders"),
        ("Expensive products in Electronics", "SELECT * FROM products WHERE category = 'Electronics' AND price > 20000 ORDER BY price DESC", "products"),
        ("High-paid engineers", "SELECT * FROM employees WHERE dept_id = 1 AND salary > 80000 ORDER BY salary DESC", "employees"),
        # Nested aggregates
        ("Cities with above average customer spending", "SELECT city, AVG(total_spent) as avg_spent FROM customers GROUP BY city HAVING AVG(total_spent) > (SELECT AVG(total_spent) FROM customers)", "customers"),
        # Complex joins with conditions
        ("Customers who ordered Electronics", "SELECT DISTINCT c.first_name, c.last_name FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id WHERE p.category = 'Electronics'", "customers"),
        ("Employees in high-budget departments", "SELECT e.first_name, e.last_name, d.budget FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.budget > 2000000", "employees"),
        # Subquery in SELECT
        ("Customers with order count", "SELECT c.first_name, c.last_name, (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.customer_id) as order_count FROM customers c", "customers"),
        # Most popular product category
        ("Most popular product category", "SELECT category FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY category ORDER BY COUNT(*) DESC LIMIT 1", "products"),
        # Complex filtering with LIKE
        ("Customers with email from gmail", "SELECT * FROM customers WHERE email LIKE '%@gmail.com%'", "customers"),
        # Date range queries
        ("Orders from 2024", "SELECT * FROM orders WHERE order_date >= '2024-01-01' AND order_date <= '2024-12-31'", "orders"),
        # IN with subquery
        ("Products in categories with high sales", "SELECT * FROM products WHERE category IN (SELECT category FROM sales GROUP BY category HAVING SUM(sales_amount) > 50000)", "products"),
    ]

    for i, (question, query, table) in enumerate(complex_templates[:32]):
        samples.append({
            "id": f"expanded_{224+i+1:06d}",
            "question": question,
            "query": query,
            "table_id": table,
            "schema": SCHEMAS[table].copy(),
            "intent": "COMPLEX",
            "source": "synthetic_expanded",
            "raw_question": question,
            "nl_tokens": tokenize_nl(question),
            "sql_tokens": tokenize_sql(query)
        })

    return samples

def generate_all_synthetic_data() -> List[Dict[str, Any]]:
    """Generate all 256 synthetic samples."""
    all_samples = []

    all_samples.extend(generate_select_samples()[:32])
    all_samples.extend(generate_select_where_samples()[:32])
    all_samples.extend(generate_select_aggregate_samples()[:32])
    all_samples.extend(generate_select_order_samples()[:32])
    all_samples.extend(generate_select_join_samples()[:32])
    all_samples.extend(generate_select_group_samples()[:32])
    all_samples.extend(generate_select_limit_samples()[:32])
    all_samples.extend(generate_complex_samples()[:32])

    logger.info(f"Generated {len(all_samples)} synthetic samples")
    return all_samples

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to expand synthetic data and merge with existing."""
    logger.info("="*80)
    logger.info("EXPANDING SYNTHETIC DATA")
    logger.info("="*80)

    # Ensure directories exist
    WIKISQL_DATA.mkdir(parents=True, exist_ok=True)

    # Generate synthetic samples
    synthetic_data = generate_all_synthetic_data()

    logger.info(f"Generated {len(synthetic_data)} synthetic samples")
    logger.info(f"Intent distribution: {Counter(s['intent'] for s in synthetic_data).most_common()}")

    # Split 70/15/15 with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(synthetic_data)

    n = len(synthetic_data)
    train_split = int(0.70 * n)
    val_split = int(0.85 * n)

    train_synth = synthetic_data[:train_split]
    val_synth = synthetic_data[train_split:val_split]
    test_synth = synthetic_data[val_split:]

    logger.info(f"Split: train={len(train_synth)}, val={len(val_synth)}, test={len(test_synth)}")

    # Load existing data (if exists)
    train_path = WIKISQL_DATA / "train.json"
    val_path = WIKISQL_DATA / "validation.json"
    test_path = WIKISQL_DATA / "test.json"

    def load_or_create(path, default_empty=True):
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} existing records from {path.name}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}, creating fresh")
        if default_empty:
            return []
        return []

    train_existing = load_or_create(train_path)
    val_existing = load_or_create(val_path)
    test_existing = load_or_create(test_path)

    # Merge synthetic with existing
    train_merged = train_existing + train_synth
    val_merged = val_existing + val_synth
    test_merged = test_existing + test_synth

    # Save merged data
    def save_data(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} records to {path.name}")

    save_data(train_path, train_merged)
    save_data(val_path, val_merged)
    save_data(test_path, test_merged)

    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("FINAL DATASET STATISTICS")
    logger.info("="*60)
    logger.info(f"Train: {len(train_merged)} records")
    logger.info(f"Validation: {len(val_merged)} records")
    logger.info(f"Test: {len(test_merged)} records")
    logger.info(f"Total: {len(train_merged) + len(val_merged) + len(test_merged)} records")

    def print_intent_dist(data, split_name):
        if data:
            cnt = Counter(record.get("intent", "UNKNOWN") for record in data)
            logger.info(f"\n{split_name} Intent Distribution:")
            for intent, count in sorted(cnt.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(data)) * 100
                logger.info(f"  {intent:20s}: {count:5d} ({pct:5.1f}%)")

    print_intent_dist(train_merged, "Train")
    print_intent_dist(val_merged, "Validation")
    print_intent_dist(test_merged, "Test")

    logger.info("="*60)
    logger.info("Synthetic data expansion complete!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("  python data/build_vocab.py")
    logger.info("  python database/create_sample_db.py")
    logger.info("  python ml/trainer.py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
