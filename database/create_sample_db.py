"""
Sample Database Creation Module
Creates a realistic multi-table SQLite database for NL2SQL testing

Author: Pranav
Date: 2026-04-02

Database Schema:
  - customers (9 columns)
  - products (5 columns)
  - orders (6 columns)
  - order_items (6 columns)
  - employees (6 columns)
  - departments (3 columns)
  - sales (5 columns)
"""

import sqlite3
import random
import faker
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add project root to sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATABASE, SAMPLE_DB, SCHEMA_FILE
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# FAKER SETUP
# ============================================================================

fake = faker.Faker('en_IN')  # Indian locale for realistic names

# ============================================================================
# INDIAN CITIES
# ============================================================================

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
    "Bhopal", "Visakhapatnam", "Pimpri-Chinchwad", "Patna", "Vadodara"
]

# ============================================================================
# PRODUCT CATEGORIES
# ============================================================================

CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books",
    "Sports & Outdoors", "Beauty & Personal Care",
    "Toys & Games", "Automotive"
]

PRODUCT_NAMES = {
    "Electronics": [
        "Smartphone", "Laptop", "Wireless Headphones", "Smart TV",
        "Tablet", "Bluetooth Speaker", "Smart Watch", "Power Bank"
    ],
    "Clothing": [
        "T-Shirt", "Jeans", "Shirt", "Dress", "Jacket",
        "Sweater", "Pants", "Shorts"
    ],
    "Home & Kitchen": [
        "Cookware Set", "Bedding Set", "Kitchen Utensils",
        "Home Decor", "Furniture", "Lighting"
    ],
    "Books": [
        "Fiction Novel", "Non-Fiction", "Textbook", "Comics",
        "Biography", "Self-Help", "Cookbook"
    ],
    "Sports & Outdoors": [
        "Running Shoes", "Yoga Mat", "Dumbbells", "Camping Tent",
        "Basketball", "Tennis Racket", "Cycling Helmet"
    ],
    "Beauty & Personal Care": [
        "Face Wash", "Moisturizer", "Shampoo", "Perfume",
        "Makeup Set", "Hair Dryer", "Skincare Kit"
    ],
    "Toys & Games": [
        "Puzzle", "Board Game", "Action Figure", "Doll",
        "Building Blocks", "Remote Control Car"
    ],
    "Automotive": [
        "Car Charger", "Seat Cover", "Floor Mat", "Tool Kit",
        "Car freshener"
    ]
}

# ============================================================================
# DEPARTMENT NAMES
# ============================================================================

DEPARTMENTS = [
    ("Engineering", "Tech Park, Bangalore"),
    ("Sales", "Corporate Office, Mumbai"),
    ("Marketing", "Creative Hub, Delhi"),
    ("HR", "People Center, Chennai"),
    ("Finance", "Finance Tower, Mumbai"),
    ("Operations", "Ops Center, Hyderabad"),
    ("Support", "Support Hub, Pune"),
    ("R&D", "Innovation Lab, Bangalore")
]

# ============================================================================
# ORDER STATUS
# ============================================================================

ORDER_STATUSES = ["pending", "processing", "shipped", "delivered", "cancelled"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_phone_number() -> str:
    """Generate Indian phone number format."""
    return f"+91 {random.randint(70000, 99999)}{random.randint(10000, 99999)}"


def generate_email(first_name: str, last_name: str) -> str:
    """Generate email address."""
    domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
    return f"{first_name.lower()}.{last_name.lower()}@{random.choice(domains)}"


def random_date(start: datetime, end: datetime) -> datetime:
    """Generate random date between start and end."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


# ============================================================================
# DATABASE CREATION
# ============================================================================

def create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables with foreign key constraints."""
    cursor = conn.cursor()

    # Customers table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT NOT NULL,
        city TEXT NOT NULL,
        state TEXT DEFAULT 'Maharashtra',
        join_date DATE NOT NULL,
        total_orders INTEGER DEFAULT 0,
        total_spent REAL DEFAULT 0.0
    )
    """)

    # Products table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL CHECK(price > 0),
        stock_qty INTEGER DEFAULT 0,
        rating REAL DEFAULT 5.0 CHECK(rating >= 1.0 AND rating <= 5.0)
    )
    """)

    # Orders table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        order_date DATE NOT NULL,
        status TEXT NOT NULL,
        total_amount REAL NOT NULL,
        city TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    """)

    # Order Items table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS order_items (
        item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL CHECK(quantity > 0),
        unit_price REAL NOT NULL,
        subtotal REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    """)

    # Employees table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        emp_id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        dept_id INTEGER NOT NULL,
        salary REAL NOT NULL,
        hire_date DATE NOT NULL,
        city TEXT NOT NULL,
        FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
    )
    """)

    # Departments table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS departments (
        dept_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dept_name TEXT UNIQUE NOT NULL,
        location TEXT NOT NULL,
        budget REAL DEFAULT 1000000.0
    )
    """)

    # Sales table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        month INTEGER NOT NULL CHECK(month BETWEEN 1 AND 12),
        year INTEGER NOT NULL,
        sales_amount REAL NOT NULL,
        city TEXT NOT NULL,
        category TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    """)

    conn.commit()
    logger.info("All tables created successfully")


def populate_departments(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Populate departments table and return inserted rows."""
    cursor = conn.cursor()
    departments = []

    for dept_name, location in DEPARTMENTS:
        budget = random.uniform(500000, 5000000)
        cursor.execute(
            "INSERT INTO departments (dept_name, location, budget) VALUES (?, ?, ?)",
            (dept_name, location, budget)
        )
        departments.append({
            "dept_id": cursor.lastrowid,
            "dept_name": dept_name,
            "location": location,
            "budget": budget
        })

    conn.commit()
    logger.info(f"Populated {len(departments)} departments")
    return departments


def populate_customers(conn: sqlite3.Connection, count: int = 200) -> List[Dict[str, Any]]:
    """Populate customers table."""
    cursor = conn.cursor()
    customers = []

    since_date = datetime(2020, 1, 1)
    until_date = datetime(2025, 12, 31)

    for i in range(count):
        first_name = fake.first_name()
        last_name = fake.last_name()
        email = generate_email(first_name, last_name)
        phone = generate_phone_number()
        city = random.choice(INDIAN_CITIES)
        state = "Maharashtra" if city in ["Mumbai", "Pune", "Nagpur"] else \
                "Karnataka" if city in ["Bangalore"] else \
                "Tamil Nadu" if city in ["Chennai"] else \
                "Telangana" if city in ["Hyderabad"] else \
                "Delhi" if city in ["Delhi"] else \
                random.choice(["Gujarat", "Rajasthan", "Uttar Pradesh", "West Bengal"])

        join_date = random_date(since_date, until_date).strftime("%Y-%m-%d")
        total_orders = random.randint(0, 50)
        total_spent = round(random.uniform(0, 100000), 2)

        cursor.execute("""
            INSERT INTO customers
            (first_name, last_name, email, phone, city, state, join_date, total_orders, total_spent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (first_name, last_name, email, phone, city, state, join_date, total_orders, total_spent))

        customers.append({
            "customer_id": cursor.lastrowid,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "city": city
        })

    conn.commit()
    logger.info(f"Populated {len(customers)} customers")
    return customers


def populate_products(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Populate products table."""
    cursor = conn.cursor()
    products = []

    product_id = 1
    for category in CATEGORIES:
        for name in PRODUCT_NAMES[category]:
            price = round(random.uniform(100, 50000), 2)
            stock_qty = random.randint(10, 1000)
            rating = round(random.uniform(3.5, 5.0), 1)

            cursor.execute("""
                INSERT INTO products (product_name, category, price, stock_qty, rating)
                VALUES (?, ?, ?, ?, ?)
            """, (name, category, price, stock_qty, rating))

            products.append({
                "product_id": product_id,
                "product_name": name,
                "category": category,
                "price": price
            })
            product_id += 1

    conn.commit()
    logger.info(f"Populated {len(products)} products")
    return products


def populate_orders(
    conn: sqlite3.Connection,
    customers: List[Dict[str, Any]],
    count: int = 500
) -> List[Dict[str, Any]]:
    """Populate orders table."""
    cursor = conn.cursor()
    orders = []

    since_date = datetime(2023, 1, 1)
    until_date = datetime(2025, 12, 31)

    for i in range(count):
        customer = random.choice(customers)
        order_date = random_date(since_date, until_date).strftime("%Y-%m-%d")
        status = random.choice(ORDER_STATUSES)
        total_amount = round(random.uniform(500, 50000), 2)
        city = customer["city"] if random.random() > 0.2 else random.choice(INDIAN_CITIES)

        cursor.execute("""
            INSERT INTO orders (customer_id, order_date, status, total_amount, city)
            VALUES (?, ?, ?, ?, ?)
        """, (customer["customer_id"], order_date, status, total_amount, city))

        orders.append({
            "order_id": cursor.lastrowid,
            "customer_id": customer["customer_id"],
            "city": city
        })

    conn.commit()
    logger.info(f"Populated {len(orders)} orders")
    return orders


def populate_order_items(
    conn: sqlite3.Connection,
    orders: List[Dict[str, Any]],
    products: List[Dict[str, Any]]
) -> None:
    """Populate order_items table."""
    cursor = conn.cursor()

    for order in orders:
        # Each order has 1-5 items
        num_items = random.randint(1, 5)
        selected_products = random.sample(products, num_items)

        for product in selected_products:
            quantity = random.randint(1, 10)
            unit_price = product["price"]
            subtotal = round(quantity * unit_price, 2)

            cursor.execute("""
                INSERT INTO order_items (order_id, product_id, quantity, unit_price, subtotal)
                VALUES (?, ?, ?, ?, ?)
            """, (order["order_id"], product["product_id"], quantity, unit_price, subtotal))

    conn.commit()
    logger.info(f"Populated order_items")


def populate_employees(
    conn: sqlite3.Connection,
    departments: List[Dict[str, Any]],
    count: int = 50
) -> List[Dict[str, Any]]:
    """Populate employees table."""
    cursor = conn.cursor()
    employees = []

    since_date = datetime(2015, 1, 1)
    until_date = datetime(2025, 12, 31)

    for i in range(count):
        first_name = fake.first_name()
        last_name = fake.last_name()
        email = generate_email(first_name, last_name)
        dept = random.choice(departments)
        salary = round(random.uniform(300000, 2500000), 2)  # 3L - 25L annually
        hire_date = random_date(since_date, until_date).strftime("%Y-%m-%d")
        city = dept["location"].split(",")[-1].strip() if "," in dept["location"] else "Bangalore"

        cursor.execute("""
            INSERT INTO employees (first_name, last_name, email, dept_id, salary, hire_date, city)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (first_name, last_name, email, dept["dept_id"], salary, hire_date, city))

        employees.append({"emp_id": cursor.lastrowid, "dept_id": dept["dept_id"]})

    conn.commit()
    logger.info(f"Populated {len(employees)} employees")
    return employees


def populate_sales(
    conn: sqlite3.Connection,
    customers: List[Dict[str, Any]],
    months_per_year: int = 3
) -> None:
    """Populate sales table with historical data."""
    cursor = conn.cursor()

    current_year = 2025
    years = [2023, 2024, 2025]
    categories = CATEGORIES[:5]  # Use first 5 categories for sales

    for customer in customers:
        for year in years:
            # Each customer has sales in random months
            num_months = random.randint(1, months_per_year)
            months = random.sample(range(1, 13), num_months)

            for month in months:
                sales_amount = round(random.uniform(5000, 200000), 2)
                city = next(c["city"] for c in customers if c["customer_id"] == customer["customer_id"])
                category = random.choice(categories)

                cursor.execute("""
                    INSERT INTO sales (customer_id, month, year, sales_amount, city, category)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (customer["customer_id"], month, year, sales_amount, city, category))

    conn.commit()
    logger.info(f"Populated sales table")


def update_customer_stats(conn: sqlite3.Connection) -> None:
    """Update customer total_orders and total_spent based on orders."""
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE customers
        SET total_orders = (
            SELECT COUNT(*) FROM orders WHERE orders.customer_id = customers.customer_id
        ),
        total_spent = (
            SELECT COALESCE(SUM(total_amount), 0)
            FROM orders WHERE orders.customer_id = customers.customer_id
        )
    """)

    conn.commit()
    logger.info("Updated customer statistics")


def create_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes for performance."""
    cursor = conn.cursor()

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id)",
        "CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date)",
        "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
        "CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id)",
        "CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id)",
        "CREATE INDEX IF NOT EXISTS idx_customers_city ON customers(city)",
        "CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)",
        "CREATE INDEX IF NOT EXISTS idx_sales_year_month ON sales(year, month)",
        "CREATE INDEX IF NOT EXISTS idx_sales_customer_id ON sales(customer_id)",
        "CREATE INDEX IF NOT EXISTS idx_employees_dept_id ON employees(dept_id)",
    ]

    for index_sql in indexes:
        cursor.execute(index_sql)

    conn.commit()
    logger.info(f"Created {len(indexes)} indexes")


def export_schema(conn: sqlite3.Connection) -> None:
    """Export database schema to JSON file."""
    import json

    cursor = conn.cursor()
    schema = {"tables": {}}

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()

    for (table_name,) in tables:
        # Get table creation SQL
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        create_sql = cursor.fetchone()[0]

        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = []
        for row in cursor.fetchall():
            columns.append({
                "cid": row[0],
                "name": row[1],
                "type": row[2],
                "notnull": bool(row[3]),
                "default": row[4],
                "pk": bool(row[5])
            })

        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = []
        for row in cursor.fetchall():
            foreign_keys.append({
                "id": row[0],
                "seq": row[1],
                "table": row[2],
                "from": row[3],
                "to": row[4],
                "on_update": row[5],
                "on_delete": row[6],
                "match": row[7]
            })

        schema["tables"][table_name] = {
            "create_sql": create_sql,
            "columns": columns,
            "foreign_keys": foreign_keys
        }

    with open(SCHEMA_FILE, 'w') as f:
        json.dump(schema, f, indent=2)

    logger.info(f"Exported schema to {SCHEMA_FILE}")


def verify_database(conn: sqlite3.Connection) -> None:
    """Verify database integrity and print statistics."""
    cursor = conn.cursor()

    print("\n" + "="*60)
    print("DATABASE VERIFICATION")
    print("="*60)

    # Count records in each table
    tables = ["customers", "products", "orders", "order_items", "employees", "departments", "sales"]
    total_records = 0

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        total_records += count
        print(f"  {table:20s}: {count:6d} records")

    print("-"*60)
    print(f"  {'TOTAL':20s}: {total_records:6d} records")
    print("="*60)

    # Verify foreign key constraints
    cursor.execute("PRAGMA foreign_keys")
    fk_enabled = cursor.fetchone()[0]
    print(f"\n  Foreign Keys: {'ENABLED' if fk_enabled else 'DISABLED'}")

    # Check integrity
    cursor.execute("PRAGMA integrity_check")
    integrity = cursor.fetchone()[0]
    print(f"  Integrity Check: {integrity}")

    # Sample queries
    print("\n" + "-"*60)
    print("  Sample Queries:")
    print("-"*60)

    try:
        # Top 5 customers by spending
        cursor.execute("""
            SELECT first_name, last_name, total_spent
            FROM customers
            ORDER BY total_spent DESC
            LIMIT 5
        """)
        print("  Top 5 Customers by Spending:")
        for row in cursor.fetchall():
            print(f"    {row[0]} {row[1]}: INR {row[2]:,.2f}")

        # Product count by category
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM products
            GROUP BY category
            ORDER BY count DESC
        """)
        print("\n  Products by Category:")
        for row in cursor.fetchall():
            print(f"    {row[0]}: {row[1]} products")

    except Exception as e:
        logger.error(f"Error running sample queries: {e}")

    print("="*60 + "\n")


def create_database() -> sqlite3.Connection:
    """
    Main function to create and populate the entire database.

    Returns:
        sqlite3.Connection: Database connection
    """
    logger.info("Starting database creation...")

    try:
        # Delete existing database for fresh start
        if SAMPLE_DB.exists():
            SAMPLE_DB.unlink()
            logger.info(f"Removed existing database: {SAMPLE_DB}")

        # Connect to database (creates file if not exists)
        conn = sqlite3.connect(SAMPLE_DB)
        cursor = conn.cursor()

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        logger.info(f"Connected to database: {SAMPLE_DB}")

        # Create tables
        create_tables(conn)

        # Populate data in order (respecting foreign keys)
        departments = populate_departments(conn)
        customers = populate_customers(conn, count=200)
        products = populate_products(conn)
        orders = populate_orders(conn, customers, count=500)
        populate_order_items(conn, orders, products)
        employees = populate_employees(conn, departments, count=50)
        populate_sales(conn, customers, months_per_year=3)
        update_customer_stats(conn)
        create_indexes(conn)
        export_schema(conn)

        # Verify
        verify_database(conn)

        logger.info("Database creation completed successfully!")
        return conn

    except Exception as e:
        logger.error(f"Database creation failed: {e}")
        raise


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        conn = create_database()
        conn.close()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
