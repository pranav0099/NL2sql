import sys
sys.path.insert(0, ".")
from engine.nl2sql_engine import NL2SQLEngine

db_path = r"database\uploads\customers-100.db"
engine = NL2SQLEngine(db_path=db_path)
session_id = engine.create_session()

queries = [
    "Show first 10 rows from customers_100",
    "Count total rows in customers_100",
    "Show all records from customers_100",
]

for query in queries:
    result = engine.query(query, session_id)
    print(f"\nQ: {query}")
    print(f"SQL: {result['sql']}")
    print(f"Success: {result['success']}")
    rows = result['results']['row_count'] if result['results'] else 0
    print(f"Rows: {rows}")
    if result['error']:
        print(f"Error: {result['error']}")
