"""
Test script for WHERE clause fix.
Tests filter extraction and WHERE rebuilding.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from nlp.preprocessor import QueryPreprocessor
from nlp.entity_extractor import EntityExtractor

pp = QueryPreprocessor()
ee = EntityExtractor()

# Mock schema covering all test queries
schema = {
    "staff": {
        "columns": ["id", "name", "salary", "department", "city"],
        "types": ["INT", "TEXT", "REAL", "TEXT", "TEXT"]
    },
    "employees": {
        "columns": ["emp_id", "name", "salary", "department", "city"],
        "types": ["INT", "TEXT", "REAL", "TEXT", "TEXT"]
    },
    "customers": {
        "columns": ["id", "name", "city", "email"],
        "types": ["INT", "TEXT", "TEXT", "TEXT"]
    },
    "products": {
        "columns": ["id", "name", "price", "category"],
        "types": ["INT", "TEXT", "REAL", "TEXT"]
    },
    "orders": {
        "columns": ["id", "customer_id", "status", "total"],
        "types": ["INT", "INT", "TEXT", "REAL"]
    },
    "departments": {
        "columns": ["id", "dept_name", "budget"],
        "types": ["INT", "TEXT", "REAL"]
    },
}

queries = [
    "show staff that salary has below 40000",
    "show employees with salary above 60000",
    "show customers from Mumbai",
    "find products under 5000",
    "show staff whose salary is at least 50000",
    "show departments with budget no more than 100000",
]

print("=" * 65)
print("WHERE Clause Fix - Filter Extraction Test")
print("=" * 65)

for q in queries:
    prep = pp.preprocess(q)
    ents = ee.extract(prep, schema)
    filt = ents.get("filters", [])
    kws = sorted(prep.get("keywords", set()))
    print(f"\nQuery: {q}")
    print(f"  Keywords: {kws}")
    print(f"  Filters:  {filt}")
    if filt:
        print("  [OK] Filters detected")
    else:
        print("  [FAIL] No filters detected!")

print("\n" + "=" * 65)
print("Test complete")
print("=" * 65)
