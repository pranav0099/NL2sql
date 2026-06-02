#!/usr/bin/env python
"""Final test for the exact queries from the problem statement"""

import sys
import re
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from engine.nl2sql_engine import NL2SQLEngine

def normalize_sql(sql):
    """Normalize SQL for comparison: remove extra spaces, uppercase"""
    return ' '.join(sql.split()).upper()

def test_queries():
    engine = NL2SQLEngine(db_path="database/sample.db")
    session_id = engine.create_session()

    # List of (query, expected_normalized_sql_pattern)
    tests = [
        # Original working case
        ("Count all customers", "SELECT COUNT(*) FROM CUSTOMERS"),

        # The specific failing cases that should now work
        ("Count customers from Mumbai", "SELECT COUNT(*) FROM CUSTOMERS WHERE CITY = 'MUMBAI'"),
        ("Count employees with salary above 50000", "SELECT COUNT(*) FROM EMPLOYEES WHERE SALARY > 50000"),
        ("How many orders were delivered?", "SELECT COUNT(*) FROM ORDERS WHERE STATUS = 'DELIVERED'"),
        ("Sum of sales above 100000", "SELECT SUM(SALES_AMOUNT) FROM SALES WHERE SALES_AMOUNT > 100000"),
        ("Average salary of employees in Mumbai", "SELECT AVG(SALARY) FROM EMPLOYEES WHERE CITY = 'MUMBAI'"),
    ]

    print("Final test for specific aggregate queries with conditions")
    print("=" * 60)

    all_passed = True

    for i, (query, expected_pattern) in enumerate(tests, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)

        result = engine.query(query, session_id)

        if result["success"]:
            print("SUCCESS")
            print(f"Generated SQL: {result['sql']}")
            print(f"Explanation: {result['explanation']}")

            # Normalize generated SQL
            gen_norm = normalize_sql(result['sql'])
            exp_norm = normalize_sql(expected_pattern)

            # Check if the expected pattern is in the generated SQL
            if exp_norm in gen_norm:
                print("PASS: SQL matches expected pattern")
            else:
                print("FAIL: SQL does not match expected pattern")
                print(f"  Expected pattern: {expected_pattern}")
                print(f"  Generated SQL:    {result['sql']}")
                all_passed = False
        else:
            print("FAILED")
            print(f"Error: {result['error']}")
            all_passed = False

        print(f"Row count: {result['results']['row_count'] if result['results'] else 0}")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("The NL2SQL system now correctly handles the specific aggregate queries.")
    else:
        print("SOME TESTS FAILED")
        print("The fixes may need further adjustment.")

    return all_passed

if __name__ == "__main__":
    test_queries()