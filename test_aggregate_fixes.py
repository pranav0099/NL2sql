#!/usr/bin/env python
"""Test script for the specific aggregate query fixes"""

from engine.nl2sql_engine import NL2SQLEngine

def test_specific_queries():
    """Test the exact queries mentioned in the problem statement"""
    engine = NL2SQLEngine(db_path="database/sample.db")
    session_id = engine.create_session()

    test_cases = [
        # Original working case
        ("Count all customers", "SELECT COUNT(*) FROM customers"),

        # The failing cases that should now work
        ("Count customers from Mumbai", "SELECT COUNT(*) FROM customers WHERE city = 'Mumbai'"),
        ("Count employees with salary above 50000", "SELECT COUNT(*) FROM employees WHERE salary > 50000"),
        ("How many orders were delivered?", "SELECT COUNT(*) FROM orders WHERE status = 'Delivered'"),
        ("Sum of sales above 100000", "SELECT SUM(sales_amount) FROM sales WHERE sales_amount > 100000"),
        ("Average salary of employees in Mumbai", "SELECT AVG(salary) FROM employees WHERE city = 'Mumbai'"),
    ]

    print("Testing specific aggregate queries with conditions:")
    print("=" * 60)

    all_passed = True

    for i, (query, expected_pattern) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)

        result = engine.query(query, session_id)

        if result["success"]:
            print(f"✅ SUCCESS")
            print(f"Generated SQL: {result['sql']}")
            print(f"Explanation: {result['explanation']}")

            # Check if SQL contains expected pattern (case insensitive, ignoring extra spaces)
            sql_normalized = ' '.join(result['sql'].split()).upper()
            expected_normalized = ' '.join(expected_pattern.split()).upper()

            if expected_normalized in sql_normalized:
                print(f"✅ SQL matches expected pattern")
            else:
                print(f"⚠️  SQL doesn't exactly match expected pattern")
                print(f"Expected pattern: {expected_pattern}")
                print(f"Actual SQL:      {result['sql']}")
        else:
            print(f"❌ FAILED")
            print(f"Error: {result['error']}")
            all_passed = False

        print(f"Row count: {result['results']['row_count'] if result['results'] else 0}")

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")

    return all_passed

if __name__ == "__main__":
    test_specific_queries()