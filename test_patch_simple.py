#!/usr/bin/env python
"""Test the _patch_sql method in isolation - simple version without unicode"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from engine.nl2sql_engine import NL2SQLEngine

def test_patch_isolated():
    engine = NL2SQLEngine(db_path="database/sample.db")

    # Test cases: (input_sql, nlp_result, expected_output)
    test_cases = [
        # Missing value after operator
        (
            "SELECT COUNT(*) FROM customers WHERE city =",
            {"preprocessed": {"numbers": []}, "entities": {"raw_cities": ["Mumbai"], "raw_names": []}, "schema_links": {"matched_tables": ["customers"]}},
            "SELECT COUNT(*) FROM customers WHERE city = 'Mumbai'"
        ),
        # Missing value after operator with number
        (
            "SELECT COUNT(*) FROM employees WHERE salary >",
            {"preprocessed": {"numbers": [50000]}, "entities": {"raw_cities": [], "raw_names": []}, "schema_links": {"matched_tables": ["employees"]}},
            "SELECT COUNT(*) FROM employees WHERE salary > 50000"
        ),
        # Missing aggregate function value
        (
            "SELECT COUNT(",
            {"preprocessed": {"numbers": []}, "entities": {"raw_cities": [], "raw_names": []}, "schema_links": {"matched_tables": ["customers"]}},
            "SELECT COUNT(*)"
        ),
        # Missing table after FROM
        (
            "SELECT COUNT(*) FROM",
            {"preprocessed": {"numbers": []}, "entities": {"raw_cities": [], "raw_names": []}, "schema_links": {"matched_tables": ["customers"]}},
            "SELECT COUNT(*) FROM customers"
        ),
        # AND condition missing value
        (
            "SELECT COUNT(*) FROM customers WHERE city = 'Mumbai' AND age >",
            {"preprocessed": {"numbers": [25]}, "entities": {"raw_cities": [], "raw_names": []}, "schema_links": {"matched_tables": ["customers"]}},
            "SELECT COUNT(*) FROM customers WHERE city = 'Mumbai' AND age > 25"
        ),
    ]

    print("Testing _patch_sql method in isolation")
    print("=" * 50)

    all_passed = True

    for i, (input_sql, nlp_result, expected) in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Input SQL:    {input_sql}")
        print(f"NLP result:   {nlp_result}")
        print(f"Expected:     {expected}")

        # We need to access the private method, so we'll use a bit of introspection
        # Alternatively, we can call it via the engine's query method but that's more complex.
        # Let's just call the private method directly for this test.
        patched = engine._patch_sql(input_sql, nlp_result)

        print(f"Actual:       {patched}")

        if patched == expected:
            print("PASS")
        else:
            print("FAIL")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")

    return all_passed

if __name__ == "__main__":
    test_patch_isolated()