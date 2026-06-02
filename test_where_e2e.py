"""
End-to-end test for WHERE clause fix.
Tests the full NL2SQL pipeline from query to final SQL.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from engine.nl2sql_engine import NL2SQLEngine

print("=" * 65)
print("WHERE Clause Fix - End-to-End Pipeline Test")
print("=" * 65)

engine = NL2SQLEngine(db_path="database/sample.db")
session_id = engine.create_session()

test_cases = [
    {
        "query": "show staff that salary has below 40000",
        "expect_where": True,
        "expect_op": "<",
    },
    {
        "query": "show employees with salary above 60000",
        "expect_where": True,
        "expect_op": ">",
    },
    {
        "query": "show customers from Mumbai",
        "expect_where": True,
        "expect_op": "=",
    },
    {
        "query": "find products under 5000",
        "expect_where": True,
        "expect_op": "<",
    },
    {
        "query": "show staff whose salary is at least 50000",
        "expect_where": True,
        "expect_op": ">=",
    },
    {
        "query": "show departments with budget no more than 100000",
        "expect_where": True,
        "expect_op": "<=",
    },
]

passed = 0
failed = 0

for i, tc in enumerate(test_cases, 1):
    result = engine.query(tc["query"], session_id)
    sql = result.get("sql", "") or ""
    has_where = "WHERE" in sql.upper()
    has_op = tc["expect_op"] in sql if tc["expect_op"] else True

    ok = has_where == tc["expect_where"] and has_op
    status = "OK" if ok else "FAIL"

    if ok:
        passed += 1
    else:
        failed += 1

    print(f"\n[{i}] [{status}] {tc['query']}")
    print(f"     SQL: {sql}")
    print(f"     WHERE present: {has_where} (expected: {tc['expect_where']})")
    print(f"     Operator '{tc['expect_op']}' present: {has_op}")
    if result.get("error"):
        print(f"     Error: {result['error']}")

print(f"\n{'=' * 65}")
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)}")
print(f"{'=' * 65}")
