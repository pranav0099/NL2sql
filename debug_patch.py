#!/usr/bin/env python
"""Debug script to see what SQL is generated before and after patching"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.nl2sql_engine import NL2SQLEngine
from nlp.pipeline import NLPPipeline

# Monkey patch to add debug prints
original_patch_sql = NL2SQLEngine._patch_sql

def debug_patch_sql(self, sql, nlp_result):
    print(f"\n=== PATCH SQL DEBUG ===")
    print(f"Input SQL: {repr(sql)}")
    print(f"NLP result keys: {list(nlp_result.keys())}")

    # Call original method
    result = original_patch_sql(self, sql, nlp_result)

    print(f"Output SQL: {repr(result)}")
    print(f"=======================\n")
    return result

NL2SQLEngine._patch_sql = debug_patch_sql

def test_queries():
    engine = NL2SQLEngine(db_path="database/sample.db")
    session_id = engine.create_session()

    test_queries = [
        "Count customers from Mumbai",
        "Count employees with salary above 50000",
        "How many orders were delivered?"
    ]

    for query in test_queries:
        print(f"\n{'#'*60}")
        print(f"Testing: {query}")
        print('#'*60)
        result = engine.query(query, session_id)
        print(f"Final SQL: {result['sql']}")
        print(f"Success: {result['success']}")

if __name__ == "__main__":
    test_queries()