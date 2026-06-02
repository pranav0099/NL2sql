#!/usr/bin/env python
"""Debug script to see what raw SQL the DL model generates before patching"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from engine.nl2sql_engine import NL2SQLEngine
from nlp.pipeline import NLPPipeline

# Monkey patch to capture raw SQL before patching
original_patch_sql = NL2SQLEngine._patch_sql

def debug_patch_sql(self, sql, nlp_result):
    print(f"\n=== RAW SQL FROM DL MODEL ===")
    print(f"Raw SQL: {repr(sql)}")
    print(f"NLP entities: {nlp_result.get('entities', {})}")
    print(f"NLP preprocessed: {nlp_result.get('preprocessed', {})}")
    print(f"NLP schema_links: {nlp_result.get('schema_links', {})}")
    print(f"=================================\n")
    # Call original method
    result = original_patch_sql(self, sql, nlp_result)
    print(f"SQL AFTER PATCH: {repr(result)}")
    return result

NL2SQLEngine._patch_sql = debug_patch_sql

def test_raw_sql():
    engine = NL2SQLEngine(db_path="database/sample.db")
    session_id = engine.create_session()

    test_queries = [
        "Count employees with salary above 50000",
        "How many orders were delivered?",
        "Sum of sales above 100000"
    ]

    for query in test_queries:
        print(f"\n{'#'*60}")
        print(f"Testing: {query}")
        print('#'*60)
        result = engine.query(query, session_id)
        print(f"Final SQL: {result['sql']}")
        print(f"Success: {result['success']}")

if __name__ == "__main__":
    test_raw_sql()