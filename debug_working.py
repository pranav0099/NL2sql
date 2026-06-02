#!/usr/bin/env python
"""Debug script to see what happens with a working query"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from engine.nl2sql_engine import NL2SQLEngine
from nlp.pipeline import NLPPipeline

# Monkey patch to capture detailed info
original_patch_sql = NL2SQLEngine._patch_sql

def debug_patch_sql(self, sql, nlp_result):
    print(f"\n=== PATCH SQL INPUT ===")
    print(f"SQL before patch: {repr(sql)}")
    print(f"Aggregations in NLP: {nlp_result.get('entities', {}).get('aggregations', [])}")
    print(f"Numbers in NLP: {nlp_result.get('preprocessed', {}).get('numbers', [])}")
    print(f"Raw cities in NLP: {nlp_result.get('entities', {}).get('raw_cities', [])}")
    print(f"Raw names in NLP: {nlp_result.get('entities', {}).get('raw_names', [])}")
    print(f"Filters in NLP: {nlp_result.get('entities', {}).get('filters', [])}")
    print(f"NL entities: {nlp_result.get('preprocessed', {}).get('entities', [])}")
    print(f"Schema links: {nlp_result.get('schema_links', {}).get('matched_tables', [])}")

    # Call original method
    result = original_patch_sql(self, sql, nlp_result)

    print(f"SQL after patch: {repr(result)}")
    print(f"=====================\n")
    return result

def debug_query(self, user_input: str, session_id: str = "default"):
    print(f"\n{'='*60}")
    print(f"QUERY: {user_input}")
    print('='*60)

    # Call original query method but capture intermediate steps
    result = original_query(self, user_input, session_id)

    print(f"FINAL RESULT:")
    print(f"  Success: {result['success']}")
    print(f"  SQL: {result['sql']}")
    print(f"  Explanation: {result['explanation']}")
    if result['error']:
        print(f"  Error: {result['error']}")
    print(f"{'='*60}\n")
    return result

NL2SQLEngine._patch_sql = debug_patch_sql
NL2SQLEngine.query = debug_query

def test_working_query():
    engine = NL2SQLEngine(db_path="database/sample.db")
    session_id = engine.create_session()

    # Test a working query first
    query = "Show all customers from Mumbai"
    print(f"Testing working query: {query}")
    result = engine.query(query, session_id)

    # Now test the problematic ones
    test_queries = [
        "Count customers from Mumbai",
    ]

    for query in test_queries:
        print(f"Testing problematic query: {query}")
        result = engine.query(query, session_id)

if __name__ == "__main__":
    test_working_query()