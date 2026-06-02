#!/usr/bin/env python
"""Debug script to see what NLP pipeline extracts from queries"""

from nlp.pipeline import NLPPipeline

def debug_query(query):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    pipeline = NLPPipeline(db_path="database/sample.db")
    result = pipeline.process(query)

    print(f"Original query: {result['original_query']}")
    print(f"Preprocessed: {result['preprocessed']}")
    print(f"Schema links: {result['schema_links']}")
    print(f"Entities: {result['entities']}")
    print(f"SQL hints: {result['sql_hints']}")

if __name__ == "__main__":
    # Test the problematic queries
    debug_query("Count customers from Mumbai")
    debug_query("Count employees with salary above 50000")
    debug_query("How many orders were delivered?")
    debug_query("Sum of sales above 100000")
    debug_query("Average salary of employees in Mumbai")