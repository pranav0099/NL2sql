#!/usr/bin/env python
"""Test the SQLGenerator directly to see what it produces"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from dl.generator import SQLGenerator
from nlp.pipeline import NLPPipeline

def test_generator():
    print("Testing SQLGenerator directly")
    print("=" * 50)

    # Initialize components
    generator = SQLGenerator()
    pipeline = NLPPipeline(db_path="database/sample.db")

    test_queries = [
        "Count customers from Mumbai",
        "Count employees with salary above 50000",
        "How many orders were delivered?",
        "Sum of sales above 100000",
        "Average salary of employees in Mumbai"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)

        # Get NLP result
        nlp_result = pipeline.process(query)
        print(f"Intent: {nlp_result['sql_hints']['intent_hint']}")
        print(f"Entities: {nlp_result['entities']}")

        # Generate SQL
        result = generator.generate(query, nlp_result)
        print(f"Generated SQL: {result['sql']}")
        print(f"Method: {result['method']}")
        print(f"Fallback used: {result['fallback_used']}")

if __name__ == "__main__":
    test_generator()