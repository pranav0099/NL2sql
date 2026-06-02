"""
Phase 2 Validation Script — Runs the NLP pipeline and produces a clean summary.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import INTENTS
from nlp.pipeline import NLPPipeline

def main():
    results = []
    summary = {"total": 0, "passed": 0, "failed": 0, "details": []}

    try:
        pipeline = NLPPipeline(db_path="database/sample.db")
        init_ok = True
    except Exception as e:
        init_ok = False
        summary["init_error"] = str(e)
        json.dump(summary, open("phase2_results.json", "w"), indent=2)
        return

    TEST_QUERIES = [
        "Show all customers from Mumbai",
        "What is the average salary of employees?",
        "Find top 5 products with price greater than 5000",
        "Count total orders grouped by city",
        "Show customer names and their order totals",
        "List employees with salary above 60000 in Sales department",
        "What is the total sales amount for each category in 2024?",
        "Show me the most expensive product",
    ]

    for i, query in enumerate(TEST_QUERIES, 1):
        summary["total"] += 1
        try:
            result = pipeline.process(query)
            intent = result["sql_hints"]["intent_hint"]
            passed = intent in INTENTS

            detail = {
                "id": i,
                "query": query,
                "intent_hint": intent,
                "tables": result["schema_links"]["matched_tables"],
                "filters": result["entities"]["filters"],
                "aggregations": result["entities"]["aggregations"],
                "order": result["entities"]["order"],
                "keywords": sorted(result["preprocessed"]["keywords"]),
                "passed": passed
            }

            if passed:
                summary["passed"] += 1
            else:
                summary["failed"] += 1

            summary["details"].append(detail)

        except Exception as e:
            summary["failed"] += 1
            summary["details"].append({
                "id": i,
                "query": query,
                "error": str(e),
                "passed": False
            })

    summary["all_passed"] = summary["failed"] == 0
    summary["init_ok"] = init_ok

    with open("phase2_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"PHASE2_DONE:{summary['passed']}/{summary['total']}")

if __name__ == "__main__":
    main()
