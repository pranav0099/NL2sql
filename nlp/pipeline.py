"""
NL2SQL — Phase 2: Master NLP Pipeline
Orchestrates preprocessor, schema linker, and entity extractor.
Run: python nlp/pipeline.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Set
import logging

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger
from config.config import INTENTS

logger = get_logger(__name__)

# Import with proper error handling
try:
    from nlp.preprocessor import QueryPreprocessor
except ImportError as e:
    logger.error(f"Failed to import QueryPreprocessor: {e}")
    raise

try:
    from nlp.schema_linker import SchemaLinker
except ImportError as e:
    logger.error(f"Failed to import SchemaLinker: {e}")
    raise

try:
    from nlp.entity_extractor import EntityExtractor
except ImportError as e:
    logger.error(f"Failed to import EntityExtractor: {e}")
    raise


# =============================================================================
# NLP PIPELINE CLASS
# =============================================================================

class NLPPipeline:
    """
    Master NLP pipeline — orchestrates all 3 modules.

    This is the ONLY class imported by Phase 3+ modules.
    """

    def __init__(self, db_path: str = "database/sample.db"):
        """
        Initialize NLP pipeline with all components.

        Args:
            db_path: Path to SQLite database file
        """
        logger.info("Initializing NLPPipeline")
        logger.info(f"Database path: {db_path}")

        try:
            self.preprocessor = QueryPreprocessor()
            self.schema_linker = SchemaLinker(db_path)
            self.entity_extractor = EntityExtractor()
            logger.info("NLPPipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NLPPipeline: {e}")
            raise

    def process(self, query: str) -> Dict[str, Any]:
        """
        Process a raw user query through the full NLP pipeline.

        NEVER crashes on any input — wraps full pipeline in try/except
        and returns a safe default result dict on any unexpected error.

        Args:
            query: Raw user query string

        Returns:
            Complete NLP result dictionary:
            {
                "original_query": str,
                "preprocessed": dict,
                "schema_links": dict,
                "entities": dict,
                "schema_context": str,
                "sql_hints": dict
            }
        """
        logger.info(f"Processing query: '{query}'")

        try:
            # 1. Preprocess
            preprocessed = self.preprocessor.preprocess(query)

            # 2. Link schema
            schema_links = self.schema_linker.link(
                preprocessed["normalized"],
                []  # We could pass entities here, but linker works on query text
            )

            # 3. Extract entities
            entities = self.entity_extractor.extract(
                preprocessed,
                self.schema_linker.schema
            )

            # 4. Generate schema context
            schema_context = self.schema_linker.get_schema_context(
                schema_links["matched_tables"]
            )

            # 5. Derive intent hint
            intent_hint = self._derive_intent_hint(
                preprocessed["keywords"],
                entities
            )

            # 6. Build SQL hints
            sql_hints = self._build_sql_hints(
                intent_hint,
                schema_links["matched_tables"],
                entities,
                preprocessed
            )

            result = {
                "original_query": query,
                "preprocessed": preprocessed,
                "schema_links": schema_links,
                "entities": entities,
                "schema_context": schema_context,
                "sql_hints": sql_hints
            }

            logger.info(f"Query processing complete: intent={intent_hint}")
            return result

        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)

            # Return safe default
            safe_result = {
                "original_query": query,
                "preprocessed": {
                    "original": query,
                    "normalized": query,
                    "tokens": [],
                    "lemmas": [],
                    "pos_tags": [],
                    "entities": [],
                    "noun_chunks": [],
                    "numbers": [],
                    "keywords": set()
                },
                "schema_links": {
                    "matched_tables": [],
                    "matched_columns": {},
                    "confidence": {},
                    "suggested_joins": []
                },
                "entities": {
                    "filters": [],
                    "aggregations": [],
                    "order": None,
                    "group_by": [],
                    "select_all": True,
                    "raw_numbers": [],
                    "raw_cities": [],
                    "raw_names": []
                },
                "schema_context": "",
                "sql_hints": {
                    "intent_hint": "SELECT",
                    "tables": [],
                    "filters": [],
                    "aggregations": [],
                    "order": None,
                    "group_by": [],
                    "select_columns": ["*"]
                }
            }
            return safe_result

    def _derive_intent_hint(self, keywords: Set[str], entities: Dict[str, Any]) -> str:
        """
        Pure rule-based intent derivation from extracted keywords.

        Priority order (check in this exact order):
        1. "JOIN" in keywords              → "SELECT_JOIN"
        2. "GROUP BY" in keywords          → "SELECT_GROUP"
        3. Any of COUNT/SUM/AVG/MAX/MIN    → "SELECT_AGGREGATE"
        4. "ORDER BY" in keywords AND "LIMIT" in keywords → "SELECT_ORDER"
        5. "WHERE" in keywords             → "SELECT_WHERE"
        6. "LIMIT" in keywords             → "SELECT_LIMIT"
        7. default                         → "SELECT"

        Args:
            keywords: Set of SQL keyword hints from preprocessor
            entities: Extracted entities dict

        Returns:
            Intent label string (one of INTENTS)
        """
        logger.debug(f"Deriving intent from keywords: {keywords}")

        # 1. JOIN
        if "JOIN" in keywords:
            return "SELECT_JOIN"

        # 2. GROUP BY
        if "GROUP BY" in keywords or any(kw == "GROUP BY" for kw in keywords):
            return "SELECT_GROUP"

        # 3. Aggregates
        agg_keywords = {"COUNT", "SUM", "AVG", "MAX", "MIN"}
        if any(kw in agg_keywords for kw in keywords):
            return "SELECT_AGGREGATE"

        # 4. ORDER BY + LIMIT
        if "ORDER BY" in keywords and "LIMIT" in keywords:
            return "SELECT_ORDER"

        # 5. WHERE
        if "WHERE" in keywords:
            return "SELECT_WHERE"

        # 6. LIMIT only
        if "LIMIT" in keywords:
            return "SELECT_LIMIT"

        # 7. Default
        logger.debug("Default intent: SELECT")
        return "SELECT"

    def _build_sql_hints(self, intent_hint: str, tables: List[str],
                        entities: Dict[str, Any],
                        preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build structured SQL hints from pipeline outputs.

        Args:
            intent_hint: Derived intent label
            tables: Matched table names
            entities: Extracted entities
            preprocessed: Preprocessed query data

        Returns:
            SQL hints dictionary:
            {
                "intent_hint": str,
                "tables": List[str],
                "filters": List[dict],
                "aggregations": List[dict],
                "order": dict or None,
                "group_by": List[str],
                "select_columns": List[str]
            }
        """
        filters = entities.get("filters", [])
        aggregations = entities.get("aggregations", [])
        order = entities.get("order")
        group_by = entities.get("group_by", [])
        select_all = entities.get("select_all", True)

        # Build select_columns
        if select_all:
            select_columns = ["*"]
        elif aggregations:
            # For aggregations, select aggregation expressions
            select_columns = [
                f"{agg['function']}({agg['column']})" for agg in aggregations
            ]
            # Also include group by columns if present
            for gb_col in group_by:
                select_columns.append(gb_col)
        else:
            # No specific columns, default to *
            select_columns = ["*"]

        sql_hints = {
            "intent_hint": intent_hint,
            "tables": tables,
            "filters": filters,
            "aggregations": aggregations,
            "order": order,
            "group_by": group_by,
            "select_columns": select_columns
        }

        logger.debug(f"SQL hints built: {sql_hints}")
        return sql_hints

    def batch_process(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries.

        Args:
            queries: List of query strings

        Returns:
            List of result dictionaries (same as process())
        """
        logger.info(f"Batch processing {len(queries)} queries")
        results = []

        for i, query in enumerate(queries, 1):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(queries)} queries")

            result = self.process(query)
            results.append(result)

        logger.info(f"Batch processing complete: {len(results)} results")
        return results


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("NLP Pipeline — Phase 2 Test")
    print("=" * 65)

    try:
        pipeline = NLPPipeline(db_path="database/sample.db")
        print("\n[OK] Pipeline initialized")
    except Exception as e:
        print(f"\n[FAIL] Failed to initialize pipeline: {e}")
        print("Make sure database/sample.db exists")
        exit(1)

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

    print("\nRunning queries...\n")

    all_passed = True
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}] Query: {query}")
        try:
            result = pipeline.process(query)

            print(f"     Intent hint  : {result['sql_hints']['intent_hint']}")
            print(f"     Tables       : {result['schema_links']['matched_tables']}")
            print(f"     Filters      : {result['entities']['filters']}")
            print(f"     Aggregations : {result['entities']['aggregations']}")
            print(f"     Order        : {result['entities']['order']}")
            print(f"     Keywords     : {sorted(result['preprocessed']['keywords'])}")

            intent = result['sql_hints']['intent_hint']
            valid_intents = INTENTS

            if intent not in valid_intents:
                print(f"     FAIL: unexpected intent '{intent}'")
                all_passed = False
            else:
                print(f"     [PASS]")

        except Exception as e:
            print(f"     [FAIL] processing error: {e}")
            all_passed = False
            logger.error(f"Error processing query '{query}': {e}", exc_info=True)

        print()

    print("=" * 65)
    if all_passed:
        print("Result: ALL TESTS PASSED")
        print("Phase 2 NLP Pipeline is ready for Phase 3 ML Classifier")
    else:
        print("Result: SOME TESTS FAILED")
        print("Review failed queries above")
    print("=" * 65)
