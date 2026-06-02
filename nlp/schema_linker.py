"""
NL2SQL — Phase 2: Schema Linker for Database Table/Column Mapping
Run: python nlp/schema_linker.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import logging
import sqlite3

# Add project root to sys.path BEFORE importing utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class SchemaLinker:
    """
    Maps user query entities to real database table and column names.

    Uses TF-IDF vectorization and fuzzy matching to link query terms
    to database schema elements. Dynamically loads schema from SQLite.
    """

    def __init__(self, db_path: str = "database/sample.db"):
        """
        Initialize SchemaLinker.

        Args:
            db_path: Path to SQLite database file
        """
        logger.info(f"Initializing SchemaLinker with db_path: {db_path}")
        self.db_path = Path(db_path)
        self.schema = self._load_schema()
        self.vectorizer = self._build_vectorizer()
        logger.info("SchemaLinker initialized successfully")

    def _load_schema(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load database schema from SQLite.

        Returns:
            Dictionary mapping table names to their columns and types:
            {
                "table_name": {
                    "columns": ["col1", "col2", ...],
                    "types": ["INTEGER", "TEXT", ...]
                },
                ...
            }
        """
        logger.info(f"Loading schema from database: {self.db_path}")

        if not self.db_path.exists():
            logger.error(f"Database not found: {self.db_path}")
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        schema = {}
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get all table names
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            logger.debug(f"Found tables: {tables}")

            for table in tables:
                # Get column info for this table
                cursor.execute(f"PRAGMA table_info({table})")
                columns_info = cursor.fetchall()

                # PRAGMA returns: (cid, name, type, notnull, default, pk)
                columns = [info[1] for info in columns_info]
                types = [info[2] for info in columns_info]

                schema[table] = {
                    "columns": columns,
                    "types": types
                }
                logger.debug(f"Table '{table}': {len(columns)} columns")

            conn.close()
            logger.info(f"Schema loaded: {len(schema)} tables")
            return schema

        except sqlite3.Error as e:
            logger.error(f"Database error while loading schema: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading schema: {e}")
            raise

    def _build_vectorizer(self):
        """
        Build and fit TF-IDF vectorizer on all table and column names.

        Returns:
            Fitted TfidfVectorizer instance
        """
        logger.info("Building TF-IDF vectorizer from schema")

        from sklearn.feature_extraction.text import TfidfVectorizer

        # Collect all candidate names: table names and column names
        candidates = list(self.schema.keys())
        for table_info in self.schema.values():
            candidates.extend(table_info["columns"])

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate.lower() not in seen:
                seen.add(candidate.lower())
                unique_candidates.append(candidate)

        logger.debug(f"Training TF-IDF on {len(unique_candidates)} candidates")

        # Build vectorizer
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words='english'
        )

        # Fit on candidate names
        vectorizer.fit(unique_candidates)

        # Store candidates for later use
        self.candidate_names = unique_candidates

        logger.info("TF-IDF vectorizer built successfully")
        return vectorizer

    def link(self, query: str, extracted_entities: List,
             top_k: int = 3) -> Dict[str, Any]:
        """
        Link query to database tables and columns.

        Args:
            query: Preprocessed query text
            extracted_entities: List from EntityExtractor (filters, etc.)
            top_k: Number of top matches to return per category

        Returns:
            Dictionary with:
            {
                "matched_tables": List[str],
                "matched_columns": Dict[str, List[str]],
                "confidence": Dict[str, float],
                "suggested_joins": List[Dict]
            }
        """
        logger.info(f"Linking query: '{query}'")
        logger.debug(f"Extracted entities: {extracted_entities}")

        try:
            # Match tables
            matched_tables, table_scores = self._match_tables(query, top_k)

            # Match columns for each matched table
            matched_columns = {}
            column_confidences = {}

            for table in matched_tables:
                cols, confs = self._match_columns(query, table, top_k)
                matched_columns[table] = cols
                column_confidences[table] = confs

            # Calculate table confidence (average of column confidences)
            table_confidence = {}
            for table in matched_tables:
                confs = column_confidences.get(table, [])
                if confs:
                    table_confidence[table] = sum(confs) / len(confs)
                else:
                    table_confidence[table] = 0.0

            # Suggest joins if multiple tables
            suggested_joins = []
            if len(matched_tables) > 1:
                for i in range(len(matched_tables)):
                    for j in range(i + 1, len(matched_tables)):
                        join_hint = self._get_join_hint(
                            matched_tables[i], matched_tables[j]
                        )
                        if join_hint:
                            suggested_joins.append(join_hint)

            result = {
                "matched_tables": matched_tables,
                "matched_columns": matched_columns,
                "confidence": table_confidence,
                "suggested_joins": suggested_joins
            }

            logger.info(f"Link result: tables={matched_tables}, joins={len(suggested_joins)}")
            return result

        except Exception as e:
            logger.error(f"Error during schema linking: {e}")
            # Return safe default
            return {
                "matched_tables": [],
                "matched_columns": {},
                "confidence": {},
                "suggested_joins": []
            }

    def _match_tables(self, query: str, top_k: int) -> Tuple[List[str], List[float]]:
        """
        Match query to tables using cosine similarity.

        Args:
            query: Query text
            top_k: Number of top matches to return

        Returns:
            Tuple of (table_names, scores)
        """
        table_names = list(self.schema.keys())
        return self._cosine_match(query, table_names, top_k)

    def _match_columns(self, query: str, table: str,
                       top_k: int) -> Tuple[List[str], List[float]]:
        """
        Match query to columns of a specific table.

        Args:
            query: Query text
            table: Table name
            top_k: Number of top matches to return

        Returns:
            Tuple of (column_names, scores)
        """
        columns = self.schema[table]["columns"]
        return self._cosine_match(query, columns, top_k)

    def _cosine_match(self, query_text: str, candidates: List[str],
                      top_k: int) -> Tuple[List[str], List[float]]:
        """
        Compute cosine similarity between query and candidates.

        Args:
            query_text: Query string
            candidates: List of candidate strings (tables/columns)
            top_k: Number of top matches to return

        Returns:
            Tuple of (top_k_matches_sorted, top_k_scores_sorted)
        """
        if not candidates:
            return [], []

        try:
            # Transform query and candidates
            query_vec = self.vectorizer.transform([query_text])
            candidate_vecs = self.vectorizer.transform(candidates)

            # Compute cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vec, candidate_vecs).flatten()

            # Get top_k indices
            import numpy as np
            top_indices = np.argsort(similarities)[::-1][:top_k]

            matches = [candidates[idx] for idx in top_indices]
            scores = [float(similarities[idx]) for idx in top_indices]

            logger.debug(f"Cosine match top {top_k}: {list(zip(matches, scores))}")
            return matches, scores

        except Exception as e:
            logger.warning(f"Cosine matching failed: {e}")
            return [], []

    def _fuzzy_match(self, token: str, candidates: List[str],
                     limit: int = 3) -> List[Tuple[str, float]]:
        """
        Fuzzy match using fuzzywuzzy.

        Args:
            token: Token to match
            candidates: List of candidate strings
            limit: Number of top matches to return

        Returns:
            List of (candidate, normalized_score) tuples, score in [0.0, 1.0]
        """
        try:
            from fuzzywuzzy import process as fuzz_process
        except ImportError:
            logger.warning("fuzzywuzzy not installed, fuzzy matching disabled")
            return []

        if not candidates or not token:
            return []

        try:
            # Get matches with scores (0-100)
            matches = fuzz_process.extract(token, candidates, limit=limit)

            # Normalize to 0.0-1.0
            normalized = [(match, score / 100.0) for match, score in matches]

            logger.debug(f"Fuzzy match for '{token}': {normalized}")
            return normalized

        except Exception as e:
            logger.warning(f"Fuzzy matching error: {e}")
            return []

    def _get_join_hint(self, table_a: str, table_b: str) -> Optional[Dict[str, str]]:
        """
        Get foreign key relationship hint between two tables.

        Known relationships in sample.db:
          - orders.customer_id → customers.customer_id
          - order_items.order_id → orders.order_id
          - order_items.product_id → products.product_id
          - employees.dept_id → departments.dept_id
          - sales.customer_id → customers.customer_id

        Args:
            table_a: First table name
            table_b: Second table name

        Returns:
            Join hint dict or None if no relationship known
        """
        # Define known foreign key relationships
        relationships = [
            ("orders", "customers", "customer_id"),
            ("order_items", "orders", "order_id"),
            ("order_items", "products", "product_id"),
            ("employees", "departments", "dept_id"),
            ("sales", "customers", "customer_id")
        ]

        for from_tbl, to_tbl, column in relationships:
            if (table_a == from_tbl and table_b == to_tbl) or \
               (table_a == to_tbl and table_b == from_tbl):
                # Determine which table has the FK
                if table_a == from_tbl:
                    join_condition = f"{table_a}.{column} = {table_b}.{column}"
                else:
                    join_condition = f"{table_b}.{column} = {table_a}.{column}"

                return {
                    "from_table": from_tbl,
                    "to_table": to_tbl,
                    "on": join_condition
                }

        return None

    def get_schema_context(self, matched_tables: List[str]) -> str:
        """
        Generate schema context string for feeding into DL model encoder.

        Format: "customers: customer_id(INTEGER) first_name(TEXT) city(TEXT) | orders: ..."

        Args:
            matched_tables: List of table names to include

        Returns:
            Formatted schema string
        """
        parts = []

        for table in matched_tables:
            if table not in self.schema:
                logger.warning(f"Table '{table}' not in schema, skipping")
                continue

            columns = self.schema[table]["columns"]
            types = self.schema[table]["types"]

            # Format: col1(TYPE) col2(TYPE) ...
            col_type_pairs = [
                f"{col}({typ})" for col, typ in zip(columns, types)
            ]
            table_str = f"{table}: " + " ".join(col_type_pairs)
            parts.append(table_str)

        context = " | ".join(parts)
        logger.debug(f"Generated schema context: {context}")
        return context


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SchemaLinker — Standalone Test")
    print("=" * 70)

    db_path = "database/sample.db"

    try:
        linker = SchemaLinker(db_path)
        print(f"\n[OK] Schema loaded with {len(linker.schema)} tables")
    except Exception as e:
        print(f"\n[FAIL] Failed to initialize: {e}")
        print("Make sure database/sample.db exists")
        exit(1)

    test_cases = [
        {
            "query": "show customers from mumbai",
            "expected_tables": ["customers"],
            "description": "Simple table match"
        },
        {
            "query": "total sales amount by city",
            "expected_tables": ["sales"],
            "description": "Sales table with amounts"
        },
        {
            "query": "customer names with their orders",
            "expected_tables": ["customers", "orders"],
            "description": "Join between customers and orders"
        },
        {
            "query": "employees and their department",
            "expected_tables": ["employees", "departments"],
            "description": "Join between employees and departments"
        },
        {
            "query": "top products by order quantity",
            "expected_tables": ["products", "order_items"],
            "description": "Join between products and order_items"
        }
    ]

    all_passed = True

    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}] {test['description']}")
        print(f"     Query: {test['query']}")
        print(f"     Expected tables: {test['expected_tables']}")

        result = linker.link(test["query"], [])

        print(f"     Matched tables   : {result['matched_tables']}")
        print(f"     Confidence       : {result['confidence']}")
        print(f"     Suggested joins  : {len(result['suggested_joins'])}")

        # Check if expected tables are in matched tables (at least one overlap)
        matched_set = set(result['matched_tables'])
        expected_set = set(test['expected_tables'])

        if expected_set.issubset(matched_set) or matched_set.intersection(expected_set):
            print(f"     [PASS]")
        else:
            print(f"     [FAIL] No overlap with expected tables")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("Result: ALL TESTS PASSED")
    else:
        print("Result: SOME TESTS FAILED")
    print("=" * 70)
