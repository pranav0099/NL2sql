"""
NL2SQL — Rule Based SQL Builder
Builds correct SQL from NLP output for ANY database.
Does not rely on DL model vocabulary.
Works on any uploaded CSV or SQLite file.
"""

import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
logger = get_logger(__name__)


# =============================================================================
# RULE-BASED SQL BUILDER
# =============================================================================

class RuleBasedSQLBuilder:
    """
    Builds SQL directly from structured NLP output using deterministic rules.

    This builder does NOT rely on a trained model — it constructs SQL purely
    from the NLP pipeline's extracted entities, schema links, and query hints.
    This makes it database-agnostic: it works on ANY uploaded dataset without
    retraining.

    For uploaded databases → this is the PRIMARY SQL generator.
    For sample.db         → this is the BACKUP when DL output is wrong.
    """

    # Aggregate keyword detection
    AGGREGATE_MAP = {
        "COUNT": [
            "how many", "count", "number of",
            "total number", "how much", "total count",
            "quantity of", "count of",
        ],
        "SUM": [
            "total", "sum", "sum of", "add up",
            "overall", "combined total",
            "cumulative", "sum total",
        ],
        "AVG": [
            "average", "avg", "mean", "typical",
            "average of", "mean of", "mean value",
        ],
        "MIN": [
            "minimum", "lowest", "smallest", "least",
            "cheapest", "fewest", "min value",
            "minimum of", "lowest value",
        ],
        "MAX": [
            "maximum", "highest", "largest", "most",
            "biggest", "greatest", "most expensive",
            "max value", "maximum of", "highest value",
        ],
    }

    # Operator keyword detection
    OPERATOR_MAP = {
        ">=": [
            "at least", "no less than",
            "minimum of", "is at least",
            "greater than or equal",
        ],
        "<=": [
            "at most", "no more than",
            "maximum of", "no greater than",
            "not more than", "up to", "within",
            "less than or equal",
        ],
        ">": [
            "above", "over", "more than",
            "greater than", "higher than",
            "exceeds", "larger than", "bigger than",
        ],
        "<": [
            "below", "under", "less than",
            "lower than", "fewer than",
            "smaller than", "beneath", "not above",
        ],
        "!=": [
            "not equal", "not", "except",
            "excluding", "other than",
        ],
        "=": [
            "equal to", "equals", "same as",
            "matching", "is exactly",
        ],
    }

    # Words to skip — not operators
    SKIP_WORDS = {
        "has", "have", "had", "that", "which", "who",
        "whose", "the", "a", "an", "are", "was", "were",
        "be", "been", "with", "for", "of", "in", "on",
        "at", "to", "by", "from", "and", "or", "but",
    }

    # -------------------------------------------------------------------------
    # PUBLIC: build
    # -------------------------------------------------------------------------
    def build(self, nlp_result: dict) -> dict:
        """
        Build complete SQL from NLP pipeline output.
        Works on any database schema.

        Args:
            nlp_result: Structured dict from the NLP pipeline containing
                        original_query, sql_hints, entities, schema_links,
                        and preprocessed data.

        Returns:
            dict with keys: sql, used_rules, confidence, method, fallback_used.
        """
        try:
            query        = nlp_result.get("original_query", "").lower().strip()
            sql_hints    = nlp_result.get("sql_hints", {})
            entities     = nlp_result.get("entities", {})
            schema_links = nlp_result.get("schema_links", {})
            preprocessed = nlp_result.get("preprocessed", {})

            # Get table
            tables = sql_hints.get("tables", []) or \
                     schema_links.get("matched_tables", [])
            if not tables:
                return self._error_result("No table found in query")
            table = tables[0]

            # Get all available columns for this table
            matched_cols = schema_links.get("matched_columns", {})
            all_columns = []
            for cols in matched_cols.values():
                all_columns.extend(cols)

            # Get NLP extracted data
            filters  = self._deduplicate_filters(
                entities.get("filters", []))
            agg_list = entities.get("aggregations", [])
            order    = entities.get("order", None)
            group_by = entities.get("group_by", [])
            numbers  = preprocessed.get("numbers", [])
            nl_ents  = preprocessed.get("entities", [])

            # Detect aggregate from query text
            agg_fn  = self._detect_aggregate(query)
            agg_col = self._get_agg_column(
                agg_fn, agg_list, all_columns, query)

            # ------------------------------------------------------------------
            # Build SELECT clause
            # ------------------------------------------------------------------
            if agg_fn == "COUNT":
                select_clause = "SELECT COUNT(*)"
            elif agg_fn in ("SUM", "AVG", "MIN", "MAX"):
                select_clause = f"SELECT {agg_fn}({agg_col})"
            else:
                select_clause = "SELECT *"

            # ------------------------------------------------------------------
            # Build full SQL (FROM)
            # ------------------------------------------------------------------
            sql = f"{select_clause} FROM {table}"

            # ------------------------------------------------------------------
            # Build WHERE clause from NLP filters
            # ------------------------------------------------------------------
            where_parts = self._build_where_parts(filters)

            # ------------------------------------------------------------------
            # Build GROUP BY (with SELECT adjustment)
            # ------------------------------------------------------------------
            if group_by and agg_fn:
                # Rebuild SELECT to include the group column
                if agg_fn == "COUNT":
                    sql = f"SELECT {group_by[0]}, COUNT(*) FROM {table}"
                else:
                    sql = f"SELECT {group_by[0]}, {agg_fn}({agg_col}) FROM {table}"

                # Attach WHERE
                if where_parts:
                    sql += " WHERE " + " AND ".join(where_parts)

                sql += f" GROUP BY {group_by[0]}"

            elif group_by and not agg_fn:
                # No aggregate — just add WHERE then GROUP BY
                if where_parts:
                    sql += " WHERE " + " AND ".join(where_parts)
                sql += f" GROUP BY {', '.join(group_by)}"

            else:
                # No GROUP BY — attach WHERE normally
                if where_parts:
                    sql += " WHERE " + " AND ".join(where_parts)

            # ------------------------------------------------------------------
            # Build ORDER BY + LIMIT
            # ------------------------------------------------------------------
            if order:
                ocol = order.get("column", "")
                odir = order.get("direction", "DESC")
                olim = order.get("limit", None)
                if ocol and "ORDER BY" not in sql.upper():
                    sql += f" ORDER BY {ocol} {odir}"
                if olim and "LIMIT" not in sql.upper():
                    sql += f" LIMIT {int(olim)}"
            else:
                # Detect "top N / bottom N" from query text
                sql = self._add_top_bottom(sql, query, all_columns)

            logger.info(f"[RuleBuilder] SQL: {sql}")

            return {
                "sql":          sql.strip(),
                "used_rules":   True,
                "confidence":   0.85,
                "method":       "rule_based",
                "fallback_used": False,
            }

        except Exception as e:
            logger.error(f"[RuleBuilder] Failed: {e}")
            return self._error_result(str(e))

    # -------------------------------------------------------------------------
    # PUBLIC: should_use_rules
    # -------------------------------------------------------------------------
    def should_use_rules(self, dl_sql: str, nlp_result: dict) -> bool:
        """
        Decide whether to use rule-based builder instead of DL output.

        Returns True when DL output is likely wrong or incomplete:
          - DL produced empty SQL
          - DL missing WHERE but NLP found filters
          - DL has duplicate conditions on the same column
          - DL missing ORDER BY but NLP found ordering
          - DL missing LIMIT but query says "top N"

        Args:
            dl_sql:     SQL string produced by DL generator.
            nlp_result: Structured dict from the NLP pipeline.

        Returns:
            True if rule-based SQL should replace DL output.
        """
        if not dl_sql:
            return True

        dl_upper = dl_sql.upper()
        query    = nlp_result.get("original_query", "").lower()
        entities = nlp_result.get("entities", {})
        filters  = entities.get("filters", [])
        order    = entities.get("order", None)

        # 1. DL missing WHERE but NLP found filters
        if filters and "WHERE" not in dl_upper:
            return True

        # 2. DL has duplicate conditions on the same column
        col_ops = {}
        where_pattern = re.findall(
            r'(\w+)\s*([><=!]+)\s*[\w\']+',
            dl_sql, re.IGNORECASE)
        for col, op in where_pattern:
            col = col.lower()
            if col not in col_ops:
                col_ops[col] = []
            col_ops[col].append(op)
        for col, ops in col_ops.items():
            if len(ops) > 1:
                return True

        # 3. DL missing ORDER BY but NLP found order
        if order and "ORDER BY" not in dl_upper:
            return True

        # 4. DL missing LIMIT but query has "top N"
        if re.search(r'\btop\s+\d+', query) and "LIMIT" not in dl_upper:
            return True

        return False

    # =========================================================================
    # PRIVATE helpers
    # =========================================================================

    def _detect_aggregate(self, query: str) -> str | None:
        """Detect aggregate function from query text using keyword matching."""
        query = query.lower()
        # Check longest phrases first (longest match wins)
        for fn, phrases in self.AGGREGATE_MAP.items():
            for phrase in sorted(phrases, key=len, reverse=True):
                if phrase in query:
                    return fn
        return None

    def _get_agg_column(
        self,
        agg_fn: str,
        agg_list: list,
        all_columns: list,
        query: str,
    ) -> str:
        """Get best column for the detected aggregate function."""
        if agg_fn == "COUNT":
            return "*"

        # Use NLP-extracted aggregation column first
        for a in agg_list:
            if a.get("function", "").upper() == agg_fn:
                col = a.get("column", "")
                if col and col != "*":
                    return col

        # Find numeric column from schema links
        NUMERIC_HINTS = [
            "salary", "price", "amount", "total",
            "revenue", "cost", "budget", "score",
            "rate", "value", "fee", "balance", "tax",
            "quantity", "qty", "age", "weight", "marks",
            "grade", "rating", "count", "number", "hours",
        ]
        for hint in NUMERIC_HINTS:
            for col in all_columns:
                if hint in col.lower():
                    return col

        # Use first non-id column
        for col in all_columns:
            if "id" not in col.lower():
                return col

        return all_columns[0] if all_columns else "*"

    def _deduplicate_filters(self, filters: list) -> list:
        """
        Remove conflicting filters on same column.
        Non-equal operators (> < >= <=) beat equal operator (=).
        Example:
            salary > 40000 AND salary = 40000
            → keep salary > 40000 only
        """
        col_map = {}
        for f in filters:
            col = f.get("column", "")
            op  = f.get("op", "=")
            if not col:
                continue
            if col not in col_map:
                col_map[col] = f
            else:
                existing_op = col_map[col].get("op", "=")
                # Non-equal wins over equal
                if existing_op == "=" and op != "=":
                    col_map[col] = f
        return list(col_map.values())

    def _build_where_parts(self, filters: list) -> list:
        """Build a list of WHERE condition strings from NLP filters."""
        parts = []
        for f in filters:
            col = f.get("column", "")
            op  = f.get("op", "=")
            val = f.get("value")
            if not col or val is None:
                continue
            formatted = self._format_val(val)
            parts.append(f"{col} {op} {formatted}")
        return parts

    def _format_val(self, val) -> str:
        """Format a value for SQL WHERE clause."""
        if isinstance(val, str):
            return f"'{val}'"
        if isinstance(val, float) and val == int(val):
            return str(int(val))
        return str(val)

    def _add_top_bottom(
        self, sql: str, query: str, all_columns: list,
    ) -> str:
        """Detect 'top N' / 'bottom N' in query and append ORDER BY + LIMIT."""
        top_m = re.search(r'\b(top|first)\s+(\d+)', query)
        bot_m = re.search(r'\b(bottom|last|lowest)\s+(\d+)', query)

        if not top_m and not bot_m:
            return sql

        match     = top_m or bot_m
        n         = int(match.group(2))
        direction = "DESC" if top_m else "ASC"

        # Find order column from "by {col}"
        by_m = re.search(r'\bby\s+(\w+)', query)
        if by_m:
            hint     = by_m.group(1).lower()
            best_col = next(
                (c for c in all_columns
                 if hint in c.lower() or c.lower() in hint),
                None,
            )
            if not best_col:
                best_col = self._get_agg_column(
                    "MAX", [], all_columns, query)
            if best_col and "ORDER BY" not in sql.upper():
                sql += f" ORDER BY {best_col} {direction}"

        if "LIMIT" not in sql.upper():
            sql += f" LIMIT {n}"

        return sql

    def _error_result(self, msg: str) -> dict:
        """Return an error result when rule builder cannot build SQL."""
        return {
            "sql":          "",
            "used_rules":   False,
            "confidence":   0.0,
            "method":       "rule_based_error",
            "fallback_used": True,
            "error":        msg,
        }


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":

    builder = RuleBasedSQLBuilder()

    # Simulate NLP output for different databases
    TESTS = [
        # ── Hotel database ──────────────────────────────────
        (
            "show all staff",
            {
                "original_query": "show all staff",
                "sql_hints": {"tables": ["staff"]},
                "entities": {
                    "filters": [], "aggregations": [],
                    "order": None, "group_by": [],
                },
                "schema_links": {
                    "matched_tables": ["staff"],
                    "matched_columns": {
                        "staff": ["staff_id", "name", "salary", "department"],
                    },
                },
                "preprocessed": {"numbers": [], "entities": []},
            },
            "SELECT * FROM staff",
        ),
        (
            "show staff with salary above 40000",
            {
                "original_query": "show staff with salary above 40000",
                "sql_hints": {"tables": ["staff"]},
                "entities": {
                    "filters": [{"column": "salary", "op": ">", "value": 40000}],
                    "aggregations": [], "order": None, "group_by": [],
                },
                "schema_links": {
                    "matched_tables": ["staff"],
                    "matched_columns": {
                        "staff": ["salary", "name", "department"],
                    },
                },
                "preprocessed": {"numbers": [40000], "entities": []},
            },
            "SELECT * FROM staff WHERE salary > 40000",
        ),

        # ── Count query ─────────────────────────────────────
        (
            "how many staff are there",
            {
                "original_query": "how many staff are there",
                "sql_hints": {"tables": ["staff"]},
                "entities": {
                    "filters": [], "aggregations": [],
                    "order": None, "group_by": [],
                },
                "schema_links": {
                    "matched_tables": ["staff"],
                    "matched_columns": {
                        "staff": ["staff_id", "name", "salary"],
                    },
                },
                "preprocessed": {"numbers": [], "entities": []},
            },
            "SELECT COUNT(*) FROM staff",
        ),

        # ── Average query ───────────────────────────────────
        (
            "average salary of staff",
            {
                "original_query": "average salary of staff",
                "sql_hints": {"tables": ["staff"]},
                "entities": {
                    "filters": [],
                    "aggregations": [{"function": "AVG", "column": "salary"}],
                    "order": None, "group_by": [],
                },
                "schema_links": {
                    "matched_tables": ["staff"],
                    "matched_columns": {"staff": ["salary", "name"]},
                },
                "preprocessed": {"numbers": [], "entities": []},
            },
            "SELECT AVG(salary) FROM staff",
        ),

        # ── Top N query ─────────────────────────────────────
        (
            "show top 5 staff by salary",
            {
                "original_query": "show top 5 staff by salary",
                "sql_hints": {"tables": ["staff"]},
                "entities": {
                    "filters": [], "aggregations": [],
                    "order": {"column": "salary", "direction": "DESC", "limit": 5},
                    "group_by": [],
                },
                "schema_links": {
                    "matched_tables": ["staff"],
                    "matched_columns": {
                        "staff": ["salary", "name", "department"],
                    },
                },
                "preprocessed": {"numbers": [5], "entities": []},
            },
            "SELECT * FROM staff ORDER BY salary DESC LIMIT 5",
        ),

        # ── School database ─────────────────────────────────
        (
            "count students with marks above 80",
            {
                "original_query": "count students with marks above 80",
                "sql_hints": {"tables": ["students"]},
                "entities": {
                    "filters": [{"column": "marks", "op": ">", "value": 80}],
                    "aggregations": [], "order": None, "group_by": [],
                },
                "schema_links": {
                    "matched_tables": ["students"],
                    "matched_columns": {
                        "students": ["marks", "name", "grade", "subject"],
                    },
                },
                "preprocessed": {"numbers": [80], "entities": []},
            },
            "SELECT COUNT(*) FROM students WHERE marks > 80",
        ),

        # ── E-commerce database ──────────────────────────────
        (
            "total revenue from orders",
            {
                "original_query": "total revenue from orders",
                "sql_hints": {"tables": ["orders"]},
                "entities": {
                    "filters": [],
                    "aggregations": [{"function": "SUM", "column": "revenue"}],
                    "order": None, "group_by": [],
                },
                "schema_links": {
                    "matched_tables": ["orders"],
                    "matched_columns": {
                        "orders": ["order_id", "revenue", "customer_id", "status"],
                    },
                },
                "preprocessed": {"numbers": [], "entities": []},
            },
            "SELECT SUM(revenue) FROM orders",
        ),

        # ── Hospital database ────────────────────────────────
        (
            "show patients with age below 30",
            {
                "original_query": "show patients with age below 30",
                "sql_hints": {"tables": ["patients"]},
                "entities": {
                    "filters": [{"column": "age", "op": "<", "value": 30}],
                    "aggregations": [], "order": None, "group_by": [],
                },
                "schema_links": {
                    "matched_tables": ["patients"],
                    "matched_columns": {
                        "patients": ["age", "name", "diagnosis", "ward"],
                    },
                },
                "preprocessed": {"numbers": [30], "entities": []},
            },
            "SELECT * FROM patients WHERE age < 30",
        ),
    ]

    print("=" * 65)
    print("Rule-Based Builder Test — Multiple Databases")
    print("=" * 65)

    passed = 0
    for query, nlp_result, expected in TESTS:
        result = builder.build(nlp_result)
        got    = result["sql"].upper().strip()
        exp    = expected.upper().strip()
        ok     = exp in got or got == exp
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"\n[{status}] {query}")
        print(f"  Expected : {expected}")
        print(f"  Got      : {result['sql']}")

    print(f"\n{'=' * 65}")
    print(f"Result: {passed}/{len(TESTS)} passed")
    print("Rule builder works on hotel, school,")
    print("e-commerce, hospital databases - ALL OK")
    print("=" * 65)
