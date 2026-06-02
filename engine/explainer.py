"""
NL2SQL — Phase 6: SQL Explanation Module
Run: python engine/explainer.py

Converts generated SQL into plain English explanations using
regex-based pattern matching.  Also produces result summaries,
follow-up query suggestions, and clarification messages for
low-confidence predictions.

Pure Python — no external dependencies.
"""

import sys
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

import re

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SQL EXPLAINER CLASS
# =============================================================================

class SQLExplainer:
    """
    Translates SQL queries into human-readable summaries, generates
    result summaries, follow-up suggestions, and clarification prompts.

    Entirely stateless — no database access, no ML models.
    All logic is pure regex-based string pattern matching.

    Public methods:
        explain(sql, result)            → dict with explanation / summary / suggestions
        generate_clarification(…)       → clarification message string
    """

    def __init__(self):
        """
        Initialize SQLExplainer.

        No external dependencies are required.
        The class is purely functional and stateless.
        """
        logger.info("SQLExplainer initialized")

    # -------------------------------------------------------------------------
    # PUBLIC: explain
    # -------------------------------------------------------------------------
    def explain(self, sql: str, result: dict = None) -> dict:
        """
        Produce a complete explanation package for a SQL query.

        Args:
            sql:    SQL query string to explain.
            result: Optional execution result dict from SQLExecutor.
                    Expected keys: success, row_count, truncated, error.

        Returns:
            dict:
                explanation (str)  — Plain-English description of the query.
                summary     (str)  — One-line result summary.
                suggestions (list) — Up to 3 follow-up query suggestions.
        """
        explanation = self._explain_sql(sql)
        summary     = self._make_summary(result) if result else "No execution result available"
        suggestions = self._make_suggestions(sql, result)

        return {
            "explanation": explanation,
            "summary":     summary,
            "suggestions": suggestions,
        }

    # -------------------------------------------------------------------------
    # PRIVATE: _explain_sql
    # -------------------------------------------------------------------------
    def _explain_sql(self, sql: str) -> str:
        """
        Convert a SQL query to plain English via ordered regex matching.

        Handles these patterns (checked in this order):
            SELECT COUNT(*) FROM {table}
            SELECT AVG({col}) FROM {table}
            SELECT SUM({col}) FROM {table}
            SELECT MAX({col}) FROM {table}
            SELECT MIN({col}) FROM {table}
            SELECT … ORDER BY {col} DESC LIMIT {n}
            SELECT … ORDER BY {col} ASC  LIMIT {n}
            SELECT … GROUP BY {col}
            SELECT … JOIN …
            SELECT * FROM {table} WHERE {col} = '{val}'
            SELECT * FROM {table} WHERE {col} > {val}
            SELECT * FROM {table} WHERE {col} < {val}
            SELECT * FROM {table}

        Falls back to "Querying the database" when nothing matches.

        Args:
            sql: SQL query string.

        Returns:
            Plain English explanation string.
        """
        sql_clean = sql.strip()
        sql_upper = sql_clean.upper()

        # -- Aggregate + WHERE patterns (most specific — checked FIRST) --------

        # SELECT COUNT(*) FROM {table} WHERE {col} = {val}
        match = re.match(
            r"SELECT\s+COUNT\([^)]*\)\s+FROM\s+(\w+)"
            r"\s+WHERE\s+(\w+)\s*([=><]+)\s*(.+)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            table = match.group(1)
            col   = match.group(2)
            op    = match.group(3)
            val   = match.group(4).strip()
            # Strip surrounding quotes if present
            if val.startswith("'") and val.endswith("'") and len(val) >= 2:
                val = val[1:-1]
            op_words = {
                '=': 'is', '!=': 'is not',
                '>': 'greater than', '<': 'less than',
                '>=': 'at least', '<=': 'at most'
            }
            op_word = op_words.get(op, op)
            return f"Counting records in {table} where {col} {op_word} {val}"

        # SELECT SUM/AVG/MIN/MAX({col}) FROM {table} WHERE {col2} = {val}
        match = re.match(
            r"SELECT\s+(SUM|AVG|MIN|MAX)\((\w+)\)\s+FROM\s+(\w+)"
            r"\s+WHERE\s+(\w+)\s*([=><]+)\s*(.+)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            fn    = match.group(1).upper()
            col   = match.group(2)
            table = match.group(3)
            wcol  = match.group(4)
            op    = match.group(5)
            val   = match.group(6).strip()
            # Strip surrounding quotes if present
            if val.startswith("'") and val.endswith("'") and len(val) >= 2:
                val = val[1:-1]
            fn_words = {
                'SUM': 'Total', 'AVG': 'Average',
                'MIN': 'Minimum', 'MAX': 'Maximum'
            }
            return f"{fn_words.get(fn, fn)} {col} from {table} where {wcol} is {val}"

        # -- Simple aggregate patterns (no WHERE) -----------------------------

        # SELECT COUNT(*) FROM {table}
        match = re.match(
            r"SELECT\s+COUNT\(\*\)\s+FROM\s+(\w+)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            table = match.group(1)
            return f"Counting total records in {table}"

        # SELECT AVG({col}) FROM {table}
        match = re.match(
            r"SELECT\s+AVG\((\w+)\)\s+FROM\s+(\w+)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            col, table = match.groups()
            return f"Calculating average {col} from {table}"

        # SELECT SUM({col}) FROM {table}
        match = re.match(
            r"SELECT\s+SUM\((\w+)\)\s+FROM\s+(\w+)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            col, table = match.groups()
            return f"Calculating total {col} from {table}"

        # SELECT MAX({col}) FROM {table}
        match = re.match(
            r"SELECT\s+MAX\((\w+)\)\s+FROM\s+(\w+)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            col, table = match.groups()
            return f"Finding maximum {col} from {table}"

        # SELECT MIN({col}) FROM {table}
        match = re.match(
            r"SELECT\s+MIN\((\w+)\)\s+FROM\s+(\w+)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            col, table = match.groups()
            return f"Finding minimum {col} from {table}"

        # -- ORDER BY + LIMIT patterns ----------------------------------------

        # SELECT … ORDER BY {col} DESC LIMIT {n}
        match = re.search(
            r"ORDER\s+BY\s+(\w+)\s+DESC\s+LIMIT\s+(\d+)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            col, n = match.groups()
            return f"Showing top {n} records by {col}"

        # SELECT … ORDER BY {col} ASC LIMIT {n}
        match = re.search(
            r"ORDER\s+BY\s+(\w+)\s+ASC\s+LIMIT\s+(\d+)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            col, n = match.groups()
            return f"Showing bottom {n} records by {col}"

        # -- GROUP BY pattern --------------------------------------------------

        # SELECT … GROUP BY {col}
        match = re.search(
            r"GROUP\s+BY\s+(\w+)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            col = match.group(1)
            return f"Grouping results by {col}"

        # -- JOIN pattern ------------------------------------------------------

        if re.search(r"\bJOIN\b", sql_upper):
            return "Combining data from multiple tables"

        # -- WHERE patterns (SELECT *) -----------------------------------------

        # SELECT * FROM {table} WHERE {col} = '{val}'
        match = re.match(
            r"SELECT\s+\*\s+FROM\s+(\w+)\s+WHERE\s+(\w+)\s*=\s*'([^']+)'",
            sql_clean, re.IGNORECASE,
        )
        if match:
            table, col, val = match.groups()
            return f"Showing all {table} where {col} is {val}"

        # SELECT * FROM {table} WHERE {col} > {val}
        match = re.match(
            r"SELECT\s+\*\s+FROM\s+(\w+)\s+WHERE\s+(\w+)\s*>\s*(\d+\.?\d*)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            table, col, val = match.groups()
            return f"Showing all {table} where {col} is greater than {val}"

        # SELECT * FROM {table} WHERE {col} < {val}
        match = re.match(
            r"SELECT\s+\*\s+FROM\s+(\w+)\s+WHERE\s+(\w+)\s*<\s*(\d+\.?\d*)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            table, col, val = match.groups()
            return f"Showing all {table} where {col} is less than {val}"

        # -- Bare SELECT * FROM {table} [LIMIT n] ---------------------------------

        match = re.match(
            r"SELECT\s+\*\s+FROM\s+(\[?[\w\s]+\]?)\s+LIMIT\s+(\d+)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            table = match.group(1).strip("[]")
            n = match.group(2)
            return f"Showing all records from {table}, {n}"

        match = re.match(
            r"SELECT\s+\*\s+FROM\s+(\[?[\w\s]+\]?)",
            sql_clean, re.IGNORECASE,
        )
        if match:
            table = match.group(1).strip("[]")
            return f"Showing all records from {table}"

        # -- Default fallback --------------------------------------------------
        return "Querying the database"

    # -------------------------------------------------------------------------
    # PRIVATE: _make_summary
    # -------------------------------------------------------------------------
    def _make_summary(self, result: dict) -> str:
        """
        Generate a one-line result summary from the execution result dict.

        Rules:
            • success=False   → "Query failed: {error}"
            • row_count == 0  → "No records found"
            • row_count == 1  → "1 record found"
            • row_count > 1   → "{n} records found"
            • truncated=True  → append "(showing first 1000)"

        Args:
            result: Execution result dict from SQLExecutor.

        Returns:
            Summary string.
        """
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            return f"Query failed: {error}"

        row_count = result.get("row_count", 0)
        truncated = result.get("truncated", False)

        if row_count == 0:
            return "No records found"
        if row_count == 1:
            return "1 record found"

        summary = f"{row_count} records found"
        if truncated:
            summary += " (showing first 1000)"
        return summary

    # -------------------------------------------------------------------------
    # PRIVATE: _make_suggestions
    # -------------------------------------------------------------------------
    def _make_suggestions(self, sql: str, result: dict) -> list:
        """
        Generate up to 3 follow-up query suggestions based on the current SQL.

        Suggestion rules (applied in order, capped at 3):
            • SELECT * with no GROUP BY → suggest COUNT, suggest ORDER BY
            • SELECT with WHERE         → suggest removing filter, suggest adding ORDER BY
            • COUNT query               → suggest showing full records instead
            • GROUP BY                  → suggest ORDER BY on the aggregation

        Args:
            sql:    SQL query string.
            result: Execution result dict from SQLExecutor (may be None).

        Returns:
            List of suggestion strings (max 3).
        """
        suggestions: list = []
        sql_upper = sql.upper()

        # Rule 1: SELECT * with no GROUP BY
        if (
            re.match(r"SELECT\s+\*\s+FROM", sql, re.IGNORECASE)
            and "GROUP BY" not in sql_upper
        ):
            table_match = re.match(r"SELECT\s+\*\s+FROM\s+(\w+)", sql, re.IGNORECASE)
            if table_match:
                table = table_match.group(1)
                suggestions.append(f"Try: Count records in {table}")
                suggestions.append(f"Try: Show {table} ordered by a column")

        # Rule 2: SELECT with WHERE
        if "WHERE" in sql_upper and len(suggestions) < 3:
            table_match = re.match(
                r"SELECT\s+.+?\s+FROM\s+(\w+)\s+WHERE",
                sql, re.IGNORECASE,
            )
            if table_match:
                table = table_match.group(1)
                if len(suggestions) < 3:
                    suggestions.append(f"Try: Show all records from {table} (remove filter)")
                if len(suggestions) < 3:
                    suggestions.append(f"Try: Order {table} results by a column")

        # Rule 3: COUNT(*) query
        if (
            re.match(r"SELECT\s+COUNT\(\*\)", sql, re.IGNORECASE)
            and len(suggestions) < 3
        ):
            table_match = re.match(
                r"SELECT\s+COUNT\(\*\)\s+FROM\s+(\w+)",
                sql, re.IGNORECASE,
            )
            if table_match:
                table = table_match.group(1)
                suggestions.append(f"Try: Show full records from {table} instead of count")

        # Rule 4: GROUP BY query
        if "GROUP BY" in sql_upper and len(suggestions) < 3:
            gb_match = re.search(r"GROUP\s+BY\s+(\w+)", sql, re.IGNORECASE)
            if gb_match:
                col = gb_match.group(1)
                suggestions.append(f"Try: Order grouped results by {col}")

        # Enforce hard cap of 3 suggestions
        return suggestions[:3]

    # -------------------------------------------------------------------------
    # PUBLIC: generate_clarification
    # -------------------------------------------------------------------------
    def generate_clarification(
        self,
        question: str,
        confidence: float,
        intent: str,
    ) -> str:
        """
        Generate a helpful clarification message when the system's
        confidence in the user's intent is below threshold.

        Args:
            question:   Original user question string.
            confidence: Confidence score (0.0–1.0).
            intent:     Predicted intent label.

        Returns:
            Formatted clarification message string.
        """
        conf_pct = int(confidence * 100)
        return (
            f"I'm not fully sure what you mean (confidence: {conf_pct}%).\n"
            "Did you mean:\n"
            "• To filter records: 'Show customers where city is Mumbai'\n"
            "• To count records: 'How many customers are in Mumbai'\n"
            "• To sort records: 'Show customers ordered by name'\n"
            "Please rephrase your question."
        )


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    explainer = SQLExplainer()

    test_sqls = [
        "SELECT * FROM customers WHERE city = 'Mumbai'",
        "SELECT AVG(salary) FROM employees",
        "SELECT city, COUNT(*) FROM orders GROUP BY city",
        "SELECT * FROM products ORDER BY price DESC LIMIT 5",
        "SELECT * FROM customers JOIN orders ON customers.customer_id = orders.customer_id",
        "SELECT COUNT(*) FROM customers",
    ]

    for sql in test_sqls:
        mock_result = {
            "success":   True,
            "row_count": 42,
            "truncated": False,
            "error":     None,
        }
        result = explainer.explain(sql, mock_result)
        print(f"SQL  : {sql[:60]}")
        print(f"Expl : {result['explanation']}")
        print(f"Sum  : {result['summary']}")
        print(f"Sugg : {result['suggestions']}")
        print()
