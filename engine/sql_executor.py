"""
NL2SQL — Phase 6: SQL Execution Module
Run: python engine/sql_executor.py

Validates and executes SQL queries against a SQLite database.
Returns structured result dictionaries with rows, columns,
timing information, and user-friendly error classification.

All connections use context managers with a 30-second timeout.
Dangerous DDL/DML statements (DROP TABLE, TRUNCATE, etc.) are
blocked at validation time so they never reach the database.
"""

import sys
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

import sqlite3
import re
import time

from config.config import SAMPLE_DB
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SQL EXECUTOR CLASS
# =============================================================================

class SQLExecutor:
    """
    Validates and executes SQL queries on a SQLite database,
    returning structured result dictionaries.

    Features:
        - Pre-execution validation (empty check, keyword check, danger check)
        - Context-managed SQLite connections with 30s timeout
        - Automatic row truncation at 1 000 rows
        - Millisecond-precision timing via time.perf_counter()
        - User-friendly error classification (table / column / syntax / empty)
        - Schema introspection via get_schema()
        - Health check via test_connection()

    Attributes:
        db_path (Path): Resolved path to the SQLite database file.
    """

    # Maximum number of rows returned in a single query
    _MAX_ROWS = 1000

    def __init__(self, db_path: str = SAMPLE_DB):
        """
        Initialize SQLExecutor with the database path.

        Args:
            db_path: Path to the SQLite database file.
                     Defaults to SAMPLE_DB from config.config.

        Side-effects:
            - Logs an error if the database file does not exist.
            - Logs initialization message with the resolved path.
        """
        self.db_path = Path(db_path)

        # Verify that the database file exists on disk
        if not self.db_path.exists():
            logger.error(f"Database file not found at: {self.db_path}")

        logger.info(f"SQLExecutor initialized: {self.db_path}")

    # -------------------------------------------------------------------------
    # PUBLIC: execute
    # -------------------------------------------------------------------------
    def execute(self, sql: str) -> dict:
        """
        Execute a SQL query against the SQLite database.

        Pipeline:
            1. Reject empty / whitespace-only queries.
            2. Run _validate_sql() for keyword and safety checks.
            3. Open a context-managed connection (timeout=30s).
            4. Execute the query and fetch up to 1 001 rows.
            5. Truncate to 1 000 if necessary, setting truncated=True.
            6. Extract column names from cursor.description.
            7. Measure wall-clock execution time in milliseconds.
            8. On any SQLite error, classify it with _classify_error().

        Args:
            sql: SQL query string to execute.

        Returns:
            dict with keys:
                success      (bool)  — True if query executed without error.
                sql          (str)   — The original SQL string.
                rows         (list)  — List of row tuples (max 1 000).
                columns      (list)  — Column name strings from cursor.description.
                row_count    (int)   — Number of rows returned (after truncation).
                truncated    (bool)  — True if result was capped at 1 000 rows.
                error        (str|None) — Raw error message, or None on success.
                error_type   (str|None) — Classified error type, or None.
                execution_ms (float) — Query execution time in milliseconds.
        """
        start_total = time.perf_counter()

        # Base result dict — always returned, even on failure
        result = {
            "success":      False,
            "sql":          sql,
            "rows":         [],
            "columns":      [],
            "row_count":    0,
            "truncated":    False,
            "error":        None,
            "error_type":   None,
            "execution_ms": 0.0,
        }

        # ------------------------------------------------------------------
        # Step 1 — Reject empty / whitespace-only input
        # ------------------------------------------------------------------
        if not sql or not sql.strip():
            result["error"]        = "SQL query is empty"
            result["error_type"]   = "syntax"
            result["execution_ms"] = (time.perf_counter() - start_total) * 1000
            logger.warning("SQL execution skipped: empty query")
            return result

        # ------------------------------------------------------------------
        # Step 2 — Validate SQL (keyword presence + danger patterns)
        # ------------------------------------------------------------------
        is_valid, validation_msg = self._validate_sql(sql)
        if not is_valid:
            result["error"]        = validation_msg
            result["error_type"]   = "syntax"
            result["execution_ms"] = (time.perf_counter() - start_total) * 1000
            logger.warning(f"SQL validation failed: {validation_msg}")
            return result

        # ------------------------------------------------------------------
        # Steps 3-7 — Connect, execute, fetch, truncate, measure
        # ------------------------------------------------------------------
        try:
            with sqlite3.connect(str(self.db_path), timeout=30) as conn:
                cursor = conn.cursor()

                # Time only the actual query execution
                exec_start = time.perf_counter()
                cursor.execute(sql)
                exec_elapsed_ms = (time.perf_counter() - exec_start) * 1000

                # Fetch one extra row to detect truncation
                rows = cursor.fetchmany(self._MAX_ROWS + 1)

                if len(rows) > self._MAX_ROWS:
                    result["truncated"] = True
                    rows = rows[: self._MAX_ROWS]

                result["rows"] = rows

                # Column names from cursor.description (None for non-SELECT)
                if cursor.description:
                    result["columns"] = [desc[0] for desc in cursor.description]

                result["row_count"]    = len(rows)
                result["success"]      = True
                result["execution_ms"] = exec_elapsed_ms

                logger.info(
                    f"SQL executed in {exec_elapsed_ms:.2f}ms — "
                    f"{result['row_count']} row(s) returned"
                )

        except sqlite3.OperationalError as e:
            raw_error = str(e)
            error_type = self._classify_error(raw_error)
            result["error"]        = self._format_error_msg(error_type, sql, raw_error)
            result["error_type"]   = error_type
            result["execution_ms"] = (time.perf_counter() - start_total) * 1000
            logger.error(f"SQLite OperationalError: {raw_error}")

        except Exception as e:
            result["error"]        = self._format_error_msg("unknown", sql, str(e))
            result["error_type"]   = "unknown"
            result["execution_ms"] = (time.perf_counter() - start_total) * 1000
            logger.error(f"Unexpected error during SQL execution: {e}")

        return result

    # -------------------------------------------------------------------------
    # PRIVATE: _validate_sql
    # -------------------------------------------------------------------------
    def _validate_sql(self, sql: str) -> tuple:
        """
        Run basic safety and syntax checks before executing SQL.

        Checks performed (in order):
            1. Query is not empty / whitespace-only.
            2. Contains at least one recognised SQL keyword
               (SELECT, INSERT, UPDATE, DELETE, CREATE, DROP).
            3. Does not contain dangerous patterns:
               DROP TABLE, DROP DATABASE, TRUNCATE, ALTER TABLE,
               DELETE FROM without WHERE.

        Args:
            sql: SQL query string to validate.

        Returns:
            (is_valid, error_message) tuple.
            is_valid is True when all checks pass; error_message is "".
        """
        sql_stripped = sql.strip()

        # Check 1 — non-empty
        if not sql_stripped:
            return (False, "SQL query is empty or whitespace only")

        # Check 2 — must contain at least one valid keyword
        valid_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP"]
        has_keyword = any(
            re.search(rf"\b{kw}\b", sql_stripped, re.IGNORECASE)
            for kw in valid_keywords
        )
        if not has_keyword:
            return (
                False,
                "SQL must contain at least one valid keyword: "
                "SELECT, INSERT, UPDATE, DELETE, CREATE, or DROP",
            )

        # Check 3 — block dangerous patterns
        sql_upper = sql_stripped.upper()

        dangerous = {
            "DROP TABLE":   "DROP TABLE is not allowed",
            "DROP DATABASE": "DROP DATABASE is not allowed",
            "TRUNCATE":     "TRUNCATE is not allowed",
            "ALTER TABLE":  "ALTER TABLE is not allowed",
        }
        for pattern, message in dangerous.items():
            if pattern in sql_upper:
                return (False, message)

        # DELETE FROM without WHERE is a data-safety violation
        if re.search(r"\bDELETE\s+FROM\b", sql_upper):
            if not re.search(r"\bWHERE\b", sql_upper):
                return (False, "DELETE FROM without WHERE clause is not allowed (data safety)")

        return (True, "")

    # -------------------------------------------------------------------------
    # PRIVATE: _classify_error
    # -------------------------------------------------------------------------
    def _classify_error(self, error_msg: str) -> str:
        """
        Classify a raw SQLite error string into a user-friendly category.

        Mapping:
            "no such table"  → "table"
            "no such column" → "column"
            "syntax error"   → "syntax"
            "no results"     → "empty"
            anything else    → "unknown"

        Args:
            error_msg: Raw error message string from SQLite.

        Returns:
            One of: "table", "column", "syntax", "empty", "unknown".
        """
        msg = error_msg.lower()

        if "no such table" in msg:
            return "table"
        if "no such column" in msg:
            return "column"
        if "syntax error" in msg:
            return "syntax"
        if "no results" in msg:
            return "empty"

        return "unknown"

    # -------------------------------------------------------------------------
    # PRIVATE: _format_error_msg
    # -------------------------------------------------------------------------
    def _format_error_msg(self, error_type: str, sql: str, raw_error: str) -> str:
        """
        Return a user-friendly error message based on the classified type.

        Args:
            error_type: Classified error type from _classify_error().
            sql:        Original SQL query string (for logging context).
            raw_error:  Raw error message from the database engine.

        Returns:
            Human-readable error string suitable for display in the UI.
        """
        friendly_messages = {
            "table":  "Table not found. Check table name in query.",
            "column": "Column not found. Check column name in query.",
            "syntax": "SQL syntax error. The generated query has a formatting issue.",
            "empty":  "Query returned no results.",
        }
        return friendly_messages.get(
            error_type,
            "Query execution failed. Please rephrase.",
        )

    # -------------------------------------------------------------------------
    # PUBLIC: get_schema
    # -------------------------------------------------------------------------
    def get_schema(self) -> dict:
        """
        Retrieve the full database schema (tables → columns).

        Connects to the database, reads sqlite_master for table names,
        then runs PRAGMA table_info(…) for each table.

        Returns:
            dict mapping table name (str) to a list of column dicts,
            each containing:
                {"name": <column_name>, "type": <sqlite_type>}

            Returns an empty dict if the database is unreachable.
        """
        schema: dict = {}
        try:
            with sqlite3.connect(str(self.db_path), timeout=30) as conn:
                cursor = conn.cursor()

                # Get all user tables (exclude internal sqlite tables)
                cursor.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' ORDER BY name"
                )
                tables = [row[0] for row in cursor.fetchall()]

                for table in tables:
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [
                        {"name": col[1], "type": col[2]}
                        for col in cursor.fetchall()
                    ]
                    schema[table] = columns

            logger.info(f"Schema retrieved for {len(schema)} table(s)")

        except Exception as e:
            logger.error(f"Failed to retrieve schema: {e}")

        return schema

    # -------------------------------------------------------------------------
    # PUBLIC: test_connection
    # -------------------------------------------------------------------------
    def test_connection(self) -> bool:
        """
        Verify that the database is reachable by executing SELECT 1.

        Returns:
            True if the connection and trivial query succeed, False otherwise.
        """
        try:
            with sqlite3.connect(str(self.db_path), timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
            logger.info("Database connection test: PASSED")
            return True
        except Exception as e:
            logger.error(f"Database connection test FAILED: {e}")
            return False


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    executor = SQLExecutor("database/sample.db")
    print("Connection:", executor.test_connection())

    tests = [
        ("SELECT * FROM customers LIMIT 5",                True),
        ("SELECT AVG(salary) FROM employees",              True),
        ("SELECT city, COUNT(*) FROM orders GROUP BY city", True),
        ("SELECT * FROM nonexistent_table",                False),
        ("INVALID SQL HERE",                               False),
        ("SELECT * FROM customers WHERE city = 'Mumbai'",  True),
    ]

    for sql, expect_success in tests:
        result = executor.execute(sql)
        status = "PASS" if result["success"] == expect_success else "FAIL"
        print(
            f"[{status}] {sql[:55]:<55} "
            f"rows={result['row_count']} "
            f"err={result['error_type']}"
        )
