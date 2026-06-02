"""
NL2SQL — Smart Value Suggester
When query returns 0 results, finds closest
matching values from actual database data.
"""

import sqlite3
import re
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class ValueSuggester:
    """
    Detects when user typed wrong value and
    suggests correct values from database.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_suggestions(self,
            sql: str,
            nlp_result: dict) -> dict:
        """
        Main method. Called when result has 0 rows.
        Analyzes SQL WHERE clause values against
        actual database values.
        Returns suggestion dict.
        """
        try:
            # Extract WHERE conditions from SQL
            conditions = self._extract_conditions(sql)
            if not conditions:
                return self._empty()

            suggestions = []
            for col, op, val in conditions:
                # Only check = conditions with strings
                if op != "=" or not isinstance(val, str):
                    continue

                # Find table for this column
                table = self._find_table_for_col(
                    col, sql)
                if not table:
                    continue

                # Get all unique values for this col
                db_values = self._get_col_values(
                    table, col)
                if not db_values:
                    continue

                # Find closest matches
                matches = self._find_closest(
                    val, db_values, top_k=3)

                if matches:
                    suggestions.append({
                        "column":       col,
                        "table":        table,
                        "typed":        val,
                        "matches":      matches,
                        "sql_template": sql
                    })

            if not suggestions:
                return self._empty()

            return {
                "has_suggestions": True,
                "suggestions":     suggestions,
                "message":         self._build_message(
                    suggestions)
            }

        except Exception as e:
            logger.debug(
                f"ValueSuggester error: {e}")
            return self._empty()

    def _extract_conditions(self,
            sql: str) -> list:
        """
        Extract (column, operator, value) tuples
        from SQL WHERE clause.
        Returns list of tuples.
        """
        conditions = []
        if not sql or "WHERE" not in sql.upper():
            return conditions

        # Extract WHERE clause
        where_match = re.search(
            r'WHERE\s+(.+?)(?:\s+GROUP\s+BY'
            r'|\s+ORDER\s+BY|\s+LIMIT'
            r'|\s+HAVING|$)',
            sql,
            re.IGNORECASE | re.DOTALL)

        if not where_match:
            return conditions

        where_clause = where_match.group(1).strip()

        # Split by AND
        parts = re.split(
            r'\bAND\b', where_clause,
            flags=re.IGNORECASE)

        for part in parts:
            part = part.strip()

            # Match: col = 'value' or col = value
            m = re.match(
                r"(\w+)\s*([=!<>]+)\s*'([^']*)'",
                part, re.IGNORECASE)

            if not m:
                # Try without quotes
                m = re.match(
                    r'(\w+)\s*([=!<>]+)\s*(\S+)',
                    part, re.IGNORECASE)

            if m:
                col = m.group(1).strip()
                op  = m.group(2).strip()
                val = m.group(3).strip().strip("'")

                if val:
                    # Try numeric
                    try:
                        val = float(val)
                        if val == int(val):
                            val = int(val)
                    except ValueError:
                        pass  # keep as string

                    conditions.append((col, op, val))

        return conditions

    def _find_table_for_col(self,
            col: str,
            sql: str) -> str:
        """Find which table a column belongs to."""
        # Extract table from FROM clause
        from_match = re.search(
            r'FROM\s+(\w+)',
            sql, re.IGNORECASE)
        if from_match:
            return from_match.group(1)

        # Try to find in database schema
        try:
            conn   = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            tables = cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table'"
            ).fetchall()

            for (table,) in tables:
                cols = cursor.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
                col_names = [c[1].lower()
                             for c in cols]
                if col.lower() in col_names:
                    conn.close()
                    return table

            conn.close()
        except Exception:
            pass

        return ""

    def _get_col_values(self,
            table: str,
            col: str) -> list:
        """
        Get all unique non-null values
        for a column from database.
        Only text columns are useful for matching.
        """
        try:
            conn   = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            rows = cursor.execute(
                f"SELECT DISTINCT [{col}] "
                f"FROM [{table}] "
                f"WHERE [{col}] IS NOT NULL "
                f"ORDER BY [{col}] "
                f"LIMIT 100"
            ).fetchall()

            conn.close()
            values = [str(r[0]) for r in rows
                      if r[0] is not None]
            return values

        except Exception as e:
            logger.debug(
                f"Get col values error: {e}")
            return []

    def _find_closest(self,
            typed: str,
            db_values: list,
            top_k: int = 3) -> list:
        """
        Find closest matching values to what
        user typed using multiple methods.
        Returns list of values sorted by
        similarity score descending.
        """
        if not db_values:
            return []

        typed_lower = typed.lower()
        scored = []

        for val in db_values:
            val_lower = val.lower()
            score = 0

            # Method 1: Exact match (skip — no typo)
            if val_lower == typed_lower:
                return []  # exact match exists,
                           # no suggestion needed

            # Method 2: Fuzzy string similarity
            try:
                from fuzzywuzzy import fuzz
                fuzzy_score = fuzz.ratio(
                    typed_lower, val_lower)
                score = max(score, fuzzy_score)

                # Also check partial ratio
                partial = fuzz.partial_ratio(
                    typed_lower, val_lower)
                score = max(score,
                            int(partial * 0.9))

            except ImportError:
                # Fallback: simple character overlap
                score = self._char_overlap(
                    typed_lower, val_lower)

            # Method 3: Starts with same letters
            min_len = min(
                len(typed_lower),
                len(val_lower))
            if min_len >= 2:
                prefix = min(3, min_len)
                if typed_lower[:prefix] == \
                   val_lower[:prefix]:
                    score = max(score, 60)

            # Method 4: Contains substring
            if typed_lower in val_lower or \
               val_lower in typed_lower:
                score = max(score, 70)

            if score >= 50:
                scored.append((val, score))

        # Sort by score descending
        scored.sort(key=lambda x: -x[1])

        # Return top_k values only
        return [v for v, s in scored[:top_k]]

    def _char_overlap(self,
            a: str, b: str) -> float:
        """Simple character overlap score 0-100."""
        if not a or not b:
            return 0
        set_a = set(a)
        set_b = set(b)
        overlap = len(set_a & set_b)
        total   = len(set_a | set_b)
        return int(overlap / total * 100) \
            if total > 0 else 0

    def _build_message(self,
            suggestions: list) -> str:
        """Build human readable message."""
        if not suggestions:
            return ""
        s   = suggestions[0]
        col = s["column"]
        val = s["typed"]
        return (f"'{val}' not found in {col}. "
                f"Did you mean one of these?")

    def make_corrected_sql(self,
            original_sql: str,
            column: str,
            old_value: str,
            new_value: str) -> str:
        """
        Replace old value with new value in SQL.
        Used when user clicks a suggestion button.
        """
        # Replace quoted string value
        corrected = re.sub(
            rf"({column}\s*=\s*)'?"
            rf"{re.escape(old_value)}'?",
            f"\\1'{new_value}'",
            original_sql,
            flags=re.IGNORECASE)
        return corrected

    def _empty(self) -> dict:
        """Return empty suggestion result."""
        return {
            "has_suggestions": False,
            "suggestions":     [],
            "message":         ""
        }


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import sqlite3
    import tempfile
    import os

    # Create test database
    tmp = tempfile.mktemp(suffix='.db')
    conn = sqlite3.connect(tmp)
    conn.execute("""
        CREATE TABLE staff (
            id INTEGER, name TEXT,
            city TEXT, department TEXT,
            role TEXT, salary REAL)""")
    conn.executemany(
        "INSERT INTO staff VALUES (?,?,?,?,?,?)",
        [(1, 'Deepak', 'Mumbai', 'Administration',
          'Manager', 85000),
         (2, 'Sneha', 'Pune', 'Front Desk',
          'Receptionist', 35000),
         (3, 'Arjun', 'Delhi', 'Kitchen',
          'Chef', 50000),
         (4, 'Amit', 'Mumbai', 'Administration',
          'Manager', 70000),
         (5, 'Kavita', 'Bangalore', 'Administration',
          'Manager', 75000)])
    conn.commit()
    conn.close()

    suggester = ValueSuggester(tmp)

    TESTS = [
        {
            "desc":   "Typo in city",
            "sql":    "SELECT * FROM staff "
                      "WHERE city = 'Mumabi'",
            "expect": "Mumbai"
        },
        {
            "desc":   "Typo in department",
            "sql":    "SELECT * FROM staff "
                      "WHERE department = "
                      "'Adminstration'",
            "expect": "Administration"
        },
        {
            "desc":   "Typo in role",
            "sql":    "SELECT * FROM staff "
                      "WHERE role = 'Cheif'",
            "expect": "Chef"
        },
        {
            "desc":   "Completely wrong city",
            "sql":    "SELECT * FROM staff "
                      "WHERE city = 'Satara'",
            "expect": None  # no close match
        },
        {
            "desc":   "Correct value (no suggestion)",
            "sql":    "SELECT * FROM staff "
                      "WHERE city = 'Mumbai'",
            "expect": None  # exact match, no suggestion
        },
    ]

    print("=" * 55)
    print("ValueSuggester Test")
    print("=" * 55)
    passed = 0
    for tc in TESTS:
        result = suggester.get_suggestions(
            tc["sql"], {})
        has_sugg = result["has_suggestions"]
        if tc["expect"] is None:
            ok = not has_sugg
        else:
            matches = []
            for s in result["suggestions"]:
                matches.extend(s["matches"])
            ok = tc["expect"] in matches
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {tc['desc']}")
        if has_sugg:
            for s in result["suggestions"]:
                print(f"  Typed  : {s['typed']}")
                print(f"  Matches: {s['matches']}")
        else:
            print(f"  No suggestions")

    print(f"\n{'=' * 55}")
    print(f"Result: {passed}/{len(TESTS)} passed")
    os.unlink(tmp)
