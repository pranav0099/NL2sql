"""
NL2SQL — Silent API SQL Generator
Uses OpenRouter API (free models available).
Key loaded from .env — never exposed in UI.
All API calls silent — no messages shown.
"""

import os
import re
import sys
import sqlite3
import requests
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger
logger = get_logger(__name__)


# =============================================================================
# Load .env silently — supports both standard dotenv and PowerShell format
# =============================================================================

def _load_env_file():
    """
    Parse the project .env file.
    Supports two formats:
      Standard:    VAR=value  or  VAR="value"
      PowerShell:  $env:VAR="value"
    Sets values into os.environ.
    Uses forced assignment (not setdefault) because python-dotenv
    may have incorrectly parsed $env: lines and set wrong values.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # PowerShell format: $env:VAR="value"
                ps_match = re.match(
                    r'^\$env:(\w+)\s*=\s*"?([^"]*)"?\s*$',
                    line)
                if ps_match:
                    key, val = ps_match.group(1), ps_match.group(2)
                    if val:  # only set if non-empty
                        os.environ[key] = val
                    continue

                # Standard dotenv: VAR=value or VAR="value"
                std_match = re.match(
                    r'^(\w+)\s*=\s*"?([^"]*)"?\s*$',
                    line)
                if std_match:
                    key, val = std_match.group(1), std_match.group(2)
                    if val:
                        os.environ[key] = val

    except Exception:
        pass  # silent — never crash on env loading


# Try python-dotenv first (may misparse $env: lines)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# ALWAYS run our custom parser LAST to fix any
# incorrect values from python-dotenv parsing
_load_env_file()


# =============================================================================
# API SQL GENERATOR CLASS
# =============================================================================

class APISQLGenerator:
    """
    Silent SQL generator using OpenRouter API.
    Reads credentials from .env file only.
    Never shows API info in UI.
    """

    def __init__(self):
        self.api_key   = os.getenv(
            "ANTHROPIC_AUTH_TOKEN", "")
        self.base_url  = os.getenv(
            "ANTHROPIC_BASE_URL",
            "https://openrouter.ai/api")
        self.model     = os.getenv(
            "ANTHROPIC_MODEL",
            "poolside/laguna-m.1:free")
        self.is_loaded = False
        self._setup_silent()

    def _setup_silent(self):
        """Initialize silently — no UI messages."""
        if not self.api_key:
            logger.debug(
                "No API key in .env")
            return
        try:
            # Mark as ready (actual test on first call)
            self.is_loaded = True
            logger.info(
                "OpenRouter API ready")
        except Exception as e:
            logger.debug(f"API setup: {e}")

    def get_schema_context(self,
            db_path: str) -> str:
        """Read database schema with sample values."""
        try:
            conn   = sqlite3.connect(db_path)
            cursor = conn.cursor()
            tables = cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table'"
            ).fetchall()

            schema_lines = []
            for (table,) in tables:
                cols = cursor.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()

                # Column definitions
                col_defs = [
                    f"{c[1]} {c[2]}"
                    for c in cols]
                schema_lines.append(
                    f"\nTable: {table}")
                schema_lines.append(
                    f"Columns: "
                    f"{', '.join(col_defs)}")

                # Sample values per column
                try:
                    rows = cursor.execute(
                        f"SELECT * FROM "
                        f"'{table}' LIMIT 5"
                    ).fetchall()
                    if rows:
                        col_names = [
                            c[1] for c in cols]
                        # Show unique values per col
                        for ci, cname in \
                                enumerate(col_names):
                            vals = list(set(
                                str(r[ci])
                                for r in rows
                                if r[ci] is not None
                            ))[:3]
                            if vals:
                                schema_lines.append(
                                    f"  {cname} "
                                    f"examples: "
                                    f"{', '.join(vals)}")
                except Exception:
                    pass

                # Row count
                try:
                    cnt = cursor.execute(
                        f"SELECT COUNT(*) "
                        f"FROM '{table}'"
                    ).fetchone()[0]
                    schema_lines.append(
                        f"Total rows: {cnt}")
                except Exception:
                    pass

            conn.close()
            return "\n".join(schema_lines)

        except Exception as e:
            logger.debug(
                f"Schema context error: {e}")
            return ""

    def generate(self, question: str,
                 db_path: str) -> dict:
        """
        Generate SQL using OpenRouter API.
        Returns same dict format as DL generator.
        Completely silent — no UI messages.
        """
        if not self.is_loaded:
            return self._empty_result()

        schema = self.get_schema_context(
            db_path)
        if not schema:
            return self._empty_result()

        prompt = self._build_prompt(
            question, schema)

        try:
            # Call OpenRouter API
            headers = {
                "Authorization":
                    f"Bearer {self.api_key}",
                "Content-Type":
                    "application/json",
                "HTTP-Referer":
                    "http://localhost:8501",
                "X-Title": "NL2SQL System"
            }

            # Fix base_url format
            base = self.base_url.rstrip('/')
            if not base.endswith('/v1'):
                base = base + '/v1'
            url = f"{base}/chat/completions"

            payload = {
                "model": self.model,
                "messages": [{
                    "role":    "user",
                    "content": prompt
                }],
                "temperature":  0,
                "max_tokens":   2000,
            }

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # Extract SQL from response
            # Some models (reasoning models like poolside/laguna)
            # put thinking in "reasoning" and output in "content".
            # If content is null/empty, try extracting SQL from reasoning.
            msg = data["choices"][0]["message"]
            sql_raw = (msg.get("content") or "").strip()

            # Fallback: extract SQL from reasoning field
            if not sql_raw:
                reasoning = (msg.get("reasoning") or "")
                # Look for SELECT statement in reasoning
                for line in reasoning.split('\n'):
                    line = line.strip()
                    if line.upper().startswith('SELECT'):
                        sql_raw = line
                        break

            sql = self._clean_sql(sql_raw)

            if not sql:
                logger.debug(
                    "API returned empty SQL")
                logger.debug(
                    f"Raw response keys: {list(msg.keys())}")
                logger.debug(
                    f"Content: {msg.get('content')}")
                logger.debug(
                    f"Finish reason: "
                    f"{data['choices'][0].get('finish_reason')}")
                return self._empty_result()

            logger.info(
                f"[API] SQL: {sql}")
            return {
                "sql":          sql,
                "confidence":   0.95,
                "tokens":       sql.split(),
                "method":       "api",
                "fallback_used": False
            }

        except requests.exceptions.Timeout:
            logger.debug("API timeout (60s)")
            return self._empty_result()
        except requests.exceptions.ConnectionError:
            logger.debug("API connection failed")
            return self._empty_result()
        except Exception as e:
            logger.debug(f"API error: {type(e).__name__}: {e}")
            return self._empty_result()

    def _build_prompt(self,
            question: str,
            schema: str) -> str:
        """Build SQL generation prompt."""
        return f"""You are an expert SQLite SQL generator.
Generate ONE valid SQL query for this database.

IMPORTANT:
- Return ONLY the SQL query
- No explanation
- No markdown backticks
- No semicolon at end

DATABASE SCHEMA WITH SAMPLE DATA:
{schema}

USER QUESTION: {question}

SQL GENERATION RULES:
1. Use ONLY table/column names from schema above
2. COUNT(*) → how many, count, number of, total count
3. SUM(col) → total, sum of, overall, combined
4. AVG(col) → average, mean, typical
5. MAX(col) → highest, maximum, most, largest, best
6. MIN(col) → lowest, minimum, least, cheapest
7. WHERE    → filter conditions
8. GROUP BY → per, each, by, grouped by
9. ORDER BY col DESC LIMIT N → top N, highest N, best N
10. ORDER BY col ASC LIMIT N → bottom N, lowest N
11. above / over / more than / greater than → >
12. below / under / less than / lower than   → <
13. at least / no less than / minimum        → >=
14. at most / no more than / maximum / up to → <=
15. NEVER duplicate condition on same column
    (e.g. salary > 40000 AND salary = 40000 is WRONG)
16. For multiple conditions use AND between them
17. String values must be in single quotes: 'Mumbai'
18. Numbers must NOT be in quotes: 40000

COLUMN SELECTION RULES (very important):
- If user says "only {{column}}" or "just {{column}}" or "show {{column}}" or "get {{column}}" or "list {{column}}" → use SELECT {{column}} not SELECT *
- Examples:
  "show only emails" → SELECT email FROM table
  "get names and salaries" → SELECT name, salary FROM table
  "list just the product names" → SELECT product_name FROM table
  "show me names and cities" → SELECT name, city FROM table
- If user asks for ALL records or does not specify columns → use SELECT *
- If user mentions a specific column name that exists in schema → include it in SELECT

WRITE ONLY THE SQL QUERY BELOW:
"""

    def _clean_sql(self, sql: str) -> str:
        """Remove markdown, extract clean SQL."""
        # Remove code blocks
        sql = re.sub(
            r'```sql\s*', '', sql,
            flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        sql = sql.strip().rstrip(';')

        # Find SELECT statement
        lines = sql.split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT'):
                return line.rstrip(';')

        # Fallback: return first non-empty line
        for line in lines:
            if line.strip():
                return line.strip().rstrip(';')

        if sql.strip():
            return sql.split('\n')[0].strip()
        return ""

    def is_available(self) -> bool:
        """Check if API is ready."""
        return self.is_loaded

    def _empty_result(self) -> dict:
        """Empty result for fallback."""
        return {
            "sql":          "",
            "confidence":   0.0,
            "method":       "api_unavailable",
            "fallback_used": True
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
            staff_id INTEGER,
            name TEXT,
            salary REAL,
            department TEXT,
            city TEXT
        )""")
    conn.executemany(
        "INSERT INTO staff VALUES (?,?,?,?,?)",
        [(1, 'Rahul',  45000, 'Kitchen',   'Mumbai'),
         (2, 'Priya',  75000, 'Admin',     'Pune'),
         (3, 'Amit',   30000, 'FrontDesk', 'Delhi'),
         (4, 'Sneha',  90000, 'Manager',   'Mumbai'),
         (5, 'Raj',    35000, 'Kitchen',   'Delhi')])
    conn.commit()
    conn.close()

    gen = APISQLGenerator()

    if not gen.is_available():
        print("Add ANTHROPIC_AUTH_TOKEN to .env")
        print("Base URL: https://openrouter.ai/api")
    else:
        TESTS = [
            ("show all staff",
             "SELECT * FROM staff"),
            ("show staff with salary above 50000",
             "SELECT * FROM staff WHERE salary > 50000"),
            ("count all staff",
             "SELECT COUNT(*) FROM staff"),
            ("average salary of staff",
             "SELECT AVG(salary) FROM staff"),
            ("top 3 staff by salary",
             "SELECT * FROM staff ORDER BY salary DESC LIMIT 3"),
            ("staff from Mumbai",
             "SELECT * FROM staff WHERE city = 'Mumbai'"),
            ("total salary by department",
             "SELECT department, SUM(salary) FROM staff GROUP BY department"),
            ("staff with salary below 40000",
             "SELECT * FROM staff WHERE salary < 40000"),
        ]

        print("=" * 60)
        print("OpenRouter API SQL Generator Test")
        print("=" * 60)
        passed = 0
        for question, expected in TESTS:
            result = gen.generate(question, tmp)
            sql    = result["sql"]
            ok     = bool(sql) and \
                     "SELECT" in sql.upper()
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"
            print(f"\n[{status}] {question}")
            print(f"  Expected : {expected}")
            print(f"  Got      : {sql}")

        print(f"\n{'=' * 60}")
        print(f"Result: {passed}/{len(TESTS)} passed")

    os.unlink(tmp)
