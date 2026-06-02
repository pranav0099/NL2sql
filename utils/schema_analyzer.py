"""
NL2SQL — Schema Analyzer
Reads any uploaded CSV or SQLite database and:
1. Detects schema (table name, columns, types, samples)
2. Generates smart natural language query suggestions
   based on actual column names and data types
"""

import sqlite3
import re
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class SchemaAnalyzer:
    """Analyze a SQLite database schema and generate smart query suggestions."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema = {}       # filled by analyze()
        self.suggestions = []  # filled by generate_suggestions()

    def analyze(self) -> dict:
        """
        Read SQLite database and return full schema dict.

        Returns:
            dict with structure:
            {
              "tables": {
                "table_name": {
                  "columns": [...],
                  "types": {...},
                  "sample_values": {...},
                  "row_count": int,
                  "numeric_columns": [...],
                  "text_columns": [...],
                  "date_columns": [...]
                }
              },
              "total_tables": int
            }
        """
        logger.info(f"Analyzing database: {self.db_path}")
        schema = {"tables": {}, "total_tables": 0}

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all table names
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(tables)} table(s): {tables}")

            for table in tables:
                table_info = {}

                # Column names and declared types via PRAGMA
                cursor.execute(f"PRAGMA table_info([{table}])")
                pragma_rows = cursor.fetchall()
                columns = [row[1] for row in pragma_rows]
                declared_types = {row[1]: row[2] for row in pragma_rows}
                table_info["columns"] = columns

                # Read sample rows into pandas for type detection
                try:
                    df = pd.read_sql_query(
                        f"SELECT * FROM [{table}] LIMIT 100", conn
                    )
                except Exception as e:
                    logger.warning(f"Could not read sample from {table}: {e}")
                    df = pd.DataFrame(columns=columns)

                # Detect types per column
                types = {}
                numeric_columns = []
                text_columns = []
                date_columns = []

                for col in columns:
                    if col in df.columns:
                        col_type = self._detect_type(col, df[col])
                    else:
                        col_type = "text"
                    types[col] = col_type

                    if col_type == "numeric":
                        numeric_columns.append(col)
                    elif col_type == "date":
                        date_columns.append(col)
                    else:
                        text_columns.append(col)

                table_info["types"] = types
                table_info["numeric_columns"] = numeric_columns
                table_info["text_columns"] = text_columns
                table_info["date_columns"] = date_columns

                # Sample values: top 3 unique per column
                sample_values = {}
                for col in columns:
                    if col in df.columns and not df[col].dropna().empty:
                        uniques = df[col].dropna().unique()[:3]
                        sample_values[col] = [
                            v.item() if hasattr(v, "item") else v
                            for v in uniques
                        ]
                    else:
                        sample_values[col] = []
                table_info["sample_values"] = sample_values

                # Row count
                cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
                table_info["row_count"] = cursor.fetchone()[0]

                schema["tables"][table] = table_info
                logger.info(
                    f"  Table '{table}': {len(columns)} cols, "
                    f"{table_info['row_count']} rows"
                )

            schema["total_tables"] = len(tables)
            conn.close()

        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            raise

        self.schema = schema
        return schema

    def _detect_type(self, col_name: str, series: pd.Series) -> str:
        """
        Detect column type from actual data.

        Priority:
        1. Column name contains date/time keywords → "date"
        2. >80% of values are numeric → "numeric"
        3. Otherwise → "text"
        """
        name_lower = col_name.lower()

        # Check for date-like column names
        date_keywords = ["date", "time", "day", "month", "year", "week"]
        for kw in date_keywords:
            if kw in name_lower:
                return "date"

        # Try numeric conversion
        if not series.dropna().empty:
            numeric_converted = pd.to_numeric(series.dropna(), errors="coerce")
            success_rate = numeric_converted.notna().sum() / max(
                len(series.dropna()), 1
            )
            if success_rate > 0.8:
                return "numeric"

        return "text"

    def generate_suggestions(self, table_name: str = None) -> list:
        """
        Generate smart NL query suggestions for a table based on its schema.

        Args:
            table_name: Table to generate suggestions for.
                        Defaults to first table in schema.

        Returns:
            List of suggestion dicts with keys: query, category, icon
        """
        if not self.schema or not self.schema.get("tables"):
            logger.warning("No schema available — call analyze() first")
            return []

        if table_name is None:
            table_name = list(self.schema["tables"].keys())[0]

        if table_name not in self.schema["tables"]:
            logger.warning(f"Table '{table_name}' not in schema")
            return []

        info = self.schema["tables"][table_name]
        numeric_cols = info["numeric_columns"]
        text_cols = info["text_columns"]
        date_cols = info["date_columns"]

        suggestions = []
        seen = set()  # for dedup

        def _add(query: str, category: str, icon: str):
            if query not in seen:
                seen.add(query)
                suggestions.append(
                    {"query": query, "category": category, "icon": icon}
                )

        # ── Basic queries (always) ──────────────────────────────────────
        _add(f"Show all records from {table_name}", "basic", "📋")
        _add(f"Count total rows in {table_name}", "basic", "🔢")
        _add(f"Show first 10 rows from {table_name}", "basic", "👁")

        # ── Aggregate queries (one per numeric column) ──────────────────
        for col in numeric_cols:
            _add(f"What is the total {col}?", "aggregate", "📊")
            _add(f"What is the average {col}?", "aggregate", "📊")
            _add(f"What is the maximum {col}?", "aggregate", "⬆")
            _add(f"What is the minimum {col}?", "aggregate", "⬇")

        # ── Group / filter queries (text columns) ───────────────────────
        for col in text_cols:
            _add(f"Count records grouped by {col}", "group", "📈")
            _add(f"Show all unique {col} values", "group", "🔍")

        # ── Ranking queries (numeric + text exist) ──────────────────────
        if numeric_cols and text_cols:
            _add(
                f"Top 10 {table_name} by {numeric_cols[0]}",
                "ranking",
                "🏆",
            )
            _add(
                f"Show {text_cols[0]} with highest {numeric_cols[0]}",
                "ranking",
                "🥇",
            )

        # ── Trend queries (date column exists) ──────────────────────────
        if date_cols and numeric_cols:
            _add(
                f"Show {numeric_cols[0]} trend by {date_cols[0]}",
                "trend",
                "📅",
            )
            _add(
                f"Total {numeric_cols[0]} per {date_cols[0]}",
                "trend",
                "📅",
            )

        # ── Combination queries (multiple text columns) ─────────────────
        if len(text_cols) >= 2:
            _add(
                f"Show {text_cols[0]} and {text_cols[1]} "
                f"grouped by {text_cols[0]}",
                "group",
                "🔗",
            )

        # ── Sort by category priority, then limit ───────────────────────
        category_order = {
            "basic": 0,
            "aggregate": 1,
            "ranking": 2,
            "trend": 3,
            "group": 4,
        }
        suggestions.sort(key=lambda s: category_order.get(s["category"], 99))
        suggestions = suggestions[:12]

        self.suggestions = suggestions
        logger.info(f"Generated {len(suggestions)} suggestions for '{table_name}'")
        return suggestions

    def get_schema_summary(self, table_name: str = None) -> str:
        """
        Return a human-readable schema summary string for display.

        Args:
            table_name: Table to summarise. Defaults to first table.

        Returns:
            Multi-line summary string.
        """
        if not self.schema or not self.schema.get("tables"):
            return "No schema available"

        if table_name is None:
            table_name = list(self.schema["tables"].keys())[0]

        if table_name not in self.schema["tables"]:
            return f"Table '{table_name}' not found"

        info = self.schema["tables"][table_name]
        cols_str = ", ".join(
            f"{c} ({info['types'].get(c, 'text')})" for c in info["columns"]
        )
        sample_parts = []
        for col in info["columns"][:4]:
            samples = info["sample_values"].get(col, [])
            if samples:
                sample_parts.append(
                    f"{col}={[str(v) for v in samples[:2]]}"
                )

        summary = (
            f"Table: {table_name} ({info['row_count']} rows)\n"
            f"Columns: {cols_str}\n"
            f"Sample: {', '.join(sample_parts)}"
        )
        return summary


# =============================================================================
# CSV → SQLite converter
# =============================================================================

def csv_to_sqlite(csv_path: str, output_dir: str = "database/uploads") -> str:
    """
    Convert a CSV file to a SQLite database.

    Steps:
    1. Read CSV with pandas (handle encoding errors)
    2. Clean column names: lowercase, replace spaces with _, remove specials
    3. Table name = CSV filename without extension, cleaned same way
    4. Save to SQLite at output_dir/{table_name}.db
    5. Return the sqlite path as string

    Handles edge cases:
    - CSV with no headers → auto-generates col_0, col_1, ...
    - Duplicate column names → appends _2, _3, ...
    - Very large CSV (>100k rows) → logs a warning
    """
    logger.info(f"Converting CSV to SQLite: {csv_path}")
    csv_path_obj = Path(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Read CSV robustly ───────────────────────────────────────────────
    df = None
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue

    if df is None:
        raise ValueError(f"Could not read CSV file: {csv_path}")

    # Detect headerless CSV (all columns look like data values)
    if all(
        isinstance(c, int)
        or (isinstance(c, str) and c.replace(".", "", 1).isdigit())
        for c in df.columns
    ):
        logger.info("CSV appears to have no headers — generating column names")
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    # ── Clean column names ──────────────────────────────────────────────
    clean_cols = []
    col_counts = {}
    for col in df.columns:
        clean = str(col).strip().lower()
        clean = re.sub(r"[^a-z0-9_]", "_", clean)
        clean = re.sub(r"_+", "_", clean).strip("_")
        if not clean:
            clean = "column"

        # Handle duplicates
        if clean in col_counts:
            col_counts[clean] += 1
            clean = f"{clean}_{col_counts[clean]}"
        else:
            col_counts[clean] = 1

        clean_cols.append(clean)

    df.columns = clean_cols

    # ── Table name from filename ────────────────────────────────────────
    table_name = csv_path_obj.stem.lower()
    table_name = re.sub(r"[^a-z0-9_]", "_", table_name)
    table_name = re.sub(r"_+", "_", table_name).strip("_")
    if not table_name:
        table_name = "uploaded_data"

    # ── Warn for large CSV ──────────────────────────────────────────────
    if len(df) > 100_000:
        logger.warning(
            f"Large CSV detected: {len(df):,} rows — conversion may take a moment"
        )

    # ── Write to SQLite ─────────────────────────────────────────────────
    sqlite_path = out_dir / f"{table_name}.db"
    conn = sqlite3.connect(str(sqlite_path))
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    logger.info(
        f"SQLite created: {sqlite_path}  "
        f"(table='{table_name}', {len(df)} rows, {len(df.columns)} cols)"
    )
    return str(sqlite_path)


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os

    # Fix Windows console encoding for emoji output
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Create a test CSV
    test_csv = """product,region,revenue,month,year
Phone,Mumbai,45000,Jan,2024
Laptop,Pune,72000,Feb,2024
Tablet,Delhi,31000,Jan,2024
Phone,Mumbai,52000,Mar,2024
Laptop,Bangalore,68000,Feb,2024"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        f.write(test_csv)
        csv_path = f.name

    sqlite_path = csv_to_sqlite(csv_path, "database/uploads")
    print(f"SQLite created: {sqlite_path}")

    analyzer = SchemaAnalyzer(sqlite_path)
    schema = analyzer.analyze()
    print(f"Tables: {list(schema['tables'].keys())}")

    table = list(schema["tables"].keys())[0]
    print(f"Columns: {schema['tables'][table]['columns']}")
    print(f"Types: {schema['tables'][table]['types']}")
    print(f"Rows: {schema['tables'][table]['row_count']}")

    suggestions = analyzer.generate_suggestions(table)
    print(f"\nGenerated {len(suggestions)} suggestions:")
    for s in suggestions:
        print(f"  {s['icon']} {s['query']}")

    print("\nSchema summary:")
    print(analyzer.get_schema_summary(table))

    os.unlink(csv_path)
    print("\nTest PASSED")
