"""
NL2SQL — Phase 7: Auto Visualization Engine
Run: python visualization/chart_engine.py

Analyzes SQL query results and automatically selects the best
chart type (bar, line, pie, scalar, table, or none). Renders
publication-quality Plotly figures with a consistent design
language used throughout the Streamlit UI.

Pure Python + Plotly — no database or ML dependencies.
"""

import sys
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import List, Optional, Dict, Any

import plotly.graph_objects as go

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# CHART ENGINE CLASS
# =============================================================================

class ChartEngine:
    """
    Auto-detects the best visualization for SQL query results
    and renders interactive Plotly charts.

    Detection priority:
        1. Empty results       → "none"
        2. Single scalar value → "scalar"
        3. Aggregate + 1 row   → "scalar"
        4. 2 cols, numeric, ≤20 rows → "pie"
        5. 2 cols, numeric, >20 rows → "bar"
        6. Time-related column  → "line"
        7. Any numeric column   → "bar"
        8. Fallback             → "table"

    All Plotly figures share a cohesive colour palette:
        Primary:   #2E75B6  (steel blue)
        Secondary: #ED7D31  (warm orange)
        Accent:    #70AD47  (green)
        Palette:   px.colors.qualitative.Set2
    """

    # ── Colour palette ───────────────────────────────────────────────────────
    _PRIMARY   = "#2E75B6"
    _PALETTE   = [
        "#2E75B6", "#ED7D31", "#70AD47", "#FFC000",
        "#5B9BD5", "#A5A5A5", "#264478", "#9B59B6",
        "#1ABC9C", "#E74C3C",
    ]

    # Column-name substrings that hint at a time axis
    _TIME_HINTS = {"month", "year", "date", "time", "day", "week", "quarter"}

    def __init__(self):
        """
        Initialize ChartEngine.

        No external dependencies — uses only Plotly and pure Python.
        """
        logger.info("ChartEngine initialized")

    # -------------------------------------------------------------------------
    # PUBLIC: detect_chart_type
    # -------------------------------------------------------------------------
    def detect_chart_type(
        self,
        columns: List[str],
        rows: list,
        intent: str,
    ) -> str:
        """
        Analyze the result set and return the best chart type.

        Args:
            columns: Column name strings from the SQL result.
            rows:    List of row tuples / lists.
            intent:  Predicted SQL intent label (e.g. "SELECT_GROUP").

        Returns:
            One of: "bar", "line", "pie", "scalar", "table", "none".
        """
        # Rule 1 — empty / None
        if not rows or not columns:
            return "none"

        row_count = len(rows)
        col_count = len(columns)

        # Rule 2 — single cell → scalar
        if row_count == 1 and col_count == 1:
            return "scalar"

        # Rule 3 — aggregate intent with single row → scalar
        if intent == "SELECT_AGGREGATE" and row_count == 1:
            return "scalar"

        # Rule 4 — plain SELECT / SELECT_LIMIT / SELECT_WHERE with many columns → table
        # Multi-column detail rows (SELECT * ...) must be shown as a data grid.
        # This MUST run before the time-hint check, otherwise columns like
        # "subscription_date" cause wide tables to be rendered as line charts,
        # which crashes the Streamlit/Plotly React integration (error #185).
        if col_count > 3:
            return "table"
        
        if col_count > 2 and intent not in ("SELECT_GROUP", "SELECT_AGGREGATE"):
            return "table"

        # Rule 5 — any column name looks like a time dimension → line chart
        lower_cols = [c.lower() for c in columns]
        for hint in self._TIME_HINTS:
            for col_name in lower_cols:
                if hint in col_name:
                    # Need at least one numeric column for a line chart
                    for idx in range(col_count):
                        if self.is_numeric_column(rows, idx):
                            return "line"

        # Rule 6 — two columns where second is numeric
        # Pie charts work best for a small number of categories (≤ 3);
        # larger category sets are better compared with bar charts.
        if col_count == 2 and self.is_numeric_column(rows, 1):
            if row_count <= 3:
                return "pie"
            return "bar"

        # Rule 7 — at least 2 cols with one numeric → bar
        if col_count >= 2:
            for idx in range(col_count):
                if self.is_numeric_column(rows, idx):
                    return "bar"

        # Rule 8 — default
        return "table"

    # -------------------------------------------------------------------------
    # PUBLIC: is_numeric_column
    # -------------------------------------------------------------------------
    def is_numeric_column(self, rows: list, col_idx: int) -> bool:
        """
        Check whether a column is predominantly numeric.

        Samples the first 10 rows and returns True if >70 %
        of non-None values are int or float.

        Args:
            rows:    Row data (list of lists / tuples).
            col_idx: Zero-based column index to inspect.

        Returns:
            True if the column is mostly numeric.
        """
        sample = rows[:10]
        if not sample:
            return False

        numeric_count = 0
        total_checked = 0

        for row in sample:
            if col_idx >= len(row):
                continue
            val = row[col_idx]
            if val is None:
                continue
            total_checked += 1
            if isinstance(val, (int, float)):
                numeric_count += 1
            else:
                # Try to parse stringified numbers
                try:
                    float(str(val))
                    numeric_count += 1
                except (ValueError, TypeError):
                    pass

        if total_checked == 0:
            return False

        return (numeric_count / total_checked) > 0.70

    # -------------------------------------------------------------------------
    # PUBLIC: render
    # -------------------------------------------------------------------------
    def render(
        self,
        columns: List[str],
        rows: list,
        intent: str,
        title: str = "",
    ) -> Dict[str, Any]:
        """
        Auto-detect the best chart type and render a Plotly figure.

        Args:
            columns: Column name strings.
            rows:    Result row data.
            intent:  SQL intent label.
            title:   Optional chart title override.

        Returns:
            dict:
                chart_type   (str)         — Detected chart type.
                figure       (go.Figure|None) — Plotly figure, or None.
                scalar_value (str|None)    — Formatted scalar, or None.
                description  (str)         — Human-readable chart description.
        """
        # FIX: React error #185 — safe rendering
        try:
            if not rows or not columns:
                return {
                    "chart_type":   "none",
                    "figure":       None,
                    "scalar_value": None,
                    "description":  "No data to display"
                }

            # limit rows for chart rendering
            chart_rows = rows[:200]

            chart_type = self.detect_chart_type(
                columns, chart_rows, intent)

            if chart_type == "none":
                return {
                    "chart_type":   "none",
                    "figure":       None,
                    "scalar_value": None,
                    "description":  "No chart needed"
                }

            elif chart_type == "scalar":
                val = self._render_scalar(
                    columns, chart_rows, title)
                return {
                    "chart_type":   "scalar",
                    "figure":       None,
                    "scalar_value": val,
                    "description":  f"Value: {val}"
                }

            elif chart_type == "bar":
                fig = self._render_bar(
                    columns, chart_rows, title)
                return {
                    "chart_type":   "bar",
                    "figure":       fig,
                    "scalar_value": None,
                    "description":  "Bar chart"
                }

            elif chart_type == "line":
                fig = self._render_line(
                    columns, chart_rows, title)
                return {
                    "chart_type":   "line",
                    "figure":       fig,
                    "scalar_value": None,
                    "description":  "Line chart"
                }

            elif chart_type == "pie":
                fig = self._render_pie(
                    columns, chart_rows, title)
                return {
                    "chart_type":   "pie",
                    "figure":       fig,
                    "scalar_value": None,
                    "description":  "Pie chart"
                }

            else:
                return {
                    "chart_type":   "table",
                    "figure":       None,
                    "scalar_value": None,
                    "description":  "Table display"
                }

        except Exception as e:
            logger.warning(f"Chart render failed: {e}")
            return {
                "chart_type":   "none",
                "figure":       None,
                "scalar_value": None,
                "description":  "Chart unavailable"
            }

    # -------------------------------------------------------------------------
    # PRIVATE: _render_bar
    # -------------------------------------------------------------------------
    def _render_bar(
        self,
        columns: List[str],
        rows: list,
        title: str,
    ) -> go.Figure:
        """
        Create a Plotly bar chart.

        x-axis: First column (categories).
        y-axis: First numeric column (values).
        Value labels are shown on top of each bar.

        Args:
            columns: Column names.
            rows:    Row data.
            title:   Chart title.

        Returns:
            Plotly Figure object.
        """
        # Find first numeric column for y-axis (prefer col index 1)
        y_idx = 1
        for idx in range(1, len(columns)):
            if self.is_numeric_column(rows, idx):
                y_idx = idx
                break

        x_vals = [row[0] for row in rows]
        y_vals = [row[y_idx] for row in rows]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=x_vals,
                    y=y_vals,
                    marker_color=self._PRIMARY,
                    text=y_vals,
                    textposition="outside",
                    textfont=dict(size=11),
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title or f"{columns[0]} vs {columns[y_idx]}", font=dict(size=16)),
            xaxis_title=columns[0],
            yaxis_title=columns[y_idx],
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#E8E8E8"),
            margin=dict(l=60, r=30, t=50, b=50),
            height=400,
        )

        return fig

    # -------------------------------------------------------------------------
    # PRIVATE: _render_line
    # -------------------------------------------------------------------------
    def _render_line(
        self,
        columns: List[str],
        rows: list,
        title: str,
    ) -> go.Figure:
        """
        Create a Plotly line chart with markers.

        x-axis: First column (time / category dimension).
        y-axis: First numeric column.

        Args:
            columns: Column names.
            rows:    Row data.
            title:   Chart title.

        Returns:
            Plotly Figure object.
        """
        # Find first numeric column for y-axis
        y_idx = 1
        for idx in range(len(columns)):
            if self.is_numeric_column(rows, idx):
                y_idx = idx
                break

        # Use column 0 for x if y_idx is not 0, otherwise column 1
        x_idx = 0 if y_idx != 0 else (1 if len(columns) > 1 else 0)

        x_vals = [row[x_idx] for row in rows]
        y_vals = [row[y_idx] for row in rows]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    line=dict(color=self._PRIMARY, width=2.5),
                    marker=dict(size=8, color=self._PRIMARY),
                    fill="tozeroy",
                    fillcolor="rgba(46,117,182,0.08)",
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title or f"{columns[x_idx]} Trend", font=dict(size=16)),
            xaxis_title=columns[x_idx],
            yaxis_title=columns[y_idx],
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#E8E8E8"),
            margin=dict(l=60, r=30, t=50, b=50),
            height=400,
        )

        return fig

    # -------------------------------------------------------------------------
    # PRIVATE: _render_pie
    # -------------------------------------------------------------------------
    def _render_pie(
        self,
        columns: List[str],
        rows: list,
        title: str,
    ) -> go.Figure:
        """
        Create a Plotly donut-style pie chart.

        labels: First column.
        values: Second (numeric) column.
        hole=0.3 for the donut aesthetic.

        Args:
            columns: Column names.
            rows:    Row data.
            title:   Chart title.

        Returns:
            Plotly Figure object.
        """
        labels = [str(row[0]) for row in rows]
        values = [row[1] for row in rows]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,
                    marker=dict(colors=self._PALETTE[: len(labels)]),
                    textinfo="label+percent",
                    textposition="outside",
                    pull=[0.03] * len(labels),
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title or f"{columns[0]} Distribution", font=dict(size=16)),
            paper_bgcolor="white",
            margin=dict(l=30, r=30, t=50, b=30),
            height=420,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )

        return fig

    # -------------------------------------------------------------------------
    # PRIVATE: _render_scalar
    # -------------------------------------------------------------------------
    def _render_scalar(
        self,
        columns: List[str],
        rows: list,
        title: str,
    ) -> str:
        """
        Format a single scalar result value for display.

        Formatting rules:
            • float   → rounded to 2 decimal places, comma-separated
            • int     → comma-separated (Indian/international)
            • other   → str()

        Args:
            columns: Column names (used for context, not displayed).
            rows:    Row data — expected to be [[value]] or [[v1,v2,…]].
            title:   Unused for scalar but kept for API consistency.

        Returns:
            Formatted string representation of the scalar value.
        """
        if not rows or not rows[0]:
            return "N/A"

        # Take the first value from the first row
        value = rows[0][0]

        if value is None:
            return "N/A"

        if isinstance(value, float):
            return f"{value:,.2f}"

        if isinstance(value, int):
            return f"{value:,}"

        # Try to parse stringified numbers
        try:
            num = float(str(value))
            if num == int(num):
                return f"{int(num):,}"
            return f"{num:,.2f}"
        except (ValueError, TypeError):
            return str(value)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    engine = ChartEngine()

    test_cases = [
        {
            "name": "Bar chart — city counts",
            "columns": ["city", "count"],
            "rows": [
                ["Mumbai", 45], ["Pune", 32],
                ["Delhi", 28], ["Bangalore", 22],
            ],
            "intent": "SELECT_GROUP",
            "expected": "bar",
        },
        {
            "name": "Scalar — single count",
            "columns": ["COUNT(*)"],
            "rows": [[142]],
            "intent": "SELECT_AGGREGATE",
            "expected": "scalar",
        },
        {
            "name": "Pie chart — category split",
            "columns": ["category", "total"],
            "rows": [
                ["Electronics", 45000],
                ["Clothing", 23000],
                ["Food", 12000],
            ],
            "intent": "SELECT_GROUP",
            "expected": "pie",
        },
        {
            "name": "Line chart — monthly trend",
            "columns": ["month", "sales_amount"],
            "rows": [
                ["Jan", 50000], ["Feb", 62000],
                ["Mar", 71000], ["Apr", 58000],
            ],
            "intent": "SELECT_GROUP",
            "expected": "line",
        },
        {
            "name": "Table — multiple columns",
            "columns": ["customer_id", "first_name", "city", "total_spent"],
            "rows": [[1, "Rahul", "Mumbai", 72000]],
            "intent": "SELECT",
            "expected": "table",
        },
    ]

    all_passed = True
    for tc in test_cases:
        detected = engine.detect_chart_type(
            tc["columns"], tc["rows"], tc["intent"]
        )
        ok = detected == tc["expected"]
        if not ok:
            all_passed = False
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {tc['name']:<35} "
            f"detected={detected} expected={tc['expected']}"
        )

    # Also test render() to make sure figures are created
    print("\n--- Render tests ---")
    for tc in test_cases:
        result = engine.render(
            tc["columns"], tc["rows"], tc["intent"],
            title=tc["name"],
        )
        has_fig   = result["figure"] is not None
        has_scalar = result["scalar_value"] is not None
        print(
            f"  {tc['name']:<35} "
            f"type={result['chart_type']:<8} "
            f"fig={has_fig} scalar={has_scalar}"
        )

    print(f"\nAll passed: {all_passed}")
    print("Phase 7 Visualization ready")
