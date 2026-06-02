"""
NL2SQL — Phase 7: Auto Visualization Package

Provides automatic chart type detection and Plotly rendering
for SQL query results.

Usage:
    from visualization.chart_engine import ChartEngine

    chart = ChartEngine()
    result = chart.render(columns, rows, intent, title)
"""

from visualization.chart_engine import ChartEngine

__all__ = ["ChartEngine"]
