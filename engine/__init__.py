"""
NL2SQL — Phase 6: Integration Engine Package

This package wires all 5 phases (NLP, ML, DL, Memory, Execution)
into one unified pipeline. Phase 7 (Visualization) and Phase 8
(Streamlit UI) import ONLY from this package.

Usage:
    from engine import NL2SQLEngine

    engine = NL2SQLEngine(db_path="database/sample.db")
    result = engine.query("show customers from Mumbai")
"""

from engine.nl2sql_engine import NL2SQLEngine

__all__ = ["NL2SQLEngine"]
