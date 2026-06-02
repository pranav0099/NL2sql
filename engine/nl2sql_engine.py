"""
NL2SQL — Phase 6: Main Integration Engine
Run: python engine/nl2sql_engine.py

THE BRAIN OF THE SYSTEM.

Wires all 5 phases together into one clean pipeline:
    Phase 1  →  Config + Database + Schema
    Phase 2  →  NLP Pipeline (preprocessor, schema linker, entity extractor)
    Phase 3  →  ML Intent Classifier
    Phase 4  →  DL SQL Generator (Transformer + fallback)
    Phase 5  →  Session Manager (multi-turn context resolution)
    Phase 6  →  SQL Executor + Explainer + this engine

User types a query → NL2SQLEngine.query() → structured result dict.

This is the ONLY file imported by Phase 7 (Visualization) and Phase 8 (UI).
"""

import sys
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

import re
import uuid
import time
import traceback

from config.config import SAMPLE_DB
from nlp.pipeline import NLPPipeline
from ml.classifier import IntentClassifier
from dl.generator import SQLGenerator
from memory.session import SessionManager
from engine.sql_executor import SQLExecutor
from engine.explainer import SQLExplainer
from engine.rule_based_builder import RuleBasedSQLBuilder
from engine.api_sql_generator import APISQLGenerator
from engine.value_suggester import ValueSuggester
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# NL2SQL ENGINE CLASS
# =============================================================================

class NL2SQLEngine:
    """
    Production integration engine that connects NLP, ML, DL,
    Memory, Executor, and Explainer into one unified pipeline.

    Lifecycle:
        1. Instantiate once with database and model paths.
        2. Call create_session() to get a session_id.
        3. Call query(user_input, session_id) for every user message.
        4. The returned dict contains everything Phase 7/8 needs:
           SQL, results, explanation, suggestions, errors, timing.

    The query() method is crash-proof — it always returns a valid
    dict with success=True or success=False.

    Attributes:
        nlp_pipeline    (NLPPipeline)      Phase 2
        ml_classifier   (IntentClassifier) Phase 3
        sql_generator   (SQLGenerator)     Phase 4
        session_manager (SessionManager)   Phase 5
        sql_executor    (SQLExecutor)      Phase 6
        explainer       (SQLExplainer)     Phase 6
        db_path         (str)              Active database path
        query_count     (int)              Total queries processed
        success_count   (int)              Queries that returned success=True
    """

    def __init__(
        self,
        db_path:         str = SAMPLE_DB,
        model_path:      str = "models/saved/transformer_nl2sql.pt",
        vocab_path:      str = "data/processed/vocab.json",
        ml_model_path:   str = "models/saved/intent_classifier.pkl",
        vectorizer_path: str = "models/saved/tfidf_vectorizer.pkl",
    ):
        """
        Initialize all NL2SQL components in the canonical order.

        Each component is loaded sequentially so that failures
        surface early with clear log messages.

        Args:
            db_path:         Path to the SQLite database.
            model_path:      Path to the trained Transformer checkpoint (.pt).
            vocab_path:      Path to the shared vocabulary JSON.
            ml_model_path:   Path to the scikit-learn intent classifier (.pkl).
            vectorizer_path: Path to the TF-IDF vectorizer (.pkl).
        """
        # 1. NLP Pipeline (Phase 2)
        self.nlp_pipeline = NLPPipeline(db_path)
        logger.info("NLP Pipeline loaded")

        # 2. ML Classifier (Phase 3)
        self.ml_classifier = IntentClassifier(ml_model_path, vectorizer_path)
        logger.info("ML Classifier loaded")

        # 3. SQL Generator (Phase 4)
        self.sql_generator = SQLGenerator(model_path, vocab_path, db_path)
        logger.info("SQL Generator loaded")

        # 4. Session Manager (Phase 5)
        self.session_manager = SessionManager(db_path)
        logger.info("Session Manager loaded")

        # 5. SQL Executor (Phase 6 — executor)
        self.sql_executor = SQLExecutor(db_path)
        logger.info("SQL Executor loaded")

        # 6. Explainer (Phase 6 — explainer)
        self.explainer = SQLExplainer()
        logger.info("Explainer loaded")

        # 7. Rule-Based SQL Builder (backup for API)
        self.rule_builder = RuleBasedSQLBuilder()
        logger.info("Rule-Based SQL Builder loaded")

        # 8. API SQL Generator (primary for ALL databases)
        self.api_generator = APISQLGenerator()
        logger.info("API SQL Generator loaded")

        # 9. Value Suggester (smart suggestions for 0-result queries)
        self.value_suggester = ValueSuggester(str(db_path))
        logger.info("Value Suggester loaded")

        # 9-11. Engine-level state
        self.db_path       = db_path
        self.default_db    = str(Path(SAMPLE_DB).resolve())
        self.query_count   = 0
        self.success_count = 0

        logger.info("NL2SQL Engine ready — all components loaded")

    # -------------------------------------------------------------------------
    # PUBLIC: query  (THE MAIN METHOD)
    # -------------------------------------------------------------------------
    def query(self, user_input: str, session_id: str = "default") -> dict:
        """
        Process a raw user question through the full NL2SQL pipeline.

        Ten-step pipeline:
            1. Session & context resolution
            2. NLP processing
            3. ML intent classification
            4. Clarification check (early return if needed)
            5. SQL generation
            6. SQL patching (incomplete value fix)
            7. SQL execution
            8. Explanation generation
            9. Record result in session memory
           10. Build and return final result dict

        This method NEVER raises — every code path returns a valid dict
        with success=True or success=False.

        Args:
            user_input: Raw natural-language query from the user.
            session_id: Session identifier for multi-turn context.
                        Defaults to "default".

        Returns:
            dict with keys:
                success, session_id, turn_number, query, resolved_query,
                is_followup, followup_type, intent, ml_confidence,
                gen_confidence, sql, fallback_used, results
                  (rows, columns, row_count, truncated),
                explanation, summary, suggestions,
                error, error_type,
                needs_clarification, clarification_msg,
                execution_ms, pipeline_ms
        """
        pipeline_start = time.perf_counter()

        # Pre-initialise local variables so the except block can reference them
        # even if an exception fires before they are assigned in the try block.
        resolved_query  = user_input
        is_followup     = False
        followup_type   = "none"
        turn_number     = 0
        intent          = "UNKNOWN"
        ml_conf         = 0.0
        gen_conf        = 0.0
        sql             = None
        fallback        = False
        processed       = {}

        try:
            # ------------------------------------------------------------------
            # Step 1 — Session & Context Resolution
            # ------------------------------------------------------------------
            processed      = self.session_manager.process_query(session_id, user_input)
            resolved_query = processed["resolved_query"]
            is_followup    = processed["is_followup"]
            followup_type  = processed.get("followup_type", "none")
            turn_number    = processed["turn_number"]
            context        = processed["context"]

            # ------------------------------------------------------------------
            # Step 2 — NLP Processing
            # ------------------------------------------------------------------
            nlp_result = self.nlp_pipeline.process(resolved_query)

            # ------------------------------------------------------------------
            # Step 3 — ML Intent Classification
            # ------------------------------------------------------------------
            ml_result           = self.ml_classifier.predict(resolved_query, nlp_result)
            intent              = ml_result["intent"]
            ml_conf             = ml_result["confidence"]
            needs_clarification = ml_result["needs_clarification"]

            # ------------------------------------------------------------------
            # Step 3b — Override clarification when NLP keywords are clear
            # ------------------------------------------------------------------
            # The ML classifier was trained on the sample database schema,
            # so custom database columns (x1, y1, etc.) produce low confidence.
            # If the NLP preprocessor has already detected clear SQL keywords
            # (AVG, MAX, MIN, COUNT, SUM, WHERE, ORDER BY, GROUP BY, etc.),
            # the query intent is unambiguous — skip clarification.
            if needs_clarification:
                nlp_keywords = nlp_result.get("preprocessed", {}).get("keywords", set())
                clear_intent_keywords = {
                    "AVG", "MAX", "MIN", "COUNT", "SUM",       # aggregates
                    "WHERE", ">", "<", ">=", "<=", "!=", "=",  # filters
                    "ORDER BY", "ORDER BY DESC LIMIT",         # ordering
                    "ORDER BY ASC LIMIT", "GROUP BY",          # grouping
                    "JOIN", "LIMIT",                           # joins / limits
                }
                if nlp_keywords & clear_intent_keywords:
                    logger.info(
                        f"Overriding clarification: NLP keywords "
                        f"{nlp_keywords & clear_intent_keywords} indicate clear intent"
                    )
                    needs_clarification = False
                    # Also fix the intent from NLP hint if ML was confused
                    nlp_intent = nlp_result.get("sql_hints", {}).get("intent_hint", "")
                    if nlp_intent and nlp_intent != "SELECT":
                        intent = nlp_intent

            # ------------------------------------------------------------------
            # Step 4 — Clarification Check (early return)
            # ------------------------------------------------------------------
            # Only ask clarification for complex queries.
            # Simple show/list/display queries always proceed.
            # Extended list includes informal/noisy words users commonly type.
            simple_keywords = [
                "show", "list", "display", "get", "fetch",
                "find", "give", "tell", "what", "which",
                "who", "all", "records", "data",
                # Informal / noisy words that indicate user intent
                "plz", "pls", "please", "gimme", "want",
                "count", "total", "average", "maximum", "minimum",
                "highest", "lowest", "sum", "top", "bottom",
                "how", "many", "much",
                # Common misspellings of key action words
                "shw", "sho", "shwo", "shoe", "lst", "lsit",
                "dispaly", "fnd", "cont", "coutn", "totl",
                "avrage", "averge", "higest", "highst",
                "lowst", "lowets", "maxium", "minmum",
            ]
            query_lower = resolved_query.lower()
            is_simple = any(w in query_lower for w in simple_keywords)

            # Also check if query contains a number (strong filter intent)
            has_number = bool(re.search(r'\d+', query_lower))

            # If API is available, it handles noisy input well — skip clarification
            api_available = self.api_generator.is_available()

            # Lower threshold for simple queries or when API is available
            if api_available:
                effective_threshold = 0.15  # API handles everything
            elif is_simple or has_number:
                effective_threshold = 0.20  # Very permissive for simple/numeric
            else:
                effective_threshold = 0.70

            if needs_clarification and ml_result.get("confidence", 0) < effective_threshold:
                # Prefer the classifier's own message; fall back to explainer's
                clarification_msg = ml_result.get("clarification_msg") or \
                    self.explainer.generate_clarification(user_input, ml_conf, intent)

                logger.info(f"Clarification needed for: {user_input!r}")

                return {
                    "success":             False,
                    "needs_clarification": True,
                    "clarification_msg":   clarification_msg,
                    "session_id":          session_id,
                    "turn_number":         turn_number,
                    "query":               user_input,
                    "resolved_query":      resolved_query,
                    "sql":                 None,
                    "results":             None,
                    "explanation":         None,
                    "summary":             None,
                    "suggestions":         [],
                    "is_followup":         is_followup,
                    "followup_type":       followup_type,
                    "intent":              intent,
                    "ml_confidence":       ml_conf,
                    "gen_confidence":      0.0,
                    "fallback_used":       False,
                    "error":               None,
                    "error_type":          None,
                    "execution_ms":        0.0,
                    "pipeline_ms":         (time.perf_counter() - pipeline_start) * 1000,
                    "value_suggestions":   {"has_suggestions": False, "suggestions": [], "message": ""},
                }
            # else: proceed even with low confidence for simple queries

            # ══════════════════════════════════════════
            # Step 5 — SQL Generation
            # Priority: API first → Rule builder backup
            # API works on ANY database correctly
            # ══════════════════════════════════════════

            sql      = ""
            gen_conf = 0.0
            fallback = False
            gen_method = "none"  # Track which layer generated the SQL

            # LAYER 1: OpenRouter API (primary)
            # Works on any database, any schema
            if self.api_generator.is_available():
                try:
                    api_result = self.api_generator.generate(
                        resolved_query,
                        str(self.db_path))
                    sql      = api_result.get("sql", "")
                    gen_conf = api_result.get(
                        "confidence", 0.0)
                    fallback = api_result.get(
                        "fallback_used", False)

                    if sql:
                        gen_method = "api"
                        logger.info(
                            f"API generated SQL: {sql}")
                except Exception as e:
                    logger.debug(f"API failed: {e}")
                    sql = ""

            # LAYER 2: Rule-based builder (backup)
            # Used when API unavailable or failed
            if not sql:
                try:
                    rule_result = \
                        self.rule_builder.build(nlp_result)
                    sql      = rule_result.get("sql", "")
                    gen_conf = rule_result.get(
                        "confidence", 0.5)
                    fallback = True
                    if sql:
                        gen_method = "rule"
                        logger.info(
                            f"Rule builder SQL: {sql}")
                except Exception as e:
                    logger.debug(f"Rule builder failed: {e}")
                    sql = ""

            # LAYER 3: DL model (last resort only)
            # Kept for research purposes
            # Only used if both API and rules fail
            if not sql:
                try:
                    dl_result = self.sql_generator.generate(
                        resolved_query, nlp_result)
                    sql      = dl_result.get("sql", "")
                    gen_conf = dl_result.get(
                        "confidence", 0.0)
                    fallback = True
                    gen_method = "dl"
                    logger.info(
                        f"DL model fallback SQL: {sql}")
                except Exception as e:
                    logger.debug(f"DL model failed: {e}")
                    sql = ""

            # Safety: if everything failed
            if not sql:
                tables = nlp_result.get(
                    "sql_hints", {}).get("tables", [])
                table  = tables[0] if tables else "unknown"
                sql    = f"SELECT * FROM {table}"
                gen_conf = 0.3
                fallback = True
                gen_method = "fallback"

            # ------------------------------------------------------------------
            # Step 6 — SQL Patch (fix incomplete SQL)
            # ------------------------------------------------------------------
            sql = self._patch_sql(sql, nlp_result, gen_method=gen_method)

            # ------------------------------------------------------------------
            # Step 7 — SQL Execution
            # ------------------------------------------------------------------
            exec_result = self.sql_executor.execute(sql)

            # ------------------------------------------------------------------
            # Step 7.5 — Smart Value Suggestions (when 0 results)
            # ------------------------------------------------------------------
            value_suggestions = {
                "has_suggestions": False,
                "suggestions": [],
                "message": ""
            }

            if exec_result.get("success") and \
               exec_result.get("row_count", 0) == 0:
                # Reload suggester with current db_path
                # (handles uploaded databases too)
                suggester = ValueSuggester(str(self.db_path))
                value_suggestions = suggester.get_suggestions(
                    sql, nlp_result)

            # ------------------------------------------------------------------
            # Step 8 — Explanation Generation
            # ------------------------------------------------------------------
            explanation_result = self.explainer.explain(sql, exec_result)
            explanation = explanation_result["explanation"]
            summary     = explanation_result["summary"]
            suggestions = explanation_result["suggestions"]

            # ------------------------------------------------------------------
            # Step 9 — Record in Session Memory
            # ------------------------------------------------------------------
            entities = nlp_result["entities"]
            self.session_manager.record_result(
                session_id     = session_id,
                raw_query      = user_input,
                resolved_query = resolved_query,
                intent         = intent,
                sql            = sql,
                result_count   = exec_result["row_count"],
                result_columns = exec_result["columns"],
                entities       = {
                    "tables":       nlp_result["schema_links"]["matched_tables"],
                    "columns":      list(entities.get("filters", [])),
                    "filters":      entities.get("filters", []),
                    "aggregations": entities.get("aggregations", []),
                    "order":        entities.get("order", None),
                    "group_by":     entities.get("group_by", []),
                },
                confidence     = gen_conf,
                success        = exec_result["success"],
            )

            # ------------------------------------------------------------------
            # Step 10 — Build & Return Final Result
            # ------------------------------------------------------------------
            self.query_count += 1
            if exec_result["success"]:
                self.success_count += 1

            pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

            return {
                "success":             exec_result["success"],
                "session_id":          session_id,
                "turn_number":         turn_number,
                "query":               user_input,
                "resolved_query":      resolved_query,
                "is_followup":         is_followup,
                "followup_type":       followup_type,
                "intent":              intent,
                "ml_confidence":       ml_conf,
                "gen_confidence":      gen_conf,
                "sql":                 sql,
                "fallback_used":       fallback,
                "results": {
                    "rows":      exec_result["rows"],
                    "columns":   exec_result["columns"],
                    "row_count": exec_result["row_count"],
                    "truncated": exec_result["truncated"],
                },
                "explanation":         explanation,
                "summary":             summary,
                "suggestions":         suggestions,
                "error":               exec_result["error"],
                "error_type":          exec_result["error_type"],
                "needs_clarification": False,
                "clarification_msg":   None,
                "execution_ms":        exec_result["execution_ms"],
                "pipeline_ms":         pipeline_ms,
                "value_suggestions":   value_suggestions,
            }

        except Exception as e:
            # -----------------------------------------------------------------
            # Catch-all — engine.query() must NEVER crash
            # -----------------------------------------------------------------
            logger.error(f"Unexpected pipeline error: {e}")
            logger.error(traceback.format_exc())

            self.query_count += 1
            pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

            return {
                "success":             False,
                "session_id":          session_id,
                "turn_number":         processed.get("turn_number", 0) if processed else 0,
                "query":               user_input,
                "resolved_query":      resolved_query,
                "is_followup":         is_followup,
                "followup_type":       followup_type,
                "intent":              intent,
                "ml_confidence":       ml_conf,
                "gen_confidence":      gen_conf,
                "sql":                 sql,
                "fallback_used":       fallback,
                "results":             None,
                "explanation":         None,
                "summary":             None,
                "suggestions":         [],
                "error":               str(e),
                "error_type":          "unknown",
                "needs_clarification": False,
                "clarification_msg":   None,
                "execution_ms":        0.0,
                "pipeline_ms":         pipeline_ms,
                "value_suggestions":   {"has_suggestions": False, "suggestions": [], "message": ""},
            }

    # -------------------------------------------------------------------------
    # PRIVATE: _is_uploaded_db
    # -------------------------------------------------------------------------
    def _is_uploaded_db(self) -> bool:
        """Check if using an uploaded database vs the default sample.db."""
        return str(Path(self.db_path).resolve()) != self.default_db

    # -------------------------------------------------------------------------
    # PRIVATE: _patch_sql
    # -------------------------------------------------------------------------
    def _deduplicate_filters(self, filters: list) -> list:
        """
        Remove duplicate/conflicting filters on same column.
        Keep only the most specific filter per column.
        Priority: >/</>=/<= OVER =
        Example:
          salary > 40000 AND salary = 40000
          → keep salary > 40000 only
        """
        # Group filters by column name
        col_filters = {}
        for f in filters:
            col = f.get("column","")
            op  = f.get("op","=")
            if col not in col_filters:
                col_filters[col] = []
            col_filters[col].append(f)

        result = []
        for col, col_f_list in col_filters.items():
            if len(col_f_list) == 1:
                result.append(col_f_list[0])
                continue

            # Multiple filters on same column
            # Priority: >,<,>=,<= wins over =
            non_equal = [f for f in col_f_list
                         if f.get("op","=") != "="]
            equal     = [f for f in col_f_list
                         if f.get("op","=") == "="]

            if non_equal:
                # Use only the non-equal filter
                result.extend(non_equal)
            else:
                # All equal — use only first one
                result.append(equal[0])

        return result

    def _patch_sql(self, sql: str, nlp_result: dict, gen_method: str = "unknown") -> str:
        """
        Fix incomplete SQL generated by DL model.
        Handles:
        1. Missing value after comparison operator
           WHERE city =        -> WHERE city = 'Mumbai'
           WHERE salary >      -> WHERE salary > 50000
        2. Missing aggregate function value
           COUNT(              -> COUNT(*)
        3. Missing table after FROM
           SELECT COUNT(*) FROM  -> SELECT COUNT(*) FROM customers

        When gen_method='api', skip aggressive WHERE overrides (FIX 0b)
        since the API already generates correct SQL for noisy input.
        """
        import re

        if not sql or not nlp_result:
            return sql

        sql = sql.strip().rstrip(';')

        preprocessed = nlp_result.get("preprocessed", {})
        entities     = nlp_result.get("entities", {})
        schema_links = nlp_result.get("schema_links", {})

        # Extract all useful values from NLP result
        numbers      = preprocessed.get("numbers", [])
        raw_cities   = entities.get("raw_cities", [])
        raw_names    = entities.get("raw_names", [])
        filters      = entities.get("filters", [])
        nl_entities  = preprocessed.get("entities", [])
        matched_tables = schema_links.get(
            "matched_tables", [])

        # Collect all string values from NLP entities
        string_values = []
        for ent_text, ent_label in nl_entities:
            if ent_label in ["GPE","ORG","PERSON",
                           "PRODUCT","LOC","FAC"]:
                string_values.append(ent_text)
        string_values.extend(raw_cities)
        string_values.extend(raw_names)

        # ── Deduplicate filters BEFORE processing ──────
        # Remove duplicate/conflicting filters on same column
        filters = self._deduplicate_filters(filters)

        # ── FIX 0: WHERE completely missing but filters exist ──────
        # If SQL has no WHERE but NLP found filters → add them
        has_where = bool(re.search(
            r'\bWHERE\b', sql, re.IGNORECASE))
        has_group = bool(re.search(
            r'\bGROUP\s+BY\b', sql, re.IGNORECASE))
        has_order = bool(re.search(
            r'\bORDER\s+BY\b', sql, re.IGNORECASE))
        has_limit = bool(re.search(
            r'\bLIMIT\b', sql, re.IGNORECASE))

        if not has_where and filters:
            where_parts = []
            for f in filters:
                col = f.get("column", "")
                op  = f.get("operator", f.get("op", "="))
                val = f.get("value")

                if not col or val is None:
                    continue

                # Format value correctly
                if isinstance(val, str):
                    formatted_val = f"'{val}'"
                elif isinstance(val, float):
                    if val == int(val):
                        formatted_val = str(int(val))
                    else:
                        formatted_val = str(val)
                else:
                    formatted_val = str(val)

                where_parts.append(
                    f"{col} {op} {formatted_val}")

            if where_parts:
                where_clause = " AND ".join(where_parts)

                # Insert WHERE before GROUP BY/ORDER BY/LIMIT
                # or append to end
                if has_group:
                    sql = re.sub(
                        r'\bGROUP\s+BY\b',
                        f"WHERE {where_clause} GROUP BY",
                        sql, flags=re.IGNORECASE)
                elif has_order:
                    sql = re.sub(
                        r'\bORDER\s+BY\b',
                        f"WHERE {where_clause} ORDER BY",
                        sql, flags=re.IGNORECASE)
                elif has_limit:
                    sql = re.sub(
                        r'\bLIMIT\b',
                        f"WHERE {where_clause} LIMIT",
                        sql, flags=re.IGNORECASE)
                else:
                    sql = f"{sql} WHERE {where_clause}"

        # ── FIX 0b: WHERE exists but operator/value is wrong ──────
        # The DL model often gets the WHERE column right but uses the
        # wrong operator (e.g. ">" instead of ">=") or wrong value.
        # NLP keyword-based operator detection is more reliable for this.
        # Compare the SQL's WHERE clause against NLP-extracted filters
        # and replace if they disagree.
        # SKIP this fix when SQL came from API — API is more accurate
        # than NLP entity extraction, especially for noisy/misspelled queries.
        if gen_method != "api" and filters and bool(re.search(r'\bWHERE\b', sql, re.IGNORECASE)):
            # Parse existing WHERE conditions from SQL
            where_match = re.search(
                r'\bWHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*$)',
                sql, re.IGNORECASE)
            if where_match:
                existing_where = where_match.group(1).strip()

                # Build the correct WHERE clause from NLP filters
                nlp_where_parts = []
                for f in filters:
                    col = f.get("column", "")
                    op  = f.get("operator", f.get("op", "="))
                    val = f.get("value")
                    if not col or val is None:
                        continue
                    if isinstance(val, str):
                        formatted_val = f"'{val}'"
                    elif isinstance(val, float):
                        formatted_val = str(int(val)) if val == int(val) else str(val)
                    else:
                        formatted_val = str(val)
                    nlp_where_parts.append(f"{col} {op} {formatted_val}")

                if nlp_where_parts:
                    nlp_where = " AND ".join(nlp_where_parts)

                    # Check if the existing WHERE differs from NLP's version
                    # Normalize for comparison: strip spaces around operators
                    def _normalize_clause(c):
                        c = re.sub(r'\s+', ' ', c.strip().lower())
                        c = re.sub(r'\s*(>=|<=|!=|=|>|<)\s*', r' \1 ', c)
                        return c

                    if _normalize_clause(existing_where) != _normalize_clause(nlp_where):
                        # Replace the WHERE clause content
                        sql = re.sub(
                            r'(\bWHERE\s+).+?(\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*$)',
                            rf'\1{nlp_where}\2',
                            sql, count=1, flags=re.IGNORECASE)
                        logger.info(
                            f"FIX 0b: Corrected WHERE clause: "
                            f"'{existing_where}' -> '{nlp_where}'"
                        )

        # ── ORDER BY + LIMIT fix ──────────────────────
        order_info = entities.get("order", None)

        if order_info:
            col_order = order_info.get("column","")
            direction = order_info.get("direction","DESC")
            limit_val = order_info.get("limit", None)

            # Only add if not already in SQL
            has_order = bool(re.search(
                r'\bORDER\s+BY\b', sql, re.IGNORECASE))
            has_limit = bool(re.search(
                r'\bLIMIT\b', sql, re.IGNORECASE))

            if col_order and not has_order:
                sql = f"{sql} ORDER BY {col_order} {direction}"

            if limit_val and not has_limit:
                sql = f"{sql} LIMIT {int(limit_val)}"

        # Also handle keywords if order_info is None
        # but "top N" or "bottom N" found in query
        original_query = nlp_result.get(
            "original_query", "").lower()

        # Pattern: "top N" or "first N" or "bottom N"
        top_match = re.search(
            r'\b(top|first|show\s+\w+\s+top)\s+(\d+)',
            original_query)
        bottom_match = re.search(
            r'\b(bottom|last|lowest)\s+(\d+)',
            original_query)

        has_limit = bool(re.search(
            r'\bLIMIT\b', sql, re.IGNORECASE))
        has_order = bool(re.search(
            r'\bORDER\s+BY\b', sql, re.IGNORECASE))

        if top_match:
            n = int(top_match.group(2))
            by_match = re.search(
                r'\bby\s+(\w+)', original_query)
            if by_match:
                order_col = by_match.group(1)
                if not has_order:
                    sql = f"{sql} ORDER BY {order_col} DESC"
            # Always set the limit to n, overriding any existing LIMIT
            sql = re.sub(r'\bLIMIT\s+\d+', f'LIMIT {n}', sql, flags=re.IGNORECASE)
            if not re.search(r'\bLIMIT\s+\d+', sql, flags=re.IGNORECASE):
                sql = f"{sql} LIMIT {n}"

        elif bottom_match:
            n = int(bottom_match.group(2))
            by_match = re.search(
                r'\bby\s+(\w+)', original_query)
            if by_match:
                order_col = by_match.group(1)
                # Always set ORDER BY for bottom matches (ASC)
                if not has_order:
                    sql = f"{sql} ORDER BY {order_col} ASC"
                else:
                    # Replace existing ORDER BY direction with ASC for bottom matches
                    sql = re.sub(
                        r'(\bORDER\s+BY\s+\w+)\s+(ASC|DESC)',
                        rf'\1 ASC',
                        sql,
                        flags=re.IGNORECASE)
            # Always set the limit to n, overriding any existing LIMIT
            sql = re.sub(r'\bLIMIT\s+\d+', f'LIMIT {n}', sql, flags=re.IGNORECASE)
            if not re.search(r'\bLIMIT\s+\d+', sql, flags=re.IGNORECASE):
                sql = f"{sql} LIMIT {n}"

        # ── FIX 1: Missing value after operator ──────
        # Pattern: WHERE col_name = (nothing after)
        # Pattern: WHERE col_name > (nothing after)
        operators = ['>=', '<=', '!=', '=', '>', '<']

        for op in operators:
            pattern = rf'WHERE\s+(\w+)\s*{re.escape(op)}\s*$'
            match   = re.search(pattern, sql,
                                 re.IGNORECASE)
            if match:
                col_name = match.group(1).lower()

                # Decide value based on column name type
                value = self._get_value_for_column(
                    col_name, op,
                    numbers, string_values, filters)

                if value is not None:
                    sql = sql + f" {value}"
                    break

        # ── FIX 2: AND condition missing value ────────
        # Pattern: AND col_name = (nothing after)
        for op in operators:
            pattern = rf'AND\s+(\w+)\s*{re.escape(op)}\s*$'
            match   = re.search(pattern, sql,
                                 re.IGNORECASE)
            if match:
                col_name = match.group(1).lower()
                value = self._get_value_for_column(
                    col_name, op,
                    numbers, string_values, filters)
                if value is not None:
                    sql = sql + f" {value}"
                    break

        # ── FIX 3: Incomplete COUNT/SUM/AVG/MIN/MAX ──
        # Pattern: COUNT( or SUM( without closing )
        agg_pattern = r'(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*$'
        if re.search(agg_pattern, sql, re.IGNORECASE):
            sql = re.sub(agg_pattern, r'\1(*)',
                         sql, flags=re.IGNORECASE)

        # ── FIX 4: Missing closing ) in aggregate ────
        # COUNT(salary  → COUNT(salary)
        open_agg = r'(COUNT|SUM|AVG|MIN|MAX)\s*\((\w+)\s*$'
        if re.search(open_agg, sql, re.IGNORECASE):
            sql = re.sub(open_agg, r'\1(\2)',
                         sql, flags=re.IGNORECASE)

        # ── FIX 5: Missing table after FROM ──────────
        from_end = r'(FROM)\s*$'
        if re.search(from_end, sql, re.IGNORECASE):
            if matched_tables:
                sql = sql + f" {matched_tables[0]}"

        # ── FIX 6: GROUP BY missing column ───────────
        group_end = r'GROUP\s+BY\s*$'
        if re.search(group_end, sql, re.IGNORECASE):
            matched_cols = schema_links.get(
                "matched_columns", {})
            if matched_cols:
                first_table = list(matched_cols.keys())[0]
                cols = matched_cols[first_table]
                # Pick first text-like column
                for col in cols:
                    if not any(w in col.lower() for w in
                               ["id","amount","salary",
                                "price","total","count"]):
                        sql = sql + f" {col}"
                        break

        # ── FIX 7: SELECT * when specific column requested ────
        sql = self._fix_column_selection(sql, nlp_result)

        return sql.strip()

    def _get_value_for_column(self,
        col_name: str,
        operator: str,
        numbers: list,
        string_values: list,
        filters: list) -> str:
        """
        Determine the correct WHERE value for a column
        based on its name and the operator used.
        Returns formatted SQL value string.
        """

        col_lower = col_name.lower()

        # Check existing filters first (most accurate)
        for f in filters:
            if f.get("column","").lower() == col_lower:
                val = f.get("value")
                if val is not None:
                    if isinstance(val, str):
                        return f"'{val}'"
                    return str(val)

        # Numeric columns → use extracted numbers
        numeric_keywords = [
            "salary","amount","price","total","revenue",
            "sales","cost","budget","score","age","rate",
            "quantity","qty","value","count","number",
            "balance","fee","tax","discount","profit"
        ]
        is_numeric = any(w in col_lower
                         for w in numeric_keywords)

        if is_numeric and numbers:
            return str(int(numbers[0])
                       if numbers[0] == int(numbers[0])
                       else numbers[0])

        # Text/category columns → use string entities
        text_keywords = [
            "city","name","status","category","type",
            "department","dept","region","country",
            "gender","class","group","month","year",
            "state","product","brand","color","size"
        ]
        is_text = any(w in col_lower
                      for w in text_keywords)

        if is_text and string_values:
            return f"'{string_values[0]}'"

        # Fallback: if operator is = and we have strings
        if operator == '=' and string_values:
            return f"'{string_values[0]}'"

        # Fallback: if operator is > < and we have numbers
        if operator in ['>', '<', '>=', '<='] and numbers:
            return str(int(numbers[0])
                       if numbers[0] == int(numbers[0])
                       else numbers[0])

        return None

    # -------------------------------------------------------------------------
    # PRIVATE: _fix_column_selection
    # -------------------------------------------------------------------------
    def _fix_column_selection(self, sql: str, nlp_result: dict) -> str:
        """
        If user asked for a specific column but SQL has SELECT * — fix it.
        Triggers on keywords like 'only', 'just', 'show', 'get', 'list', 'display'.
        """
        import re

        query = nlp_result.get("original_query", "").lower()
        schema_links = nlp_result.get("schema_links", {})

        # Check if SELECT * in SQL
        if "SELECT *" not in sql.upper():
            return sql  # already specific

        # Check for "only" or "just" keywords
        only_match = re.search(
            r'\b(only|just|show|get|list|display)\s+(?:the\s+)?(\w+)',
            query)

        if not only_match:
            return sql

        requested = only_match.group(2).lower()

        # Find matching column in schema
        all_cols = []
        for cols in schema_links.get("matched_columns", {}).values():
            all_cols.extend(cols)

        matched_col = None
        for col in all_cols:
            if requested in col.lower() or col.lower() in requested:
                matched_col = col
                break

        if matched_col:
            sql = sql.replace("SELECT *", f"SELECT {matched_col}", 1)

        return sql

    # -------------------------------------------------------------------------
    # PUBLIC: get_stats
    # -------------------------------------------------------------------------
    def get_stats(self) -> dict:
        """
        Return engine usage statistics.

        Returns:
            dict:
                total_queries  (int)   — Queries processed since init.
                successful     (int)   — Queries with success=True.
                failed         (int)   — Queries with success=False.
                success_rate   (float) — Ratio of successful to total.
                active_sessions (int)  — Currently tracked sessions.
        """
        failed       = self.query_count - self.success_count
        success_rate = (
            self.success_count / self.query_count
            if self.query_count > 0 else 0.0
        )

        active_sessions = 0
        if hasattr(self.session_manager, "sessions"):
            active_sessions = len(self.session_manager.sessions)

        return {
            "total_queries":   self.query_count,
            "successful":      self.success_count,
            "failed":          failed,
            "success_rate":    success_rate,
            "active_sessions": active_sessions,
        }

    # -------------------------------------------------------------------------
    # PUBLIC: reset_session
    # -------------------------------------------------------------------------
    def reset_session(self, session_id: str) -> None:
        """
        Clear all conversation memory for the given session.

        Args:
            session_id: Session identifier to reset.
        """
        if hasattr(self.session_manager, "clear_session"):
            self.session_manager.clear_session(session_id)
            logger.info(f"Session reset: {session_id}")
        else:
            logger.warning("SessionManager does not support clear_session")

    # -------------------------------------------------------------------------
    # PUBLIC: get_session_history
    # -------------------------------------------------------------------------
    def get_session_history(self, session_id: str) -> dict:
        """
        Retrieve the full session summary from the session manager.

        Args:
            session_id: Session identifier to query.

        Returns:
            Session summary dict (turn count, tables used, query log, etc.)
            or an empty dict if the session does not exist.
        """
        if hasattr(self.session_manager, "get_session_summary"):
            return self.session_manager.get_session_summary(session_id)
        logger.warning("SessionManager does not support get_session_summary")
        return {}

    # -------------------------------------------------------------------------
    # PUBLIC: create_session
    # -------------------------------------------------------------------------
    def create_session(self) -> str:
        """
        Create a new conversation session and return its identifier.

        Returns:
            New session_id string (8-character UUID segment).
        """
        if hasattr(self.session_manager, "create_session"):
            session_id = self.session_manager.create_session()
            logger.info(f"Created session: {session_id}")
            return session_id

        # Fallback if SessionManager lacks create_session (defensive)
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"Created session (fallback): {session_id}")
        return session_id


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 65)
    print("NL2SQL Engine — Phase 6 Integration Test")
    print("=" * 65)

    engine = NL2SQLEngine(db_path="database/sample.db")
    session_id = engine.create_session()

    CONVERSATION = [
        "Show all customers from Mumbai",
        "Now filter by total spent above 50000",
        "How many are there?",
        "What is the average salary of employees?",
        "Show top 5 products by price",
        "Count orders grouped by city",
        "Show employees with salary above 60000",
        "List all departments",
    ]

    passed = 0
    failed = 0

    for i, query in enumerate(CONVERSATION, 1):
        print(f"\n{'=' * 65}")
        print(f"Turn {i}: {query}")
        print(f"{'=' * 65}")

        result = engine.query(query, session_id)

        print(f"Intent      : {result['intent']}")
        print(f"SQL         : {result['sql']}")
        print(f"Explanation : {result['explanation']}")
        print(f"Summary     : {result['summary']}")
        print(f"Rows        : {result['results']['row_count'] if result['results'] else 0}")
        print(f"Followup    : {result['is_followup']} ({result['followup_type']})")
        print(f"ML Conf     : {result['ml_confidence']:.0%}")
        print(f"Gen Conf    : {result['gen_confidence']:.0%}")
        print(f"Fallback    : {result['fallback_used']}")
        print(f"Success     : {result['success']}")

        if result["suggestions"]:
            print(f"Suggestions : {result['suggestions'][0]}")

        if result["success"]:
            passed += 1
        else:
            failed += 1
            print(f"Error       : {result['error']}")

    stats = engine.get_stats()
    print(f"\n{'=' * 65}")
    print(f"PHASE 6 INTEGRATION TEST COMPLETE")
    print(f"{'=' * 65}")
    print(f"Passed       : {passed}/{len(CONVERSATION)}")
    print(f"Success rate : {stats['success_rate']:.0%}")
    print(f"Total queries: {stats['total_queries']}")
    print(f"\nEngine is ready for Phase 7 Visualization")
    print(f"and Phase 8 Streamlit UI")
    print(f"{'=' * 65}")
