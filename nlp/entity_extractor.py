"""
NL2SQL — Phase 2: Entity Extractor for Structured Query Components
Run: python nlp/entity_extractor.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
import re

# Add project root to sys.path BEFORE importing utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class EntityExtractor:
    """
    Extracts structured entities from preprocessed query output.

    Takes preprocessed query data and extracts:
    - Filters (WHERE conditions)
    - Aggregations (COUNT, SUM, AVG, etc.)
    - ORDER BY clauses with direction and limit
    - GROUP BY columns
    - Select all flag
    """

    def __init__(self):
        """Initialize EntityExtractor (no external dependencies)."""
        logger.info("Initializing EntityExtractor")
        logger.info("EntityExtractor initialized successfully")

    def extract(self, preprocessed: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured entities from preprocessed query.

        Args:
            preprocessed: Output from QueryPreprocessor.preprocess()
            schema: Database schema dict from SchemaLinker._load_schema()

        Returns:
            Dictionary containing:
            {
                "filters": [
                    {"column": "city", "operator": "=", "value": "Mumbai"}
                ],
                "aggregations": [
                    {"function": "AVG", "column": "salary"}
                ],
                "order": {
                    "column": "total_spent",
                    "direction": "DESC",
                    "limit": 10
                } or None,
                "group_by": ["city"],
                "select_all": True,
                "raw_numbers": [50000],
                "raw_cities": ["Mumbai"],
                "raw_names": ["Rahul"]
            }
        """
        logger.info("Extracting entities from preprocessed query")

        tokens = preprocessed.get("tokens", [])
        lemmas = preprocessed.get("lemmas", [])
        entities = preprocessed.get("entities", [])
        numbers = preprocessed.get("numbers", [])
        keywords = preprocessed.get("keywords", set())

        # Flatten all schema columns for matching
        schema_columns = []
        table_names = list(schema.keys())
        for table_info in schema.values():
            schema_columns.extend(table_info["columns"])

        # Extract each component
        filters = self._extract_filters(tokens, lemmas, entities, numbers, keywords, schema_columns, schema)
        aggregations = self._extract_aggregations(keywords, lemmas, schema_columns, tokens, table_names)
        order = self._extract_order(keywords, tokens, lemmas, numbers, schema_columns)
        group_by = self._extract_group_by(keywords, tokens, schema_columns)

        # Determine select_all flag
        select_all = self._determine_select_all(lemmas, keywords, aggregations, filters)

        # Extract raw named entities by type
        raw_numbers = numbers
        raw_cities = [ent_text for ent_text, ent_label in entities if ent_label in ("GPE", "LOC")]
        raw_names = [ent_text for ent_text, ent_label in entities if ent_label in ("PERSON",)]

        result = {
            "filters": filters,
            "aggregations": aggregations,
            "order": order,
            "group_by": group_by,
            "select_all": select_all,
            "raw_numbers": raw_numbers,
            "raw_cities": raw_cities,
            "raw_names": raw_names
        }

        logger.debug(f"Extraction complete: filters={len(filters)}, aggregations={len(aggregations)}")
        return result

    def _extract_filters(self, tokens: List[str], lemmas: List[str],
                        entities: List[Tuple[str, str]], numbers: List[float],
                        keywords: Set[str], schema_columns: List[str],
                        schema: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Extract WHERE clause filters.

        Logic:
        - Scan tokens for comparison operators: >, <, =, >=, <=, !=
        - Scan tokens for multi-word operator phrases: "below", "above", etc.
        - Match adjacent tokens to column names
        - Extract value (number, quoted string, or named entity)
        """
        filters = []
        operators = [">", "<", "=", ">=", "<=", "!=", "BETWEEN", "LIKE"]

        # Words to skip when scanning for operators
        SKIP_WORDS = [
            "has", "have", "had", "that", "which",
            "who", "whose", "where", "the", "a", "an",
            "is", "are", "was", "were", "be", "been"
        ]

        # ── Multi-word operator phrase patterns ──
        OPERATOR_PATTERNS = {
            ">":  ["greater than", "more than", "above",
                   "higher than", "exceeds", "over",
                   "larger than", "bigger than"],
            "<":  ["less than", "below", "under",
                   "lower than", "fewer than",
                   "smaller than", "beneath",
                   "not above", "not over"],
            ">=": ["at least", "no less than",
                   "minimum of", "greater than or equal",
                   "more than or equal"],
            "<=": ["at most", "no more than",
                   "maximum of", "less than or equal",
                   "not more than", "no greater than"],
            "!=": ["not equal", "except",
                   "excluding", "other than"],
            "=":  ["equal to", "equals", "same as",
                   "matching", "is"],
        }

        # ── Phase 1: single-token operator detection (original logic) ──
        for i, token in enumerate(tokens):
            # Skip noise words
            if token.lower() in SKIP_WORDS:
                continue
            token_upper = token.upper()
            if token_upper in operators or token in operators:
                operator = token_upper if token_upper in operators else token

                # Look for column before operator
                column = None
                if i > 0:
                    for j in range(max(0, i - 3), i):
                        candidate = tokens[j]
                        matched_col = self._match_to_column(candidate, schema_columns)
                        if matched_col:
                            column = matched_col
                            break

                if not column and i > 0:
                    for j in range(max(0, i - 3), i):
                        candidate = lemmas[j]
                        matched_col = self._match_to_column(candidate, schema_columns)
                        if matched_col:
                            column = matched_col
                            break

                value = None
                if i + 1 < len(tokens):
                    value_token = tokens[i + 1]
                    if value_token.replace(".", "", 1).isdigit():
                        try:
                            value = float(value_token)
                            if value.is_integer():
                                value = int(value)
                        except ValueError:
                            value = value_token
                    elif value_token.startswith(("'", '"')) or value_token.endswith(("'", '"')):
                        value = value_token.strip("'\"")
                    elif numbers:
                        for num in numbers:
                            if str(num) in value_token or str(int(num)) in value_token:
                                value = num
                                break
                    else:
                        value = value_token

                if column and operator:
                    filter_dict = {
                        "column": column,
                        "operator": operator,
                        "value": value
                    }
                    filters.append(filter_dict)
                    logger.debug(f"Extracted filter: {filter_dict}")

        # ── Phase 2: multi-word operator phrase detection ──
        # Build a flat list of (phrase, op_symbol) sorted longest-first
        # so "no more than" matches before "more than", etc.
        all_phrases = []
        for op_symbol, phrases in OPERATOR_PATTERNS.items():
            for phrase in phrases:
                all_phrases.append((phrase, op_symbol))
        all_phrases.sort(key=lambda x: len(x[0].split()), reverse=True)

        # Track which token positions have been consumed by a matched phrase
        consumed_positions = set()

        for phrase, op_symbol in all_phrases:
            phrase_tokens_list = phrase.split()
            n = len(phrase_tokens_list)

            for i in range(len(tokens) - n + 1):
                # Skip if any position in this window was already consumed
                window_positions = set(range(i, i + n))
                if window_positions & consumed_positions:
                    continue

                window = tokens[i:i + n]
                # Case-insensitive match
                if [t.lower() for t in window] == phrase_tokens_list:
                    # Skip connector words ("has","have","is") when they
                    # appear right before another operator keyword.
                    # e.g. "salary has below 40000" — "has" is a connector,
                    # not an "=" operator.  Only skip single-word = phrases.
                    if op_symbol == "=" and n == 1 and phrase in ("has", "have", "is"):
                        # Check if the NEXT token is itself an operator phrase
                        next_is_op = False
                        if i + 1 < len(tokens):
                            next_tok = tokens[i + 1].lower()
                            for chk_phrase, _ in all_phrases:
                                chk_parts = chk_phrase.split()
                                if chk_parts[0] == next_tok:
                                    next_is_op = True
                                    break
                        if next_is_op:
                            continue  # skip — "has" is just a connector

                    # Found operator phrase starting at position i
                    col = self._find_column_left(tokens, i, schema_columns, lemmas, schema)
                    val = self._find_value_right(tokens, i + n, numbers)

                    if col and val is not None:
                        filter_dict = {
                            "column": col,
                            "operator": op_symbol,
                            "value": val
                        }
                        # Avoid duplicate filters
                        if filter_dict not in filters:
                            filters.append(filter_dict)
                            consumed_positions.update(window_positions)
                            logger.debug(f"Extracted filter (phrase): {filter_dict}")

        # ── Phase 3: Named-entity-based filter detection ──
        # Detect spaCy GPE/LOC entities (cities, countries) and PERSON entities
        # and map them to matching schema columns like "city", "state", "name".
        # Handles patterns like "show customers from Mumbai" → city = 'Mumbai'
        location_col_hints = ["city", "location", "state", "country", "region", "address"]
        name_col_hints = ["name", "first_name", "last_name", "customer_name", "employee_name"]

        for ent_text, ent_label in entities:
            if ent_label in ("GPE", "LOC"):
                # Check if this entity is already covered by an existing filter
                already_covered = any(
                    f.get("value") and str(f["value"]).lower() == ent_text.lower()
                    for f in filters
                )
                if already_covered:
                    continue

                # Find a matching location column in the schema
                matched_col = None
                for col in schema_columns:
                    if any(h in col.lower() for h in location_col_hints):
                        matched_col = col
                        break

                if matched_col:
                    filter_dict = {
                        "column": matched_col,
                        "operator": "=",
                        "value": ent_text
                    }
                    if filter_dict not in filters:
                        filters.append(filter_dict)
                        logger.debug(f"Extracted filter (entity): {filter_dict}")

            elif ent_label == "PERSON":
                already_covered = any(
                    f.get("value") and str(f["value"]).lower() == ent_text.lower()
                    for f in filters
                )
                if already_covered:
                    continue

                matched_col = None
                for col in schema_columns:
                    if any(h in col.lower() for h in name_col_hints):
                        matched_col = col
                        break

                if matched_col:
                    filter_dict = {
                        "column": matched_col,
                        "operator": "=",
                        "value": ent_text
                    }
                    if filter_dict not in filters:
                        filters.append(filter_dict)
                        logger.debug(f"Extracted filter (entity): {filter_dict}")

        return filters

    def _find_column_left(self, tokens: List[str], pos: int,
                          schema_columns: List[str],
                          lemmas: List[str] = None,
                          schema: Dict[str, Any] = None) -> Optional[str]:
        """
        Search left from pos for a matching schema column name.
        Falls back to inferring a likely column from the table schema
        when only a table name is found (e.g. "products" → "price").

        Args:
            tokens: Original tokens
            pos: Position of the operator phrase start
            schema_columns: Available column names
            lemmas: Optional lemmatized tokens
            schema: Optional full schema dict for table-to-column inference

        Returns:
            Matched column name or None
        """
        for j in range(pos - 1, max(pos - 5, -1), -1):
            token = tokens[j]
            # Direct match
            if self._match_to_column(token, schema_columns):
                return self._match_to_column(token, schema_columns)
            # Partial match (but not table-name-only matches)
            for col in schema_columns:
                if token.lower() in col.lower() or col.lower() in token.lower():
                    return col
            # Lemma match
            if lemmas and j < len(lemmas):
                lemma = lemmas[j]
                if self._match_to_column(lemma, schema_columns):
                    return self._match_to_column(lemma, schema_columns)

        # ── Fallback: infer column from table name ──
        # If no column was found but a table name appears to the left,
        # pick the most likely numeric column from that table's schema.
        # e.g. "find products under 5000" → table "products" → col "price"
        if schema:
            numeric_col_hints = [
                "price", "salary", "amount", "total", "cost", "budget",
                "revenue", "balance", "fee", "score", "rating", "quantity",
                "qty", "age", "rate", "value", "discount", "profit",
                "stock", "sales",
            ]
            for j in range(pos - 1, max(pos - 5, -1), -1):
                tok = tokens[j].lower().rstrip("s")  # "products" → "product"
                for table_name, table_info in schema.items():
                    tbl_lower = table_name.lower().rstrip("s")
                    if tok == tbl_lower or tok == table_name.lower():
                        # Found a table name — pick a likely column
                        cols = table_info.get("columns", [])
                        types = table_info.get("types", [])
                        # Prefer columns whose name hints at a numeric value
                        for col in cols:
                            if any(h in col.lower() for h in numeric_col_hints):
                                return col
                        # Else pick first REAL/INT column that isn't an id
                        for ci, col in enumerate(cols):
                            col_type = types[ci].upper() if ci < len(types) else ""
                            if col_type in ("REAL", "FLOAT", "DOUBLE", "NUMERIC", "INTEGER", "INT"):
                                if "id" not in col.lower():
                                    return col
        return None

    def _find_value_right(self, tokens: List[str], pos: int,
                          numbers: List[float]) -> Optional[Any]:
        """
        Get value immediately after operator phrase.

        Args:
            tokens: Original tokens
            pos: Position right after operator phrase ends
            numbers: Extracted numeric values

        Returns:
            Parsed value (int, float, or str) or None
        """
        if pos >= len(tokens):
            # Fallback to extracted numbers
            if numbers:
                return numbers[0]
            return None

        token = tokens[pos]

        # Try numeric
        try:
            num = float(token)
            if num.is_integer():
                return int(num)
            return num
        except ValueError:
            pass

        # Try extracting number from token like "40000" embedded in text
        # Check if token contains digits
        import re as _re
        num_match = _re.search(r'[\d]+\.?[\d]*', token)
        if num_match:
            try:
                num = float(num_match.group())
                if num.is_integer():
                    return int(num)
                return num
            except ValueError:
                pass

        # Check if token is in extracted numbers
        if numbers:
            for num in numbers:
                if str(num) in token or str(int(num)) in token:
                    return num
            return numbers[0]

        # Return as string value if meaningful
        if len(token) > 1:
            return token

        return None

    def _extract_aggregations(self, keywords: Set[str], lemmas: List[str],
                             schema_columns: List[str],
                             tokens: List[str] = None,
                             table_names: List[str] = None) -> List[Dict[str, str]]:
        """
        Extract aggregate functions (COUNT, SUM, AVG, MAX, MIN).

        Args:
            keywords: Set of SQL keyword hints
            lemmas: Lemmatized tokens
            schema_columns: Available column names
            tokens: Original tokens (optional, for fallback matching)

        Returns:
            List of aggregation dicts: [{"function": "AVG", "column": "salary"}, ...]
        """
        aggregations = []
        agg_functions = ["COUNT", "SUM", "AVG", "MAX", "MIN"]

        # Map from NL words to SQL aggregate functions
        agg_word_map = {
            "average": "AVG", "avg": "AVG", "mean": "AVG",
            "maximum": "MAX", "max": "MAX", "highest": "MAX",
            "largest": "MAX", "biggest": "MAX",
            "minimum": "MIN", "min": "MIN", "lowest": "MIN",
            "smallest": "MIN", "least": "MIN",
            "count": "COUNT", "total": "SUM", "sum": "SUM",
        }

        # Check for aggregate keywords
        found_aggs = [kw for kw in keywords if kw in agg_functions]

        for agg_func in found_aggs:
            # Try to find associated column
            column = None

            # Search lemmas for column match
            for lemma in lemmas:
                matched_col = self._match_to_column(lemma, schema_columns)
                if matched_col:
                    column = matched_col
                    break

            # Fallback: If no column matched from schema, look for the token
            # immediately after the aggregate keyword word in the original tokens.
            # This handles custom database columns like "x1", "y1" that don't
            # appear in the sample schema.
            if not column and tokens:
                # Build a set of table name words for filtering
                tbl_words = set()
                for tbl_name in (table_names or []):
                    tbl_words.add(tbl_name.lower())
                    # Also add individual words from multi-word table names
                    for part in tbl_name.lower().replace('_', ' ').split():
                        tbl_words.add(part)

                for i, tok in enumerate(tokens):
                    tok_lower = tok.lower()
                    if tok_lower in agg_word_map and agg_word_map[tok_lower] == agg_func:
                        # Look at tokens after the aggregate word, skipping
                        # common filler words and counting phrases
                        skip_words = {
                            "of", "the", "in", "for", "from", "a", "an", "is", "are",
                            "total", "rows", "records", "entries", "count", "all",
                            "how", "many", "number", "sum", "average", "max", "min",
                        }
                        # Also skip tokens that are part of table names
                        skip_words |= tbl_words

                        for j in range(i + 1, min(i + 5, len(tokens))):
                            candidate = tokens[j].lower()
                            if candidate not in skip_words and len(candidate) >= 1:
                                # Prefer tokens that match schema columns
                                matched = self._match_to_column(tokens[j], schema_columns)
                                if matched:
                                    column = matched
                                else:
                                    # Use the raw token as column name
                                    # (for custom databases with unknown columns)
                                    column = tokens[j]
                                break
                        break

            # For COUNT with no specific column, default to "*"
            if agg_func == "COUNT" and not column:
                column = "*"

            if column:
                aggregations.append({
                    "function": agg_func,
                    "column": column
                })
                logger.debug(f"Extracted aggregation: {aggregations[-1]}")

        return aggregations

    def _extract_order(self, keywords: Set[str], tokens: List[str],
                      lemmas: List[str], numbers: List[float],
                      schema_columns: List[str]) -> Optional[Dict[str, Any]]:
        """
        Extract ORDER BY clause with direction and LIMIT.

        Args:
            keywords: Set of SQL keyword hints
            tokens: Original tokens
            lemmas: Lemmatized tokens
            numbers: Extracted numeric values
            schema_columns: Available column names

        Returns:
            Order dict or None
        """
        order_result = None
        query_text = " ".join(tokens).lower()

        # Pattern: top/first N by column
        import re
        top_match = re.search(
            r'\b(top|first)\s+(\d+)\b', query_text)
        bot_match = re.search(
            r'\b(bottom|last|lowest)\s+(\d+)\b',
            query_text)

        # Find column after "by"
        by_match = re.search(
            r'\bby\s+(\w+)', query_text)

        if top_match or bot_match:
            match   = top_match or bot_match
            n       = int(match.group(2))
            direction = "DESC" if top_match else "ASC"

            # Find order column
            order_col = None
            if by_match:
                col_hint = by_match.group(1)
                for sc in schema_columns:
                    if col_hint in sc.lower() \
                       or sc.lower() in col_hint:
                        order_col = sc
                        break

            # Fallback: use first numeric column
            if not order_col:
                for sc in schema_columns:
                    scl = sc.lower()
                    if any(w in scl for w in
                           ["salary","price","amount",
                            "total","revenue","cost",
                            "score","rate","value"]):
                        order_col = sc
                        break

            if order_col:
                order_result = {
                    "column":    order_col,
                    "direction": direction,
                    "limit":     n
                }

        return order_result

    def _extract_group_by(self, keywords: Set[str], tokens: List[str],
                         schema_columns: List[str]) -> List[str]:
        """
        Extract GROUP BY columns.

        Args:
            keywords: Set of SQL keyword hints
            tokens: Original tokens
            schema_columns: Available column names

        Returns:
            List of column names
        """
        if "GROUP BY" not in keywords:
            return []

        group_columns = []

        # Find "group" token
        for i, token in enumerate(tokens):
            if token.lower() in ("group", "grouped", "per", "each", "every"):
                # Look for "by" next
                if i + 1 < len(tokens) and tokens[i + 1].lower() == "by":
                    # Search after "by"
                    for j in range(i + 2, min(i + 6, len(tokens))):
                        candidate = tokens[j]
                        matched_col = self._match_to_column(candidate, schema_columns)
                        if matched_col:
                            group_columns.append(matched_col)
                            break
                    break
                # If "group by" as single token
                elif token.lower() == "group" and i + 1 < len(tokens) and tokens[i + 1].lower() == "by":
                    for j in range(i + 2, min(i + 6, len(tokens))):
                        candidate = tokens[j]
                        matched_col = self._match_to_column(candidate, schema_columns)
                        if matched_col:
                            group_columns.append(matched_col)
                            break
                    break

        logger.debug(f"Extracted GROUP BY columns: {group_columns}")
        return group_columns

    def _determine_select_all(self, lemmas: List[str], keywords: Set[str],
                             aggregations: List[Dict], filters: List[Dict]) -> bool:
        """
        Determine if query intends SELECT * (all columns).

        Args:
            lemmas: Lemmatized tokens
            keywords: SQL keyword hints
            aggregations: Extracted aggregations
            filters: Extracted filters

        Returns:
            True if SELECT * is intended, False otherwise
        """
        # If there's a COUNT(*) aggregation, it's not select_all
        for agg in aggregations:
            if agg["function"] == "COUNT" and agg["column"] == "*":
                return False

        # If there are aggregations but not COUNT(*), likely select specific columns
        if aggregations:
            return False

        # If there are no explicit column references in lemmas that match schema
        # This is a heuristic - if query contains "all" or "show all" keywords
        select_keywords = {"all", "everything", "every"}
        if any(kw in lemmas for kw in select_keywords):
            return True

        # Default to True for simple queries without column specification
        return True

    def _match_to_column(self, token: str, schema_columns: List[str]) -> Optional[str]:
        """
        Match a token to the closest schema column.

        Matching strategy:
        1. Exact match (case-insensitive)
        2. Singular/plural match
        3. Partial string match (token is substring of column name)

        Args:
            token: Token to match
            schema_columns: List of valid column names

        Returns:
            Matched column name or None
        """
        token_lower = token.lower().strip()

        # 1. Exact match
        for col in schema_columns:
            if col.lower() == token_lower:
                logger.debug(f"Exact match: '{token}' -> '{col}'")
                return col

        # 2. Singular/plural match
        # Remove trailing 's' for plural -> singular
        token_singular = token_lower.rstrip('s')
        token_plural = token_lower + 's' if not token_lower.endswith('s') else token_lower

        for col in schema_columns:
            col_lower = col.lower()
            # Check if token matches singular of column or vice versa
            if (token_singular == col_lower or
                token_plural == col_lower or
                token_lower == col_lower.rstrip('s')):
                logger.debug(f"Singular/plural match: '{token}' -> '{col}'")
                return col

        # 3. Partial string match (token is substring of column)
        for col in schema_columns:
            col_lower = col.lower()
            if token_lower in col_lower or col_lower in token_lower:
                # Only match if reasonable length overlap
                if len(token_lower) >= 2 and len(col_lower) >= 2:
                    logger.debug(f"Partial match: '{token}' -> '{col}'")
                    return col

        return None


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EntityExtractor — Standalone Test")
    print("=" * 70)

    # Mock schema for testing
    mock_schema = {
        "customers": {
            "columns": ["customer_id", "first_name", "last_name", "email",
                       "phone", "city", "state", "join_date", "total_orders", "total_spent"],
            "types": ["INTEGER", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "TEXT", "INTEGER", "REAL"]
        },
        "products": {
            "columns": ["product_id", "product_name", "category", "price", "stock_qty", "rating"],
            "types": ["INTEGER", "TEXT", "TEXT", "REAL", "INTEGER", "REAL"]
        },
        "orders": {
            "columns": ["order_id", "customer_id", "order_date", "status", "total_amount", "city"],
            "types": ["INTEGER", "INTEGER", "TEXT", "TEXT", "REAL", "TEXT"]
        },
        "employees": {
            "columns": ["emp_id", "first_name", "last_name", "email", "dept_id", "salary", "hire_date", "city"],
            "types": ["INTEGER", "TEXT", "TEXT", "TEXT", "INTEGER", "REAL", "TEXT", "TEXT"]
        },
        "departments": {
            "columns": ["dept_id", "dept_name", "location", "budget"],
            "types": ["INTEGER", "TEXT", "TEXT", "REAL"]
        }
    }

    extractor = EntityExtractor()

    test_cases = [
        {
            "name": "Filter query",
            "preprocessed": {
                "tokens": ["show", "all", "customers", "where", "city", "=", "mumbai"],
                "lemmas": ["show", "all", "customer", "where", "city", "=", "mumbai"],
                "entities": [("mumbai", "GPE")],
                "numbers": [],
                "keywords": {"SELECT", "WHERE"}
            },
            "expected_filters": 1
        },
        {
            "name": "Aggregate query",
            "preprocessed": {
                "tokens": ["what", "is", "the", "average", "salary", "of", "employees"],
                "lemmas": ["what", "be", "the", "average", "salary", "of", "employee"],
                "entities": [],
                "numbers": [],
                "keywords": {"SELECT", "AVG"}
            },
            "expected_aggregations": 1
        },
        {
            "name": "Order + limit query",
            "preprocessed": {
                "tokens": ["find", "top", "5", "products", "by", "price"],
                "lemmas": ["find", "top", "5", "product", "by", "price"],
                "entities": [],
                "numbers": [5],
                "keywords": {"SELECT", "ORDER BY", "LIMIT"}
            },
            "expected_order": 1
        },
        {
            "name": "Group by query",
            "preprocessed": {
                "tokens": ["count", "orders", "group", "by", "city"],
                "lemmas": ["count", "order", "group", "by", "city"],
                "entities": [],
                "numbers": [],
                "keywords": {"SELECT", "COUNT", "GROUP BY"}
            },
            "expected_group_by": 1
        }
    ]

    all_passed = True

    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}] {test['name']}")
        print(f"     Input preprocessed: {test['preprocessed']}")

        result = extractor.extract(test["preprocessed"], mock_schema)

        print(f"     Filters:      {result['filters']}")
        print(f"     Aggregations: {result['aggregations']}")
        print(f"     Order:        {result['order']}")
        print(f"     Group by:     {result['group_by']}")
        print(f"     Select all:   {result['select_all']}")

        # Validate expected counts
        checks = []
        if "expected_filters" in test:
            checks.append(("filters", len(result['filters']), test['expected_filters']))
        if "expected_aggregations" in test:
            checks.append(("aggregations", len(result['aggregations']), test['expected_aggregations']))
        if "expected_order" in test:
            passed = 1 if result['order'] is not None else 0
            checks.append(("order", passed, test['expected_order']))
        if "expected_group_by" in test:
            checks.append(("group_by", len(result['group_by']), test['expected_group_by']))

        for field, actual, expected in checks:
            if actual >= expected:
                print(f"     [OK] {field}: expected ~{expected}, got {actual}")
            else:
                print(f"     [FAIL] {field}: expected {expected}, got {actual}")
                all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("Result: ALL TESTS PASSED")
    else:
        print("Result: SOME TESTS FAILED")
    print("=" * 70)
