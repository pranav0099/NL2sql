"""
NL2SQL — Phase 5: Query Resolution for Multi-Turn Conversations
Run: python memory/resolver.py
"""
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


class QueryResolver:
    """Resolves follow-up queries by injecting context from prior conversation turns."""

    FOLLOWUP_TRIGGERS = [
        "now", "also", "additionally", "furthermore",
        "and also", "what about", "how about"
    ]

    REFERENCE_WORDS = [
        "them", "they", "those", "these", "it", "that",
        "this", "the same", "same", "such", "the ones",
        "there", "their", "ones", "which"
    ]

    ADDITIVE_WORDS = [
        "also", "too", "as well", "additionally",
        "in addition", "furthermore", "moreover"
    ]

    REFINEMENT_WORDS = [
        "but", "except", "however", "instead",
        "rather", "only", "just"
    ]

    COUNT_WORDS = [
        "how many", "count", "total number",
        "number of", "how much"
    ]

    SORT_WORDS = [
        "sort", "order", "rank", "arrange",
        "top", "bottom", "highest", "lowest",
        "ascending", "descending", "asc", "desc"
    ]

    FILTER_WORDS = [
        "cost", "price", "above", "below", "greater", "less",
        "equal", "at least", "at most", ">", "<", "="
    ]

    def __init__(self):
        """Initialize resolver with no external dependencies."""
        pass

    def resolve(self, current_query: str, context: dict) -> dict:
        """
        Main resolution method. Returns a dict with resolved query and metadata.
        Never crashes — returns original query on failure.
        """
        try:
            original_query = current_query

            if not context.get("turn_count", 0):
                return {
                    "resolved_query": original_query,
                    "is_followup": False,
                    "followup_type": "independent",
                    "injected_context": {},
                    "original_query": original_query
                }

            query_type = self._detect_query_type(current_query, context)
            is_followup = query_type != "independent"

            if query_type == "aggregate":
                resolved = self._resolve_aggregate(current_query, context)
            elif query_type == "sort_add":
                resolved = self._resolve_sort(current_query, context)
            elif query_type == "filter_add":
                resolved = self._resolve_filter_add(current_query, context)
            elif query_type == "filter_refine":
                resolved = self._resolve_filter_refine(current_query, context)
            elif query_type == "full_reference":
                resolved = self._resolve_full_reference(current_query, context)
            else:
                resolved = current_query

            return {
                "resolved_query": resolved,
                "is_followup": is_followup,
                "followup_type": query_type,
                "injected_context": {},
                "original_query": original_query
            }

        except Exception as e:
            logger.error(f"Resolution failed: {e}")
            return {
                "resolved_query": current_query,
                "is_followup": False,
                "followup_type": "independent",
                "injected_context": {},
                "original_query": current_query
            }

    def _detect_query_type(self, query: str, context: dict) -> str:
        """Detect the type of follow-up query by checking patterns in order."""
        query_lower = query.lower()

        has_count = any(cw in query_lower for cw in self.COUNT_WORDS)
        has_ref = any(rw in query_lower for rw in self.REFERENCE_WORDS)
        has_sort = any(sw in query_lower for sw in self.SORT_WORDS)
        has_filter = any(fw in query_lower for fw in self.FILTER_WORDS)

        if has_count and has_ref:
            return "aggregate"
        if has_sort and has_ref:
            return "sort_add"
        if any(query_lower.startswith(ft) for ft in self.FOLLOWUP_TRIGGERS):
            return "filter_add"
        if any(aw in query_lower for aw in self.ADDITIVE_WORDS):
            return "filter_add"
        if any(rw in query_lower for rw in self.REFINEMENT_WORDS):
            return "filter_refine"
        if has_ref and has_filter:
            return "filter_add"
        if has_ref:
            return "full_reference"
        return "independent"

    def _resolve_aggregate(self, query: str, context: dict) -> str:
        """Resolve aggregate queries (how many/count) using prior table and filters."""
        last_table = context.get("last_tables", [""])[0]
        filters_text = self._extract_filters_as_text(context.get("last_filters", []))
        resolved = f"Count {last_table}"
        if filters_text:
            resolved += f" {filters_text}"
        return resolved

    def _resolve_sort(self, query: str, context: dict) -> str:
        """Resolve sort queries by injecting prior context and sort column."""
        last_table = context.get("last_tables", [""])[0]
        filters_text = self._extract_filters_as_text(context.get("last_filters", []))
        sort_col = query.lower().split("by")[-1].strip() if "by" in query.lower() else "name"

        resolved = f"Show {last_table}"
        if filters_text:
            resolved += f" {filters_text}"
        resolved += f" sorted by {sort_col}"
        return resolved

    def _resolve_filter_add(self, query: str, context: dict) -> str:
        """Resolve filter addition queries by combining prior and new filters."""
        last_table = context.get("last_tables", [""])[0]
        filters_text = self._extract_filters_as_text(context.get("last_filters", []))
        new_filter = query.lower()

        for trigger in ["filter by", "where", "and also", "now filter by", "also where"]:
            if trigger in new_filter:
                new_filter = new_filter.split(trigger, 1)[1].strip()
                break

        # Remove reference words from the new filter
        for rw in self.REFERENCE_WORDS:
            pattern = re.compile(r'\b' + re.escape(rw) + r'\b', re.IGNORECASE)
            new_filter = pattern.sub('', new_filter)
        new_filter = re.sub(r'\s+', ' ', new_filter).strip()
        # Clean up leading/trailing punctuation
        new_filter = new_filter.strip(' ?')

        resolved = f"Show {last_table}"
        if filters_text:
            resolved += f" where {filters_text}"
        if new_filter:
            resolved += f" and {new_filter}" if filters_text else f" where {new_filter}"
        return resolved

    def _resolve_filter_refine(self, query: str, context: dict) -> str:
        """Resolve filter refinement queries by replacing prior filters."""
        last_table = context.get("last_tables", [""])[0]
        new_filter = query.lower()
        for rw in self.REFINEMENT_WORDS:
            if rw in new_filter:
                new_filter = new_filter.split(rw, 1)[1].strip()
                break
        return f"Show {last_table} where {new_filter}"

    def _resolve_full_reference(self, query: str, context: dict) -> str:
        """Resolve full reference queries by replacing pronouns with entities."""
        last_table = context.get("last_tables", [""])[0]
        resolved = query
        for rw in self.REFERENCE_WORDS:
            pattern = re.compile(r'\b' + re.escape(rw) + r'\b', re.IGNORECASE)
            resolved = pattern.sub(last_table, resolved)
        return resolved.strip()

    def _extract_table_from_sql(self, sql: str) -> str:
        """Extract the first table name from a SQL string."""
        sql_lower = sql.lower()
        if ' from ' not in sql_lower:
            return ""
        after_from = sql_lower.split(' from ', 1)[1].strip()
        table = after_from.split()[0] if after_from else ""
        return table.rstrip(',()')

    def _extract_filters_as_text(self, filters: list[dict]) -> str:
        """Convert a list of filter dicts to a natural language string."""
        op_map = {'=': 'is', '>': 'above', '<': 'below', '>=': 'at least', '<=': 'at most', '!=': 'not'}
        parts = []
        for f in filters:
            col = f.get("column", "")
            op = f.get("op", "=")
            val = f.get("value", "")
            parts.append(f"{col} {op_map.get(op, op)} {val}")
        return " and ".join(parts)


if __name__ == "__main__":
    resolver = QueryResolver()
    mock_context = {
        "turn_count": 2,
        "last_sql": "SELECT * FROM customers WHERE city = 'Mumbai'",
        "last_intent": "SELECT_WHERE",
        "last_tables": ["customers"],
        "last_columns": ["city"],
        "last_filters": [{"column": "city", "op": "=", "value": "Mumbai"}],
        "recent_turns": []
    }

    TEST_CASES = [
        ("how many are there?", "aggregate"),
        ("sort them by name", "sort_add"),
        ("now filter by sales above 50000", "filter_add"),
        ("also show their email", "filter_add"),
        ("but only from Pune", "filter_refine"),
        ("show me their total spending", "full_reference"),
        ("show all products", "independent"),
        ("what is the average salary", "independent"),
    ]

    print("=" * 65)
    print("Query Resolver — Phase 5 Test")
    print("=" * 65)
    passed = 0
    for query, expected in TEST_CASES:
        result = resolver.resolve(query, mock_context)
        ok = result["followup_type"] == expected
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {query:<40} type={result['followup_type']:<15}")
        if result["is_followup"]:
            print(f"       resolved -> {result['resolved_query']}")

    print(f"\nResult: {passed}/{len(TEST_CASES)} correct")
    print("=" * 65)
