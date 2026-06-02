"""
NL2SQL — Phase 2: Text Preprocessing with spaCy
Run: python nlp/preprocessor.py
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any
import logging

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger
from config.config import INTENTS

logger = get_logger(__name__)

# =============================================================================
# SPACY MODEL MANAGEMENT
# =============================================================================

def _ensure_spacy_model(model_name: str = "en_core_web_sm") -> None:
    """
    Ensure the spaCy model is installed. Download if missing.

    Args:
        model_name: Name of the spaCy model to ensure
    """
    try:
        import spacy
        spacy.load(model_name)
        logger.debug(f"spaCy model '{model_name}' is available")
    except ImportError:
        logger.error("spaCy is not installed. Please install: pip install spacy")
        raise
    except OSError:
        logger.warning(f"spaCy model '{model_name}' not found. Downloading...")
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Successfully downloaded spaCy model '{model_name}'")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download spaCy model: {e.stderr}")
            raise


# =============================================================================
# CONTRACTION MAP
# =============================================================================

CONTRACTION_MAP = {
    "what's": "what is",
    "don't": "do not",
    "i'm": "i am",
    "it's": "it is",
    "there's": "there is",
    "that's": "that is",
    "how's": "how is",
    "where's": "where is",
    "who's": "who is",
    "when's": "when is",
    "why's": "why is",
    "can't": "cannot",
    "won't": "will not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not",
    "mightn't": "might not",
    "needn't": "need not",
    "let's": "let us",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
}


# =============================================================================
# SQL KEYWORD MAP
# =============================================================================

SQL_KEYWORD_MAP = {
    # SELECT triggers
    "show": "SELECT",
    "list": "SELECT",
    "display": "SELECT",
    "get": "SELECT",
    "find": "SELECT",
    "fetch": "SELECT",
    "give": "SELECT",
    "tell": "SELECT",
    "what": "SELECT",
    "which": "SELECT",
    "who": "SELECT",
    "all": "SELECT",

    # Aggregate triggers
    "count total": "COUNT",
    "total rows": "COUNT",
    "total records": "COUNT",
    "count": "COUNT",
    "how many": "COUNT",
    "total": "SUM",
    "sum": "SUM",
    "average": "AVG",
    "avg": "AVG",
    "mean": "AVG",
    "maximum": "MAX",
    "max": "MAX",
    "highest": "ORDER BY DESC",
    "largest": "MAX",
    "biggest": "MAX",
    "minimum": "MIN",
    "min": "MIN",
    "lowest": "ORDER BY ASC",
    "smallest": "MIN",
    "least": "MIN",

    # WHERE triggers
    "where": "WHERE",
    "filter": "WHERE",
    "with": "WHERE",
    "having": "HAVING",

    # Comparison operators
    "greater than": ">",
    "more than": ">",
    "above": ">",
    "higher than": ">",
    "exceeds": ">",
    "over": ">",
    "larger than": ">",
    "bigger than": ">",
    "less than": "<",
    "below": "<",
    "under": "<",
    "lower than": "<",
    "fewer than": "<",
    "smaller than": "<",
    "beneath": "<",
    "not more than": "<=",
    "at most": "<=",
    "no more than": "<=",
    "no greater than": "<=",
    "at least": ">=",
    "no less than": ">=",
    "minimum of": ">=",
    "equal to": "=",
    "equals": "=",
    "not equal": "!=",
    "not": "!=",
    "with": "WHERE",
    "that has": "WHERE",
    "that have": "WHERE",
    "where": "WHERE",
    "filter": "WHERE",
    "whose": "WHERE",

    # ORDER triggers
    "order by": "ORDER BY",
    "sort by": "ORDER BY",
    "sorted by": "ORDER BY",
    "ranked by": "ORDER BY",
    "order": "ORDER BY",
    "sort": "ORDER BY",
    "sorted": "ORDER BY",
    "rank": "ORDER BY",
    "ranked": "ORDER BY",
    "ascending": "ASC",
    "descending": "DESC",
    "top": "ORDER BY DESC LIMIT",
    "bottom": "ORDER BY ASC LIMIT",
    "first": "LIMIT",
    "last": "ORDER BY DESC LIMIT",
    "highest": "ORDER BY DESC",
    "lowest": "ORDER BY ASC",

    # GROUP triggers
    "group by": "GROUP BY",
    "grouped by": "GROUP BY",
    "per": "GROUP BY",
    "each": "GROUP BY",
    "every": "GROUP BY",

    # JOIN triggers
    "join": "JOIN",
    "with their": "JOIN",
    "along with": "JOIN",
    "including": "JOIN",
    "combined with": "JOIN",

    # LIMIT triggers
    "limit": "LIMIT",
    "only": "LIMIT",

    # Logical
    "and": "AND",
    "or": "OR"
}


# =============================================================================
# PREPROCESSOR CLASS
# =============================================================================

class QueryPreprocessor:
    """
    spaCy-based text preprocessing pipeline for natural language queries.

    Handles tokenization, lemmatization, POS tagging, entity recognition,
    keyword extraction for SQL clause hints, and numeric value extraction.
    """

    # Common typos → correct spellings (checked before NLP processing)
    COMMON_TYPOS = {
        "disignation":  "designation",
        "desgination":  "designation",
        "desigation":   "designation",
        "coustomers":   "customers",
        "custommers":   "customers",
        "cusotmers":    "customers",
        "emplyees":     "employees",
        "employes":     "employees",
        "employess":    "employees",
        "produts":      "products",
        "producst":     "products",
        "ordres":       "orders",
        "ordes":        "orders",
        "salay":        "salary",
        "sallary":      "salary",
        "deparment":    "department",
        "departmnet":   "department",
        "departement":  "department",
        "recods":       "records",
        "recrods":      "records",
        "recorts":      "records",
        "shwo":         "show",
        "sohw":         "show",
        "form":         "from",
        "fom":          "from",
        "eamil":        "email",
        "emial":        "email",
        "emai":         "email",
        "adress":       "address",
        "adres":        "address",
        "ciy":          "city",
        "ciyt":         "city",
        "cites":        "cities",
        "categroy":     "category",
        "catgeory":     "category",
        "catagory":     "category",
        "statsu":       "status",
        "satus":        "status",
        "detailes":     "details",
        "infomation":   "information",
        "manger":       "manager",
        "managr":       "manager",
    }

    def __init__(self):
        """Initialize spaCy model, downloading if necessary."""
        logger.info("Initializing QueryPreprocessor")
        _ensure_spacy_model("en_core_web_sm")
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
        logger.info("QueryPreprocessor initialized successfully")

    def _fix_typos(self, text: str) -> str:
        """
        Fix common typos before NLP processing.
        Also uses fuzzy matching for unknown typos.
        """
        # Fix known typos
        words = text.split()
        fixed = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.COMMON_TYPOS:
                fixed.append(self.COMMON_TYPOS[word_lower])
            else:
                fixed.append(word)
        text = " ".join(fixed)

        # Fuzzy match against schema words if fuzzywuzzy available
        try:
            from fuzzywuzzy import process
            schema_words = [
                "show", "list", "display",
                "customers", "employees", "orders",
                "products", "departments", "sales",
                "designation", "salary", "department",
                "email", "city", "status", "category",
                "records", "data", "all", "from",
                "where", "count", "average", "total",
                "maximum", "minimum", "above", "below"
            ]
            words = text.split()
            corrected = []
            for word in words:
                if len(word) > 4:
                    match, score = process.extractOne(
                        word, schema_words)
                    if score >= 85 and match != word.lower():
                        corrected.append(match)
                    else:
                        corrected.append(word)
                else:
                    corrected.append(word)
            return " ".join(corrected)
        except ImportError:
            return text

    def preprocess(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query into structured components.

        Args:
            query: Raw user query string

        Returns:
            Dictionary containing:
            {
                "original": str,
                "normalized": str,
                "tokens": List[str],
                "lemmas": List[str],
                "pos_tags": List[Tuple[str, str]],
                "entities": List[Tuple[str, str]],
                "noun_chunks": List[str],
                "numbers": List[float],
                "keywords": Set[str]
            }
        """
        # Fix typos first (before any NLP processing)
        query = self._fix_typos(query)
        logger.info(f"Preprocessing query: '{query}'")

        original = query.strip()
        normalized = self._normalize(original)

        # Process with spaCy
        doc = self.nlp(normalized)

        # Extract components
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        pos_tags = [(token.text, token.pos_) for token in doc]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        numbers = self._extract_numbers(doc)
        keywords = self._extract_keywords(normalized, entities)

        result = {
            "original": original,
            "normalized": normalized,
            "tokens": tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "entities": entities,
            "noun_chunks": noun_chunks,
            "numbers": numbers,
            "keywords": keywords
        }

        logger.debug(f"Preprocessing complete: tokens={len(tokens)}, keywords={len(keywords)}")
        return result

    def _normalize(self, text: str) -> str:
        """
        Normalize text: lowercase, expand contractions, clean special chars.

        Args:
            text: Input text string

        Returns:
            Normalized text string
        """
        # Lowercase
        text = text.lower().strip()

        # Expand contractions
        for contraction, expansion in CONTRACTION_MAP.items():
            text = text.replace(contraction, expansion)

        # Remove special characters except letters, digits, spaces, apostrophes
        # Keep alphanumeric, spaces, and apostrophes
        cleaned = []
        for char in text:
            if char.isalnum() or char.isspace() or char == "'":
                cleaned.append(char)
            else:
                cleaned.append(' ')

        text = ''.join(cleaned)

        # Collapse multiple spaces
        text = ' '.join(text.split())

        return text

    def _extract_keywords(self, text: str, entities: List[Tuple[str, str]]) -> Set[str]:
        """
        Extract SQL keyword hints from text using multi-word phrase matching.

        Matching strategy:
        1. Check multi-word phrases first (longer phrases before shorter)
        2. Track matched spans to avoid shorter phrases re-matching
        3. Prevent 'total' -> SUM when 'total rows' -> COUNT already matched

        Args:
            text: Normalized query text
            entities: List of (entity_text, entity_label) from spaCy

        Returns:
            Set of SQL keyword strings (e.g., "SELECT", "WHERE", ">")
        """
        keywords = set()
        text_lower = text.lower()

        # Sort map keys by length (descending) to match longer phrases first
        sorted_phrases = sorted(SQL_KEYWORD_MAP.keys(), key=len, reverse=True)

        # Track which positions in text have been consumed by longer phrases
        consumed_spans = []

        for phrase in sorted_phrases:
            phrase_lower = phrase.lower()
            # Check if phrase appears in text
            idx = text_lower.find(phrase_lower)
            if idx >= 0:
                span_end = idx + len(phrase_lower)

                # Check if this span overlaps with an already-consumed span
                # that mapped to a DIFFERENT keyword value
                is_overlap = False
                for (cs, ce, ck) in consumed_spans:
                    if idx < ce and span_end > cs:
                        # Overlap detected — skip if they map to different keywords
                        if SQL_KEYWORD_MAP[phrase] != ck:
                            is_overlap = True
                            break

                if not is_overlap:
                    kw = SQL_KEYWORD_MAP[phrase]
                    keywords.add(kw)
                    consumed_spans.append((idx, span_end, kw))

        return keywords

    def _extract_numbers(self, doc) -> List[float]:
        """
        Extract all numeric values from spaCy doc.

        Handles:
        - Cardinal numbers (e.g., "5", "50000")
        - Numeric tokens
        - Attempts word-to-number conversion for written numbers

        Args:
            doc: spaCy Doc object

        Returns:
            List of numeric values (int or float)
        """
        numbers = []

        # Try to import word2number for written number conversion
        try:
            from word2number import w2n
            word2number_available = True
        except ImportError:
            word2number_available = False
            logger.debug("word2number not available, using digit extraction only")

        for token in doc:
            # Direct numeric tokens
            if token.like_num:
                try:
                    # Convert to float/int
                    num = float(token.text)
                    if num.is_integer():
                        num = int(num)
                    numbers.append(num)
                except ValueError:
                    pass

            # Try word-to-number conversion for potential written numbers
            if word2number_available and token.pos_ in ("NOUN", "ADJ", "NUM"):
                try:
                    # Only try conversion for single tokens that aren't pure digits
                    if not token.text.isdigit() and len(token.text) > 1:
                        num = w2n.word_to_num(token.text)
                        numbers.append(num)
                except (ValueError, AttributeError):
                    pass

        # Also check entities for cardinals
        for ent in doc.ents:
            ent_text = ent.text
            ent_label = ent.label_
            if ent_label == "CARDINAL":
                try:
                    num = float(ent_text.replace(",", ""))
                    if num.is_integer():
                        num = int(num)
                    if num not in numbers:
                        numbers.append(num)
                except ValueError:
                    pass

        logger.debug(f"Extracted numbers: {numbers}")
        return numbers


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QueryPreprocessor — Standalone Test")
    print("=" * 70)

    preprocessor = QueryPreprocessor()

    test_queries = [
        "Show me all customers from Mumbai",
        "What is the average salary of employees?",
        "Find top 10 products with price greater than 5000",
        "Count orders grouped by city where total is above 100000",
        "Show customers and their orders joined together"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Query: {query}")
        print("-" * 70)
        result = preprocessor.preprocess(query)

        print(f"Original    : {result['original']}")
        print(f"Normalized  : {result['normalized']}")
        print(f"Tokens      : {result['tokens']}")
        print(f"Lemmas      : {result['lemmas']}")
        print(f"POS Tags    : {result['pos_tags']}")
        print(f"Entities    : {result['entities']}")
        print(f"Noun Chunks : {result['noun_chunks']}")
        print(f"Numbers     : {result['numbers']}")
        print(f"Keywords    : {sorted(result['keywords'])}")

    print("\n" + "=" * 70)
    print("Preprocessor test complete")
    print("=" * 70)
