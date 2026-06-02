"""
NL2SQL — Phase 5: Multi-Turn Conversation Memory
Run: python memory/context_manager.py
"""
import sys
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import MAX_CONTEXT_TURNS
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in a multi-turn conversation."""
    turn_id: int
    timestamp: str
    raw_query: str
    resolved_query: str
    intent: str
    sql: str
    result_count: int
    result_columns: List[str]
    entities: Dict[str, Any]
    confidence: float
    success: bool


class ContextManager:
    """Manages conversation history and context for a single session."""

    def __init__(self, session_id: str = None, max_turns: int = MAX_CONTEXT_TURNS):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.current_topic: Dict[str, Any] = {
            "tables": [],
            "columns": [],
            "filters": [],
            "intent": None
        }
        logger.info(f"Session started: {self.session_id}")

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a new turn to history, trim old turns, and update current topic."""
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

        if turn.entities.get("tables"):
            self.current_topic["tables"] = turn.entities["tables"]
        if turn.entities.get("columns"):
            self.current_topic["columns"] = turn.entities["columns"]
        self.current_topic["filters"] = turn.entities.get("filters", [])
        self.current_topic["intent"] = turn.intent

        logger.info(f"Turn {turn.turn_id} added: {turn.raw_query[:50]}")

    def get_context(self) -> Dict[str, Any]:
        """Return current context summary for injection into the NLP pipeline."""
        last_turn = self.get_last_turn()
        recent_turns = [asdict(t) for t in self.turns[-3:]] if self.turns else []

        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "current_topic": self.current_topic.copy(),
            "last_sql": last_turn.sql if last_turn else None,
            "last_intent": last_turn.intent if last_turn else None,
            "last_tables": last_turn.entities.get("tables", []) if last_turn else [],
            "last_columns": last_turn.entities.get("columns", []) if last_turn else [],
            "last_filters": last_turn.entities.get("filters", []) if last_turn else [],
            "recent_turns": recent_turns
        }

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Return the most recent turn or None if no turns exist."""
        return self.turns[-1] if self.turns else None

    def get_turn(self, turn_id: int) -> Optional[ConversationTurn]:
        """Return a specific turn by ID or None if not found."""
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None

    def clear(self) -> None:
        """Clear all turns and reset current topic to initial state."""
        self.turns = []
        self.current_topic = {
            "tables": [],
            "columns": [],
            "filters": [],
            "intent": None
        }
        logger.info(f"Session {self.session_id} cleared")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire session to a dictionary for saving."""
        return {
            "session_id": self.session_id,
            "max_turns": self.max_turns,
            "turns": [asdict(t) for t in self.turns],
            "current_topic": self.current_topic.copy()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextManager":
        """Reconstruct a ContextManager instance from a serialized dictionary."""
        cm = cls(session_id=data["session_id"], max_turns=data["max_turns"])
        cm.current_topic = data["current_topic"].copy()
        for turn_data in data["turns"]:
            cm.turns.append(ConversationTurn(**turn_data))
        return cm

    def save(self, path: str) -> None:
        """Save the session to a JSON file at the given path."""
        file_path = Path(path)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Session {self.session_id} saved to {file_path}")

    @classmethod
    def load(cls, path: str) -> "ContextManager":
        """Load a session from a JSON file and return a ContextManager instance."""
        file_path = Path(path)
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


if __name__ == "__main__":
    cm = ContextManager(session_id="test_001")

    turns_data = [
        {
            "raw_query": "Show customers from Mumbai",
            "resolved": "Show customers from Mumbai",
            "intent": "SELECT_WHERE",
            "sql": "SELECT * FROM customers WHERE city='Mumbai'",
            "result_count": 45,
            "result_columns": ["customer_id", "first_name", "city"],
            "entities": {
                "tables": ["customers"],
                "columns": ["city"],
                "filters": [{"column": "city", "op": "=", "value": "Mumbai"}],
                "aggregations": [], "order": None, "group_by": []
            },
            "confidence": 0.94,
            "success": True
        },
        {
            "raw_query": "Now filter by sales above 50000",
            "resolved": "Show customers from Mumbai where sales above 50000",
            "intent": "SELECT_WHERE",
            "sql": "SELECT * FROM customers WHERE city='Mumbai' AND total_spent > 50000",
            "result_count": 12,
            "result_columns": ["customer_id", "first_name", "city", "total_spent"],
            "entities": {
                "tables": ["customers"],
                "columns": ["city", "total_spent"],
                "filters": [
                    {"column": "city", "op": "=", "value": "Mumbai"},
                    {"column": "total_spent", "op": ">", "value": 50000}
                ],
                "aggregations": [], "order": None, "group_by": []
            },
            "confidence": 0.88,
            "success": True
        },
        {
            "raw_query": "How many are there?",
            "resolved": "Count customers from Mumbai where total_spent > 50000",
            "intent": "SELECT_AGGREGATE",
            "sql": "SELECT COUNT(*) FROM customers WHERE city='Mumbai' AND total_spent > 50000",
            "result_count": 1,
            "result_columns": ["COUNT(*)"],
            "entities": {
                "tables": ["customers"],
                "columns": ["customer_id"],
                "filters": [
                    {"column": "city", "op": "=", "value": "Mumbai"},
                    {"column": "total_spent", "op": ">", "value": 50000}
                ],
                "aggregations": [{"function": "COUNT", "column": "*"}],
                "order": None, "group_by": []
            },
            "confidence": 0.91,
            "success": True
        },
    ]

    for i, td in enumerate(turns_data):
        turn = ConversationTurn(
            turn_id=i + 1,
            timestamp=datetime.now().isoformat(),
            raw_query=td["raw_query"],
            resolved_query=td["resolved"],
            intent=td["intent"],
            sql=td["sql"],
            result_count=td["result_count"],
            result_columns=td["result_columns"],
            entities=td["entities"],
            confidence=td["confidence"],
            success=td["success"]
        )
        cm.add_turn(turn)

    ctx = cm.get_context()
    print(f"Session      : {ctx['session_id']}")
    print(f"Turns        : {ctx['turn_count']}")
    print(f"Last SQL     : {ctx['last_sql']}")
    print(f"Last tables  : {ctx['last_tables']}")
    print(f"Last filters : {ctx['last_filters']}")
    print("Context manager test PASSED")
