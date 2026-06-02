"""
NL2SQL — Phase 5: Session Orchestration for Multi-Turn Conversations
Run: python memory/session.py
"""
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import MAX_CONTEXT_TURNS
from memory.context_manager import ContextManager, ConversationTurn
from memory.resolver import QueryResolver
from utils.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Master session manager that orchestrates context and resolution for all sessions."""

    def __init__(self, db_path: str = "database/sample.db", max_turns: int = MAX_CONTEXT_TURNS):
        self.sessions: Dict[str, ContextManager] = {}
        self.resolver = QueryResolver()
        self.max_turns = max_turns
        self.db_path = db_path
        logger.info("SessionManager initialized")

    def create_session(self, session_id: str = None) -> str:
        """Create a new session and return its ID. Reuses existing ID if provided and exists."""
        if session_id and session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists, returning existing")
            return session_id
        sid = session_id or str(uuid.uuid4())[:8]
        self.sessions[sid] = ContextManager(session_id=sid, max_turns=self.max_turns)
        logger.info(f"Created session: {sid}")
        return sid

    def get_session(self, session_id: str) -> ContextManager:
        """Get an existing session or create a new one if not found."""
        if session_id not in self.sessions:
            self.create_session(session_id)
        return self.sessions[session_id]

    def process_query(self, session_id: str, raw_query: str) -> Dict[str, Any]:
        """
        Process a user query: resolve context, return structured result.
        Never crashes — always returns a valid dict.
        """
        try:
            cm = self.get_session(session_id)
            context = cm.get_context()
            resolved = self.resolver.resolve(raw_query, context)

            return {
                "session_id": session_id,
                "turn_number": context.get("turn_count", 0) + 1,
                "raw_query": raw_query,
                "resolved_query": resolved["resolved_query"],
                "is_followup": resolved["is_followup"],
                "followup_type": resolved["followup_type"],
                "context": {
                    "last_sql": context.get("last_sql"),
                    "last_tables": context.get("last_tables", []),
                    "last_filters": context.get("last_filters", []),
                    "last_intent": context.get("last_intent"),
                    "turn_count": context.get("turn_count", 0)
                },
                "ready_for_pipeline": True
            }
        except Exception as e:
            logger.error(f"Process query failed for session {session_id}: {e}")
            return {
                "session_id": session_id,
                "turn_number": 0,
                "raw_query": raw_query,
                "resolved_query": raw_query,
                "is_followup": False,
                "followup_type": "independent",
                "context": {},
                "ready_for_pipeline": True
            }

    def record_result(self, session_id: str, raw_query: str, resolved_query: str,
                      intent: str, sql: str, result_count: int, result_columns: List[str],
                      entities: Dict[str, Any], confidence: float, success: bool) -> ConversationTurn:
        """Record a completed turn (after SQL generation and execution) into the session."""
        cm = self.get_session(session_id)
        turn_id = len(cm.turns) + 1
        turn = ConversationTurn(
            turn_id=turn_id,
            timestamp=datetime.now().isoformat(),
            raw_query=raw_query,
            resolved_query=resolved_query,
            intent=intent,
            sql=sql,
            result_count=result_count,
            result_columns=result_columns,
            entities=entities,
            confidence=confidence,
            success=success
        )
        cm.add_turn(turn)
        return turn

    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Shortcut to retrieve context for a specific session."""
        if session_id in self.sessions:
            return self.sessions[session_id].get_context()
        return {}

    def clear_session(self, session_id: str) -> None:
        """Clear all turns and reset context for a session."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            logger.info(f"Session {session_id} cleared")

    def delete_session(self, session_id: str) -> None:
        """Remove a session entirely from memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session {session_id} deleted")

    def get_all_sessions(self) -> List[str]:
        """Return a list of all active session IDs."""
        return list(self.sessions.keys())

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Return a summary of the session suitable for UI display."""
        if session_id not in self.sessions:
            return {}
        cm = self.sessions[session_id]
        turns = cm.turns
        tables_used = list({t.entities.get("tables", [""])[0] for t in turns if t.entities.get("tables")})

        return {
            "session_id": session_id,
            "turn_count": len(turns),
            "start_time": turns[0].timestamp if turns else "",
            "last_activity": turns[-1].timestamp if turns else "",
            "tables_used": tables_used,
            "queries": [
                {"turn_id": t.turn_id, "query": t.raw_query, "sql": t.sql, "success": t.success}
                for t in turns[-10:]
            ]
        }

    def save_session(self, session_id: str, path: str = None) -> None:
        """Save a session to a JSON file. Defaults to logs/session_{id}.json."""
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not found")
            return
        if path is None:
            path = Path("logs") / f"session_{session_id}.json"
        else:
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions[session_id].save(str(path))
        logger.info(f"Session {session_id} saved to {path}")

    def load_session(self, path: str) -> str:
        """Load a session from a JSON file and register it in memory."""
        cm = ContextManager.load(path)
        sid = cm.session_id
        self.sessions[sid] = cm
        logger.info(f"Loaded session {sid} from {path}")
        return sid


if __name__ == "__main__":
    sm = SessionManager(db_path="database/sample.db")
    sid = sm.create_session("demo_session")

    CONVERSATION = [
        "Show all customers from Mumbai",
        "Now filter by sales above 50000",
        "Sort them by name",
        "How many are there?",
        "Show all products",
        "Which ones cost above 5000?",
    ]

    print("=" * 65)
    print("SessionManager — Phase 5 Conversation Test")
    print("=" * 65)

    turn_entities = [
        {  # Turn 1: Show all customers from Mumbai
            "tables": ["customers"],
            "columns": ["city"],
            "filters": [{"column": "city", "op": "=", "value": "Mumbai"}],
            "aggregations": [],
            "order": None,
            "group_by": []
        },
        {  # Turn 2: Now filter by sales above 50000
            "tables": ["customers"],
            "columns": ["city", "total_spent"],
            "filters": [
                {"column": "city", "op": "=", "value": "Mumbai"},
                {"column": "total_spent", "op": ">", "value": 50000}
            ],
            "aggregations": [],
            "order": None,
            "group_by": []
        },
        {  # Turn 3: Sort them by name
            "tables": ["customers"],
            "columns": ["city", "total_spent"],
            "filters": [
                {"column": "city", "op": "=", "value": "Mumbai"},
                {"column": "total_spent", "op": ">", "value": 50000}
            ],
            "aggregations": [],
            "order": {"column": "first_name", "direction": "ASC"},
            "group_by": []
        },
        {  # Turn 4: How many are there?
            "tables": ["customers"],
            "columns": ["customer_id"],
            "filters": [
                {"column": "city", "op": "=", "value": "Mumbai"},
                {"column": "total_spent", "op": ">", "value": 50000}
            ],
            "aggregations": [{"function": "COUNT", "column": "*"}],
            "order": None,
            "group_by": []
        },
        {  # Turn 5: Show all products
            "tables": ["products"],
            "columns": ["product_name"],
            "filters": [],
            "aggregations": [],
            "order": None,
            "group_by": []
        },
        {  # Turn 6: Which ones cost above 5000?
            "tables": ["products"],
            "columns": ["price"],
            "filters": [{"column": "price", "op": ">", "value": 5000}],
            "aggregations": [],
            "order": None,
            "group_by": []
        }
    ]

    for i, query in enumerate(CONVERSATION, 1):
        result = sm.process_query(sid, query)
        print(f"\nTurn {i}: {query}")
        print(f"  Followup     : {result['is_followup']}")
        print(f"  Type         : {result['followup_type']}")
        print(f"  Resolved     : {result['resolved_query']}")

        sm.record_result(
            session_id=sid,
            raw_query=query,
            resolved_query=result["resolved_query"],
            intent="SELECT_WHERE",
            sql=f"SELECT * FROM table_{i}",
            result_count=10 * i,
            result_columns=["id", "name", "city"],
            entities=turn_entities[i-1],
            confidence=0.90,
            success=True
        )

    summary = sm.get_session_summary(sid)
    print(f"\nSession Summary:")
    print(f"  Turns      : {summary['turn_count']}")
    print(f"  Tables used: {summary['tables_used']}")
    print(f"  Queries:")
    for q in summary["queries"]:
        print(f"    [{q['turn_id']}] {q['query']}")

    sm.save_session(sid)
    print(f"\nSession saved.")
    print("=" * 65)
    print("Phase 5 Memory complete — ready for Phase 6")
    print("=" * 65)
