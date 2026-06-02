"""
NL2SQL — Phase 8: Streamlit Conversational UI
Launch: streamlit run ui/app.py

Complete conversational interface for querying a SQLite database
using natural language. Integrates the NL2SQL engine (Phase 6)
and the auto-visualization chart engine (Phase 7).
"""

# =============================================================================
# SECTION 1 — Page config and imports
# =============================================================================

import streamlit as st

# ── Page config must be the FIRST Streamlit call ────────────────────────────
st.set_page_config(
    page_title="NL2SQL — Talk to Your Database",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import time
import json
import sqlite3

# ── Project root on sys.path ────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

from engine.nl2sql_engine import NL2SQLEngine
from visualization.chart_engine import ChartEngine
from utils.schema_analyzer import SchemaAnalyzer, csv_to_sqlite


# =============================================================================
# HELPER: generate smart follow-up suggestions
# =============================================================================

def generate_next_suggestions(
        result: dict,
        schema_suggestions: list,
        db_path: str = "database/sample.db") -> list:
    """
    Generate 4 smart follow-up query suggestions
    based on the current query result and intent.
    Always returns exactly 4 suggestions.
    Never returns the same query just asked.
    """
    import sqlite3
    from pathlib import Path

    suggestions = []
    intent      = result.get("intent", "SELECT")
    success     = result.get("success", False)
    current_q   = result.get("query", "").lower().strip()
    row_count   = 0
    columns     = []

    if result.get("results"):
        row_count = result["results"].get("row_count", 0)
        columns   = result["results"].get("columns", [])

    # Get table name from SQL
    sql = result.get("sql", "") or ""
    table_name = ""
    if "FROM" in sql.upper():
        parts = sql.upper().split("FROM")
        if len(parts) > 1:
            table_name = parts[1].strip().split()[0].lower()
            table_name = table_name.replace(";", "").strip()

    # Get real columns from database for better suggestions
    real_columns = columns.copy()
    if table_name and Path(db_path).exists():
        try:
            conn = sqlite3.connect(db_path)
            pragma = conn.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()
            real_columns = [row[1] for row in pragma]
            conn.close()
        except Exception:
            real_columns = columns

    # Find a numeric column and text column for suggestions
    numeric_col = None
    text_col    = None
    for col in real_columns:
        col_lower = col.lower()
        if any(w in col_lower for w in
               ["amount", "salary", "price", "revenue", "total",
                "count", "score", "age", "value", "cost", "qty",
                "quantity", "rate", "budget", "sales"]):
            if numeric_col is None:
                numeric_col = col
        elif any(w in col_lower for w in
                 ["name", "city", "category", "status", "type",
                  "department", "region", "country", "gender",
                  "month", "year", "date", "class", "group"]):
            if text_col is None:
                text_col = col

    # Fallback column names
    if not numeric_col and len(real_columns) > 1:
        numeric_col = real_columns[-1]
    if not text_col and len(real_columns) > 0:
        text_col = real_columns[0]

    # Build intent-based suggestions
    if not success:
        if table_name:
            suggestions = [
                f"Show all records from {table_name}",
                f"Count total rows in {table_name}",
                f"Show first 10 rows from {table_name}",
                f"Show all columns in {table_name}",
            ]
        else:
            suggestions = [
                "Show all records",
                "Count total records",
                "Show first 10 rows",
                "List all tables",
            ]

    elif intent == "SELECT_AGGREGATE":
        if table_name:
            suggestions = [
                f"Show all records from {table_name}",
                f"Show first 10 rows from {table_name}",
            ]
            if text_col:
                suggestions.append(
                    f"Count records grouped by {text_col}")
            if numeric_col and text_col:
                suggestions.append(
                    f"Show average {numeric_col} by {text_col}")
            elif numeric_col:
                suggestions.append(
                    f"Show maximum {numeric_col}")

    elif intent in ("SELECT", "SELECT_LIMIT"):
        suggestions = []
        if table_name:
            suggestions.append(
                f"Count total rows in {table_name}")
        if numeric_col:
            suggestions.append(
                f"What is the average {numeric_col}?")
            suggestions.append(
                f"Show top 10 by {numeric_col}")
        if text_col:
            suggestions.append(
                f"Count records grouped by {text_col}")
        if len(suggestions) < 4 and table_name:
            suggestions.append(
                f"Show maximum {numeric_col or 'value'}"
                f" from {table_name}")

    elif intent == "SELECT_WHERE":
        suggestions = [
            "How many are there?",
            "Sort them by name",
        ]
        if numeric_col:
            suggestions.append(
                f"What is the average {numeric_col}?")
        if table_name:
            suggestions.append(
                f"Show all records from {table_name}")

    elif intent == "SELECT_GROUP":
        suggestions = [
            "Sort by highest value",
            "Show top 5 results",
        ]
        if table_name:
            suggestions.append(
                f"Count total rows in {table_name}")
        if numeric_col:
            suggestions.append(
                f"What is the total {numeric_col}?")

    elif intent == "SELECT_ORDER":
        suggestions = []
        if table_name:
            suggestions.append(
                f"Count total rows in {table_name}")
        if numeric_col:
            suggestions.append(
                f"What is the average {numeric_col}?")
        if text_col:
            suggestions.append(
                f"Count records grouped by {text_col}")
        suggestions.append("Show first 10 rows")

    elif intent == "SELECT_JOIN":
        suggestions = [
            "Count total records",
            "Show top 10 results",
            "Group by category",
            "Show all records",
        ]

    else:
        if table_name:
            suggestions = [
                f"Show all records from {table_name}",
                f"Count total rows in {table_name}",
                f"Show first 10 rows from {table_name}",
                "Show average value",
            ]
        else:
            suggestions = [
                "Show all records",
                "Count total records",
                "Show first 10 rows",
                "Group by category",
            ]

    # Add from schema suggestions if not enough
    for s in schema_suggestions:
        if len(suggestions) >= 4:
            break
        sq = s["query"] if isinstance(s, dict) else s
        if sq.lower().strip() != current_q:
            suggestions.append(sq)

    # Remove current query from suggestions
    suggestions = [
        s for s in suggestions
        if s.lower().strip() != current_q
    ]

    # Remove duplicates keeping order
    seen  = set()
    final = []
    for s in suggestions:
        if s.lower() not in seen:
            seen.add(s.lower())
            final.append(s)

    # Always return exactly 4
    while len(final) < 4:
        defaults = [
            "Show all records",
            "Count total records",
            "Show first 10 rows",
            "Show average value",
        ]
        for d in defaults:
            if d.lower() not in seen and len(final) < 4:
                final.append(d)
                seen.add(d.lower())

    return final[:4]


# =============================================================================
# SECTION 2 — Custom CSS
# =============================================================================

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F3864;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .sql-box {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Cascadia Code', 'Fira Code', monospace;
        font-size: 0.85rem;
        margin: 0.5rem 0;
        overflow-x: auto;
        border: 1px solid #333;
    }
    .sql-keyword {
        color: #569CD6;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E75B6;
        margin: 0.3rem 0;
    }
    .success-badge {
        background: #d4edda;
        color: #155724;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .error-badge {
        background: #f8d7da;
        color: #721c24;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .followup-badge {
        background: #cce5ff;
        color: #004085;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    .welcome-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
    }
    .welcome-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .welcome-card h4 {
        color: #1F3864;
        margin-bottom: 0.5rem;
    }
    .welcome-card p {
        color: #666;
        font-size: 0.9rem;
        margin: 0;
    }
    .sidebar-stat {
        background: linear-gradient(135deg, #f0f4f8, #e2e8f0);
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    .response-time {
        color: #888;
        font-size: 0.8rem;
        font-style: italic;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 0.3rem;
    }
</style>""", unsafe_allow_html=True)


# =============================================================================
# SECTION 3 — Session state initialization
# =============================================================================

st.session_state.setdefault("messages",         [])
st.session_state.setdefault("engine",           None)
st.session_state.setdefault("chart_engine",     None)
st.session_state.setdefault("session_id",       None)
st.session_state.setdefault("query_count",      0)
st.session_state.setdefault("db_path",          "database/sample.db")
st.session_state.setdefault("pending_query",    None)
st.session_state.setdefault("uploaded_db_path", None)
st.session_state.setdefault("schema_analyzer",  None)
st.session_state.setdefault("suggestions",      [])
st.session_state.setdefault("last_upload_key",  None)


# =============================================================================
# SECTION 4 — Engine loader (cached)
# =============================================================================

@st.cache_resource
def load_engine(db_path: str) -> NL2SQLEngine:
    """Load the NL2SQL engine once and cache across reruns."""
    return NL2SQLEngine(db_path=db_path)


@st.cache_resource
def load_chart_engine() -> ChartEngine:
    """Load the chart engine once and cache across reruns."""
    return ChartEngine()


def initialize_engine():
    """Initialize or retrieve the cached engine and session."""
    try:
        if st.session_state.engine is None:
            st.session_state.engine = load_engine(st.session_state.db_path)
            st.session_state.chart_engine = load_chart_engine()
            st.session_state.session_id = st.session_state.engine.create_session()
        return True
    except Exception as e:
        st.error(f"⚠️ Failed to load NL2SQL Engine: {e}")
        st.info(
            "Make sure all model files exist:\n"
            "- `models/saved/intent_classifier.pkl`\n"
            "- `models/saved/tfidf_vectorizer.pkl`\n"
            "- `models/saved/transformer_nl2sql.pt`\n"
            "- `data/processed/vocab.json`\n"
            "- `database/sample.db`"
        )
        return False


# ── Load engine ─────────────────────────────────────────────────────────────
engine_ready = initialize_engine()


# =============================================================================
# SECTION 5 — Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## 🗄️ NL2SQL")
    st.caption("Conversational Database Query System")
    st.divider()

    # ── Database section ────────────────────────────────────────────────────
    st.subheader("📁 Database")
    current_db = st.session_state.get("db_path", "database/sample.db")
    st.code(current_db)
    if current_db != "database/sample.db":
        st.caption("📤 Custom database active")
    else:
        st.caption("✅ Default database")

    if st.button("🔄 Reset Session", use_container_width=True):
        # Clear conversation
        st.session_state.session_id       = None
        st.session_state.messages         = []
        st.session_state.query_count      = 0

        # Reset database back to default
        st.session_state.db_path          = "database/sample.db"
        st.session_state.engine           = None

        # Clear all upload state
        st.session_state.uploaded_db_path = None
        st.session_state.schema_analyzer  = None
        st.session_state.suggestions      = []
        st.session_state.last_upload_key  = None
        st.session_state.pending_query    = None

        st.success("Session reset. Using default database.")
        st.rerun()

    st.divider()

    # ── Upload database section ─────────────────────────────────────────────
    st.subheader("📤 Upload Your Data")

    uploaded_file = st.file_uploader(
        "Upload CSV or SQLite database",
        type=["csv", "db", "sqlite", "sqlite3"],
        help="Upload any CSV or SQLite file to query it in plain English",
    )

    if uploaded_file is not None:
        upload_dir = Path("database/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.get("last_upload_key") != file_key:
            with st.spinner("Reading your data..."):

                # Save raw file
                raw_path = upload_dir / uploaded_file.name
                with open(raw_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Convert CSV → SQLite if needed
                if uploaded_file.name.endswith(".csv"):
                    db_path = csv_to_sqlite(
                        str(raw_path), str(upload_dir)
                    )
                    st.success("CSV converted to database")
                else:
                    db_path = str(raw_path)

                # Analyze schema
                analyzer = SchemaAnalyzer(db_path)
                schema = analyzer.analyze()
                first_table = list(schema["tables"].keys())[0]
                suggestions = analyzer.generate_suggestions(first_table)

                # Update session state
                st.session_state.uploaded_db_path = db_path
                st.session_state.schema_analyzer = analyzer
                st.session_state.suggestions = suggestions
                st.session_state.last_upload_key = file_key
                st.session_state.db_path = db_path
                st.session_state.engine = None
                st.session_state.session_id = None
                st.session_state.messages = []

                st.rerun()

    # ── Show schema if uploaded ─────────────────────────────────────────────
    if st.session_state.get("schema_analyzer"):
        analyzer = st.session_state.schema_analyzer
        schema = analyzer.schema

        st.success("Custom database active")

        for table_name, table_info in schema["tables"].items():
            with st.expander(
                f"🗂 {table_name} ({table_info['row_count']} rows)"
            ):
                for col in table_info["columns"]:
                    col_type = table_info["types"].get(col, "text")
                    icon = (
                        "🔢" if col_type == "numeric"
                        else "📅" if col_type == "date"
                        else "🔤"
                    )
                    samples = table_info["sample_values"].get(col, [])
                    sample_str = ", ".join(str(v) for v in samples[:3])
                    st.text(f"{icon} {col} ({col_type})")
                    if sample_str:
                        st.caption(f"   e.g. {sample_str}")

        # Reset to default DB button
        if st.button("🔄 Back to default database"):
            st.session_state.uploaded_db_path = None
            st.session_state.schema_analyzer = None
            st.session_state.suggestions = []
            st.session_state.last_upload_key = None
            st.session_state.db_path = "database/sample.db"
            st.session_state.engine = None
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # ── Stats section ───────────────────────────────────────────────────────
    st.subheader("📊 Session Stats")
    if engine_ready and st.session_state.engine:
        stats = st.session_state.engine.get_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", stats["total_queries"])
        with col2:
            st.metric("Successful", stats["successful"])
        st.metric(
            "Success Rate",
            f"{stats['success_rate']:.0%}" if stats["total_queries"] > 0 else "N/A",
        )
    else:
        st.caption("Engine not loaded")

    st.divider()

    # ── History section ─────────────────────────────────────────────────────
    st.subheader("🕐 Query History")
    if engine_ready and st.session_state.engine and st.session_state.session_id:
        history = st.session_state.engine.get_session_history(
            st.session_state.session_id
        )
        queries = history.get("queries", [])
        if queries:
            for q in queries[-5:]:
                icon = "✅" if q.get("success", False) else "❌"
                query_text = q.get("query", "")
                display_text = (
                    query_text[:40] + "…" if len(query_text) > 40 else query_text
                )
                st.markdown(f"{icon} {display_text}")
        else:
            st.caption("No queries yet — start asking!")
    else:
        st.caption("No history available")

    st.divider()

    # ── Schema section ──────────────────────────────────────────────────────
    st.subheader("🗂️ Database Schema")
    if engine_ready and st.session_state.engine:
        try:
            schema = st.session_state.engine.sql_executor.get_schema()
            for table_name, columns_info in schema.items():
                with st.expander(f"📋 {table_name}", expanded=False):
                    for col in columns_info:
                        col_name = col["name"]
                        col_type = col["type"]
                        st.markdown(
                            f"&nbsp;&nbsp;`{col_name}` — *{col_type}*"
                        )
        except Exception:
            st.caption("Could not load schema")
    else:
        st.caption("Engine not loaded")

    st.divider()

    # ── Example queries section ─────────────────────────────────────────────
    st.subheader("💡 Example Queries")

    example_queries = [
        "Show all customers from Mumbai",
        "What is the average salary?",
        "Count orders by city",
        "Top 5 products by price",
        "Show sales grouped by category",
        "List employees with salary above 60000",
    ]

    for eq in example_queries:
        if st.button(eq, key=f"example_{eq}", use_container_width=True):
            st.session_state.pending_query = eq
            st.rerun()

    st.divider()

    # ── Session footer ──────────────────────────────────────────────────────
    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id}`")


# =============================================================================
# SECTION 6 — Main area header
# =============================================================================

st.markdown(
    '<h1 class="main-header">🗄️ Talk to Your Database</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">'
    "Ask questions in plain English — no SQL knowledge required"
    "</p>",
    unsafe_allow_html=True,
)

# ── Info banner when using uploaded database ────────────────────────────────
if st.session_state.db_path != "database/sample.db":
    st.info("Using your uploaded database")


# =============================================================================
# HELPER: format SQL for display
# =============================================================================

def format_sql_display(sql: str) -> str:
    """Wrap SQL in a dark code block for display."""
    if not sql:
        return ""
    return f'<div class="sql-box">{sql}</div>'


# =============================================================================
# HELPER: render an assistant message
# =============================================================================

def render_assistant_message(msg: dict):
    """Render a full assistant response inside a chat_message block."""
    # FIX: React error #185 — safe rendering
    try:

        # ── Badges ──────────────────────────────────────────────────────────
        badges = []
        if msg.get("success"):
            badges.append('<span class="success-badge">✅ Success</span>')
        elif msg.get("needs_clarification"):
            badges.append('<span class="followup-badge">❓ Clarification</span>')
        else:
            badges.append('<span class="error-badge">❌ Error</span>')
    
        if badges:
            st.markdown(" ".join(badges), unsafe_allow_html=True)
    
        # ── Clarification ───────────────────────────────────────────────────
        if msg.get("needs_clarification") and msg.get("clarification_msg"):
            st.warning(msg["clarification_msg"])
            return
    
        # ── Error ───────────────────────────────────────────────────────────
        if not msg.get("success") and not msg.get("needs_clarification"):
            if msg.get("error"):
                st.error(msg["error"])
            if msg.get("sql"):
                st.markdown("**Generated SQL:**")
                st.markdown(format_sql_display(msg["sql"]), unsafe_allow_html=True)
            return
    
        # ── SQL ─────────────────────────────────────────────────────────────
        if msg.get("sql"):
            st.markdown("**Generated SQL:**")
            st.markdown(format_sql_display(msg["sql"]), unsafe_allow_html=True)
    
        # ── Explanation ─────────────────────────────────────────────────────
        if msg.get("explanation"):
            st.markdown(f"**Explanation:** {msg['explanation']}")
    
        # ── Scalar metric ───────────────────────────────────────────────────
        # FIX: React error #185 — safe rendering
        if msg.get("chart_type") == "scalar" and msg.get("scalar_value"):
            label = msg.get("columns", ["Result"])[0] if msg.get("columns") else "Result"
            try:
                if msg.get("scalar_value") is not None:
                    st.metric(
                        label=label,
                        value=str(msg["scalar_value"]),
                        key=f"metric_{msg.get('turn', 0)}"
                    )
            except Exception:
                st.write(f"**Result:** {msg['scalar_value']}")
    
        # ── Results table ───────────────────────────────────────────────────
        # FIX: React error #185 — safe rendering
        rows = msg.get("rows", [])
        columns = msg.get("columns", [])
        row_count = msg.get("row_count", 0)
    
        if row_count > 0 and columns:
            try:
                df = pd.DataFrame(rows, columns=columns)
                display_df = df.head(50)  # show max 50 rows
    
                if len(df) > 50:
                    st.caption(
                        f"Showing 50 of {len(df)} rows. "
                        f"Full result has {len(df)} records."
                    )
    
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=min(300, (len(display_df) + 1) * 35 + 38),
                    key=f"df_{msg.get('turn', 0)}"
                )
            except Exception as e:
                st.warning("Could not render table. Showing raw data.")
                if rows and columns:
                    for i, row in enumerate(rows[:20]):
                        st.text(str(dict(zip(columns, row))))
    
        # ── Plotly chart ────────────────────────────────────────────────────
        # FIX: React error #185 — safe rendering
        chart_type = msg.get("chart_type", "")
        chart_fig = msg.get("chart_fig")
    
        if chart_fig is not None and chart_type not in ("none", "table", "scalar"):
            try:
                import plotly.graph_objects as go
                # Verify it is a valid plotly figure
                if isinstance(chart_fig, go.Figure):
                    st.plotly_chart(
                        chart_fig,
                        use_container_width=True,
                        key=f"chart_{msg.get('turn', 0)}"
                    )
            except Exception:
                pass  # silently skip chart if it fails
    
        # ── Summary ─────────────────────────────────────────────────────────
        if msg.get("summary"):
            st.markdown(
                f'<div class="metric-card">📝 <strong>Summary:</strong> '
                f'{msg["summary"]}</div>',
                unsafe_allow_html=True,
            )
    
        # ── Response time ───────────────────────────────────────────────────
        if msg.get("pipeline_ms"):
            st.markdown(
                f'<span class="response-time">'
                f"⏱️ Response time: {msg['pipeline_ms']:.0f}ms</span>",
                unsafe_allow_html=True,
            )

        # ── Smart value suggestions ("Did you mean?") ────────────────────
        vs = msg.get("value_suggestions", {})
        if vs.get("has_suggestions") and msg.get("row_count", 0) == 0:
            st.markdown("---")
            st.warning(vs.get("message", ""))

            for sugg in vs.get("suggestions", []):
                col_name = sugg["column"]
                typed    = sugg["typed"]
                matches  = sugg["matches"]

                if matches:
                    st.caption(
                        f"Closest matches for "
                        f"**{col_name}** = "
                        f"'{typed}':")

                    # Show as clickable buttons
                    btn_cols = st.columns(
                        len(matches))
                    for i, match_val in \
                            enumerate(matches):
                        with btn_cols[i]:
                            btn_key = (
                                f"sugg_"
                                f"{msg.get('turn_number', msg.get('turn', 0))}"
                                f"_{col_name}_{i}")
                            if st.button(
                                f"🔍 {match_val}",
                                key=btn_key,
                                use_container_width=True,
                                type="secondary"):

                                # Build corrected query
                                corrected_q = \
                                    msg.get(
                                        "query", "") \
                                    .replace(
                                        typed,
                                        match_val)

                                st.session_state \
                                    .pending_query = \
                                    corrected_q
                                st.rerun()
    
        # ── Suggestions ─────────────────────────────────────────────────────
        suggestions = msg.get("suggestions", [])
        if suggestions:
            st.markdown("**💡 Try next:**")
            suggestion_cols = st.columns(min(len(suggestions), 3))
            for i, sug in enumerate(suggestions[:3]):
                with suggestion_cols[i]:
                    if st.button(
                        sug,
                        key=f"btn_{msg.get('turn', 0)}_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_query = sug
                        st.rerun()

        # ── Next suggestions ────────────────────────────────────────────────
        try:
            next_sugg = msg.get("next_suggestions", [])
            turn_num  = msg.get("turn_number", msg.get("turn", 0))

            if next_sugg:
                st.markdown("---")
                st.caption("💡 **You might also want to ask:**")

                # 2 buttons per row
                col1, col2 = st.columns(2)
                for idx_s, suggestion in enumerate(next_sugg[:4]):
                    target_col = col1 if idx_s % 2 == 0 else col2
                    btn_key = f"nxt_{turn_num}_{idx_s}_{suggestion[:8]}"
                    with target_col:
                        if st.button(
                            suggestion,
                            key=btn_key,
                            use_container_width=True,
                            type="secondary"
                        ):
                            st.session_state.pending_query = suggestion
                            st.rerun()
        except Exception:
            pass  # never crash on suggestion display
    
    except Exception as e:
        st.error(f"Display error: {str(e)[:200]}")
        # Still show the SQL and summary safely
        if msg.get("sql"):
            st.code(msg["sql"], language="sql")
        if msg.get("summary"):
            st.info(msg["summary"])


# =============================================================================
# SECTION 7 — Chat history display
# =============================================================================

for idx, message in enumerate(st.session_state.messages):
    role = message.get("role", "user")

    with st.chat_message(role):
        if role == "user":
            st.markdown(message.get("content", ""))
        else:
            render_assistant_message(message)


# =============================================================================
# SECTION 9 — Welcome message (shown when no messages yet)
# =============================================================================

if not st.session_state.messages and engine_ready:

    # ── Smart suggestions from uploaded data ─────────────────────────────
    if st.session_state.suggestions:
        st.info("Your database is loaded! Here are suggested queries:")

        suggestions = st.session_state.suggestions
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions[:8]):
            col = cols[i % 2]
            with col:
                if st.button(
                    f"{suggestion['icon']} {suggestion['query']}",
                    key=f"sug_{i}",
                    use_container_width=True,
                ):
                    st.session_state.pending_query = suggestion["query"]
                    st.rerun()

    # ── Default welcome (sample.db) ─────────────────────────────────────
    else:
        st.markdown("---")
        st.markdown("### 👋 Welcome! Try one of these queries to get started:")
        st.markdown("")

        welcome_queries = [
            {
                "icon": "👥",
                "title": "Customer Query",
                "query": "Show all customers from Mumbai",
                "desc": "Find customers filtered by city",
            },
            {
                "icon": "💰",
                "title": "Aggregate Query",
                "query": "What is the average salary?",
                "desc": "Calculate aggregate statistics",
            },
            {
                "icon": "📦",
                "title": "Ranking Query",
                "query": "Top 5 products by price",
                "desc": "Sort and limit results",
            },
        ]

        cols = st.columns(3)
        for i, wq in enumerate(welcome_queries):
            with cols[i]:
                st.markdown(
                    f"""<div class="welcome-card">
                        <h3>{wq['icon']}</h3>
                        <h4>{wq['title']}</h4>
                        <p>{wq['desc']}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button(
                    wq["query"],
                    key=f"welcome_{i}",
                    use_container_width=True,
                ):
                    st.session_state.pending_query = wq["query"]
                    st.rerun()

        st.markdown("")
        st.info("💬 **Ready to answer your questions!** Type a query below or click an example above.")


# =============================================================================
# SECTION 8 — Chat input and processing
# =============================================================================

# ── Resolve input (typed or from button) ────────────────────────────────────
user_input = st.chat_input("Ask anything about your data...")

# If a pending query exists (from button click), use it
if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input and engine_ready:
    # 1. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # 3. Show "Thinking..." spinner while processing
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing your query..."):
            try:
                # 4. Call engine.query()
                result = st.session_state.engine.query(
                    user_input, st.session_state.session_id
                )

                # 5. Build chart data
                # FIX: React error #185 — safe rendering
                chart_fig = None
                chart_type = "none"
                scalar_val = None

                if (
                    result["success"]
                    and result["results"]
                    and result["results"]["row_count"] > 0
                ):
                    try:
                        chart_result = st.session_state.chart_engine.render(
                            columns=result["results"]["columns"],
                            rows=result["results"]["rows"],
                            intent=result["intent"],
                            title=user_input[:40]
                        )
                        chart_fig   = chart_result.get("figure")
                        chart_type  = chart_result.get("chart_type", "none")
                        scalar_val  = chart_result.get("scalar_value")
                    except Exception:
                        chart_fig  = None
                        chart_type = "none"
                        scalar_val = None

                # Handle scalar for aggregate with single row
                if (
                    result["success"]
                    and result["results"]
                    and result["results"]["row_count"] == 1
                    and len(result["results"]["columns"]) == 1
                    and chart_type != "scalar"
                ):
                    chart_type = "scalar"
                    value = result["results"]["rows"][0][0]
                    if isinstance(value, float):
                        scalar_val = f"{value:,.2f}"
                    elif isinstance(value, int):
                        scalar_val = f"{value:,}"
                    else:
                        scalar_val = str(value)

                # 6. Build response dict
                response = {
                    "role": "assistant",
                    "query": user_input,
                    "sql": result["sql"],
                    "explanation": result["explanation"],
                    "summary": result["summary"],
                    "suggestions": result["suggestions"],
                    "success": result["success"],
                    "error": result["error"],
                    "rows": (
                        result["results"]["rows"]
                        if result["results"]
                        else []
                    ),
                    "columns": (
                        result["results"]["columns"]
                        if result["results"]
                        else []
                    ),
                    "row_count": (
                        result["results"]["row_count"]
                        if result["results"]
                        else 0
                    ),
                    "chart_type": chart_type,
                    "chart_fig": chart_fig,
                    "scalar_value": scalar_val,
                    "needs_clarification": result["needs_clarification"],
                    "clarification_msg": result["clarification_msg"],
                    "pipeline_ms": result["pipeline_ms"],
                    "turn": result.get("turn_number", 0),
                    "turn_number": result.get("turn_number", 0),
                    "next_suggestions": generate_next_suggestions(
                        result=result,
                        schema_suggestions=st.session_state.get(
                            "suggestions", []),
                        db_path=st.session_state.get(
                            "db_path", "database/sample.db")
                    ),
                    "value_suggestions": result.get(
                        "value_suggestions",
                        {"has_suggestions": False}),
                }

                # 7. Add response to session state and increment count
                st.session_state.messages.append(response)
                st.session_state.query_count += 1

            except Exception as e:
                _err_result = {
                    "success": False, "query": user_input,
                    "intent": "UNKNOWN", "sql": None,
                    "results": None,
                }
                error_response = {
                    "role": "assistant",
                    "query": user_input,
                    "sql": None,
                    "explanation": None,
                    "summary": None,
                    "suggestions": [],
                    "success": False,
                    "error": str(e),
                    "rows": [],
                    "columns": [],
                    "row_count": 0,
                    "chart_type": "none",
                    "chart_fig": None,
                    "scalar_value": None,
                    "needs_clarification": False,
                    "clarification_msg": None,
                    "pipeline_ms": 0.0,
                    "turn": 0,
                    "turn_number": 0,
                    "next_suggestions": generate_next_suggestions(
                        result=_err_result,
                        schema_suggestions=st.session_state.get(
                            "suggestions", []),
                        db_path=st.session_state.get(
                            "db_path", "database/sample.db")
                    ),
                    "value_suggestions": {"has_suggestions": False,
                                          "suggestions": [],
                                          "message": ""},
                }
                st.session_state.messages.append(error_response)

    # 8. Rerun to refresh display
    st.rerun()
