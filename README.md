NL2SQL — Talk to Your Database
Ask questions in plain English. Get SQL, results, explanations, and auto-charts. No SQL knowledge required.

An end-to-end Natural Language to SQL system that converts English questions into executable SQL, runs them against your database, and explains the result — built from scratch with a custom PyTorch Transformer, classical ML, and an NLP pipeline.

Live demo: add your Streamlit Cloud URL here

Demo
Landing page	Query + generated SQL
screenshot	screenshot
Example interactions

Show all customers from Mumbai
What is the average salary?
Top 5 products by price
Show first 10 rows from orders
Each query returns the generated SQL, the result table, a plain-English explanation, and (when applicable) an auto-generated Plotly chart — typically in under 100 ms.

Features
Natural language → SQL — type a question, get the SQL and the answer.
Bring your own data — drag-and-drop a CSV or SQLite file (up to 200 MB). Schema is analyzed automatically.
Explainable — every result shows the generated SQL plus a one-line explanation.
Multi-turn aware — follow-up questions reuse previous context via a session manager.
Auto visualization — the chart engine picks an appropriate Plotly chart for the result shape.
Graceful fallbacks — deep-learning generator → rule-based builder → API generator, so the system never silently fails.
Crash-proof contract — engine.query() always returns a structured result dict (success or failure).
Architecture
An 8-phase pipeline, each module independently testable:

User question
     │
     ▼
[Phase 2] NLP Pipeline ── spaCy preprocessing, entity extraction, schema linking
     │
     ▼
[Phase 3] ML Intent Classifier ── scikit-learn (SELECT / aggregate / join / rank)
     │
     ▼
[Phase 4] DL SQL Generator ── PyTorch Transformer (encoder–decoder, beam search)
     │                         + rule-based and API fallbacks
     ▼
[Phase 5] Session Manager ── multi-turn context resolution
     │
     ▼
[Phase 6] SQL Executor + Explainer ── runs SQL, generates explanation
     │
     ▼
[Phase 7] Chart Engine ── auto-selects Plotly chart
     │
     ▼
[Phase 8] Streamlit UI ── conversational interface
Phase	Module	Role
1	config/, database/	Configuration, sample SQLite DB, schema definitions
2	nlp/	Preprocessor, entity extractor, schema linker (spaCy + fuzzy matching)
3	ml/	TF-IDF + classifier for query intent
4	dl/	Transformer model, dataset, trainer, generator with beam search
5	memory/	Session and multi-turn context resolution
6	engine/	Main orchestrator (NL2SQLEngine), SQL executor, explainer
7	visualization/	Auto chart engine (Plotly)
8	ui/	Streamlit conversational front-end
Tech stack
Deep learning: PyTorch (custom Transformer encoder–decoder)
Classical ML: scikit-learn
NLP: spaCy (en_core_web_sm), fuzzywuzzy
UI: Streamlit
Visualization: Plotly
Data: pandas, SQLite
Training data: Spider, WikiSQL, plus a custom synthetic hotel-domain dataset
Getting started
Prerequisites
Python 3.8+
~2 GB free disk space (for spaCy model + dependencies)
Install
git clone https://github.com/pranav0099/NL2sql.git
cd NL2sql

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
Run the app
streamlit run streamlit_app.py
The app opens at http://localhost:8501. A sample SQLite database (database/sample.db) is included so you can start querying immediately, or upload your own CSV / SQLite file from the sidebar.

Project structure
.
├── streamlit_app.py          # Entry point for Streamlit Cloud
├── requirements.txt
├── config/                   # Centralized configuration
├── database/                 # Sample SQLite DB + schema
├── data/                     # Training data, generators, validators
├── nlp/                      # Preprocessor, entity extractor, schema linker
├── ml/                       # Intent classifier + trainer
├── dl/                       # Transformer model, dataset, trainer, generator
├── memory/                   # Session and context manager
├── engine/                   # NL2SQLEngine, executor, explainer, fallbacks
├── visualization/            # Plotly chart engine
├── ui/                       # Streamlit app
└── utils/                    # Logger, schema analyzer
Training your own model
The included Transformer was trained on Spider, WikiSQL, and a synthetic hotel-domain dataset.

# Generate / download datasets
python data/download_spider.py
python data/download_wikisql.py
python generate_hotel_training.py

# Train
python train_hotel_model.py
Trained weights land in models/saved/. The engine automatically loads them on startup; if no checkpoint is found, it falls back to the rule-based and API generators.

Roadmap
not done
JOIN inference across uploaded multi-table databases
not done
Pluggable LLM backend (OpenAI / local) as a fourth fallback tier
not done
PostgreSQL and MySQL adapters
not done
Query history and export
License
MIT

Author
Pranav — GitHub
