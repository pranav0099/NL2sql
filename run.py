"""
NL2SQL — Application Launcher
Run: python run.py

One-command launcher that validates all required files
exist before starting the Streamlit UI server.
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_requirements():
    """
    Verify that all required model, data, and database files
    are present before launching the Streamlit app.

    Prints a clear message and exits with code 1 if any are missing.
    """
    required_files = {
        "database/sample.db": "SQLite database",
        "models/saved/intent_classifier.pkl": "ML intent classifier",
        "models/saved/tfidf_vectorizer.pkl": "TF-IDF vectorizer",
        "models/saved/transformer_nl2sql.pt": "Transformer SQL generator",
        "data/processed/vocab.json": "Vocabulary file",
    }

    missing = []
    for rel_path, description in required_files.items():
        full_path = ROOT / rel_path
        if not full_path.exists():
            missing.append(f"  [X] {rel_path}  ({description})")

    if missing:
        print("=" * 60)
        print("ERROR: Required files are missing!")
        print("=" * 60)
        print()
        for m in missing:
            print(m)
        print()
        print("Please ensure all model files have been trained")
        print("and the database has been populated before launching.")
        print("=" * 60)
        sys.exit(1)

    print("[OK] All required files found")


def launch():
    """
    Check requirements and start the Streamlit server.
    """
    print("=" * 60)
    print("  NL2SQL — Conversational Database Query System")
    print("=" * 60)
    print()

    check_requirements()

    print()
    print("Starting NL2SQL Application...")
    print("Open browser at: http://localhost:8501")
    print("Press Ctrl+C to stop")
    print()

    os.chdir(ROOT)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/app.py",
            "--server.port",
            "8501",
            "--server.headless",
            "false",
            "--browser.gatherUsageStats",
            "false",
        ]
    )


if __name__ == "__main__":
    launch()
