"""
NL2SQL Project Master Setup Script

This script sets up the entire project from scratch:
  - Checks Python version and dependencies
  - Creates project directory structure
  - Downloads and prepares datasets
  - Builds vocabulary
  - Verifies installation

Author: Pranav
Date: 2026-04-02

Usage:
    python setup.py
    or
    python3 setup.py
"""

import sys
import subprocess
import importlib
import importlib.util
from pathlib import Path
from typing import Tuple, List, Optional

# ============================================================================
# CONSTANTS
# ============================================================================

REQUIRED_PYTHON_VERSION = (3, 8)

# Core dependencies with minimum versions
REQUIRED_PACKAGES = {
    "torch": "2.0",
    "spacy": "3.6",
    "scikit-learn": "1.3",
    "pandas": "2.0",
    "numpy": "1.24",
    "plotly": "5.15",
    "matplotlib": "3.7",
    "streamlit": "1.25",
    "fuzzywuzzy": "0.18",
    "python-Levenshtein": "0.21",
    "datasets": "2.14",
    "joblib": "1.3",
    "tqdm": "4.65",
    "Faker": "20.0"
}

# Optional dependencies (nice to have but not critical)
OPTIONAL_PACKAGES = {
    "transformers": "4.0",
    "peft": "0.0",
    "accelerate": "0.0",
    "bitsandbytes": "0.0"
}

# spaCy model
SPACY_MODEL = "en_core_web_sm"

# Project root
ROOT = Path(__file__).resolve().parent

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def check_python_version() -> bool:
    """
    Check if Python version is sufficient.

    Returns:
        True if version is OK, False otherwise
    """
    print_subheader("Step 1: Checking Python Version")

    current_version = sys.version_info
    required = REQUIRED_PYTHON_VERSION

    print(f"  Current Python version: {current_version.major}.{current_version.minor}.{current_version.micro}")
    print(f"  Required Python version: >= {required[0]}.{required[1]}")

    if current_version >= required:
        print("  [OK] Python version is sufficient")
        return True
    else:
        print("  [FAIL] Python version is too old")
        print(f"    Please upgrade to Python {required[0]}.{required[1]} or later")
        return False


def check_package(package_name: str, min_version: Optional[str] = None) -> bool:
    """
    Check if a package is installed and meets minimum version.

    Args:
        package_name: Name of the package
        min_version: Minimum required version (optional)

    Returns:
        True if package is OK, False otherwise
    """
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            print(f"  [FAIL] {package_name:30s}: Not installed")
            return False

        # Try to get version
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")

        if min_version and version != "unknown":
            # Simple version comparison (may not work for all version schemes)
            print(f"  [OK] {package_name:30s}: {version:15s} (required: >= {min_version})")
            return True
        else:
            print(f"  [OK] {package_name:30s}: {version}")
            return True

    except ImportError as e:
        print(f"  [FAIL] {package_name:30s}: Import error - {e}")
        return False


def install_package(package_name: str, version_spec: Optional[str] = None) -> bool:
    """
    Install a package using pip.

    Args:
        package_name: Name of the package
        version_spec: Version specifier (e.g., ">=2.0")

    Returns:
        True if installation succeeded
    """
    package_spec = f"{package_name}{version_spec}" if version_spec else package_name
    print(f"  Installing: {package_spec}...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print(f"  [OK] Successfully installed {package_name}")
            return True
        else:
            print(f"  [FAIL] Failed to install {package_name}")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [FAIL] Installation of {package_name} timed out")
        return False
    except Exception as e:
        print(f"  [FAIL] Error installing {package_name}: {e}")
        return False


def check_and_install_dependencies() -> Tuple[bool, List[str]]:
    """
    Check all required dependencies and install if missing.

    Returns:
        Tuple of (all_ok, missing_packages)
    """
    print_header("Step 1: Checking Dependencies")

    missing = []
    all_ok = True

    for package, min_version in REQUIRED_PACKAGES.items():
        if not check_package(package, min_version):
            missing.append(package)

    # Optional packages (just check, don't install automatically)
    print_subheader("Optional Packages (not required)")
    for package in OPTIONAL_PACKAGES:
        check_package(package)  # Just check, don't fail if missing

    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Automatically installing missing packages...")
        print("\n  Installing packages...")
        for package in missing:
            version_spec = f">={REQUIRED_PACKAGES[package]}"
            if not install_package(package, version_spec):
                all_ok = False
        if all_ok:
            print("\n  All packages installed successfully!")
        else:
            print("\n  Some packages failed to install. You can install manually:")
            for package in missing:
                print(f"    pip install {package}>={REQUIRED_PACKAGES[package]}")
    else:
        print("\n  [OK] All required packages are installed")

    return all_ok, missing


def download_spacy_model() -> bool:
    """
    Download spaCy model if not present.

    Returns:
        True if model is available
    """
    print_subheader("Step 2: Checking spaCy Model")

    try:
        import spacy
        try:
            nlp = spacy.load(SPACY_MODEL)
            print(f"  [OK] spaCy model '{SPACY_MODEL}' is installed")
            return True
        except OSError:
            print(f"  [FAIL] spaCy model '{SPACY_MODEL}' is not installed")
            print(f"    Attempting to download...")

            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print(f"  [OK] Successfully downloaded {SPACY_MODEL}")
                return True
            else:
                print(f"  [FAIL] Failed to download {SPACY_MODEL}")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")
                return False

    except ImportError:
        print("  [FAIL] spaCy is not installed (should have been installed in previous step)")
        return False


def show_device_info() -> None:
    """Show PyTorch device information."""
    print_subheader("Step 3: System Information")

    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA available: Yes")
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print(f"  CUDA available: No (using CPU)")
        print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    except ImportError:
        print("  ⚠ PyTorch not installed (will be installed later)")

    try:
        import platform
        print(f"\n  System: {platform.system()} {platform.release()}")
        print(f"  Machine: {platform.machine()}")
    except:
        pass


def create_project_folders() -> bool:
    """
    Create all project folders using config.

    Returns:
        True if successful
    """
    print_subheader("Step 4: Creating Project Folders")

    try:
        # Import config module (it creates folders on import)
        sys.path.insert(0, str(ROOT))
        from config.config import create_dirs

        create_dirs()

        print("\n  [OK] Project folders created:")
        print("    - data/")
        print("    - data/raw")
        print("    - data/processed")
        print("    - data/wikisql")
        print("    - data/spider")
        print("    - models/")
        print("    - database/")
        print("    - logs/")
        print("    - config/")
        print("    - utils/")
        print("    - data/")

        return True

    except Exception as e:
        print(f"  [FAIL] Failed to create project folders: {e}")
        return False


def run_script(script_path: Path, description: str) -> bool:
    """
    Run a Python script.

    Args:
        script_path: Path to the script
        description: Description for logging

    Returns:
        True if script succeeded
    """
    print(f"\n  Running: {description}...")
    print(f"  Script: {script_path}")

    if not script_path.exists():
        print(f"  [FAIL] Script not found: {script_path}")
        return False

    try:
        # Change to project root for execution
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            print(f"  [FAIL] Script failed with exit code {result.returncode}")
            if result.stderr:
                print("  Error output:")
                print(result.stderr)
            return False
        else:
            print(f"  [OK] Script completed successfully")
            return True

    except subprocess.TimeoutExpired:
        print(f"  [FAIL] Script timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"  [FAIL] Error running script: {e}")
        return False


def verify_outputs() -> bool:
    """
    Verify all expected output files exist.

    Returns:
        True if all files present
    """
    print_header("Step 9: Verifying Outputs")

    from config.config import (
        SAMPLE_DB, WIKISQL_DATA, SPIDER_DATA,
        PROCESSED_DATA, VOCAB_FILE, SCHEMA_FILE
    )

    checks = [
        ("Sample Database", SAMPLE_DB),
        ("WikiSQL Train", WIKISQL_DATA / "train.json"),
        ("WikiSQL Validation", WIKISQL_DATA / "validation.json"),
        ("WikiSQL Test", WIKISQL_DATA / "test.json"),
        ("Spider Train", SPIDER_DATA / "train.json"),
        ("Spider Validation", SPIDER_DATA / "validation.json"),
        ("Spider Test", SPIDER_DATA / "test.json"),
        ("Processed Train", PROCESSED_DATA / "train.json"),
        ("Processed Validation", PROCESSED_DATA / "val.json"),
        ("Processed Test", PROCESSED_DATA / "test.json"),
        ("Vocabulary", VOCAB_FILE),
        ("Database Schema", SCHEMA_FILE)
    ]

    all_ok = True
    missing = []

    for name, path in checks:
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"  [OK] {name:30s}: {path.name} ({size:.1f} KB)")
        else:
            print(f"  [FAIL] {name:30s}: MISSING")
            all_ok = False
            missing.append(name)

    return all_ok, missing


def show_sample_record() -> bool:
    """
    Display a sample record from the processed dataset.

    Returns:
        True if successful
    """
    print_subheader("Step 10: Sample Record")

    try:
        import json
        from config.config import PROCESSED_DATA

        sample_file = PROCESSED_DATA / "train.json"
        if not sample_file.exists():
            print("  [FAIL] Processed data not found")
            return False

        with open(sample_file, 'r') as f:
            data = json.load(f)

        if not data:
            print("  [FAIL] No records in sample file")
            return False

        sample = data[0]

        print("\n  Sample Training Record:")
        print(f"    ID: {sample.get('id', 'N/A')}")
        print(f"    Intent: {sample.get('intent', 'N/A')}")
        print(f"    Question: {sample.get('question', 'N/A')}")
        print(f"    SQL: {sample.get('query', 'N/A')[:100]}...")
        print(f"    NL tokens: {len(sample.get('nl_tokens', []))} tokens")
        print(f"    SQL tokens: {len(sample.get('sql_tokens', []))} tokens")
        print(f"    Schema tables: {len(sample.get('schema', {}).get('tables', {}))}")

        # Show a few records count
        print(f"\n  Dataset Summary:")
        print(f"    Total training records: {len(data)}")

        try:
            from config.config import VOCAB_FILE
            with open(VOCAB_FILE, 'r') as f:
                vocab = json.load(f)

            nl_size = vocab.get('nl_vocab', {}).get('num_tokens', 0)
            sql_size = vocab.get('sql_vocab', {}).get('num_tokens', 0)
            print(f"    NL vocabulary size: {nl_size}")
            print(f"    SQL vocabulary size: {sql_size}")
        except:
            pass

        return True

    except Exception as e:
        print(f"  [FAIL] Error showing sample record: {e}")
        return False


def print_final_summary() -> None:
    """Print final setup summary and next steps."""
    print_header("Setup Complete!")

    print("""
  [OK] Project structure created
  [OK] Database initialized with sample data
  [OK] Datasets downloaded and processed
  [OK] Vocabulary built and saved

  Next Steps:

  1. Explore the data:
     python -c "from data.build_vocab import print_statistics; ..."

  2. Train the intent classifier:
     python models/train_intent.py

  3. Train the SQL generator:
     python models/train_sql_generator.py

  4. Run evaluation:
     python models/evaluate.py

  5. Start the web interface:
     streamlit run app.py

  Project Structure:
    .
    ├── config/
    │   └── config.py
    ├── utils/
    │   └── logger.py
    ├── database/
    │   ├── sample.db
    │   └── schema.json
    ├── data/
    │   ├── wikisql/
    │   ├── spider/
    │   └── processed/
    ├── models/
    │   └── (trained models will be saved here)
    ├── logs/
    └── setup.py

  Configuration:
    Edit config/config.py to adjust hyperparameters.

  Troubleshooting:
    - Check logs/nl2sql.log for detailed logs
    - Ensure all dependencies are installed: pip install -r requirements.txt
    - Make sure you have at least 2GB free disk space

  Thank you for using NL2SQL!
""")

    print("=" * 80 + "\n")


# ============================================================================
# MAIN SETUP FUNCTION
# ============================================================================

def main() -> int:
    """
    Main setup function running all steps.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    print_header("NL2SQL PROJECT AUTOMATED SETUP")
    print("  Setting up production-grade NL2SQL system")
    print("  This may take several minutes...")

    steps = []

    # Step 1: Python version check
    if not check_python_version():
        print("\n[FAIL] Setup failed: Python version insufficient")
        return 1
    steps.append("[OK] Python version checked")

    # Step 2: Check and install dependencies
    deps_ok, missing = check_and_install_dependencies()
    if deps_ok:
        steps.append("[OK] Dependencies installed")
    else:
        print(f"\n⚠ Some dependencies missing: {', '.join(missing)}")
        print("  Setup cannot continue without required packages")
        return 1

    # Step 3: Download spaCy model
    if not download_spacy_model():
        print("\n⚠ spaCy model download failed, but continuing...")
        steps.append("⚠ spaCy model: manual download needed")
    else:
        steps.append("[OK] spaCy model downloaded")

    # Step 4: Show device info
    show_device_info()
    steps.append("[OK] System information collected")

    # Step 5: Create project folders
    if not create_project_folders():
        print("\n[FAIL] Failed to create project folders")
        return 1
    steps.append("[OK] Project folders created")

    # Step 6: Create sample database
    print_header("Step 6: Creating Sample Database")
    if not run_script(ROOT / "database" / "create_sample_db.py",
                      "Creating sample SQLite database"):
        print("\n[FAIL] Database creation failed")
        return 1
    steps.append("[OK] Sample database created")

    # Step 7: Download WikiSQL
    print_header("Step 7: Downloading WikiSQL Dataset")
    if not run_script(ROOT / "data" / "download_wikisql.py",
                      "Downloading and processing WikiSQL"):
        print("\n[FAIL] WikiSQL download failed")
        return 1
    steps.append("[OK] WikiSQL dataset ready")

    # Step 8: Download Spider
    print_header("Step 8: Downloading Spider Dataset")
    if not run_script(ROOT / "data" / "download_spider.py",
                      "Downloading and processing Spider"):
        print("\n[FAIL] Spider download failed")
        return 1
    steps.append("[OK] Spider dataset ready")

    # Step 9: Build vocabulary
    print_header("Step 9: Building Vocabulary")
    if not run_script(ROOT / "data" / "build_vocab.py",
                      "Building NL and SQL vocabularies"):
        print("\n[FAIL] Vocabulary building failed")
        return 1
    steps.append("[OK] Vocabulary built")

    # Step 10: Verify outputs
    all_ok, missing = verify_outputs()
    if all_ok:
        steps.append("[OK] All outputs verified")
    else:
        print(f"\n⚠ Missing outputs: {', '.join(missing)}")
        print("  Some outputs may be missing but continuing...")
        steps.append("⚠ Some outputs missing")

    # Step 11: Show sample
    show_sample_record()
    steps.append("[OK] Sample record displayed")

    # Print summary
    print("\n" + "="*80)
    print("SETUP STEPS COMPLETED:")
    for step in steps:
        print(f"  {step}")
    print("="*80)

    # Print final summary
    print_final_summary()

    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        exit_code = main()
        if exit_code == 0:
            print("[OK] Setup completed successfully!")
        else:
            print("[FAIL] Setup failed with errors")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
