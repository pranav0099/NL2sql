import sys
from pathlib import Path
import runpy

# Ensure the project root is in the python path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Execute ui/app.py seamlessly
app_path = root_dir / "ui" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
