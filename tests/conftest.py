from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in [ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "app"]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
