from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.storage.database import get_database_url, init_db


def main() -> None:
    init_db()
    print(f"Initialized database: {get_database_url()}")


if __name__ == "__main__":
    main()
