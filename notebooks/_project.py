"""Bootstrap pentru notebook-uri: adaugă rădăcina proiectului în sys.path."""

from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    for candidate in (Path.cwd(), Path.cwd().parent):
        if (candidate / "src" / "config.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Nu am găsit rădăcina proiectului (src/config.py). "
        "Rulează din folderul notebooks/ sau instalează: pip install -e ."
    )


def setup_path() -> Path:
    root = project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
