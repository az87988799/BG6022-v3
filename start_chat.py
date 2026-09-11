"""Small double-click entry point; all behavior lives in ``bg6022.cli``."""

from __future__ import annotations

import sys
from pathlib import Path

from bg6022.cli import main


def run() -> int:
    project_root = Path(__file__).resolve().parent
    config = project_root / "config.toml"
    if not config.is_file():
        print(f"Create {config} from config.example.toml before starting chat.", file=sys.stderr)
        return 2
    return main(["--config", str(config), "chat"])


if __name__ == "__main__":
    raise SystemExit(run())
