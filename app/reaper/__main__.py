"""Entry point: python -m app.reaper"""

from __future__ import annotations

from app.reaper.reaper import main
from app.runtime import run

if __name__ == "__main__":
    run(main())
