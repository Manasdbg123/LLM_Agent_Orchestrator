"""Entry point: python -m app.worker"""

from __future__ import annotations

from app.runtime import run
from app.worker.worker import main

if __name__ == "__main__":
    run(main())
