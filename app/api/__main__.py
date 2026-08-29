"""Entry point: python -m app.api

Preferred over `uvicorn app.api.main:app` because uvicorn builds its event loop
before this package gets a say, and on Windows the default loop is one the database
driver cannot use (see app/runtime.py).
"""

from __future__ import annotations

import argparse

import uvicorn

from app.runtime import new_event_loop


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    config = uvicorn.Config("app.api.main:app", host=args.host, port=args.port, reload=args.reload)
    server = uvicorn.Server(config)
    loop = new_event_loop()
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
