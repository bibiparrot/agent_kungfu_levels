"""Serve the catalog and each demo at its own URL using marimo's native router."""

import argparse
from pathlib import Path

import marimo
import uvicorn


def create_app():
    demos = Path(__file__).resolve().parent / "demos"
    builder = marimo.create_asgi_app()
    for index in sorted(demos.glob("*/index.py")):
        builder = builder.with_app(path=f"/{index.parent.name}/index", root=str(index))
    return builder.with_app(path="/", root=str(demos / "index.py")).build()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2718)
    args = parser.parse_args()
    print(f"Open the Demo catalog: http://127.0.0.1:{args.port}/", flush=True)
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
