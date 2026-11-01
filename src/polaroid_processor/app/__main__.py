"""``python -m polaroid_processor.app [--port N] [--no-browser]``: start the app."""
import argparse
import sys

from .server import AppState, serve


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m polaroid_processor.app", description="Local web app for the scans.")
    p.add_argument("--port", type=int, default=8780, help="port on 127.0.0.1 (default 8780, 0 = any free port)")
    p.add_argument("--no-browser", action="store_true", help="do not open the page in the browser")
    p.add_argument("--config", help="user data file (default %%APPDATA%%/polaroid-processor/app.json)")
    a = p.parse_args(argv)
    serve(AppState(a.config), port=a.port, open_browser=not a.no_browser, announce=lambda m: print(m, file=sys.stderr, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
