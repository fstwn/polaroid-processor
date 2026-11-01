"""``python -m polaroid_processor.app [FOLDER ...] [--port N] [--no-browser]``: start the app (same as ``polaroid app``)."""
import argparse
import sys

from .launch import launch


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m polaroid_processor.app", description="Local web app for the scans.")
    p.add_argument("folders", nargs="*", help="scan folders to make projects of (an output folder opens as it is)")
    p.add_argument("--port", type=int, help="port on 127.0.0.1 (default 8780; 0 = any free port)")
    p.add_argument("--no-browser", action="store_true", help="do not open the page in the browser")
    p.add_argument("--config", help="user data file (default %%APPDATA%%/polaroid-processor/app.json)")
    a = p.parse_args(argv)
    return launch(a.folders, port=a.port, open_browser=not a.no_browser, config=a.config,
                  announce=lambda m: print(m, file=sys.stderr, flush=True))


if __name__ == "__main__":
    sys.exit(main())
