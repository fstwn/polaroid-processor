"""Local web app: projects, detect run with a live review grid."""
from .server import AppState, make_server, serve

__all__ = ["AppState", "make_server", "serve"]
