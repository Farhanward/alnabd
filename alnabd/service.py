"""Alnabd content scoring as a local HTTP service.

``POST /api/score`` accepts ``{"text": "..."}`` and returns the pulse score,
PROMISING/IMPROVE decision, Arabic reasons and a rewrite suggestion. The
Naive Bayes model is loaded once at startup (``ALNABD_MODEL`` overrides the
path, default ``<project>/models/alnabd_pulse_model.json``).
"""

from __future__ import annotations

import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .core import rewrite_text
from .http_base import BaseServiceHandler, build_server
from .model import PulseModel

_MODEL: PulseModel | None = None


def model_path() -> Path:
    raw = os.environ.get("ALNABD_MODEL", "").strip()
    return Path(raw) if raw else PROJECT_ROOT / "models" / "alnabd_pulse_model.json"


def _score_route(data: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    text = str(data.get("text") or "").strip()
    if not text:
        return 400, {"ok": False, "error": "missing 'text'"}
    if _MODEL is None:
        return 503, {"ok": False, "error": f"model not loaded: {model_path()}"}
    return 200, {"ok": True, **rewrite_text(text, _MODEL)}


class Handler(BaseServiceHandler):
    post_routes = {"/api/score": staticmethod(_score_route)}


def create_server(host: str | None = None, port: int | None = None) -> ThreadingHTTPServer:
    global _MODEL
    path = model_path()
    _MODEL = PulseModel.load(path) if path.exists() else None
    return build_server(Handler, host=host, port=port)


def run_server(host: str | None = None, port: int | None = None) -> None:
    from .version import __version__

    server = create_server(host=host, port=port)
    print(f"alnabd service v{__version__}: http://{server.server_address[0]}:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
