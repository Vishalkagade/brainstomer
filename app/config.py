"""Credentials and endpoints. The only module that reads the API key.

Everything else imports from here, so there is exactly one place to look when a
key is missing or an endpoint moves.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

# The Voice Agent API. Two different hosts for the same product: REST over
# HTTPS to manage agents and read past sessions, a WebSocket to hold a call.
API_BASE = "https://agents.assemblyai.com/v1"
WS_URL = "wss://agents.assemblyai.com/v1/ws"


def load_env(path: Path = ENV_FILE) -> None:
    """Read KEY=value lines into the process environment.

    A real shell variable wins over the file, which is how a hosting platform
    supplies the key in production without a .env existing at all. The regex
    tolerates spaces around the `=` and optional quotes, because .env files are
    hand-edited and ours already has `ASSEMBLYAI_API_KEY = ...` with spaces.
    """
    try:
        text = path.read_text()
    except OSError:
        return
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        key, raw = match.group(1), match.group(2)
        if key in os.environ:
            continue
        os.environ[key] = re.sub(r"^(['\"])(.*)\1$", r"\2", raw)


def save_env(key: str, value: str, path: Path = ENV_FILE) -> None:
    """Write one key back to .env, replacing it in place if already there.

    Used for the agent id: the first publish creates an agent and we have to
    remember which one, or the next run creates a second.
    """
    os.environ[key] = value
    try:
        text = path.read_text()
    except OSError:
        text = ""
    line = f"{key}={value}"
    pattern = re.compile(rf"^[ \t]*{re.escape(key)}[ \t]*=.*$", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(line, text, count=1)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += line + "\n"
    path.write_text(text)


def api_key() -> str:
    load_env()
    key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        raise SystemExit(
            "No ASSEMBLYAI_API_KEY in .env — get one at "
            "https://www.assemblyai.com/dashboard/api-keys"
        )
    return key


def headers() -> dict:
    """Auth header for the REST API.

    This key must never reach the browser. The page gets a short-lived token
    instead (see server.py); anyone holding this key can run sessions billed to
    your account.
    """
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
