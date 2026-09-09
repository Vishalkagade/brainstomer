"""Serves the page and mints session tokens.

    .venv/bin/python app/server.py     then open http://localhost:3000

This process is the security boundary. The API key lives here and never leaves:
the browser is handed a token that is good for 60 seconds and one session. If
the page had the key instead, anyone who opened dev tools could run sessions on
your account for as long as the key lives.

It sits outside the audio path entirely. Once the page has a token it talks to
AssemblyAI directly over its own WebSocket, so nothing here has to be fast, and
nothing here breaks the call if it restarts.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import profiles  # noqa: E402
import tools  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"

# How long a minted token stays valid. Short on purpose: it only has to survive
# the moment between the page asking and the socket opening.
TOKEN_TTL_SECONDS = 60


def mint_token() -> dict:
    """Ask AssemblyAI for a browser-safe credential.

    `product=voice_agent` scopes the token to this API; a token minted for
    anything else is refused on the agents socket.
    """
    response = httpx.get(
        f"{config.API_BASE}/token",
        params={"product": "voice_agent", "expires_in_seconds": TOKEN_TTL_SECONDS},
        headers=config.headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


class Handler(BaseHTTPRequestHandler):
    # Keep-alive, so the page's fetches reuse one connection.
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802  (the base class names it this)
        parsed = urlparse(self.path)
        path = parsed.path

        # Which modes exist, for the buttons on the page.
        if path == "/profiles":
            modes = []
            for name in profiles.list_modes():
                profile = profiles.assemble(name)
                turn = profile["session"]["input"]["turn_detection"]
                # hue and min_silence travel with the button so the page can
                # recolour itself and show the silence window the moment a mode
                # is picked — before any call exists to fetch a full profile.
                modes.append({
                    "id": name,
                    "name": profile["name"],
                    "hue": profile["hue"],
                    "min_silence": turn["min_silence"],
                })
            self._send(200, json.dumps({"modes": modes}).encode(), "application/json")
            return

        # One assembled profile. The prompt layers are read fresh on every
        # request, so editing a .md file and clicking the mode button again
        # applies it without restarting anything — which is what makes these
        # files worth editing by hand.
        if path == "/profile":
            mode = parse_qs(parsed.query).get("mode", [""])[0]
            if mode not in profiles.list_modes():
                self._send(404, b'{"error":"no such mode"}', "application/json")
                return
            try:
                body = json.dumps(profiles.assemble(mode)).encode()
            except SystemExit as err:
                # A layer over its token cap. Surface it to the page rather than
                # killing the server.
                print(f"profile {mode}: {err}")
                self._send(500, json.dumps({"error": str(err)}).encode(),
                           "application/json")
                return
            self._send(200, body, "application/json")
            return

        if path == "/token":
            try:
                self._send(200, json.dumps(mint_token()).encode(), "application/json")
            except httpx.HTTPError as err:
                print(f"token request failed: {err}")
                self._send(502, b'{"error":"could not mint a token"}',
                           "application/json")
            return

        if path == "/client.js":
            self._send(200, (WEB / "client.js").read_bytes(), "text/javascript")
            return

        # Anything else is the page. The agent id is baked in at request time
        # rather than fetched, so the page has everything it needs to open a
        # socket the moment it loads.
        page = (WEB / "index.html").read_text().replace("{{AGENT_ID}}", AGENT_ID)
        self._send(200, page.encode(), "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802  (the base class names it this)
        """Run one tool on the page's behalf.

        The WebSocket lives in the browser, so `tool.call` arrives there — but
        the Exa key is in this process and has to stay here, exactly like the
        AssemblyAI key. So the page relays the call to us, we run it, and it
        gets back only the shaped result. Nothing the browser holds could be
        used to spend someone else's search quota.

        This blocks for as long as the search takes, roughly 1.7 s. That is
        fine: the server is a ThreadingHTTPServer, so the page can still fetch
        while a tool runs, and the audio never comes through here at all.
        """
        if urlparse(self.path).path != "/tool":
            self._send(404, b'{"error":"no such endpoint"}', "application/json")
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            call = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"not json"}', "application/json")
            return

        # tools.run never raises and always returns a JSON string, so this is
        # handed straight to the page and on into tool.result unchanged.
        result = tools.run(call.get("name", ""), call.get("arguments") or {})
        self._send(200, json.dumps({"result": result}).encode(), "application/json")

    def log_message(self, *args) -> None:
        """Silence the default per-request logging; real errors print above."""


def main() -> None:
    global AGENT_ID
    config.load_env()
    AGENT_ID = os.environ.get("AGENT_ID", "")
    if not AGENT_ID:
        raise SystemExit("No AGENT_ID in .env — run: .venv/bin/python app/agent.py")

    # PORT when the environment sets one (hosting platforms do), else 3000 and
    # upward until something is free.
    fixed = os.environ.get("PORT")
    port = int(fixed) if fixed else 3000
    while True:
        try:
            server = ThreadingHTTPServer(("", port), Handler)
            break
        except OSError:
            if fixed or port >= 3010:
                raise
            port += 1

    print(f"agent  {AGENT_ID}")
    print(f"talk   http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
