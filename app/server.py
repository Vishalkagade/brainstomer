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

import base64
import hmac
import json
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import jev  # noqa: E402
import memory  # noqa: E402
import profiles  # noqa: E402
import router  # noqa: E402
import spawner  # noqa: E402
import store  # noqa: E402
import tools  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"

# How long a minted token stays valid. Short on purpose: it only has to survive
# the moment between the page asking and the socket opening.
TOKEN_TTL_SECONDS = 60

READ_ONLY = os.environ.get("BRAINSTORMER_READ_ONLY") == "1"  # hosted demo: history visible, promote refused
PASSWORD = os.environ.get("BRAINSTORMER_PASSWORD", "")  # hosted demo: set, and every request needs it (any username)
JEV_MODE = os.environ.get("JEV", "on")  # on: Jev decides, embeddings fall back | shadow: logged only | off: never asked
JEV_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jev")


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


def refresh_recaps() -> None:
    """Rebuild every mode's memory from AssemblyAI. Slow (one fetch per past call), so never inline."""
    try:
        with closing(store.connect()) as conn, httpx.Client(headers=config.headers(), timeout=30) as client:
            memory.refresh_all(conn, client)
    except Exception as err:  # memory is a nicety; a failure here must not touch the call
        print(f"recap refresh failed: {err}")


def late_verdict(route_id: int, verdict: dict) -> None:
    """Runs on the Jev thread once the shadow answer lands, after the page already got its decision."""
    try:
        with closing(store.connect()) as conn:
            store.set_route_jev(conn, route_id, verdict)
    except Exception as err:
        print(f"jev verdict not logged: {err}")


def evolution_overview(conn) -> dict:
    """Every mode with its version list, for the evolution page. One request draws the whole left side."""
    modes = []
    for m in store.list_modes(conn):
        settings = store.current(conn, m["id"])["settings"]
        modes.append({"id": m["id"], "name": m["name"], "status": m["status"],
                      "hue": settings.get("hue", 190), "versions": store.history(conn, m["id"])})
    if store.is_layer(conn, store.CORE):  # the user core, last: same versions, diff and buttons as a mode
        core = store.current(conn, store.CORE)
        modes.append({"id": store.CORE, "name": core["name"], "status": "layer",
                      "hue": core["settings"].get("hue", 40), "versions": store.history(conn, store.CORE)})
    return {"modes": modes, "can_promote": not READ_ONLY}


def promote_version(conn, version_id: int, embed_one=None) -> dict:
    return router.promote(conn, version_id, embed_one)  # lives in router since 25 Sep: the evidence review promotes too


class Handler(BaseHTTPRequestHandler):
    # Keep-alive, so the page's fetches reuse one connection.
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _locked(self) -> bool:
        """With a password set, refuse anything without it. The browser asks once and resends it on every request."""
        if not PASSWORD:
            return False
        header = self.headers.get("Authorization", "")
        try:
            given = base64.b64decode(header.removeprefix("Basic ")).decode().split(":", 1)[1]
        except (ValueError, IndexError):
            given = ""
        if hmac.compare_digest(given, PASSWORD):
            return False
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Brainstormer"')  # makes the browser show the login box
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def do_GET(self) -> None:  # noqa: N802  (the base class names it this)
        if self._locked():
            return
        parsed = urlparse(self.path)
        path = parsed.path

        # Which modes exist, for the buttons on the page.
        if path == "/profiles":
            threading.Thread(target=refresh_recaps, daemon=True).start()  # page load: rebuild memory off the reply path
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
                    "version": profile["version"],  # shown on the button: "Gym v2"
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
                profile = profiles.assemble(mode)
                with closing(store.connect()) as conn:
                    profile["recap"] = memory.cached(conn, mode)  # cached only; never fetched while a swap waits
                body = json.dumps(profile).encode()
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

        if path == "/versions":
            with closing(store.connect()) as conn:
                self._send(200, json.dumps(evolution_overview(conn)).encode(), "application/json")
            return

        if path == "/changes":
            query = parse_qs(parsed.query)
            try:
                old_id, new_id = int(query["old"][0]), int(query["new"][0])
                with closing(store.connect()) as conn:
                    body = store.changes(conn, old_id, new_id)
            except (KeyError, ValueError, SystemExit) as err:  # missing, not a number, or no such version
                self._send(400, json.dumps({"error": str(err)}).encode(), "application/json")
                return
            self._send(200, json.dumps(body).encode(), "application/json")
            return

        if path == "/evolution.js":
            self._send(200, (WEB / "evolution.js").read_bytes(), "text/javascript")
            return

        if path == "/evolution":
            self._send(200, (WEB / "evolution.html").read_bytes(), "text/html; charset=utf-8")
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
        if self._locked():
            return
        path = urlparse(self.path).path
        if path not in ("/tool", "/switch", "/route", "/promote"):
            self._send(404, b'{"error":"no such endpoint"}', "application/json")
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"not json"}', "application/json")
            return

        if path == "/tool":
            self._run_tool(body)
        elif path == "/switch":
            self._log_switch(body)
        elif path == "/promote":
            self._promote(body)
        else:
            self._route(body)

    def _run_tool(self, call: dict) -> None:
        """Run one tool for the page. The Exa key stays here, like the AssemblyAI key."""
        # tools.run never raises and always returns a JSON string, handed on into tool.result unchanged
        result = tools.run(call.get("name", ""), call.get("arguments") or {})
        self._send(200, json.dumps({"result": result}).encode(), "application/json")

    def _promote(self, body: dict) -> None:
        """Make one version live. Rollback is the same request with an older id."""
        if READ_ONLY:
            self._send(403, b'{"error":"promotion is switched off on this server"}', "application/json")
            return
        try:
            with closing(store.connect()) as conn:
                result = promote_version(conn, int(body.get("version_id")))
        except (TypeError, ValueError, SystemExit) as err:
            self._send(400, json.dumps({"error": str(err)}).encode(), "application/json")
            return
        except Exception as err:  # the pointer moved but the embedding call failed: say so, do not hide it
            self._send(502, json.dumps({"error": f"promoted, but fingerprint refresh failed: {err}"}).encode(),
                       "application/json")
            return
        self._send(200, json.dumps(result).encode(), "application/json")

    def _log_switch(self, body: dict) -> None:
        """Record a profile swap. The browser is the only one who knows the session id and the moment."""
        needed = ("session_id", "ts_ms", "mode", "version_id", "source")
        missing = [key for key in needed if body.get(key) in (None, "")]
        if missing:
            self._send(400, json.dumps({"error": f"missing {', '.join(missing)}"}).encode(), "application/json")
            return
        try:
            with closing(store.connect()) as conn:
                row_id = store.log_switch(conn, body["session_id"], int(body["ts_ms"]),
                                          body["mode"], int(body["version_id"]), body["source"])
        except (sqlite3.IntegrityError, ValueError) as err:  # unknown mode/version, or a non-number
            self._send(400, json.dumps({"error": str(err)}).encode(), "application/json")
            return
        self._send(200, json.dumps({"id": row_id}).encode(), "application/json")

    def _route(self, body: dict) -> None:
        """Decide the mode for a partial transcript. Answers fast; the page decides whether to act."""
        text = (body.get("text") or "").strip()
        if not text or not body.get("session_id"):
            self._send(400, b'{"error":"need text and session_id"}', "application/json")
            return
        live_mode = body.get("live_mode") or None
        try:
            with closing(store.connect()) as conn:
                # Jev is asked in its own thread, so an ask costs max(jev, embedding), never the sum.
                #   on:     Jev decides. The embedding runs only on a finished sentence, for the vector that
                #           memory recall and the candidates table need; if Jev fails, embeddings decide.
                #   shadow: the embedding decides, Jev's verdict is logged next to it.
                final = bool(body.get("final"))
                agent_last = body.get("agent_last") or ""
                future = None
                if JEV_MODE != "off" and len(router.strip_fillers(text).split()) >= router.MIN_WORDS:
                    future = JEV_POOL.submit(jev.route, text, live_mode, agent_last, jev.mode_options(conn))
                embed_decide = lambda: router.decide(text, live_mode, router.load_fingerprints(conn),
                                                     router.keyterm_index(conn), embed_fn=router.embed, final=final)
                decision = embed_decide() if (JEV_MODE != "on" or final or future is None) else None
                if JEV_MODE == "on" and future is not None:
                    verdict = future.result(timeout=jev.TIMEOUT_S + 0.5)
                    ruled = router.decide_jev(text, live_mode, verdict, final, agent_last)
                    if ruled is None:  # Jev had no answer: embeddings decide, as before 22 Sep
                        decision = decision or embed_decide()
                        decision["jev"] = verdict
                    else:
                        vector = decision.pop("_vector", None) if decision else None
                        decision = ruled | ({"_vector": vector} if vector is not None else {})
                elif future is not None and future.done():  # shadow: never wait; a late verdict is logged below
                    decision["jev"] = future.result()
                vector = decision.pop("_vector", None)  # 4096 floats: for the store, not for the page
                if router.new_topic(decision, final):  # once per utterance, not per partial
                    cid = store.add_candidate(conn, router.GENERAL, body["session_id"], "unknown_topic", text, vector)
                    new_mode = spawner.maybe_spawn(conn, body["session_id"], cid, group_fn=spawner.group_and_name)
                    if new_mode:
                        live = store.current(conn, new_mode)
                        decision.update(mode=new_mode, switch=True, signal="spawned",
                                        spawned={"id": new_mode, "name": live["name"], "about": live["settings"].get("about", "")})
                if decision["switch"]:
                    # the mode's memory block for the page to append to the prompt; the closest exchanges too when a vector exists
                    decision["recap"] = memory.recall(conn, decision["mode"], vector, exclude_session=body["session_id"])
                if decision["signal"] not in ("too_short", "no_fingerprints"):
                    route_id = store.log_route(conn, body["session_id"], int(body.get("ts_ms") or 0), text, live_mode, decision, agent_last, final)
                    if future is not None and "jev" not in decision:
                        future.add_done_callback(lambda f, rid=route_id: late_verdict(rid, f.result()))
        except (Exception, SystemExit) as err:  # a routing failure must never take the call down; the page just stays put
            print(f"route failed: {err}")
            self._send(500, json.dumps({"error": str(err), "switch": False}).encode(), "application/json")
            return
        self._send(200, json.dumps(decision).encode(), "application/json")

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

    with closing(store.connect()) as conn:
        if not store.list_modes(conn):  # empty store (fresh disk): every mode file becomes v1
            store.seed(conn, profiles.PROMPTS / "modes")
        core = profiles.core_file()
        if store.seed_core(conn, core):
            print(f"user core seeded into the store as v1 from {core.name}")
        if not conn.execute("SELECT 1 FROM fingerprints LIMIT 1").fetchone():  # fresh store: one embedding call, once
            print(f"router fingerprints built for {', '.join(router.build(conn))}")
    if JEV_MODE != "off":
        JEV_POOL.submit(jev.warm)  # open the connection now, not on the first route ask of the first call
    print(f"agent  {AGENT_ID}")
    print(f"talk   http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
