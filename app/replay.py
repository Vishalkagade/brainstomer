"""Replay a recorded call into a fresh session, acting as the browser would.

    .venv/bin/python app/replay.py SESSION_ID [--server http://localhost:3000] [--password PW] [--mode general]

Streams the user's channel of that call's recording in real time over a new socket, routes on
transcripts through the server like client.js does, answers tool calls, then prints the new
session's timeline next to the old one. Costs the call's minutes on AssemblyAI again.
"""

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import sessions  # noqa: E402

RATE = 24_000
CHUNK_MS = 100  # one input.audio message per 100 ms, paced to the clock
CACHE = Path(__file__).resolve().parent / "data" / "recordings"


def user_pcm(session_id: str) -> bytes:
    """The user's channel (left) of the call's recording as 24 kHz mono s16le, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    raw = CACHE / f"{session_id}.raw"
    if raw.exists():
        return raw.read_bytes()
    with httpx.Client(headers=config.headers(), timeout=120) as client:
        session = sessions.get_session(client, session_id)
        url = next(a["url"] for a in session.get("artifacts", []) if a["type"] == "audio")
    ogg = CACHE / f"{session_id}.ogg"
    ogg.write_bytes(httpx.get(url, timeout=120).content)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ogg), "-map_channel", "0.0.0",
                    "-ar", str(RATE), "-f", "s16le", str(raw)], check=True)
    return raw.read_bytes()


class Client:
    """The parts of client.js the model can notice: mode swaps, routing, memory, tools."""

    def __init__(self, server: str, password: str, mode: str):
        auth = ("replay", password) if password else None
        self.http = httpx.AsyncClient(base_url=server, auth=auth, timeout=30)
        self.mode = mode
        self.live_mode = None
        self.session_id = None
        self.agent_last = ""
        self.ws = None
        self.t0 = time.time()
        self.events = []  # what happened, for the summary
        self.route_busy = False
        self.routed_text = ""
        self.last_route_at = 0.0
        self.last_words = 0

    def log(self, kind: str, text: str = "") -> None:
        t = time.time() - self.t0
        self.events.append((t, kind, text))
        print(f"{t:7.1f}s  {kind:<10} {text[:110]}", flush=True)

    async def send(self, message: dict) -> None:
        await self.ws.send(json.dumps(message))

    async def apply_mode(self, mode: str, source: str = "manual", recap=None) -> None:
        profile = (await self.http.get("/profile", params={"mode": mode})).json()
        memory = recap if recap is not None else profile.get("recap")
        if memory:
            profile["session"]["system_prompt"] += "\n\n=== MEMORY: from earlier calls ===\n\n" + memory
        await self.send({"type": "session.update", "session": profile["session"]})
        self.live_mode = mode
        td = profile["session"]["input"]["turn_detection"]
        self.log("switch", f"{profile['name']} v{profile['version']} by {source}; silence {td['min_silence']}-{td['max_silence']} ms; "
                           f"tools {[t['name'] for t in profile['session']['tools']] or 'none'}; memory {'yes' if memory else 'no'}")
        await self.http.post("/switch", json={"session_id": self.session_id, "ts_ms": int(time.time() * 1000),
                                              "mode": mode, "version_id": profile["version_id"], "source": source})

    async def maybe_route(self, text: str, final: bool) -> None:
        """Same throttle as client.js: partials once per 1.5 s when grown by three words, every final."""
        words = len(text.split())
        if self.route_busy or words < 4:
            return
        if not final and (text == self.routed_text or words - self.last_words < 3 or time.time() - self.last_route_at < 1.5):
            return
        self.route_busy = True
        self.last_route_at, self.routed_text, self.last_words = time.time(), text, words
        try:
            r = await self.http.post("/route", json={"text": text, "live_mode": self.live_mode, "session_id": self.session_id,
                                                     "ts_ms": int(time.time() * 1000), "final": final, "agent_last": self.agent_last})
            d = r.json()
            if r.status_code == 200 and d.get("switch") and d.get("mode") and d["mode"] != self.live_mode:
                self.log("route", f"{d['signal']} -> {d['mode']}")
                await self.apply_mode(d["mode"], "spawner" if d.get("spawned") else "router", d.get("recap"))
        except Exception as err:
            self.log("route", f"failed: {err}")
        finally:
            self.route_busy = False

    async def run_tool(self, msg: dict) -> None:
        started = time.time()
        try:
            r = await self.http.post("/tool", json={"name": msg["name"], "arguments": msg.get("arguments") or {}})
            result = r.json()["result"]
        except Exception as err:
            result = json.dumps({"error": f"the tool could not be reached: {err}"})
        self.log("tool", f"{msg['name']} {json.dumps(msg.get('arguments'))} -> {len(result)} chars in {time.time() - started:.1f}s")
        await self.send({"type": "tool.result", "call_id": msg["call_id"], "result": result})

    async def pump_audio(self, pcm: bytes) -> None:
        """Real-time pacing: the silence windows are measured by the clock, so the audio must arrive at speed."""
        step = RATE * 2 * CHUNK_MS // 1000
        start = time.time()
        for i, offset in enumerate(range(0, len(pcm), step)):
            await asyncio.sleep(max(0.0, start + i * CHUNK_MS / 1000 - time.time()))
            await self.send({"type": "input.audio", "audio": base64.b64encode(pcm[offset:offset + step]).decode()})
        await asyncio.sleep(8)  # let the last reply land
        self.log("end", "recording finished")
        await self.send({"type": "session.end"})

    async def call(self, pcm: bytes) -> str:
        token = (await self.http.get("/token")).json()["token"]
        async with websockets.connect(f"{config.WS_URL}?token={token}", max_size=None) as ws:
            self.ws = ws
            await self.send({"type": "session.update", "session": {"agent_id": os.environ["AGENT_ID"]}})
            pump = None
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "session.ready":
                    self.session_id = msg["session_id"]
                    self.t0 = time.time()
                    self.log("ready", self.session_id)
                    await self.apply_mode(self.mode)
                    pump = asyncio.create_task(self.pump_audio(pcm))
                elif kind == "transcript.user.delta":
                    asyncio.create_task(self.maybe_route(msg["text"], False))
                elif kind == "transcript.user":
                    self.log("you", msg["text"])
                    asyncio.create_task(self.maybe_route(msg["text"], True))
                elif kind == "transcript.agent":
                    self.agent_last = msg.get("text") or ""
                    self.log("agent", self.agent_last)
                elif kind == "reply.done" and msg.get("status") == "interrupted":
                    self.log("cut", "agent interrupted")
                elif kind == "tool.call":
                    asyncio.create_task(self.run_tool(msg))
                elif kind == "session.error":
                    self.log("error", f"{msg.get('code')}: {msg.get('message')}")
                elif kind == "session.ended":
                    break
            if pump:
                pump.cancel()
        await self.http.aclose()
        return self.session_id


def summary(client: httpx.Client, session_id: str) -> dict:
    """Counts from the stored timeline: replies, interruptions, gap from the end of speech to the reply."""
    for _ in range(12):  # the timeline artifact appears up to a minute after the socket closes
        session = sessions.get_session(client, session_id)
        turns = [t for t in sessions.get_timeline(client, session) if t.get("trigger") == "user_speech"]
        if turns:
            break
        time.sleep(5)
    gaps = sorted((t.get("agent_reply_started_at_ms") or 0) - (t.get("user_speech_ended_at_ms") or 0)
                  for t in turns if t.get("agent_reply_started_at_ms") and t.get("user_speech_ended_at_ms"))
    return {"turns": len(turns), "interrupted": sum(t.get("status") == "interrupted" for t in turns),
            "reply gap median ms": gaps[len(gaps) // 2] if gaps else None, "reply gap max ms": gaps[-1] if gaps else None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("session_id")
    p.add_argument("--server", default="http://localhost:3000")
    p.add_argument("--password", default="")
    p.add_argument("--mode", default="general")
    args = p.parse_args()
    config.load_env()
    pcm = user_pcm(args.session_id)
    print(f"replaying {len(pcm) / RATE / 2:.0f} s of the user's audio from {args.session_id}\n")
    new_id = asyncio.run(Client(args.server, args.password, args.mode).call(pcm))
    with httpx.Client(headers=config.headers(), timeout=60) as client:
        print("\nold:", summary(client, args.session_id))
        print("new:", summary(client, new_id), new_id)


if __name__ == "__main__":
    main()
