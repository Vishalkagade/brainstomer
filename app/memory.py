"""What a mode remembers from its own earlier calls.

    .venv/bin/python app/memory.py             refresh every mode's recap from AssemblyAI, then print them
    .venv/bin/python app/memory.py gym         print one mode's recap

A call starts empty on AssemblyAI's side. So when the page switches into a mode it
injects one system message: the last few exchanges that happened in that mode
before, as a plain transcript. That is what `history_depth` in a mode's settings
now means: how many exchanges come back. Capped in tokens, sent once per mode per
call, never for General (its history is everything and nothing).

Recaps are built off the reply path: at page load and by this command, never
while a swap is waiting. `/profile` only reads what is cached.
"""

import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import evolver  # noqa: E402
import profiles  # noqa: E402
import router  # noqa: E402
import store  # noqa: E402

MAX_TOKENS = 500      # hard ceiling per recap, whatever history_depth says; estimate is chars/4
RELEVANT_K = 3        # on a router switch: the closest past exchanges to what was just said, not the last N
RELEVANT_MIN = 0.45   # measured 15 Sep: true matches 0.55-0.66, same mode other exercise ~0.49, unrelated <= 0.37
MAX_AGE_DAYS = 14     # "not too old": nothing older than this comes back
MAX_SESSIONS = 5      # and never more than this many past calls
NO_RECAP = {"general"}


def recent_turns(conn, client: httpx.Client, mode_id: str, depth: int) -> list[dict]:
    """The last `depth` usable exchanges in this mode, oldest first, from recent calls only."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)).isoformat()
    convos = [c for c in evolver.conversations(conn, client, mode_id) if c["created_at"] >= cutoff][-MAX_SESSIONS:]
    turns = [t for c in convos for t in c["turns"]]
    return turns[-depth:]


def render(turns: list[dict], mode_name: str) -> str:
    """The injected message. Newest last, trimmed from the front to stay under MAX_TOKENS."""
    if not turns:
        return ""
    lines = []
    for t in turns:
        lines.append(f"He: {t['user_transcript'].strip()}")
        lines.append(f"You: {t['agent_text'].strip()}")
    head = f"Earlier conversations in {mode_name} mode, most recent last. Use them as context; do not recite them.\n"
    body = "\n".join(lines)
    while profiles.estimate_tokens(head + body) > MAX_TOKENS and len(lines) > 2:
        lines = lines[2:]  # drop the oldest exchange
        body = "\n".join(lines)
    return head + body


def index_exchanges(conn, client: httpx.Client, mode_id: str) -> int:
    """Embed and store this mode's past exchanges. Only sessions not seen before are paid for."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)).isoformat()
    seen = store.exchange_sessions(conn, mode_id)
    fresh = [c for c in evolver.conversations(conn, client, mode_id)
             if c["created_at"] >= cutoff and c["session_id"] not in seen]
    rows = [{"session_id": c["session_id"], "ts_ms": t.get("agent_reply_started_at_ms") or 0,
             "user_text": t["user_transcript"].strip(), "agent_text": t["agent_text"].strip()}
            for c in fresh for t in c["turns"]]
    if not rows:
        return 0
    vectors, _ = router.embed([router.strip_fillers(r["user_text"]) for r in rows])  # one batched call
    for r, v in zip(rows, vectors):
        r["vector"] = v
    return store.add_exchanges(conn, mode_id, rows)


def relevant(conn, mode_id: str, vector: list[float], exclude_session: str | None = None,
             k: int = RELEVANT_K) -> str:
    """The k past exchanges in this mode closest to what was just said. '' when nothing is close enough."""
    if mode_id in NO_RECAP:
        return ""
    fps = router.load_fingerprints(conn)
    filler, own = fps.get(router.FILLER), fps.get(mode_id)

    def is_memory(e: dict) -> bool:
        if e["session_id"] == exclude_session:                # this call's own turns are already in context
            return False
        if len(router.strip_fillers(e["user_text"]).split()) < router.MIN_WORDS:
            return False
        if filler and own:                                    # "okay, understood, thank you" sounds like filler, not like the mode
            return router.cosine(e["vector"], filler) < router.cosine(e["vector"], own)
        return True

    scored = [(router.cosine(vector, e["vector"]), e) for e in store.exchanges_for(conn, mode_id) if is_memory(e)]
    scored = [(s, e) for s, e in scored if s >= RELEVANT_MIN]
    scored.sort(key=lambda p: p[0], reverse=True)
    top = sorted((e for _, e in scored[:k]), key=lambda e: e["ts_ms"])  # oldest first, like a transcript
    if not top:
        return ""
    name = store.current(conn, mode_id)["name"]
    turns = [{"user_transcript": e["user_text"], "agent_text": e["agent_text"]} for e in top]
    return render(turns, name).replace("most recent last", "the ones related to what he just said")


def refresh(conn, client: httpx.Client, mode_id: str) -> str:
    live = store.current(conn, mode_id)
    if mode_id in NO_RECAP:
        text = ""
    else:
        try:
            index_exchanges(conn, client, mode_id)  # so relevant() has something to search
        except SystemExit as err:  # no embedding model answered: recency recap still works
            print(f"exchange index for {mode_id} skipped: {err}")
        depth = int(live["settings"].get("history_depth") or 10)
        text = render(recent_turns(conn, client, mode_id, depth), live["name"])
    conn.execute("INSERT OR REPLACE INTO recaps (mode_id, text, updated_at) VALUES (?, ?, ?)",
                 (mode_id, text, store.now()))
    conn.commit()
    return text


def refresh_all(conn, client: httpx.Client) -> dict[str, int]:
    """Every mode's recap. Returns mode -> token estimate."""
    out = {}
    for m in store.list_modes(conn):
        out[m["id"]] = profiles.estimate_tokens(refresh(conn, client, m["id"]))
    return out


def cached(conn, mode_id: str) -> str:
    row = conn.execute("SELECT text FROM recaps WHERE mode_id = ?", (mode_id,)).fetchone()
    return row["text"] if row else ""


def main() -> None:
    config.load_env()
    with closing(store.connect()) as conn, httpx.Client(headers=config.headers(), timeout=30) as client:
        if len(sys.argv) > 1:
            print(refresh(conn, client, sys.argv[1]) or "(nothing to remember yet)")
            return
        for mode, tokens in refresh_all(conn, client).items():
            print(f"{mode:<16} recap ~{tokens} tokens")
        print()
        for m in store.list_modes(conn):
            text = cached(conn, m["id"])
            if text:
                print(f"--- {m['id']}\n{text}\n")


if __name__ == "__main__":
    main()
