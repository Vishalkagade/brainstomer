"""What a mode remembers from its own earlier calls.

    .venv/bin/python app/memory.py             refresh every mode's recap from AssemblyAI, then print them
    .venv/bin/python app/memory.py gym         print one mode's recap

A call starts empty on AssemblyAI's side. So when the page switches into a mode it
appends a MEMORY block to that mode's system prompt: the last exchanges from the last
three calls in that mode, and on a router switch also the past exchanges closest to
what was just said. Capped in tokens. General gets an index of the areas talked about.

Why the prompt and not a conversation.message: measured 26 Sep, a conversation.message
(system or user) never reached the model in three forced replies; the same text inside
the system prompt did. Recaps are built off the reply path: at page load and by this
command. `/profile` only reads what is cached; `/route` builds one from the store.
"""

import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import evidence  # noqa: E402
import evolver  # noqa: E402
import profiles  # noqa: E402
import router  # noqa: E402
import store  # noqa: E402

MAX_TOKENS = 500      # hard ceiling per recap, whatever history_depth says; estimate is chars/4
RELEVANT_K = 3        # on a router switch: the closest past exchanges to what was just said, not the last N
RELEVANT_MIN = 0.45   # measured 15 Sep: true matches 0.55-0.66, same mode other exercise ~0.49, unrelated <= 0.37
MAX_AGE_DAYS = 14     # "not too old": nothing older than this comes back
MAX_SESSIONS = 3      # "at least the last three conversations" (Vishal, 26 Sep)
NO_RECAP = {"general"}


def render(turns: list[dict], mode_name: str, tail: str = "") -> str:
    """The memory block. Newest last, trimmed from the front to stay under MAX_TOKENS. `tail` is kept whole."""
    if not turns and not tail:
        return ""
    lines = []
    for t in turns:
        lines.append(f"He: {t['user_transcript'].strip()}")
        lines.append(f"You: {t['agent_text'].strip()}")
    head = (f"Earlier conversations in {mode_name} mode, most recent last. Use them as context. "
            f"If he asks what was said before, tell him from these; otherwise do not recite them.\n")
    body = "\n".join(lines)
    while profiles.estimate_tokens(head + body + tail) > MAX_TOKENS and len(lines) > 2:
        lines = lines[2:]  # drop the oldest exchange
        body = "\n".join(lines)
    return head + body + tail


def recent(conn, mode_id: str, depth: int, exclude_session: str | None = None) -> list[dict]:
    """The last `depth` exchanges from the last MAX_SESSIONS calls in this mode, oldest first. Store only, no fetch."""
    rows = [e for e in store.exchanges_for(conn, mode_id) if e["session_id"] != exclude_session]
    order = []
    for e in rows:
        if e["session_id"] not in order:
            order.append(e["session_id"])
    keep = set(order[-MAX_SESSIONS:])
    return [e for e in rows if e["session_id"] in keep][-depth:]


def index(conn) -> str:
    """For General: which areas earlier calls covered, so 'what do you know about me' has an answer."""
    parts = []
    for r in conn.execute("SELECT mode_id, COUNT(DISTINCT session_id) AS calls, MAX(ts_ms) AS last "
                          "FROM exchanges GROUP BY mode_id ORDER BY last DESC"):
        if r["mode_id"] in NO_RECAP or not any(m["id"] == r["mode_id"] for m in store.list_modes(conn)):
            continue
        when = datetime.fromtimestamp(r["last"] / 1000, timezone.utc).strftime("%-d %B") if r["last"] else "earlier"
        parts.append(f"{store.current(conn, r['mode_id'])['name']} ({r['calls']} call{'s' if r['calls'] != 1 else ''}, last on {when})")
    if not parts:
        return ""
    return ("Areas he has talked about with you in earlier calls: " + "; ".join(parts) + ". If he asks what you know "
            "or what was said before, name these areas; the details arrive when the call moves into one of them.")


def recall(conn, mode_id: str, vector: list[float] | None = None, exclude_session: str | None = None) -> str:
    """The MEMORY block for a switch into `mode_id`: recent exchanges, plus the closest ones when there are words."""
    if mode_id in NO_RECAP:
        return index(conn)
    live = store.current(conn, mode_id)
    rows = recent(conn, mode_id, int(live["settings"].get("history_depth") or 10), exclude_session)
    tail = ""
    if vector is not None:
        seen = {(e["session_id"], e["ts_ms"]) for e in rows}
        close = [e for e in closest(conn, mode_id, vector, exclude_session) if (e["session_id"], e["ts_ms"]) not in seen]
        if close:
            tail = "\nAlso, from earlier calls, related to what he just said:\n" + "\n".join(
                f"He: {e['user_text'].strip()}\nYou: {e['agent_text'].strip()}" for e in close)
    return render([{"user_transcript": e["user_text"], "agent_text": e["agent_text"]} for e in rows], live["name"], tail)


def index_exchanges(conn, client: httpx.Client, mode_id: str, convos: list[dict] | None = None) -> int:
    """Embed and store this mode's past exchanges. Only sessions not seen before are paid for."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)).isoformat()
    seen = store.exchange_sessions(conn, mode_id)
    if convos is None:
        convos = evolver.conversations(conn, client, mode_id)
    fresh = [c for c in convos if c["created_at"] >= cutoff and c["session_id"] not in seen]
    rows = [{"session_id": c["session_id"], "ts_ms": t.get("agent_reply_started_at_ms") or 0,
             "user_text": t["user_transcript"].strip(), "agent_text": t["agent_text"].strip()}
            for c in fresh for t in c["turns"]]
    if not rows:
        return 0
    vectors, _ = router.embed([router.strip_fillers(r["user_text"]) for r in rows])  # one batched call
    for r, v in zip(rows, vectors):
        r["vector"] = v
    return store.add_exchanges(conn, mode_id, rows)


def closest(conn, mode_id: str, vector: list[float], exclude_session: str | None = None,
            k: int = RELEVANT_K) -> list[dict]:
    """The k past exchanges in this mode closest to what was just said, oldest first. [] when nothing is close enough."""
    if mode_id in NO_RECAP:
        return []
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
    return sorted((e for _, e in scored[:k]), key=lambda e: e["ts_ms"])  # oldest first, like a transcript


def relevant(conn, mode_id: str, vector: list[float], exclude_session: str | None = None, k: int = RELEVANT_K) -> str:
    """Only the closest exchanges, rendered. '' when nothing is close enough."""
    top = closest(conn, mode_id, vector, exclude_session, k)
    if not top:
        return ""
    turns = [{"user_transcript": e["user_text"], "agent_text": e["agent_text"]} for e in top]
    return render(turns, store.current(conn, mode_id)["name"]).replace("most recent last", "the ones related to what he just said")


def refresh(conn, client: httpx.Client, mode_id: str) -> str:
    if mode_id in NO_RECAP:
        text = index(conn)
    else:
        convos = evolver.conversations(conn, client, mode_id)  # one fetch, three readers
        try:
            index_exchanges(conn, client, mode_id, convos)  # so relevant() has something to search
        except SystemExit as err:  # no embedding model answered: recency recap still works
            print(f"exchange index for {mode_id} skipped: {err}")
        try:
            evidence.judge_new(conn, mode_id, convos)  # what went wrong per turn, for the evolver; judged once
            evidence.review(conn, mode_id)  # keeps or rolls back an evolver version once enough turns ran on it
        except Exception as err:  # never lets a judging problem take the recap down
            print(f"evidence for {mode_id} skipped: {err}")
        try:
            evolver.auto_propose(conn, mode_id, convos)  # proposes the next version when the floors are met
        except Exception as err:
            print(f"evolver for {mode_id} skipped: {err}")
        text = recall(conn, mode_id)  # from the exchange index just updated: the last calls, no words to match yet
    conn.execute("INSERT OR REPLACE INTO recaps (mode_id, text, updated_at) VALUES (?, ?, ?)",
                 (mode_id, text, store.now()))
    conn.commit()
    return text


def refresh_all(conn, client: httpx.Client) -> dict[str, int]:
    """Every mode's recap, then the user core's turn. Returns mode -> token estimate."""
    out = {}
    for m in store.list_modes(conn):
        out[m["id"]] = profiles.estimate_tokens(refresh(conn, client, m["id"]))
    if store.is_layer(conn, store.CORE):
        try:
            import user_core
            evidence.review_core(conn)     # keeps or rolls back an evolver-written core
            user_core.auto_propose(conn)   # and writes the next one when a mode version was kept
        except Exception as err:
            print(f"user core skipped: {err}")
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
