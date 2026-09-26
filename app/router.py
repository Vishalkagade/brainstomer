"""Decide which mode an utterance belongs to, fast enough to run while the user is still talking.

    .venv/bin/python app/router.py build                 make one fingerprint per mode from its live version
    .venv/bin/python app/router.py "next set incline"    route one utterance and show the scores

Two signals, in order. A keyterm that belongs to exactly one mode decides in 0 ms.
Otherwise the text is embedded and compared by cosine to each mode's fingerprint.
No language model is involved; nothing here writes prose.
"""

import json
import math
import re
import sys
import time
from contextlib import closing
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import store  # noqa: E402

EMBED_URL = "https://api.fireworks.ai/inference/v1/embeddings"
EMBED_MODELS = [  # ordered; first that answers is used. Fireworks availability changes without notice
    "accounts/fireworks/models/qwen3-embedding-8b",
]
MIN_WORDS = 4      # shorter than this is not routed at all ("yes", "next set")
MARGIN = 0.05      # winner must beat the live mode's score by this much to cause a switch; a guess until measured
MARGIN_TO_GENERAL = 0.15  # falling back to general is a demotion. Measured 18 Sep: wrong exits had gaps 0.10-0.13, real ones 0.17+
GENERAL = "general"  # the landing mode; its fingerprint is its about line, so everyday topics win it outright
FILLER = "filler"    # not a mode: an anchor for acknowledgements. When it wins, nothing changes.
FILLER_TEXT = ("Filler and acknowledgements: okay, I understand, yes, that makes sense, thank you, "
               "got it, wait, hmm, let me see, uh, alright then.")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def embed(texts: list[str]) -> tuple[list[list[float]], str]:
    """Vectors for each text, plus the model that produced them."""
    key = config.require("FIREWORKS_API_KEY", "https://app.fireworks.ai/settings/users/api-keys")
    failures = []
    for model in EMBED_MODELS:
        response = httpx.post(EMBED_URL, headers={"Authorization": f"Bearer {key}"},
                              json={"model": model, "input": texts}, timeout=30)
        if response.status_code in (404, 429, 500, 502, 503):
            failures.append(f"{model.rsplit('/', 1)[-1]}: {response.status_code}")
            continue
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda d: d["index"])  # API may reorder
        return [d["embedding"] for d in data], model
    raise SystemExit("No embedding model answered: " + ", ".join(failures))


def fingerprint_text(live: dict) -> str:
    """What a mode is ABOUT, not how the agent should behave in it.

    The prompt text describes behaviour ("let pauses sit", "one sentence") and matched filler
    better than real content. `about` names the subject matter; keyterms add the vocabulary.
    """
    s = live["settings"]
    about = s.get("about") or live["prompt"]  # no about line yet: the prompt is the only text we have
    return f"{live['name']}: {about}\n\nVocabulary: {', '.join(s.get('keyterms', []))}"


def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def fingerprint_one(conn, mode_id: str, seed_vectors: list[list[float]] | None = None) -> None:
    """Store the fingerprint of a single mode. Used when a mode is spawned mid-call.

    seed_vectors are the utterances the mode was spawned from. The fingerprint is the average of
    the about-line vector and their centroid, so the mode sounds like what was actually said.
    """
    live = store.current(conn, mode_id)
    (vec,), model = embed([fingerprint_text(live)])
    if seed_vectors:
        centroid = [sum(col) / len(seed_vectors) for col in zip(*(unit(s) for s in seed_vectors))]
        vec = [(a + b) / 2 for a, b in zip(unit(vec), unit(centroid))]
    conn.execute("INSERT OR REPLACE INTO fingerprints (mode_id, version_id, model, vector, created_at) "
                 "VALUES (?, ?, ?, ?, ?)", (mode_id, live["version_id"], model, json.dumps(vec), store.now()))
    conn.commit()


def promote(conn, version_id: int, embed_one=None) -> dict:
    """Move a mode's live pointer. The fingerprint is re-embedded only if the words it is built from changed."""
    target = store.version(conn, version_id)
    if store.is_layer(conn, target["mode_id"]):  # the user core is never routed to: nothing to embed
        store.promote(conn, version_id)
        return {"mode": target["mode_id"], "live": version_id, "n": target["n"], "fingerprint_refreshed": False}
    before = fingerprint_text(store.current(conn, target["mode_id"]))
    store.promote(conn, version_id)
    after = fingerprint_text(store.current(conn, target["mode_id"]))
    refreshed = before != after
    if refreshed:
        (embed_one or fingerprint_one)(conn, target["mode_id"])
    return {"mode": target["mode_id"], "live": version_id, "n": target["n"], "fingerprint_refreshed": refreshed}


def build(conn) -> list[str]:
    """One fingerprint per mode from its live version, plus the filler anchor. Rebuild after any promote."""
    lives = [store.current(conn, m["id"]) for m in store.list_modes(conn)]
    vectors, model = embed([fingerprint_text(live) for live in lives] + [FILLER_TEXT])
    for live, vec in zip(lives, vectors):
        conn.execute(
            "INSERT OR REPLACE INTO fingerprints (mode_id, version_id, model, vector, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (live["mode"], live["version_id"], model, json.dumps(vec), store.now()))
    conn.execute("INSERT OR REPLACE INTO anchors (name, model, vector, created_at) VALUES (?, ?, ?, ?)",
                 (FILLER, model, json.dumps(vectors[-1]), store.now()))
    conn.commit()
    return [live["mode"] for live in lives]


def load_fingerprints(conn) -> dict[str, list[float]]:
    """Mode vectors plus anchors, one dict. {} until `router.py build` has run."""
    rows = conn.execute("SELECT f.mode_id AS name, f.vector FROM fingerprints f "
                        "JOIN modes m ON m.id = f.mode_id WHERE m.status != 'archived' "
                        "UNION ALL SELECT name, vector FROM anchors").fetchall()
    return {r["name"]: json.loads(r["vector"]) for r in rows}


def keyterm_index(conn) -> dict[str, set[str]]:
    """keyterm (lowercased) -> the modes that list it. A term in two modes decides nothing."""
    index: dict[str, set[str]] = {}
    for m in store.list_modes(conn):
        for term in store.current(conn, m["id"])["settings"].get("keyterms", []):
            index.setdefault(term.lower(), set()).add(m["id"])
    return index


FILLERS = re.compile(r"(?:,\s*)?\b(uh|um|umm|uhh|erm|hmm)\b[,.]?", re.IGNORECASE)  # only sounds, never words like "like"


def strip_fillers(text: str) -> str:
    """'Uh, suggest me, uh, best places' -> 'suggest me best places'. Fillers carry no topic."""
    return re.sub(r"\s+", " ", FILLERS.sub(" ", text)).strip(" ,")


def keyterm_hits(text: str, index: dict[str, set[str]]) -> dict[str, list[str]]:
    """mode -> the terms from that mode found in text, counting only terms unique to one mode."""
    lowered = f" {re.sub(r'[^a-z0-9 ]+', ' ', text.lower())} "
    hits: dict[str, list[str]] = {}
    for term, modes in index.items():
        if len(modes) != 1:
            continue
        if f" {term} " in lowered:
            hits.setdefault(next(iter(modes)), []).append(term)
    return hits


def decide(text: str, live_mode: str | None, fingerprints: dict, index: dict,
           embed_fn=embed, margin: float = MARGIN, final: bool = True) -> dict:
    """The whole decision, as a dict the page can act on and the log can keep.

    signal: 'too_short' | 'keyterm' | 'no_fingerprints' | 'filler' | 'unknown' | 'embedding'
    switch: True only when the page should change mode.
    'filler' = the acknowledgement anchor won: say nothing, change nothing.
    'unknown' = general won: none of the real modes fits; the words are worth remembering.
    final: False while he is still talking. A partial may move INTO a topic, never out to general.
    """
    text = strip_fillers(text)
    words = text.split()
    if len(words) < MIN_WORDS:
        return {"mode": live_mode, "switch": False, "signal": "too_short", "scores": {}, "terms": []}

    hits = keyterm_hits(text, index)
    if len(hits) == 1 and next(iter(hits)) != live_mode:  # a hit for the mode you are already in proves nothing; embed instead
        (mode, terms), = hits.items()
        return {"mode": mode, "switch": True, "signal": "keyterm", "scores": {}, "terms": terms}

    if not fingerprints:  # nothing to compare against; say so instead of crashing the call
        return {"mode": live_mode, "switch": False, "signal": "no_fingerprints", "scores": {}, "terms": []}

    t0 = time.perf_counter()
    (vec,), _ = embed_fn([text])
    embed_ms = round((time.perf_counter() - t0) * 1000)
    scores = {m: round(cosine(vec, fp), 4) for m, fp in fingerprints.items()}
    winner = max(scores, key=scores.get)

    if winner == FILLER:  # "okay, I see": not a topic, never a reason to move
        return {"mode": live_mode, "switch": False, "signal": "filler",
                "scores": scores, "terms": [], "embed_ms": embed_ms}

    live_score = scores.get(live_mode, -1.0)
    needed = max(margin, MARGIN_TO_GENERAL) if winner == GENERAL else margin
    switch = winner != live_mode and scores[winner] - live_score >= needed
    if winner == GENERAL and not final:  # half a sentence has no topic words yet; leaving early buys nothing
        switch = False
    signal = "unknown" if winner == GENERAL else "embedding"
    return {"mode": winner if switch else live_mode, "switch": switch, "signal": signal,
            "scores": scores, "terms": [], "embed_ms": embed_ms,
            "_vector": vec}  # for the candidates table; the server strips it before answering the page


# The Jev rule (decided 22 Sep after 23 logged asks agreed 91% with the embedding, both disagreements Jev's way).
SWITCH_P = 0.8   # Jev's probability for a mode before we move; TypeSafe's own "act automatically" band starts at 0.9 for high stakes
REPLY_P = 0.6    # above this, the words answer the agent's last question: stay whatever they look like


def decide_jev(text: str, live_mode: str | None, verdict: dict, final: bool, agent_last: str) -> dict | None:
    """The router's decision from Jev's verdict. None when Jev had no answer, so the caller falls back to embeddings.

    signal: 'reply' (answering the agent, stay) | 'jev' (a mode won or nothing changed) | 'unknown' (none fits: general).
    """
    if not verdict or "error" in verdict:
        return None
    words = strip_fillers(text).split()
    if len(words) < MIN_WORDS:
        return {"mode": live_mode, "switch": False, "signal": "too_short", "scores": {}, "terms": [], "jev": verdict}
    base = {"scores": verdict["p"], "terms": [], "jev": verdict, "confidence": verdict["confidence"]}
    target = GENERAL if verdict["mode"] == "none" else verdict["mode"]
    p = verdict["p"].get(verdict["mode"], 0.0)
    if target != GENERAL and target != live_mode and p >= SWITCH_P:  # a confident subject wins, even mid-sentence
        return {"mode": target, "switch": True, "signal": "jev", **base}
    # The reply score only ever keeps you where you are. It must not block a subject: "I wanted to learn about
    # attention" right after "what's on your mind?" scores 0.94 as a reply and is still a new subject.
    if agent_last and verdict["reply"] >= REPLY_P:  # no agent_last on the first ask of a call: the score means nothing then
        return {"mode": live_mode, "switch": False, "signal": "reply", **base}
    if target == GENERAL and target != live_mode and p >= SWITCH_P and final:  # nothing fits: general, on a finished sentence only
        return {"mode": GENERAL, "switch": True, "signal": "unknown", **base}
    return {"mode": live_mode, "switch": False, "signal": "unknown" if target == GENERAL else "jev", **base}


def compare(conn, fresh: bool = False) -> None:
    """Embedding router versus the Jev rule on every logged ask. --fresh re-asks Jev with today's modes and the logged context."""
    rows = [dict(r) for r in conn.execute("SELECT text, live_mode, mode, switch, signal, embed_ms, jev_json, agent_last, final FROM routes "
                                          "WHERE jev_json IS NOT NULL ORDER BY ts_ms")]
    if not rows:
        print("No asks with a Jev verdict yet. Make a call with the server running.")
        return
    if fresh:
        import jev
        options = jev.mode_options(conn)
        for r in rows:
            r["jev_json"] = json.dumps(jev.route(r["text"], r["live_mode"], r["agent_last"] or "", options))
    agree = errors = 0
    jev_ms, embed_ms = [], []
    print(f"{'emb':<15} {'jev rule':<15} {'conf':>5} {'reply':>5} {'push':>5} {'ms e/j':>9}  text")
    for r in rows:
        j = json.loads(r["jev_json"])
        if "error" in j:
            errors += 1
            print(f"{r['mode'] or '-':<15} {'ERROR':<15} {'':>5} {'':>5} {'':>5} {'':>9}  {j['error'][:60]}")
            continue
        rule = decide_jev(r["text"], r["live_mode"], j, final=r["final"] is None or bool(r["final"]), agent_last=r["agent_last"] or "")
        jev_mode = rule["mode"]
        same = jev_mode == r["mode"]
        agree += same
        jev_ms.append(j["ms"]); embed_ms.append(r["embed_ms"] or 0)
        print(f"{r['mode'] or '-':<15} {(jev_mode or '-') + ('' if same else ' *'):<15} {j['confidence']:>5.2f} {j['reply']:>5.2f} "
              f"{j['pushback']:>5.2f} {(r['embed_ms'] or 0):>4}/{j['ms']:<4}  {r['text'][:60]}")
    n = len(rows) - errors
    if n:
        med = lambda xs: sorted(xs)[len(xs) // 2]
        print(f"\n{n} asks: agree on {agree} ({100 * agree // n}%), * marks a disagreement. "
              f"median ms embedding {med(embed_ms)}, jev {med(jev_ms)}. {errors} Jev errors.")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: app/router.py build | compare [--fresh] | "some words" [live_mode]')
    config.load_env()
    with closing(store.connect()) as conn:
        if sys.argv[1] == "build":
            for mode in build(conn):
                print(f"{mode:<12} fingerprint stored")
            return
        if sys.argv[1] == "compare":
            compare(conn, fresh="--fresh" in sys.argv)
            return
        text = sys.argv[1]
        live_mode = sys.argv[2] if len(sys.argv) > 2 else None
        t0 = time.perf_counter()
        fingerprints = load_fingerprints(conn)
        if not fingerprints:
            raise SystemExit("No fingerprints yet. Run: app/router.py build")
        d = decide(text, live_mode, fingerprints, keyterm_index(conn))
        d["total_ms"] = round((time.perf_counter() - t0) * 1000)
        print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
