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
MARGIN_TO_GENERAL = 0.10  # falling back to general is a demotion, so it needs a clearer win than moving to a topic
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
           embed_fn=embed, margin: float = MARGIN) -> dict:
    """The whole decision, as a dict the page can act on and the log can keep.

    signal: 'too_short' | 'keyterm' | 'no_fingerprints' | 'filler' | 'unknown' | 'embedding'
    switch: True only when the page should change mode.
    'filler' = the acknowledgement anchor won: say nothing, change nothing.
    'unknown' = general won: none of the real modes fits; the words are worth remembering.
    """
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
    signal = "unknown" if winner == GENERAL else "embedding"
    return {"mode": winner if switch else live_mode, "switch": switch, "signal": signal,
            "scores": scores, "terms": [], "embed_ms": embed_ms,
            "_vector": vec}  # for the candidates table; the server strips it before answering the page


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: app/router.py build   |   app/router.py "some words" [live_mode]')
    config.load_env()
    with closing(store.connect()) as conn:
        if sys.argv[1] == "build":
            for mode in build(conn):
                print(f"{mode:<12} fingerprint stored")
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
