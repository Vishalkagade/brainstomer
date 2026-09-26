"""The L1 evolver: rewrites the user core from what the modes have learned.

    .venv/bin/python app/user_core.py            what it would read, and whether it is due
    .venv/bin/python app/user_core.py propose    write the next core version and make it LIVE
    .venv/bin/python app/user_core.py propose --hold     store it without promoting

It reads only what the store already holds: every mode's live overlay (a rule about him in
two of them is a fact seen by two independent modes), the corrections, the rolled-back
versions (the ledger: lessons the evidence undid), and his own words from the judged turns
(a stated fact enters on one hearing, marked "he said"; decided 26 Sep). It runs when a mode
version has been kept since its last proposal, goes live at once, and is judged on every
mode's turns by evidence.review_core.
"""

import json
import os
import re
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import evolver  # noqa: E402
import profiles  # noqa: E402
import router  # noqa: E402
import store  # noqa: E402

MAX_SAID = 150  # his own lines since the last proposal; newest kept


def last_proposal_at(conn) -> str:
    row = conn.execute("SELECT MAX(created_at) AS at FROM versions WHERE mode_id = ? AND source = 'evolver'",
                       (store.CORE,)).fetchone()
    return row["at"] or ""


def gather(conn) -> dict:
    """Everything the core evolver may learn from, all of it already in the store."""
    since = last_proposal_at(conn)
    overlays = [(m["name"], store.current(conn, m["id"])["prompt"]) for m in store.list_modes(conn)]
    corrections = [(r["mode_id"], r["text"]) for r in conn.execute(
        "SELECT mode_id, text FROM candidates WHERE kind = 'correction' AND created_at > ? ORDER BY id", (since,))]
    rolled = [(r["mode_id"], r["n"], r["rationale"].strip().splitlines()[0][:160], r["verdict_why"]) for r in conn.execute(
        "SELECT mode_id, n, rationale, verdict_why FROM versions WHERE verdict = 'rolled_back' AND verdict_at > ? "
        "ORDER BY verdict_at", (since,))]
    said = [(r["mode_id"], r["user_text"]) for r in conn.execute(
        "SELECT mode_id, user_text FROM evidence WHERE judged_at > ? ORDER BY judged_at", (since,))][-MAX_SAID:]
    return {"since": since, "overlays": overlays, "corrections": corrections, "rolled_back": rolled, "said": said}


def due(conn) -> tuple[bool, str]:
    """A mode version kept since the last core proposal is the signal: a mode has learned something that held."""
    live = store.current(conn, store.CORE)
    row = conn.execute("SELECT source, parent_id, verdict FROM versions WHERE id = ?", (live["version_id"],)).fetchone()
    if row["source"] == "evolver" and row["parent_id"] and not row["verdict"]:
        return False, f"core v{live['n']} is still waiting for its verdict"
    kept = conn.execute("SELECT COUNT(*) FROM versions WHERE verdict = 'kept' AND mode_id != ? AND verdict_at > ?",
                        (store.CORE, last_proposal_at(conn))).fetchone()[0]
    since = " since the last core proposal" if last_proposal_at(conn) else ""
    return kept > 0, f"{kept} mode version(s) kept{since}"


def build_messages(live: dict, material: dict) -> list[dict]:
    cap = profiles.CAPS["l1"]
    used = profiles.estimate_tokens(live["prompt"])
    rules = f"""You maintain the USER CORE of a voice agent: who this person is, true in every area of his life. It is read after a bedrock layer (safety, precedence, speaking style) and before one mode overlay (the current area). You will propose the next version from what the modes have learned.

Sections, in this order: constants (hard constraints he stated), communication style, decision style, open loops, corrections ledger. Keep the headings that exist.

Hard limits:
- Under {cap} tokens (about {cap * profiles.CHARS_PER_TOKEN} characters). It is currently about {used}. To add, you must remove, and you must list what you removed.
- Do not restate anything that belongs to the bedrock (how to speak, when to caveat).

What may enter, and nothing else:
- A rule about him that appears in TWO OR MORE different mode overlays. One mode alone is a mode's style, not him.
- A fact he stated in his own words (the "he said" lines). Write it as a fact, prefixed "He said:".
- A correction, or a lesson a rolled-back version had learned: goes in the corrections ledger as a plain "do not" line, so no later version re-learns it from the same conversations.
- Never diagnose, never add a medical, dietary or financial claim he did not state himself.
- Do not learn a lasting trait from a single moment. Being terse out of breath is the situation, not the person.
- If nothing qualifies, return the current version unchanged and say so in the rationale.

Return ONLY a JSON object with exactly these keys:
{{"prompt": <string>, "rationale": <string, 2-6 sentences, what changed and which evidence>, "evicted": [<string>, ...]}}
The prompt is read by a voice agent: plain prose under the headings, no markdown lists, nothing that cannot be spoken about."""

    parts = [f"# Current user core (v{live['n']})\n\n{live['prompt']}", "# Live mode overlays (a rule in two of them is about him)"]
    for name, prompt in material["overlays"]:
        parts.append(f"## {name}\n{prompt}")
    if material["corrections"]:
        parts.append("# Corrections (what he said right after pushing back)\n" +
                     "\n".join(f"- [{m}] {t}" for m, t in material["corrections"]))
    if material["rolled_back"]:
        parts.append("# Versions the evidence rolled back (lessons that did not hold)\n" +
                     "\n".join(f"- {m} v{n}: {r} -> {w}" for m, n, r, w in material["rolled_back"]))
    if material["said"]:
        parts.append("# He said (his own words, by mode, since the last core proposal)\n" +
                     "\n".join(f"- [{m}] {t}" for m, t in material["said"]))
    return [{"role": "system", "content": rules}, {"role": "user", "content": "\n\n".join(parts)}]


def parse(text: str) -> dict:
    body = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        data = json.loads(body)
    except json.JSONDecodeError as err:
        raise ValueError(f"model did not return JSON: {err}\n---\n{text[:400]}")
    missing = {"prompt", "rationale", "evicted"} - set(data if isinstance(data, dict) else {})
    if missing:
        raise ValueError(f"model output lacks {sorted(missing)}")
    if not isinstance(data["prompt"], str) or not data["prompt"].strip():
        raise ValueError("model returned an empty core")
    return data


def propose(conn, hold: bool = False) -> int:
    """Ask the model, check the cap, store the core version and make it live (hold=True: store only)."""
    live = store.current(conn, store.CORE)
    material = gather(conn)
    answer, model = evolver.call_model(build_messages(live, material))
    proposal = parse(answer)
    prompt = proposal["prompt"].strip()
    used = profiles.estimate_tokens(prompt)
    if used > profiles.CAPS["l1"]:
        raise SystemExit(f"Model proposed a {used}-token core, cap is {profiles.CAPS['l1']}. Nothing stored.")
    rationale = proposal["rationale"].strip()
    if proposal["evicted"]:
        rationale += "\n\nEvicted: " + "; ".join(str(e) for e in proposal["evicted"])
    rationale += (f"\n\nRead: {len(material['overlays'])} overlays, {len(material['corrections'])} corrections, "
                  f"{len(material['rolled_back'])} rolled back, {len(material['said'])} of his lines. Model: {model.rsplit('/', 1)[-1]}.")
    version_id = store.add_version(conn, store.CORE, live["settings"], prompt, rationale, source="evolver",
                                   parent_id=live["version_id"])
    if not hold:
        router.promote(conn, version_id)  # judged on every mode's turns by evidence.review_core
    return version_id


def auto_propose(conn) -> int | None:
    """The unattended path, once per refresh: propose when a mode version has been kept."""
    if os.environ.get("AUTO_EVOLVE", "on") == "off" or os.environ.get("BRAINSTORMER_READ_ONLY") == "1":
        return None
    ok, why = due(conn)
    if not ok:
        return None
    import time
    if time.time() - evolver._failed_at.get(store.CORE, 0) < evolver.RETRY_AFTER_S:
        return None
    with evolver._proposing_lock:
        if store.CORE in evolver._proposing:
            return None
        evolver._proposing.add(store.CORE)
    try:
        version_id = propose(conn)
        conn.execute("UPDATE versions SET rationale = rationale || ' Proposed by itself: ' || ? || '.' WHERE id = ?",
                     (why, version_id))
        conn.commit()
        print(f"user core: new version {version_id} live ({why})")
        return version_id
    except (SystemExit, ValueError) as err:
        evolver._failed_at[store.CORE] = time.time()
        print(f"user core not evolved: {err}")
        return None
    finally:
        with evolver._proposing_lock:
            evolver._proposing.discard(store.CORE)


def main() -> None:
    config.load_env()
    with closing(store.connect()) as conn:
        if not store.is_layer(conn, store.CORE):
            raise SystemExit("No user core in the store yet: start the server once, or run app/store.py seed")
        if "propose" in sys.argv[1:]:
            hold = "--hold" in sys.argv[1:]
            live = store.current(conn, store.CORE)
            version_id = propose(conn, hold=hold)
            row = conn.execute("SELECT n, rationale FROM versions WHERE id = ?", (version_id,)).fetchone()
            print(f"Stored user core v{row['n']} as version id {version_id}, {'NOT live (held)' if hold else 'LIVE now'}.\n")
            print(row["rationale"])
            print(f"\nReview:   app/store.py diff {live['version_id']} {version_id}")
            return
        live = store.current(conn, store.CORE)
        m = gather(conn)
        ok, why = due(conn)
        print(f"user core v{live['n']}, {profiles.estimate_tokens(live['prompt'])} of {profiles.CAPS['l1']} tokens")
        print(f"by itself: {'due' if ok else 'not yet'}: {why}")
        print(f"would read: {len(m['overlays'])} overlays, {len(m['corrections'])} corrections, "
              f"{len(m['rolled_back'])} rolled back, {len(m['said'])} of his lines since {m['since'] or 'the start'}\n")
        for mode, text in m["corrections"]:
            print(f"   correction [{mode}] {text[:100]}")
        for mode, n, r, w in m["rolled_back"]:
            print(f"   rolled back {mode} v{n}: {w}")


if __name__ == "__main__":
    main()
