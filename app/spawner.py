"""Create a new mode while the conversation is still going.

    .venv/bin/python app/spawner.py sweep                 provisional -> established / archived, by use
    .venv/bin/python app/spawner.py name "t1" "t2" "t3"   dry run: which of these group with the last, and the name

When three utterances in one call land in General and are about the same thing,
that is a topic, not a stray question. A provisional mode is created on the spot
with a vanilla prompt (a template, no waiting for prose), and the call switches
to it. L1 underneath still knows the person; the mode-specific part arrives with
the first evolution. Modes that never get used again are archived by `sweep`.
"""

import hashlib
import json
import re
import sys
from datetime import datetime
import time
from contextlib import closing
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import profiles  # noqa: E402
import router  # noqa: E402
import store  # noqa: E402

RELATED = 0.40      # cheap gate: only ask the model when the newest utterance is at least this close to another one
MIN_CLUSTER = 3     # utterances about one thing, in one call, before a mode is spawned
ESTABLISH_AFTER = 3  # distinct calls that used a provisional mode before it counts as established
ARCHIVE_PROVISIONAL_DAYS = 7    # born in a call, used in no other, and quiet this long: a likely misfire fades
ARCHIVE_ESTABLISHED_DAYS = 30   # a real area of life, quiet this long: dormant, not deleted (Vishal, 29 Sep)
REVIVE_MIN = 0.50    # a new cluster this close to an archived mode wakes it instead of spawning a twin

# Fast first; this runs mid-call. Fireworks availability changes, so an ordered list, never one id.
NAMING_MODELS = [
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/deepseek-v4-flash-0731",
    "accounts/fireworks/models/glm-5p3-flash",
    "accounts/fireworks/models/deepseek-v4-pro-0813",
]
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

TEMPLATE = """This mode was created a moment ago from what he started talking about: {about}
Nothing else is known about how he wants to be answered here yet.

Answer plainly and briefly. Do not assume expertise or the lack of it; if the level
matters, ask one short question to find it, then keep going at that level.

Prefer a concrete example over a general description. If a real answer needs
information you do not have, say what you would need rather than guessing.

Everything in the user core above still applies: how he thinks, how he wants
answers shaped, what has already been corrected."""


def related(newest: dict, others: list[dict], threshold: float = RELATED) -> list[dict]:
    """The newest utterance plus every earlier one loosely close to it. A gate, not a verdict:
    raw similarity called finance and clothes the same (0.56) and clothes and shoes different (0.45)."""
    close = [c for c in others if c["id"] != newest["id"] and router.cosine(newest["vector"], c["vector"]) >= threshold]
    return close + [newest]


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:40] or "topic"


def unique_id(conn, base: str) -> str:
    candidate, n = base, 2
    while conn.execute("SELECT 1 FROM modes WHERE id = ?", (candidate,)).fetchone():
        candidate, n = f"{base}_{n}", n + 1
    return candidate


def group_and_name(texts: list[str]) -> tuple[dict, str]:
    """One call, two jobs: which of these sentences share a subject with the LAST one, and what is it.

    Returns (fields, model id). fields["same"] holds 1-based indices of the sentences that belong with the
    last one (the last one included). Grouping by a model instead of by similarity is what lets clothes,
    shoes and jackets become "Shopping" while finance advice stays out.
    """
    key = config.require("FIREWORKS_API_KEY", "https://app.fireworks.ai/settings/users/api-keys")
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
    ask = [
        {"role": "system", "content":
            "You organise what a person says to a personal voice assistant into areas of their life. You get numbered "
            "sentences from one conversation. Decide which sentences belong to the SAME area as the LAST sentence "
            "(buying clothes, shoes and jackets are one area, Shopping; asking for finance advice is not). "
            "Name the area at UMBRELLA level, the way a person names a hobby or a part of life, never the first detail "
            "they asked about: a question on the badminton serve is the area Badminton; how to cook dal is Cooking; "
            "which Seiko to buy is Shopping; places to visit in Asia is Travel. One or two words. Return ONLY a JSON object: "
            '{"same": [<1-based numbers of every sentence in that area, including the last one>], '
            '"name": <the umbrella area, 1-2 words, e.g. "Badminton">, '
            '"about": <one sentence: the area as a whole, then what he asked about so far as examples, no advice>, '
            '"keyterms": [<up to 10 domain-specific nouns or names from the whole area a transcriber might mishear; never everyday words>]}'},
        {"role": "user", "content": numbered},
    ]
    failures = []
    for model in NAMING_MODELS:
        response = httpx.post(FIREWORKS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=60,
                              json={"model": model, "messages": ask, "temperature": 0.2, "max_tokens": 1200,
                                    "response_format": {"type": "json_object"}})
        if response.status_code in (404, 429, 500, 502, 503):
            failures.append(f"{model.rsplit('/', 1)[-1]}: {response.status_code}")
            continue
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"] or ""
        body = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip())
        data = json.loads(body)
        fields = clean_fields(data)
        same = data.get("same") if isinstance(data.get("same"), list) else []
        fields["same"] = sorted({int(i) for i in same if str(i).isdigit() and 1 <= int(i) <= len(texts)} | {len(texts)})
        return fields, model
    raise SystemExit("No grouping model answered: " + ", ".join(failures))


def clean_fields(data: dict) -> dict:
    """Trust nothing the model returned: right types, right sizes, plain strings."""
    name = str(data.get("name") or "").strip()[:40] or "New topic"
    about = str(data.get("about") or "").strip()[:300] or name
    keyterms = [str(k).strip() for k in (data.get("keyterms") or [])
                if isinstance(k, (str, int)) and len(str(k).strip()) >= 4][:10]  # "dry", "oven": too common to route on
    return {"name": name, "about": about, "keyterms": keyterms}


def far_hue(taken: list[int]) -> int:
    """The hue furthest around the wheel from every mode that exists, so a newborn mode is visibly new."""
    gap = lambda h: min(min(abs(h - t), 360 - abs(h - t)) for t in taken)
    return max(range(0, 360, 5), key=gap)


def default_settings(mode_id: str, fields: dict, taken: list[int] | None = None) -> dict:
    """A vanilla profile: neutral listening, web search, quick to evolve. Hue far from `taken`, else from the id."""
    return {
        "name": fields["name"],
        "about": fields["about"],
        "hue": far_hue(taken) if taken else int(hashlib.md5(mode_id.encode()).hexdigest(), 16) % 360,
        "tools": ["web_search"],  # a new subject usually needs a lookup before it needs a style
        "history_depth": 10,
        "model": None,
        "turn_detection": {"min_silence": 800, "max_silence": 2000, "interrupt_response": True, "interruption_delay": 300},
        "transcription_mode": "balanced",
        "keyterms": fields["keyterms"],
        "evolve_after": {"sessions": 2, "turns": 10},
    }


def spawn(conn, session_id: str, members: list[dict], fields: dict, model: str) -> str:
    """Create the provisional mode from a group and hand it the candidates. Returns the new mode id."""
    texts = [m["text"] for m in members]
    mode_id = unique_id(conn, slug(fields["name"]))
    prompt = TEMPLATE.format(about=fields["about"])
    profiles.check_cap("l2", prompt)
    rationale = (f"Spawned mid-call from {len(texts)} utterances in session {session_id} that "
                 f"{model.rsplit('/', 1)[-1]} grouped as one subject. Vanilla template prompt.\n\n"
                 + "\n".join(f"- {t}" for t in texts))
    taken = [store.current(conn, m["id"])["settings"].get("hue", 190) for m in store.list_modes(conn)]
    store.add_mode(conn, mode_id, fields["name"], default_settings(mode_id, fields, taken), prompt,
                   rationale, source="spawner", status="provisional")
    store.claim_candidates(conn, [m["id"] for m in members], mode_id)
    router.fingerprint_one(conn, mode_id, seed_vectors=[m["vector"] for m in members])  # sounds like what was said
    return mode_id


def maybe_spawn(conn, session_id: str, newest_id: int, group_fn=group_and_name) -> str | None:
    """After a candidate is saved: is this call now three-deep on one subject? Then spawn.

    The model is asked only when the cheap gate passes: at least MIN_CLUSTER unclaimed candidates in the
    call, and the newest loosely related to at least MIN_CLUSTER-1 of them. One model call at most per
    saved candidate, and none for the first two.
    """
    rows = store.session_candidates(conn, session_id)
    newest = next((r for r in rows if r["id"] == newest_id), None)
    if newest is None or len(rows) < MIN_CLUSTER or len(related(newest, rows)) < MIN_CLUSTER:
        return None
    close = related(newest, rows)
    dormant, score = closest_archived(conn, [r["vector"] for r in close])
    if dormant and score >= REVIVE_MIN:  # the subject came back: wake the old mode, and spend no model call on a name
        return revive(conn, dormant, close)
    fields, model = group_fn([r["text"] for r in rows])  # newest is last, by id order
    members = [rows[i - 1] for i in fields["same"]]
    if len(members) < MIN_CLUSTER:
        return None
    return spawn(conn, session_id, members, fields, model)


def last_used_ms(conn, mode: dict) -> int:
    """When a call was last in this mode; its creation time when it never was."""
    last = conn.execute("SELECT MAX(ts_ms) FROM switches WHERE mode_id = ?", (mode["id"],)).fetchone()[0]
    if last:
        return int(last)
    return int(datetime.fromisoformat(mode["created_at"].replace("Z", "+00:00")).timestamp() * 1000)


def sweep(conn, now_ms: int | None = None) -> list[str]:
    """Move modes on by use and by days of quiet: provisional -> established after enough calls;
    quiet for long enough -> archived. Archived is dormant, not deleted: revive() wakes it. General never sleeps."""
    now_ms = now_ms or int(time.time() * 1000)
    notes = []
    rows = conn.execute("SELECT id, name, status, created_at FROM modes WHERE status IN ('provisional', 'established')").fetchall()
    for m in rows:
        if m["id"] == router.GENERAL:
            continue
        used_in = conn.execute("SELECT count(DISTINCT session_id) FROM switches WHERE mode_id = ?", (m["id"],)).fetchone()[0]
        quiet_days = (now_ms - last_used_ms(conn, dict(m))) / 86_400_000
        if m["status"] == "provisional" and used_in >= ESTABLISH_AFTER:
            store.set_status(conn, m["id"], "established")
            notes.append(f"{m['id']}: established (used in {used_in} calls)")
        elif m["status"] == "provisional" and used_in <= 1 and quiet_days >= ARCHIVE_PROVISIONAL_DAYS:
            store.set_status(conn, m["id"], "archived")
            notes.append(f"{m['id']}: archived (used once, quiet for {quiet_days:.0f} days)")
        elif quiet_days >= ARCHIVE_ESTABLISHED_DAYS:
            store.set_status(conn, m["id"], "archived")
            notes.append(f"{m['id']}: archived (quiet for {quiet_days:.0f} days)")
    return notes


def closest_archived(conn, vectors: list[list[float]]) -> tuple[str | None, float]:
    """The archived mode whose fingerprint is closest to the centre of these sentences, and how close."""
    centre = [sum(col) / len(vectors) for col in zip(*(router.unit(v) for v in vectors))]
    best, score = None, 0.0
    for r in conn.execute("SELECT f.mode_id, f.vector FROM fingerprints f JOIN modes m ON m.id = f.mode_id "
                          "WHERE m.status = 'archived'"):
        c = router.cosine(centre, json.loads(r["vector"]))
        if c > score:
            best, score = r["mode_id"], c
    return best, score


def revive(conn, mode_id: str, members: list[dict]) -> str:
    """Wake an archived mode: its versions, notes and exchanges were never gone. Established again if it had earned that."""
    used_in = conn.execute("SELECT count(DISTINCT session_id) FROM switches WHERE mode_id = ?", (mode_id,)).fetchone()[0]
    store.set_status(conn, mode_id, "established" if used_in >= ESTABLISH_AFTER else "provisional")
    store.claim_candidates(conn, [m["id"] for m in members], mode_id)
    return mode_id


def was_revived(conn, mode_id: str, session_id: str) -> bool:
    """True when this mode existed before the call that just switched into it."""
    return bool(conn.execute("SELECT 1 FROM switches WHERE mode_id = ? AND session_id != ? LIMIT 1", (mode_id, session_id)).fetchone()
                or conn.execute("SELECT 1 FROM modes WHERE id = ? AND created_at < datetime('now', '-2 minutes')", (mode_id,)).fetchone())


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: app/spawner.py sweep   |   app/spawner.py name "t1" "t2" ...')
    config.load_env()
    if sys.argv[1] == "name":
        fields, model = group_and_name(sys.argv[2:])
        print(json.dumps(fields, indent=2), f"\n(by {model.rsplit('/', 1)[-1]}; id would be '{slug(fields['name'])}')")
        return
    with closing(store.connect()) as conn:
        for note in sweep(conn) or ["no provisional modes"]:
            print(note)


if __name__ == "__main__":
    main()
