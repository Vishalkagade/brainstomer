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
ARCHIVE_AFTER = 5    # calls since creation with no use: the button fades away

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
            "You organise what a person says to a personal voice assistant into subjects. You get numbered sentences "
            "from one conversation. Decide which sentences are about the SAME broad subject as the LAST sentence "
            "(for example: buying clothes, buying shoes and buying jackets are one subject, shopping; asking for "
            "finance advice is not). Return ONLY a JSON object: "
            '{"same": [<1-based numbers of every sentence on that subject, including the last one>], '
            '"name": <2-3 word title for that subject, e.g. "Shopping">, '
            '"about": <one sentence naming the subject matter, concrete and topical, no advice>, '
            '"keyterms": [<up to 10 domain-specific nouns or names a transcriber might mishear; never everyday words>]}'},
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
    fields, model = group_fn([r["text"] for r in rows])  # newest is last, by id order
    members = [rows[i - 1] for i in fields["same"]]
    if len(members) < MIN_CLUSTER:
        return None
    return spawn(conn, session_id, members, fields, model)


def sweep(conn) -> list[str]:
    """Move provisional modes on: used in enough calls -> established; unused for long enough -> archived."""
    notes = []
    for m in conn.execute("SELECT id, name, created_at FROM modes WHERE status = 'provisional'").fetchall():
        used_in = conn.execute("SELECT count(DISTINCT session_id) FROM switches WHERE mode_id = ?", (m["id"],)).fetchone()[0]
        calls_since = conn.execute("SELECT count(DISTINCT session_id) FROM switches WHERE ts_ms > "
                                   "(SELECT COALESCE(MIN(ts_ms), 0) FROM switches WHERE mode_id = ?)", (m["id"],)).fetchone()[0]
        if used_in >= ESTABLISH_AFTER:
            store.set_status(conn, m["id"], "established")
            notes.append(f"{m['id']}: established (used in {used_in} calls)")
        elif used_in <= 1 and calls_since >= ARCHIVE_AFTER:
            store.set_status(conn, m["id"], "archived")
            notes.append(f"{m['id']}: archived (not used again in {calls_since} calls)")
        else:
            notes.append(f"{m['id']}: provisional (used in {used_in} calls, {calls_since} calls since)")
    return notes


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
