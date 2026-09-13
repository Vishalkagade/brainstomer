"""Rewrite a mode's L2 from what was actually said in it.

    .venv/bin/python app/evolver.py gym              show the conversations gym has accumulated
    .venv/bin/python app/evolver.py gym propose      ask the model for the next version, store it UNPROMOTED
    .venv/bin/python app/evolver.py gym propose --force    ignore the evidence threshold (testing only)

Reading: a turn belongs to the mode version that was live when its reply started
(switches table joined to AssemblyAI's timeline). Writing: the model gets the live
version, its cap, and the conversations, and returns a full replacement. We
validate it, check the cap, and add it as a new version. Nothing here promotes.
"""

import json
import re
import sys
from contextlib import closing
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import profiles  # noqa: E402
import sessions  # noqa: E402
import store  # noqa: E402
import tools  # noqa: E402

MIN_CONFIDENCE = 0.7  # below this the transcript is a guess; do not learn a preference from it

# Tried in order; the first that answers is used and named in the rationale. Fireworks serverless
# availability changes without notice (deepseek-v4-pro was listed but 404'd), so never depend on one id.
MODELS = [
    "accounts/fireworks/models/deepseek-v4-pro-0813",
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/glm-5p3",
    "accounts/fireworks/models/gpt-oss-120b",
]
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_EVOLVE_AFTER = {"sessions": 3, "turns": 12}  # used when a mode's settings do not say; a guess

# What the evolver may set, and within what bounds. Anything else in its output is an error.
TRANSCRIPTION_MODES = ("balanced", "min_latency", "max_accuracy")
RANGES = {
    "min_silence": (100, 3000),
    "max_silence": (500, 6000),
    "interruption_delay": (0, 1000),
    "history_depth": (1, 50),
    "hue": (0, 359),
    "evolve_sessions": (1, 50),
    "evolve_turns": (1, 500),
}
ALLOWED_KEYS = {"name", "about", "hue", "tools", "history_depth", "model",
                "turn_detection", "transcription_mode", "keyterms", "evolve_after"}
TURN_DETECTION_KEYS = {"min_silence", "max_silence", "interrupt_response", "interruption_delay"}


def attribute(turns: list[dict], switches: list) -> list[tuple[dict, dict | None]]:
    """Pair each turn with the switch that was live when its reply started.

    A turn before the first switch gets None: no profile of ours was loaded yet.
    """
    ordered = sorted(switches, key=lambda s: s["ts_ms"])
    out = []
    for turn in turns:
        at = turn.get("agent_reply_started_at_ms") or turn.get("user_speech_started_at_ms")
        live = None
        for sw in ordered:
            if at is not None and sw["ts_ms"] <= at:
                live = sw  # keep walking; the last one at or before `at` wins
            else:
                break
        out.append((turn, live))
    return out


def usable(turn: dict) -> bool:
    """A turn the evolver may learn from: a real exchange, heard clearly."""
    if turn.get("trigger") == "greeting":
        return False
    if not turn.get("user_transcript") or not turn.get("agent_text"):
        return False
    confidence = turn.get("user_confidence")
    return confidence is None or confidence >= MIN_CONFIDENCE


def conversations(conn, client: httpx.Client, mode_id: str) -> list[dict]:
    """Every session's turns that were answered in `mode_id`, oldest session first."""
    session_ids = [r["session_id"] for r in conn.execute(
        "SELECT DISTINCT session_id FROM switches WHERE mode_id = ?", (mode_id,))]
    out = []
    for session_id in session_ids:
        try:
            session = sessions.get_session(client, session_id)
        except httpx.HTTPStatusError:
            continue  # deleted on AssemblyAI's side; nothing to read
        turns = sessions.get_timeline(client, session)
        switches = [dict(r) for r in store.switches_for(conn, session_id)]
        mine = [(t, sw) for t, sw in attribute(turns, switches)
                if sw and sw["mode_id"] == mode_id and usable(t)]
        if not mine:
            continue
        out.append({
            "session_id": session_id,
            "created_at": session.get("created_at", ""),
            "version_ids": sorted({sw["version_id"] for _, sw in mine}),
            "turns": [t for t, _ in mine],
        })
    out.sort(key=lambda c: c["created_at"])
    return out


def _in_range(errors: list, label: str, value, key: str) -> None:
    low, high = RANGES[key]
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        errors.append(f"{label} must be an integer in {low}..{high}, got {value!r}")


def validate(settings: dict) -> list[str]:
    """Every way a settings block can be wrong, as messages. Empty list means it is acceptable."""
    errors = []
    if not isinstance(settings, dict):
        return ["settings must be an object"]
    for key in set(settings) - ALLOWED_KEYS:
        errors.append(f"unknown key {key!r}")
    if not isinstance(settings.get("name"), str) or not settings.get("name", "").strip():
        errors.append("name must be a non-empty string")
    if "about" in settings and (not isinstance(settings["about"], str) or not settings["about"].strip()):
        errors.append("about must be a non-empty string when present")

    td = settings.get("turn_detection")
    if not isinstance(td, dict):
        errors.append("turn_detection must be an object")
    else:
        for key in set(td) - TURN_DETECTION_KEYS:
            errors.append(f"unknown turn_detection key {key!r}")
        _in_range(errors, "turn_detection.min_silence", td.get("min_silence"), "min_silence")
        _in_range(errors, "turn_detection.max_silence", td.get("max_silence"), "max_silence")
        _in_range(errors, "turn_detection.interruption_delay", td.get("interruption_delay", 0), "interruption_delay")
        if not isinstance(td.get("interrupt_response", True), bool):
            errors.append("turn_detection.interrupt_response must be true or false")
        if isinstance(td.get("min_silence"), int) and isinstance(td.get("max_silence"), int) \
                and td["min_silence"] > td["max_silence"]:
            errors.append("turn_detection.min_silence must not exceed max_silence")

    if settings.get("transcription_mode") not in TRANSCRIPTION_MODES:
        errors.append(f"transcription_mode must be one of {TRANSCRIPTION_MODES}")

    keyterms = settings.get("keyterms", [])
    if not isinstance(keyterms, list) or not all(isinstance(k, str) and k.strip() for k in keyterms):
        errors.append("keyterms must be a list of non-empty strings")
    elif len(keyterms) > 100:
        errors.append(f"keyterms has {len(keyterms)} entries, the API allows 100")

    for name in settings.get("tools", []) if isinstance(settings.get("tools"), list) else ["<not a list>"]:
        if name not in tools.TOOLS:
            errors.append(f"unknown tool {name!r}; have {sorted(tools.TOOLS)}")

    if "history_depth" in settings:
        _in_range(errors, "history_depth", settings["history_depth"], "history_depth")
    if "hue" in settings:
        _in_range(errors, "hue", settings["hue"], "hue")
    if settings.get("model") is not None and not isinstance(settings.get("model"), str):
        errors.append("model must be a string or null")

    ea = settings.get("evolve_after")
    if ea is not None:
        if not isinstance(ea, dict) or set(ea) != {"sessions", "turns"}:
            errors.append("evolve_after must be {sessions, turns}")
        else:
            _in_range(errors, "evolve_after.sessions", ea["sessions"], "evolve_sessions")
            _in_range(errors, "evolve_after.turns", ea["turns"], "evolve_turns")
    return errors


def enough_evidence(settings: dict, convos: list[dict]) -> tuple[bool, str]:
    """Both floors must be met: distinct sessions (independence) and usable turns (information)."""
    need = settings.get("evolve_after") or DEFAULT_EVOLVE_AFTER
    have_sessions = len(convos)
    have_turns = sum(len(c["turns"]) for c in convos)
    ok = have_sessions >= need["sessions"] and have_turns >= need["turns"]
    return ok, (f"have {have_sessions} session(s) / {have_turns} usable turn(s), "
                f"need {need['sessions']} / {need['turns']}")


def transcript(convos: list[dict]) -> str:
    lines = []
    for i, c in enumerate(convos, 1):
        lines.append(f"## Conversation {i}  ({c['created_at'][:10]})")
        for t in c["turns"]:
            marks = []
            if t.get("status") == "interrupted" or t.get("interrupted_at_ms"):
                marks.append("user interrupted this reply")
            if t.get("time_to_first_audio_ms"):
                marks.append(f"{t['time_to_first_audio_ms']} ms to first audio")
            lines.append(f"USER: {t['user_transcript']}")
            lines.append(f"AGENT: {t['agent_text']}" + (f"   [{'; '.join(marks)}]" if marks else ""))
        lines.append("")
    return "\n".join(lines)


def build_messages(live: dict, convos: list[dict]) -> list[dict]:
    """The whole ask, as chat messages. Everything the model needs to respect is in here."""
    cap = profiles.CAPS["l2"]
    used = profiles.estimate_tokens(live["prompt"])
    rules = f"""You maintain the MODE OVERLAY of a voice agent: the part of its system prompt that describes one area of the user's life, plus the runtime settings for how it listens in that area. You will propose the next version from real conversations.

The overlay is read AFTER a bedrock layer (safety, precedence) and a user-core layer (facts about the person, true in every mode). Do not restate anything that belongs there. Style within this domain is yours; facts and constraints about the person are not.

Hard limits:
- The prompt text must stay under {cap} tokens (about {cap * profiles.CHARS_PER_TOKEN} characters). It is currently about {used} tokens. To add, you must remove, and you must list what you removed.
- Settings may only use these keys: {sorted(ALLOWED_KEYS)}. turn_detection keys: {sorted(TURN_DETECTION_KEYS)}.
- Ranges: min_silence {RANGES['min_silence']}, max_silence {RANGES['max_silence']}, interruption_delay {RANGES['interruption_delay']} (all ms), history_depth {RANGES['history_depth']}, hue {RANGES['hue']}. transcription_mode is one of {TRANSCRIPTION_MODES}. tools may only name: {sorted(tools.TOOLS)}. keyterms: at most 100 short strings the transcriber should recognise.
- Turn detection semantics (AssemblyAI): min_silence is the silence after which a turn ends when the words sound finished; max_silence is the ceiling after which the turn is forced to end even mid-sentence. So: cut off on an UNFINISHED sentence -> the pause exceeded max_silence, raise max_silence. Cut off on a sentence that SOUNDED finished but was not -> raise min_silence. Raising either makes every reply slower to start; say why the trade is worth it.
- Change a setting only when the conversations give a reason (e.g. the user interrupted long replies -> ask for shorter answers in the prompt).
- `about` is one or two sentences naming what this mode covers (its subject matter, not the agent's manner). The router matches utterances against it, so keep it concrete and topical.
- keyterms are this domain's recurring vocabulary that a transcriber would otherwise mishear: jargon, product names, technical terms. Add a term only if it appears in at least two different conversations. Never add place names, people, or one-off nouns from a single conversation.
- Do not learn a lasting preference from a single moment. Terseness while out of breath is the situation, not the person.
- Never diagnose, never add medical or dietary claims about the user.
- If the conversations give no reason to change anything, return the current version unchanged and say so in the rationale.

Return ONLY a JSON object with exactly these keys:
{{"settings": <object>, "prompt": <string>, "rationale": <string, 2-6 sentences, what changed and which evidence>, "evicted": [<string>, ...]}}
This is a voice agent: the prompt must not ask for markdown, lists, or anything that cannot be spoken."""

    current = f"""# Current version (v{live['n']}) of mode "{live['name']}"

## settings
{json.dumps(live['settings'], indent=2)}

## prompt
{live['prompt']}

# Conversations answered by this mode

{transcript(convos)}"""
    return [{"role": "system", "content": rules}, {"role": "user", "content": current}]


def call_model(messages: list[dict]) -> tuple[str, str]:
    """One chat completion on Fireworks, first available model in MODELS. Returns (answer text, model id)."""
    key = config.require("FIREWORKS_API_KEY", "https://app.fireworks.ai/settings/users/api-keys")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    failures = []
    for model in MODELS:
        response = httpx.post(
            FIREWORKS_URL, headers=headers, timeout=300,
            json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 6000,
                  "response_format": {"type": "json_object"}},
        )
        if response.status_code in (404, 429, 500, 502, 503):  # not deployed, over quota, or down: try the next
            failures.append(f"{model.rsplit('/', 1)[-1]}: {response.status_code}")
            continue
        response.raise_for_status()  # anything else (401, 400) is our mistake, stop and show it
        if failures:
            print(f"note: fell back to {model.rsplit('/', 1)[-1]} after {', '.join(failures)}")
        return response.json()["choices"][0]["message"]["content"] or "", model
    raise SystemExit("No model in MODELS answered: " + ", ".join(failures))


def parse_proposal(text: str) -> dict:
    """The model's JSON, tolerant of a ```json fence around it. Raises ValueError on anything else."""
    body = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        data = json.loads(body)
    except json.JSONDecodeError as err:
        raise ValueError(f"model did not return JSON: {err}\n---\n{text[:400]}")
    missing = {"settings", "prompt", "rationale", "evicted"} - set(data if isinstance(data, dict) else {})
    if missing:
        raise ValueError(f"model output lacks {sorted(missing)}")
    if not isinstance(data["prompt"], str) or not data["prompt"].strip():
        raise ValueError("model returned an empty prompt")
    if not isinstance(data["evicted"], list):
        raise ValueError("evicted must be a list")
    return data


def propose(conn, live: dict, convos: list[dict], force: bool = False) -> int:
    """Ask the model, check the answer, store it as an unpromoted version. Returns the version id."""
    ok, why = enough_evidence(live["settings"], convos)
    if not ok and not force:
        raise SystemExit(f"Not enough evidence to evolve {live['mode']}: {why}. Pass --force to override.")

    answer, model = call_model(build_messages(live, convos))
    proposal = parse_proposal(answer)
    errors = validate(proposal["settings"])
    if errors:
        raise SystemExit("Model proposed invalid settings, nothing stored:\n  " + "\n  ".join(errors))
    prompt = proposal["prompt"].strip()
    used = profiles.estimate_tokens(prompt)
    if used > profiles.CAPS["l2"]:
        raise SystemExit(f"Model proposed a {used}-token prompt, cap is {profiles.CAPS['l2']}. Nothing stored.")

    rationale = proposal["rationale"].strip()
    if proposal["evicted"]:
        rationale += "\n\nEvicted: " + "; ".join(str(e) for e in proposal["evicted"])
    rationale += f"\n\nEvidence: {why}. Model: {model.rsplit('/', 1)[-1]}."
    return store.add_version(conn, live["mode"], proposal["settings"], prompt, rationale, source="evolver")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Which mode? e.g. app/evolver.py gym [propose] [--force]")
    mode_id = sys.argv[1]
    do_propose = "propose" in sys.argv[2:]
    force = "--force" in sys.argv[2:]
    config.load_env()

    with closing(store.connect()) as conn, httpx.Client(headers=config.headers(), timeout=30) as client:
        live = store.current(conn, mode_id)
        convos = conversations(conn, client, mode_id)
        ok, why = enough_evidence(live["settings"], convos)

        if do_propose:
            version_id = propose(conn, live, convos, force=force)
            row = conn.execute("SELECT n, rationale FROM versions WHERE id = ?", (version_id,)).fetchone()
            print(f"Stored {live['name']} v{row['n']} as version id {version_id}, NOT live.\n")
            print(row["rationale"])
            print(f"\nReview:   app/store.py diff {live['version_id']} {version_id}")
            print(f"Go live:  app/store.py promote {version_id}")
            return

    print(f"{live['name']}  live v{live['n']}  {len(convos)} conversation(s) on record  "
          f"[{'ready to evolve' if ok else 'not yet'}: {why}]\n")
    for c in convos:
        print(f"-- {c['session_id']}  {c['created_at'][:16].replace('T', ' ')}  "
              f"{len(c['turns'])} turns  answered by version id(s) {c['version_ids']}")
        for t in c["turns"]:
            print(f"   you    {t['user_transcript']}")
            print(f"   agent  {t['agent_text']}")
        print()


if __name__ == "__main__":
    main()
