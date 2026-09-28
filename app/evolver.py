"""Rewrite a mode's L2 from what was actually said in it.

    .venv/bin/python app/evolver.py gym              show the conversations gym has accumulated
    .venv/bin/python app/evolver.py gym propose      ask the model for the next version and make it LIVE
    .venv/bin/python app/evolver.py gym propose --hold     store it without promoting (the old behaviour)
    .venv/bin/python app/evolver.py gym propose --force    ignore the evidence threshold (testing only)

Reading: a turn belongs to the mode version that was live when its reply started
(switches table joined to AssemblyAI's timeline). Writing: the model gets the live
version, its cap, and the conversations, and returns a full replacement. We
validate it, check the cap, and add it as a new version. Since 25 Sep the new version goes
live at once; the evidence pass judges it after a floor of turns and rolls it back if it did
worse than the version it came from (evidence.review). The buttons on the evolution page stay.
"""

import json
import os
import re
import sys
import threading
from contextlib import closing
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import profiles  # noqa: E402
import router  # noqa: E402
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
_proposing: set[str] = set()   # modes with a proposal in flight in this process; two page loads must not pay twice
_proposing_lock = threading.Lock()
_failed_at: dict[str, float] = {}  # mode -> when its last unattended proposal failed; no retry for RETRY_AFTER_S
RETRY_AFTER_S = 3600

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


def is_echo(user_text: str, agent_before: str) -> bool:
    """The mic heard the agent's own voice: the 'user' line is the start of what the agent had just said.
    Seen 22 Sep: 'You would start by selecting' twice, then him asking whether the speakers were on."""
    words = (user_text or "").lower().split()
    return len(words) >= 3 and " ".join(words) in " ".join((agent_before or "").lower().split())


def usable(turn: dict) -> bool:
    """A turn the evolver may learn from: a real exchange, heard clearly."""
    if turn.get("trigger") == "greeting":
        return False
    if not turn.get("user_transcript") or not turn.get("agent_text"):
        return False
    if is_echo(turn.get("user_transcript"), turn.get("agent_before", "")):
        return False
    confidence = turn.get("user_confidence")
    return confidence is None or confidence >= MIN_CONFIDENCE


def neighbours(turns: list[dict]) -> list[dict]:
    """Give every turn the agent line before it and the user line after it, from the whole call.

    The evidence judge needs both, and the mode's own turn list has gaps where other modes answered.
    """
    for i, t in enumerate(turns):
        t["agent_before"] = turns[i - 1].get("agent_text") or "" if i else ""
        t["user_next"] = turns[i + 1].get("user_transcript") or "" if i + 1 < len(turns) else ""
    return turns


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
        turns = neighbours(sessions.get_timeline(client, session))
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


def tried(conn, live: dict) -> list[str]:
    """Versions proposed from the live one and rolled back: the model must not propose them again."""
    rows = conn.execute("SELECT n, verdict_why FROM versions WHERE parent_id = ? AND verdict = 'rolled_back' ORDER BY n",
                        (live["version_id"],)).fetchall()
    return [f"v{r['n']} was tried and {r['verdict_why']}" for r in rows]


def judged(rows: list[dict]) -> str:
    """Counts first, then only the turns Jev marked. Replaces the full transcript once the evidence pass has run."""
    import evidence
    lines = [f"Judged turns: {evidence.headline(rows)}."]
    bad = evidence.failing(rows)
    if not bad:
        lines.append("Nothing went wrong in any judged turn. Return the current version unchanged.")
        return "\n".join(lines)
    lines.append(f"Only the {len(bad)} turn(s) with a problem are shown; the rest went well.\n")
    for r in bad:
        marks = []
        if r["problem"] != "fine":
            marks.append(f"{r['problem'].replace('_', ' ')}, {r['confidence']:.0%} sure")
        if r["pushback"] >= evidence.jev.PUSHBACK_MIN:
            marks.append(f"user pushed back next ({r['pushback']:.2f})")
        lines.append(f"USER: {r['user_text']}")
        lines.append(f"AGENT: {r['agent_text']}   [{'; '.join(marks)}]")
    return "\n".join(lines)


def build_messages(live: dict, convos: list[dict], rows: list[dict] | None = None,
                   tried_lines: list[str] | None = None) -> list[dict]:
    """The whole ask, as chat messages. Everything the model needs to respect is in here.

    With judged rows the model sees counts and failing turns; without them (Jev off) the whole transcript.
    tried_lines: earlier proposals from this version that the evidence rolled back, so they are not repeated.
    """
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

# {"What went wrong, judged turn by turn" if rows else "Conversations answered by this mode"}

{judged(rows) if rows else transcript(convos)}"""
    if tried_lines:
        current += ("\n\n# Already tried from this version and undone by the evidence; do not propose the same again\n\n"
                    + "\n".join(tried_lines))
    return [{"role": "system", "content": rules}, {"role": "user", "content": current}]


def call_model(messages: list[dict], models: list[str] | None = None, max_tokens: int = 6000,
               params: dict | None = None) -> tuple[str, str]:
    """One chat completion on Fireworks, first available model in `models` (default MODELS). Returns (answer text, model id).
    `params` are extra request fields, e.g. reasoning_effort for a small job that must not think for a minute."""
    key = config.require("FIREWORKS_API_KEY", "https://app.fireworks.ai/settings/users/api-keys")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    failures = []
    for model in models or MODELS:
        response = httpx.post(
            FIREWORKS_URL, headers=headers, timeout=300,
            json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens,
                  "response_format": {"type": "json_object"}, **(params or {})},
        )
        if response.status_code in (404, 429, 500, 502, 503):  # not deployed, over quota, or down: try the next
            failures.append(f"{model.rsplit('/', 1)[-1]}: {response.status_code}")
            continue
        response.raise_for_status()  # anything else (401, 400) is our mistake, stop and show it
        choice = response.json()["choices"][0]
        text = choice["message"].get("content") or ""
        if not text.strip():  # a reasoning model that spent the whole budget thinking: content empty, finish_reason length
            failures.append(f"{model.rsplit('/', 1)[-1]}: empty content, finish {choice.get('finish_reason')}")
            continue
        if failures:
            print(f"note: fell back to {model.rsplit('/', 1)[-1]} after {', '.join(failures)}")
        return text, model
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


def propose(conn, live: dict, convos: list[dict], force: bool = False, hold: bool = False) -> int:
    """Ask the model, check the answer, store the version and make it live (hold=True: store only). Returns its id."""
    ok, why = enough_evidence(live["settings"], convos)
    if not ok and not force:
        raise SystemExit(f"Not enough evidence to evolve {live['mode']}: {why}. Pass --force to override.")

    # what THIS version did wrong; a version with nothing judged yet falls back to the mode's whole evidence
    rows = store.evidence_by_version(conn, live["mode"]).get(live["version_id"]) or store.evidence_for(conn, live["mode"])
    answer, model = call_model(build_messages(live, convos, rows, tried(conn, live)))
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
    import evidence
    rationale += f"\n\nEvidence: {why}; {evidence.headline(rows)}. Model: {model.rsplit('/', 1)[-1]}."
    version_id = store.add_version(conn, live["mode"], proposal["settings"], prompt, rationale, source="evolver",
                                   parent_id=live["version_id"])
    if not hold:
        router.promote(conn, version_id)  # judged by evidence.review once JUDGE_AFTER turns have run on it
    return version_id


def due(conn, live: dict) -> tuple[bool, str]:
    """Should the evolver propose from the live version by itself? Counts only judged turns on that version,
    and only those judged since its last proposal, so the same evidence never pays for two proposals."""
    row = conn.execute("SELECT source, parent_id, verdict FROM versions WHERE id = ?", (live["version_id"],)).fetchone()
    if row["source"] == "evolver" and row["parent_id"] and not row["verdict"]:
        return False, f"v{live['n']} is still waiting for its verdict"
    rows = store.evidence_by_version(conn, live["mode"]).get(live["version_id"], [])
    last = conn.execute("SELECT MAX(created_at) AS at FROM versions WHERE parent_id = ?", (live["version_id"],)).fetchone()["at"]
    if last:
        rows = [r for r in rows if r["judged_at"] > last]
    need = live["settings"].get("evolve_after") or DEFAULT_EVOLVE_AFTER
    have_sessions, have_turns = len({r["session_id"] for r in rows}), len(rows)
    ok = have_sessions >= need["sessions"] and have_turns >= need["turns"]
    since = " since the last proposal" if last else ""
    return ok, f"{have_turns} judged turn(s) in {have_sessions} call(s) on v{live['n']}{since}, need {need['turns']} in {need['sessions']}"


def auto_propose(conn, mode_id: str, convos: list[dict]) -> int | None:
    """The unattended path: propose and go live when due() says so. Returns the version id or None."""
    if os.environ.get("AUTO_EVOLVE", "on") == "off" or os.environ.get("BRAINSTORMER_READ_ONLY") == "1":
        return None
    live = store.current(conn, mode_id)
    ok, why = due(conn, live)
    if not ok:
        return None
    import time
    if time.time() - _failed_at.get(mode_id, 0) < RETRY_AFTER_S:
        return None  # a failed proposal is not retried on every page load; an hour, then again
    with _proposing_lock:
        if mode_id in _proposing:
            return None
        _proposing.add(mode_id)
    try:
        version_id = propose(conn, live, convos)
        conn.execute("UPDATE versions SET rationale = rationale || ' Proposed by itself: ' || ? || '.' WHERE id = ?",
                     (why, version_id))
        conn.commit()
        print(f"evolver: {mode_id} v{live['n']} -> new version {version_id} live ({why})")
        return version_id
    except (SystemExit, ValueError) as err:  # not enough conversations, no model, or an answer that is not a proposal
        _failed_at[mode_id] = time.time()
        print(f"evolver: {mode_id} not evolved: {err}")
        return None
    finally:
        with _proposing_lock:
            _proposing.discard(mode_id)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Which mode? e.g. app/evolver.py gym [propose] [--force]")
    mode_id = sys.argv[1]
    do_propose = "propose" in sys.argv[2:]
    force = "--force" in sys.argv[2:]
    hold = "--hold" in sys.argv[2:]
    config.load_env()

    with closing(store.connect()) as conn, httpx.Client(headers=config.headers(), timeout=30) as client:
        live = store.current(conn, mode_id)
        convos = conversations(conn, client, mode_id)
        ok, why = enough_evidence(live["settings"], convos)
        auto_ok, auto_why = due(conn, live)

        if do_propose:
            version_id = propose(conn, live, convos, force=force, hold=hold)
            row = conn.execute("SELECT n, rationale FROM versions WHERE id = ?", (version_id,)).fetchone()
            import evidence
            state = ("NOT live (held)" if hold else
                     f"LIVE now; judged against v{live['n']} after {evidence.JUDGE_AFTER['turns']} turns "
                     f"in {evidence.JUDGE_AFTER['sessions']} calls")
            print(f"Stored {live['name']} v{row['n']} as version id {version_id}, {state}.\n")
            print(row["rationale"])
            print(f"\nReview:   app/store.py diff {live['version_id']} {version_id}")
            print(f"{'Go live' if hold else 'Undo'}:  app/store.py promote {version_id if hold else live['version_id']}")
            return

    print(f"{live['name']}  live v{live['n']}  {len(convos)} conversation(s) on record  "
          f"[{'ready to evolve' if ok else 'not yet'}: {why}]")
    print(f"by itself: {'due' if auto_ok else 'not yet'}: {auto_why}\n")
    for c in convos:
        print(f"-- {c['session_id']}  {c['created_at'][:16].replace('T', ' ')}  "
              f"{len(c['turns'])} turns  answered by version id(s) {c['version_ids']}")
        for t in c["turns"]:
            print(f"   you    {t['user_transcript']}")
            print(f"   agent  {t['agent_text']}")
        print()


if __name__ == "__main__":
    main()
