"""Assemble a runtime profile from the prompt layers on disk.

    .venv/bin/python app/profiles.py          every mode, with its token budget
    .venv/bin/python app/profiles.py gym      the assembled prompt for one mode

A profile is not a prompt. It is the prompt PLUS the settings that decide how
the agent listens, and both come out of one file per mode: a JSON block for the
settings, then the prompt text under it.

Assembly order is fixed — bedrock, user core, mode overlay — because the
precedence rule in L0 only means anything if L0 is read first.
"""

import json
import sys
from contextlib import closing
from pathlib import Path

import store
import tools

PROMPTS = Path(__file__).resolve().parent / "prompts"

# Hard caps, in tokens. These are the design, not a limitation: to add something
# the evolver has to evict something and say which. A layer over its cap is an
# error, not a warning — otherwise the caps mean nothing by step 4.
CAPS = {"l0": 300, "l1": 800, "l2": 600}

# Rough English estimate: ~4 characters per token. Good enough to enforce a
# budget, and it does not pin us to one model's tokenizer. Anything close to a
# cap should be checked properly before it matters.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return round(len(text) / CHARS_PER_TOKEN)


def check_cap(layer: str, text: str) -> int:
    used = estimate_tokens(text)
    cap = CAPS[layer]
    if used > cap:
        raise SystemExit(
            f"{layer} is {used} tokens, over its {cap} cap by {used - cap}.\n"
            f"Evict something before adding more — that is the point of the cap."
        )
    return used


def read_layer(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Missing prompt layer: {path}")
    return path.read_text().strip()


def parse_mode(path: Path) -> tuple[dict, str]:
    """Split a mode file into its settings block and its prompt text.

    The file opens with a JSON object fenced by `---` lines. Keeping both halves
    in one file is deliberate: a mode IS the pair, and splitting them across two
    files invites them to drift apart.
    """
    text = path.read_text().strip()
    if not text.startswith("---"):
        raise SystemExit(f"{path.name} must start with a --- settings block")
    _, raw_settings, prompt = text.split("---", 2)
    try:
        settings = json.loads(raw_settings)
    except json.JSONDecodeError as err:
        raise SystemExit(f"{path.name}: settings block is not valid JSON — {err}")
    return settings, prompt.strip()


def list_modes() -> list[str]:
    with closing(store.connect()) as conn:  # closing() actually closes; bare `with conn:` only ends a transaction
        return [row["id"] for row in store.list_modes(conn)]


def today_line() -> str:
    """The one fact no layer can hold: the model guesses the year otherwise (it searched January 2025 on 25 Sep 2026)."""
    from datetime import datetime
    return f"Today is {datetime.now().strftime('%A, %-d %B %Y')}."


def assemble(mode: str) -> dict:
    """Build everything needed to swap into `mode` on an open socket."""
    bedrock = read_layer(PROMPTS / "l0_bedrock.md")

    # L1 and L2 come from the store's live versions, not the .md files (those were only the v1 seeds)
    with closing(store.connect()) as conn:  # closing() actually closes; bare `with conn:` only ends a transaction
        live = store.current(conn, mode)
        user_core = (store.current(conn, store.CORE)["prompt"] if store.is_layer(conn, store.CORE)
                     else read_layer(PROMPTS / "l1_user_core.md"))  # a store from before 26 Sep: the file, until seeded
    settings, overlay = live["settings"], live["prompt"]

    budget = {
        "l0": check_cap("l0", bedrock),
        "l1": check_cap("l1", user_core),
        "l2": check_cap("l2", overlay),
    }
    budget["total"] = sum(budget.values())

    # Bedrock first, mode last. The headers are not decoration: they are what
    # makes the precedence rule in L0 refer to something the model can see.
    system_prompt = (
        f"{bedrock}\n\n"
        f"=== USER CORE — facts and constraints, true in every mode ===\n\n"
        f"{today_line()}\n\n{user_core}\n\n"
        f"=== MODE: {settings.get('name', mode).upper()} — style within this domain ===\n\n"
        f"{overlay}"
    )

    # Exactly what goes inside a session.update. Only fields that are mutable
    # mid-session appear here; anything else would be rejected or silently
    # ignored on an open socket.
    session = {
        "system_prompt": system_prompt,
        "input": {
            "turn_detection": settings["turn_detection"],
            "transcription_mode": settings["transcription_mode"],
            "keyterms": settings["keyterms"],
        },
        # Always sent, even when empty. A session.update REPLACES the tools
        # array rather than merging into it, so a mode with no tools must say
        # so explicitly — omit the key and the previous mode's tools would stay
        # live after the swap. `[]` is what actually takes web_search away from
        # the agent when you switch from deep work to gym.
        "tools": tools.definitions_for(settings.get("tools", [])),
    }

    # The two dimensions the API has no knob for. They are declared in the mode
    # file so the profile is complete and the gap is visible, but nothing can
    # apply them until we run our own OpenAI-compatible endpoint behind the
    # stored agent's llm. Reporting them as unapplied is more honest than
    # quietly dropping them.
    not_applied = {
        "history_depth": settings.get("history_depth"),
        "model": settings.get("model"),
    }

    return {
        "mode": mode,
        "name": settings.get("name", mode),
        "version_id": live["version_id"],  # store row id, goes into the switch log
        "version": live["n"],  # per-mode number, for display: "Gym v1"
        # Base hue for the page's instrument, in degrees. It sits OUTSIDE
        # `session` on purpose: everything in there goes on the wire verbatim,
        # and an unknown field would be rejected. This is ours, for the browser.
        "hue": settings.get("hue", 190),
        "session": session,
        "budget": budget,
        "not_applied": not_applied,
    }


def main() -> None:
    if len(sys.argv) > 1:
        profile = assemble(sys.argv[1])
        print(profile["session"]["system_prompt"])
        print("\n" + "-" * 70)
        print(f"budget  {profile['budget']}")
        print(f"listen  {json.dumps(profile['session']['input'])}")
        return

    for mode in list_modes():
        profile = assemble(mode)
        used = profile["budget"]
        td = profile["session"]["input"]["turn_detection"]
        print(f"{profile['name']:<12} l0={used['l0']:>3}/{CAPS['l0']}  "
              f"l1={used['l1']:>3}/{CAPS['l1']}  l2={used['l2']:>3}/{CAPS['l2']}  "
              f"total={used['total']:>4}   "
              f"silence {td['min_silence']}-{td['max_silence']}ms  "
              f"{profile['session']['input']['transcription_mode']}")


if __name__ == "__main__":
    main()
