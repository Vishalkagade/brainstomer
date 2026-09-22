"""Decisions from Jev, TypeSafe AI's System One model.

    .venv/bin/python app/jev.py "I would go with the modern look" watch_shopping "Do you want a modern or classic look?"

Jev does not write text. It takes a block of state and typed questions and returns a
choice, a score or a probability, with calibrated confidence, in about 100 ms. That is
the shape of the router's job: which part of his life is he in right now?

Shadow mode first: the server asks Jev next to the embedding router and logs both
answers. Nothing switches on a Jev verdict until the log says it should.
"""

import json
import sys
import time
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import store  # noqa: E402

MODEL = "jev-latest"
TIMEOUT_S = 1.5   # a route ask that takes longer than this is useless; the reply has started
NONE = "none"     # the choice that means: none of the modes fits

_client = None


def client():
    """One client for the process. The SDK reads the key itself, but we want the missing-key error by name."""
    global _client
    if _client is None:
        import httpx
        from typesafe_sdk import TypeSafeClient
        # keepalive_expiry=None: httpx drops an idle connection after 5 s by default, and turns in a call
        # are further apart than that, so every ask paid ~600 ms to reconnect. Measured 22 Sep: 370 ms after 30 s idle.
        http = httpx.Client(timeout=TIMEOUT_S, limits=httpx.Limits(keepalive_expiry=None))
        _client = TypeSafeClient(api_key=config.require("TYPESAFE_API_KEY", "https://console.typesafe.ai/settings/keys"),
                                 model=MODEL, timeout=TIMEOUT_S, http_client=http)
    return _client


def warm() -> dict:
    """One tiny ask so the TLS connection exists before the first real one. Measured 21 Sep: cold 1.3 s, warm 0.3 s."""
    return route("hello", None, "", {NONE: "nothing"})


def mode_options(conn) -> dict[str, str]:
    """{mode_id: about line} for the choice question, plus a way out. General is not a subject, so it is not offered."""
    options = {}
    for m in store.list_modes(conn):
        if m["id"] == "general":
            continue
        settings = store.current(conn, m["id"])["settings"]
        options[m["id"]] = f"{settings.get('name', m['id'])}: {settings.get('about', '')}".strip(": ")
    options[NONE] = "None of these subjects, or small talk, or a question that fits everywhere"
    return options


def questions(options: dict[str, str]) -> dict:
    from typesafe_sdk import Choice, Noul
    return {
        "mode": Choice(
            instructions="Which area of the user's life is what they are saying now closest to? Pick the closest related "
                         "area even if the detail is new to it; pick none only when no area is related. Use what the agent "
                         "just said as context: a short answer belongs to the area of the question it answers.",
            criteria=options),
        "reply": Noul(instructions="The user is answering or reacting to what the agent just said."),
        # free in the same call: the signal the corrections ledger will want
        "pushback": Noul(instructions="The user is correcting the agent or is frustrated with its last answer."),
    }


def route(text: str, live_mode: str | None, agent_last: str, options: dict[str, str], ask=None) -> dict:  # live_mode kept for the log
    """Jev's view of one moment. Never raises: a failure is a dict with 'error', and the call goes on."""
    # No "current subject" in the state: measured 22 Sep, the same sentence got 0.94 none and then 0.79 travel
    # purely because that field changed. Staying put is the router's rule, not a bias inside the model.
    state = {"agent_just_said": agent_last or "(nothing yet)", "user_is_saying": text}
    t0 = time.perf_counter()
    try:
        response = (ask or client().system_one)(state=state, questions=questions(options))
    except (Exception, SystemExit) as err:  # SDK errors, timeouts, no key (SystemExit from config.require): all the same to the router
        return {"error": f"{type(err).__name__}: {err}", "ms": round((time.perf_counter() - t0) * 1000)}
    mode = response.answers["mode"]
    return {
        "mode": mode.choice,
        "confidence": round(mode.confidence, 3),
        "p": {k: round(v, 3) for k, v in mode.probabilities.items()},
        "reply": round(response.answers["reply"].noul, 3),
        "pushback": round(response.answers["pushback"].noul, 3),
        "tokens": response.usage.input_tokens,
        "ms": round((time.perf_counter() - t0) * 1000),
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: app/jev.py "what he said" [live_mode] ["what the agent said before"]')
    config.load_env()
    with closing(store.connect()) as conn:
        options = mode_options(conn)
    live = sys.argv[2] if len(sys.argv) > 2 else None
    agent_last = sys.argv[3] if len(sys.argv) > 3 else ""
    print(json.dumps(route(sys.argv[1], live, agent_last, options), indent=2))


if __name__ == "__main__":
    main()
