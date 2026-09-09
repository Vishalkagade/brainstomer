"""The tools an agent may call, and the code that runs them.

    .venv/bin/python app/tools.py                     list the registry
    .venv/bin/python app/tools.py "who won the F1 race last weekend"

One registry, keyed by tool name. Each entry holds the `definition` the model
sees — the name, the description and the argument schema that decide WHEN it is
called — next to the `handler` that runs when it is. Keeping them in one entry
is what makes it impossible to offer the model a tool with nothing behind it.

Why a registry instead of putting each tool in the mode file that wants it: a
mode file holds settings that are genuinely per-mode, and the rule there is that
a mode IS its settings-plus-prompt. A tool schema is not per-mode — it is the
same fifteen lines every time. Copying it into each mode would be the drift risk,
not the cure. Modes name the tools they want (`"tools": ["web_search"]`) and the
definition is written once, here.

These are CLIENT-SIDE function tools, not server-side HTTP tools. AssemblyAI
supports both: an HTTP tool means they call Exa directly and feed us the reply,
which is less code and no round trip. We do not use it, for one reason —
payload size. Measured on 9 Sep 2026 against a live current-affairs query:

    Exa /search + contents   1457 ms    95,689 chars   ~23,900 tokens
    Exa /answer              1877 ms     4,152 chars    ~1,038 tokens
    the answer text alone                  464 chars      ~116 tokens

L0 + L1 + L2 together are capped at 1,700 tokens. A raw /search result is
fourteen times the entire profile system, and even /answer is nine times bigger
than the part worth speaking — the rest is citation ids, images and authors. An
HTTP tool would put all of that into the model's context with no say from us.
Handling the call ourselves is what lets shape_answer() below cut it to ~150.

The second reason to run tools here rather than in the page: the Exa key lives
in this process and never reaches the browser. The page relays a tool.call to us
and gets back only the shaped result — the same boundary server.py already
enforces for the AssemblyAI key.
"""

import json
import re
import sys

import httpx

import config

EXA_ANSWER_URL = "https://api.exa.ai/answer"

# Exa took 1.9 s on the measurement above. 15 s leaves room for a slow day
# without leaving the caller hanging: the agent is talking over this wait
# (see execution_mode below), and it has to have something to say when we
# come back.
SEARCH_TIMEOUT_SECONDS = 15

# How many sources to name. The model can say "according to Reuters"; it cannot
# usefully speak a URL, so only titles travel and three is as many as anyone
# tracks by ear.
MAX_SOURCES = 3

# A ceiling on the ANSWER TEXT, in characters — roughly 200 tokens at the
# chars/4 estimate profiles.py uses. The source titles and the JSON envelope
# ride on top of this, so it caps the part that grows without bound rather than
# the whole result.
#
# Note this truncates where a prompt layer over its cap raises SystemExit. That
# difference is deliberate: a prompt layer is authored text, and being forced to
# evict something is the point. A tool result is runtime data arriving mid-call,
# and crashing the session because a news answer ran long would be absurd.
MAX_ANSWER_CHARS = 800

# Exa writes inline markers — "...rates unchanged on 23 July 2026 [1]. The
# Council is expected to raise [2][3][4]..." Text-to-speech reads those out
# loud, so the agent says "bracket one" mid-sentence. The leading \s* takes the
# space in front of the marker with it, so no double space is left behind.
# Strips "[1]", "[2][3]" and "[1, 2]" alike.
#
# Not really an Exa quirk: bracketed citations are the convention for cited LLM
# answers generally, and the reason to strip them — they are unspeakable — holds
# whoever produced the text. This survives a change of provider.
CITATION_MARKER = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")

# One client for the life of the process, built on first use.
#
# httpx.post() builds a fresh client per call, which means a new SSLContext and
# a new TLS handshake every search: measured at 32 ms of CPU plus 64 ms of
# network on this machine, ~96 ms added to every lookup while someone waits mid
# call. A reused client pays that once. It is built lazily rather than at import
# so that importing this module does not demand an Exa key from anyone who is
# not going to search — profiles.py imports it to read tool definitions.
_client: httpx.Client | None = None


def exa_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=SEARCH_TIMEOUT_SECONDS,
            headers={
                "x-api-key": config.require("EXA_API_KEY",
                                            "https://dashboard.exa.ai/api-keys"),
                "Content-Type": "application/json",
            },
        )
    return _client


def strip_citation_markers(text: str) -> str:
    """Remove the inline [n] markers, which TTS would otherwise read aloud."""
    return CITATION_MARKER.sub("", text).strip()


def shape_answer(payload: dict) -> dict:
    """Reduce Exa's reply to the part the agent can actually say.

    Everything dropped here — citation ids, image urls, authors, publish dates,
    the full text of five web pages — is context the model would have to read
    past to reach the one paragraph that answers the question.
    """
    answer = strip_citation_markers(payload.get("answer") or "")
    if len(answer) > MAX_ANSWER_CHARS:
        # Cut at a sentence end if there is one in the back half, so the agent
        # is not reading a half sentence aloud. If the only sentence break is
        # near the start, keeping it would throw away most of the answer, so
        # take the hard cut instead.
        clipped = answer[:MAX_ANSWER_CHARS]
        stop = clipped.rfind(". ")
        answer = clipped[: stop + 1] if stop > MAX_ANSWER_CHARS // 2 else clipped

    sources = []
    for citation in (payload.get("citations") or [])[:MAX_SOURCES]:
        title = (citation.get("title") or "").strip()
        if title:
            sources.append(title)

    return {"answer": answer, "sources": sources}


def web_search(query: str) -> dict:
    """Ask Exa a question and return an answer short enough to speak.

    `text: False` asks for the synthesised answer without the scraped page
    bodies. That single flag is the difference between ~1,000 tokens and
    ~24,000 — see the measurements at the top of this file.
    """
    response = exa_client().post(EXA_ANSWER_URL, json={"query": query, "text": False})
    response.raise_for_status()
    return shape_answer(response.json())


# name -> {definition, handler}.
#
# Every key inside `definition` is a field of a client-side function tool in a
# session.update:
#
#   type             always "function" — this is the shape that comes back to us
#                    as a tool.call over the socket, rather than one AssemblyAI
#                    resolves on its own.
#   name             what arrives in tool.call.name, and our key into TOOLS.
#   description      the model's ONLY signal for when to call. Written as a
#                    trigger, not a summary: lead with the verb, then the exact
#                    condition. This is the field to edit if the agent searches
#                    too eagerly or not enough.
#   parameters       JSON Schema for the arguments. `required` matters — without
#                    it the model may call with no query at all.
#   execution_mode   "interactive" lets the agent keep talking while we work,
#                    so a two-second lookup sounds like "let me check" rather
#                    than a dropped call. "hold" would go silent. AssemblyAI's
#                    own guidance names wrapping a slow lookup in "hold" as the
#                    common mistake.
#   timeout_seconds  the agent apologises and carries on past this; the session
#                    survives. Kept just above our own httpx timeout so ours
#                    fires first and we control the message.
#
# The handler takes the parsed `arguments` object and returns anything
# JSON-serialisable. It reads its argument out of the dict rather than being
# called with **arguments, because the model sometimes invents an extra key and
# an unexpected one should be ignored, not raise.
TOOLS = {
    "web_search": {
        "definition": {
            "type": "function",
            "name": "web_search",
            "description": (
                "Look up current information on the web. Use this whenever the "
                "answer depends on something recent, changing, or specific that "
                "you are not certain of — news, prices, results, releases, who "
                "currently holds a position. Prefer calling it over guessing or "
                "saying your knowledge may be out of date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The question to look up, as a full question in "
                            "plain English, e.g. 'what did the ECB decide about "
                            "interest rates in September 2026'. Include the "
                            "specifics the user gave; do not shorten to "
                            "keywords."
                        ),
                    }
                },
                "required": ["query"],
            },
            "execution_mode": "interactive",
            "timeout_seconds": 20,
        },
        "handler": lambda args: web_search(args["query"]),
    }
}


def definitions_for(names: list[str]) -> list[dict]:
    """The `tools` array for a session.update, for the tools a mode asks for.

    An unknown name is a typo in a mode file and stops us here, rather than
    silently handing the model a shorter toolset than the mode intended.
    server.py already turns a SystemExit out of profile assembly into a 500 for
    the page, so this surfaces as a mode that will not load.
    """
    unknown = [n for n in names if n not in TOOLS]
    if unknown:
        raise SystemExit(
            f"Unknown tool(s): {', '.join(unknown)}. "
            f"Have: {', '.join(sorted(TOOLS))}"
        )
    return [TOOLS[n]["definition"] for n in names]


def run(name: str, arguments: dict) -> str:
    """Execute one tool call and return the string that goes back as tool.result.

    Always returns a string, never raises. A tool that throws mid-call would
    leave the agent waiting out its whole timeout in silence; an error the model
    can read lets it say something honest instead. SystemExit is caught for the
    same reason — a missing EXA_API_KEY should not take a live call down.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"no such tool: {name}"})
    try:
        return json.dumps(tool["handler"](arguments))
    except KeyError as err:
        return json.dumps({"error": f"missing argument {err}"})
    except (httpx.HTTPError, SystemExit) as err:
        print(f"tool {name} failed: {err}")
        return json.dumps({"error": "the search failed; say so and carry on"})


def main() -> None:
    if len(sys.argv) > 1:
        import time

        import profiles  # imported here, not at module scope: profiles imports us

        started = time.time()
        result = run("web_search", {"query": " ".join(sys.argv[1:])})
        elapsed = (time.time() - started) * 1000
        payload = json.loads(result)
        print(f"{elapsed:.0f} ms   {len(result)} chars   "
              f"~{profiles.estimate_tokens(result)} tokens\n")
        print(payload.get("answer") or payload)
        if payload.get("sources"):
            print("\nsources: " + " | ".join(payload["sources"]))
        return

    for name, tool in TOOLS.items():
        definition = tool["definition"]
        print(f"{name:<14} {definition['execution_mode']:<12} "
              f"timeout {definition['timeout_seconds']}s")
        print(f"{'':<14} {definition['description'][:60]}...")


if __name__ == "__main__":
    main()
