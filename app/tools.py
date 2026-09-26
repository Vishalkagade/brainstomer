"""The tools an agent may call
"""

import json
import re
import sys

import httpx

import config

EXA_ANSWER_URL = "https://api.exa.ai/answer"

SEARCH_TIMEOUT_SECONDS = 15 # time to wait
MAX_SOURCES = 3 # no of sources
MAX_ANSWER_CHARS = 800
CITATION_MARKER = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]") # so it removes the inline [n] markers, which TTS would otherwise read aloud
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
        clipped = answer[:MAX_ANSWER_CHARS] # but we might loose the imp info?
        stop = clipped.rfind(". ")
        answer = clipped[: stop + 1] if stop > MAX_ANSWER_CHARS // 2 else clipped

    sources, dates = [], []
    for citation in (payload.get("citations") or []):
        title = (citation.get("title") or "").strip()
        if title and len(sources) < MAX_SOURCES:
            sources.append(title)
        if citation.get("publishedDate"):
            dates.append(citation["publishedDate"][:10])

    out = {"answer": answer, "sources": sources}
    if dates:
        out["newest_source"] = max(dates)  # so the agent can say how fresh the news is, and notice when it is old
    return out


def today() -> str:
    from datetime import datetime
    return datetime.now().strftime("%-d %B %Y")


def search_instructions() -> str:
    """Exa's own model writes the answer; without the date it also guesses the year."""
    return (f"Today is {today()}. Prefer the most recent sources and say when the events happened. "
            "If nothing recent matches, say so instead of describing older events as current.")


def web_search(query: str) -> dict:
    """Ask Exa a question and return an answer short enough to speak.

    `text: False` asks for the synthesised answer without the scraped page
    bodies. That single flag is the difference between ~1,000 tokens and
    ~24,000 — see the measurements at the top of this file.
    """
    response = exa_client().post(EXA_ANSWER_URL, json={"query": query, "text": False,
                                                       "systemPrompt": search_instructions()})
    response.raise_for_status()
    return shape_answer(response.json())

TOOLS = {
    "web_search": {
        "definition": {
            "type": "function",
            "name": "web_search",
            "description": (
                "Look up current information on the web. Use this whenever the "
                "answer depends on something recent, changing, or specific that "
                "you are not certain of such as  news, prices, results, releases, who "
                "currently holds a position. Prefer calling it over guessing or "
                "saying your knowledge may be out of date. Today's date is in your "
                "instructions: put the current month and year in the query when "
                "recency matters. The result's newest_source says how fresh the "
                "news is; mention it when it is not from the last few days."
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
