"""The stored agent: our configuration, living on AssemblyAI's side.

    .venv/bin/python app/agent.py

Run it once and it creates the agent, saving the id to .env. Run it again after
editing AGENT and it updates that same agent instead of making a second one.

Why a stored agent at all, when the whole design is about swapping config
mid-call? Because `llm` — pointing the agent at a model we choose — exists ONLY
here. It is not a field you can send inline over the WebSocket. The probes in
probes/ confirmed the two are compatible: bind this agent when the socket opens,
then keep sending `session.update` with inline fields for the rest of the call,
and the stored `llm` survives every one of them.
"""

import httpx

import config

# Every key here is a field of POST /v1/agents. Reference:
# https://www.assemblyai.com/docs/voice-agents/voice-agent-api/create-agent
AGENT = {
    # Shown in the AssemblyAI dashboard. Also how we find this agent again if
    # .env is ever lost, so it has to stay stable.
    "name": "Brainstormer",

    # For step 1 this is a plain hand-written prompt. From step 2 on, this
    # becomes the L0 + L1 + L2 assembly and gets replaced per turn over the
    # socket — what is set here is only the fallback the session starts with.
    "system_prompt": (
        "You are a thinking companion on a voice call. Keep replies to one or "
        "two spoken sentences. Ask a sharpening question when the person is "
        "still working something out, and give a direct answer when they are "
        "not. Say plainly when you are unsure. No filler openers, no lists, no "
        "exclamation marks — this is speech, not a document."
    ),

    # Spoken verbatim on connect, straight to text-to-speech, not written by the
    # model. Note for later: `greeting` is IMMUTABLE once a session starts, so a
    # mode can never change how the call opens. That is fine — the router has
    # not classified anything yet at greeting time anyway.
    "greeting": "I'm here. What's on your mind?",

    # Careful, this shape differs by endpoint. On a STORED agent it is an
    # object: {"voice_id": "alba"}. Sent INLINE over the socket it is a bare
    # string under output: {"output": {"voice": "alba"}}. Same setting, two
    # shapes, and mixing them up is a silent no-op.
    # Catalog: https://www.assemblyai.com/docs/voice-agents/voice-agent-api/voices
    "voice": {"voice_id": "alba"},

    # Deliberately NOT setting input.turn_detection here. Left unset, the agent
    # paces end-of-turn adaptively and waits longer when it hears an unfinished
    # entity. Setting min_silence or max_silence turns that off for the whole
    # session. Step 2 sets them per mode on purpose — for now the default is
    # better than anything we would guess.
    #
    # `llm` is added by resolved_agent() below, and only when MODEL is set.
}

# Which model answers. None means AssemblyAI's managed model: omit `llm`
# entirely and they pick and bill it.
#
# Naming a model here routes through the LLM Gateway instead — one key, one
# bill, and the field that only exists on a stored agent. That is what step 3
# needs for per-mode model selection.
#
# BUT: this account is entitled to exactly one gateway model,
# `qwen3.5-4b-32k-fast`. Every other id in GET /v1/models — all the Claude,
# Gemini and GPT entries — answers 400 "Your account does not have access to
# this LLM Gateway model". The catalogue endpoint lists what exists, not what
# you may call, so the only way to know is to POST a completion and see.
#
# A wrong id here fails in the worst possible way: the greeting still plays
# (that is text-to-speech, no model involved), transcription still works, and
# then every single reply silently never comes. If the agent goes mute after
# the greeting, check this first.
MODEL = None

GATEWAY_BASE_URL = "https://llm-gateway.assemblyai.com/v1"

ENV_KEY = "AGENT_ID"


def resolved_agent() -> dict:
    """The config as it goes on the wire, with credentials filled in.

    `api_key` is write-only on the API — it is encrypted on arrival and never
    read back — so every publish has to send it again.
    """
    agent = dict(AGENT)
    if MODEL:
        agent["llm"] = [{
            "base_url": GATEWAY_BASE_URL,
            "model": MODEL,
            "api_key": config.api_key(),
        }]
    else:
        # An empty list, not an omitted key. On PUT, a key that is absent is
        # left alone rather than cleared — so simply deleting the `llm` block
        # from this file would leave the old model on the agent and the agent
        # would stay mute. Sending [] is what actually detaches it.
        agent["llm"] = []
    return agent


def find_by_name(client: httpx.Client, name: str) -> str:
    """Recover the id from the account when .env has lost it."""
    response = client.get(f"{config.API_BASE}/agents")
    response.raise_for_status()
    for agent in response.json().get("agents", []):
        if agent.get("name") == name:
            return agent["id"]
    return ""


def publish() -> str:
    """Create the agent, or update it if we already have one. Returns its id.

    An id in .env decides create-versus-update. Absent, we look the name up on
    the account before creating, so a lost .env does not leave a trail of
    duplicate agents on every run.
    """
    import os

    config.load_env()
    body = resolved_agent()

    with httpx.Client(headers=config.headers(), timeout=30) as client:
        agent_id = os.environ.get(ENV_KEY) or find_by_name(client, AGENT["name"])

        if agent_id:
            response = client.put(f"{config.API_BASE}/agents/{agent_id}", json=body)
            if response.status_code == 404:
                agent_id = ""  # it was deleted on the dashboard; fall through
            else:
                response.raise_for_status()
                config.save_env(ENV_KEY, agent_id)
                return agent_id

        response = client.post(f"{config.API_BASE}/agents", json=body)
        response.raise_for_status()
        agent_id = response.json()["id"]
        config.save_env(ENV_KEY, agent_id)
        return agent_id


def main() -> None:
    agent_id = publish()
    print(f"agent   {AGENT['name']}")
    print(f"id      {agent_id}   (saved to .env as {ENV_KEY})")
    print(f"model   {MODEL or 'AssemblyAI managed (no llm block sent)'}")
    print(f"voice   {AGENT['voice']['voice_id']}")


if __name__ == "__main__":
    main()
