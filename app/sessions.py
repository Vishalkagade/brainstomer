"""Read past calls back off AssemblyAI.

    .venv/bin/python app/sessions.py              list recent calls
    .venv/bin/python app/sessions.py last         print the newest call
    .venv/bin/python app/sessions.py sess_abc123  print one call

Every connection to the agent is stored as a session, with the conversation
already paired turn-by-turn. That stored timeline is what the evolver reads in
step 4 — we never have to log conversations ourselves.

The one rule: artifact URLs are presigned and expire. Keep the session id and
re-fetch; a saved URL is dead within the hour.
"""

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def list_sessions(client: httpx.Client, limit: int = 20) -> list[dict]:
    response = client.get(f"{config.API_BASE}/sessions", params={"limit": limit})
    response.raise_for_status()
    return response.json().get("sessions", [])


def get_session(client: httpx.Client, session_id: str) -> dict:
    response = client.get(f"{config.API_BASE}/sessions/{session_id}")
    response.raise_for_status()
    return response.json()


def get_timeline(client: httpx.Client, session: dict) -> list[dict]:
    """Follow the `timeline` artifact link and return its turns.

    The artifact lives on S3 behind a presigned URL, so this request goes out
    WITHOUT our Authorization header — the signature is the credential, and
    sending a bearer token alongside it is rejected by S3.
    """
    url = next((a["url"] for a in session.get("artifacts", [])
                if a["type"] == "timeline"), None)
    if not url:
        return []
    response = httpx.get(url, timeout=60)
    response.raise_for_status()
    return response.json().get("turns", [])


def show_list(sessions: list[dict]) -> None:
    if not sessions:
        print("No sessions yet. Make a call first.")
        return
    print(f"{'session id':<40} {'when':<21} {'secs':>6}  status")
    for s in sessions:
        when = (s.get("created_at") or "")[:19].replace("T", " ")
        print(f"{s['id']:<40} {when:<21} {s.get('duration_seconds', 0):>6.0f}  "
              f"{s.get('status', '')}")


def show_timeline(session: dict, turns: list[dict]) -> None:
    print(f"session  {session['id']}")
    print(f"agent    {session.get('agent_id')}")
    print(f"ended    {session.get('public_close_reason')} "
          f"after {session.get('duration_seconds', 0):.0f}s")
    print(f"turns    {len(turns)}\n")

    latencies, interruptions, low_confidence = [], 0, 0

    for turn in turns:
        said = turn.get("user_transcript")
        if said:
            confidence = turn.get("user_confidence")
            # Anything the agent only half heard is a bad thing to learn a
            # lasting preference from. Flagged here so it is visible early.
            flag = ""
            if confidence is not None and confidence < 0.7:
                flag = f"   (heard at {confidence:.0%})"
                low_confidence += 1
            print(f"  you    {said}{flag}")

        spoke = turn.get("agent_text")
        if spoke:
            marks = []
            if turn.get("interrupted_at_ms"):
                marks.append("interrupted")
                interruptions += 1
            ttfa = turn.get("time_to_first_audio_ms")
            if ttfa:
                latencies.append(ttfa)
                marks.append(f"{ttfa}ms")
            suffix = f"   [{', '.join(marks)}]" if marks else ""
            print(f"  agent  {spoke}{suffix}")
        print()

    if latencies:
        latencies.sort()
        median = latencies[len(latencies) // 2]
        print(f"time to first audio   median {median}ms   "
              f"range {latencies[0]}-{latencies[-1]}ms")
    if interruptions:
        print(f"interrupted           {interruptions} of {len(turns)} turns")
    if low_confidence:
        print(f"low-confidence input  {low_confidence} turns under 70%")


def main() -> None:
    config.load_env()
    argument = sys.argv[1] if len(sys.argv) > 1 else ""

    with httpx.Client(headers=config.headers(), timeout=30) as client:
        if not argument:
            show_list(list_sessions(client))
            print("\nThen: app/sessions.py last   (or pass a session id)")
            return

        session_id = argument
        if argument == "last":
            sessions = list_sessions(client, limit=1)
            if not sessions:
                raise SystemExit("No sessions yet. Make a call first.")
            session_id = sessions[0]["id"]

        session = get_session(client, session_id)
        show_timeline(session, get_timeline(client, session))


if __name__ == "__main__":
    main()
