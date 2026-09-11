"""Rewrite a mode's L2 from what was actually said in it.

    .venv/bin/python app/evolver.py gym          show the conversations gym has accumulated

Piece one: reading. A mode's material is every turn that was answered while that
mode was live. AssemblyAI's timeline has the turns and their timestamps; our
switches table has when each session moved onto which mode version. Joining
the two is what this file does first. The rewrite comes after.
"""

import sys
from contextlib import closing
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import sessions  # noqa: E402
import store  # noqa: E402

MIN_CONFIDENCE = 0.7  # below this the transcript is a guess; do not learn a preference from it


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


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Which mode? e.g. app/evolver.py gym")
    mode_id = sys.argv[1]
    config.load_env()
    with closing(store.connect()) as conn, httpx.Client(headers=config.headers(), timeout=30) as client:
        live = store.current(conn, mode_id)
        convos = conversations(conn, client, mode_id)

    print(f"{live['name']}  live v{live['n']}  {len(convos)} conversation(s) on record\n")
    for c in convos:
        print(f"-- {c['session_id']}  {c['created_at'][:16].replace('T', ' ')}  "
              f"{len(c['turns'])} turns  answered by version id(s) {c['version_ids']}")
        for t in c["turns"]:
            print(f"   you    {t['user_transcript']}")
            print(f"   agent  {t['agent_text']}")
        print()


if __name__ == "__main__":
    main()
