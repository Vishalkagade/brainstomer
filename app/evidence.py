"""What went wrong in each turn, judged once by Jev, so the evolver reads counts instead of transcripts.

    .venv/bin/python app/evidence.py gym          print the judged turns of a mode, counts first
    .venv/bin/python app/evidence.py gym judge    judge the turns not judged yet (page load does this too)

Runs off the reply path, inside the memory refresh that already fetches every mode's calls.
A turn is judged once (evidence table, unique per session and time) and never again. A
strong pushback in the next turn also lands in the candidates table as a correction.
"""

import os
import sys
from contextlib import closing
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import evolver  # noqa: E402
import jev  # noqa: E402
import router  # noqa: E402
import store  # noqa: E402

JUDGE_AFTER = {"sessions": 2, "turns": 10}  # an evolver version is judged once it has this much behind it; a guess


def enabled() -> bool:
    return os.environ.get("JEV", "on") != "off"  # the same switch the router uses; off means no TypeSafe calls at all


def facts(turn: dict) -> str:
    """The timing the transcript cannot show, as plain sentences for the judge."""
    out = []
    if turn.get("status") == "interrupted" or turn.get("interrupted_at_ms"):
        out.append("the user interrupted this reply")
    if turn.get("time_to_first_audio_ms"):
        out.append(f"the reply took {turn['time_to_first_audio_ms']} ms to start")
    out.append(f"the reply is {len((turn.get('agent_text') or '').split())} words long")
    return "; ".join(out)


def turn_key(turn: dict) -> int:
    return turn.get("agent_reply_started_at_ms") or 0  # same key the exchanges table uses


def judge_new(conn, mode_id: str, convos: list[dict], ask=None) -> int:
    """Ask Jev about every usable turn not judged before. Returns how many were judged. Stops on the first error."""
    if not enabled():
        return 0
    done = store.judged_turns(conn, mode_id)
    judged = 0
    for c in convos:
        for t in c["turns"]:
            if (c["session_id"], turn_key(t)) in done:
                continue
            verdict = jev.judge(t.get("agent_before", ""), t["user_transcript"], t["agent_text"],
                                t.get("user_next", ""), facts(t), ask=ask)
            if "error" in verdict:
                print(f"evidence for {mode_id} stopped: {verdict['error']}")  # the rest waits for the next refresh
                return judged
            store.add_evidence(conn, mode_id, c["session_id"], turn_key(t), verdict, t["user_transcript"], t["agent_text"])
            if verdict["pushback"] >= jev.PUSHBACK_MIN and t.get("user_next"):
                store.add_candidate(conn, mode_id, c["session_id"], "correction", t["user_next"])
            judged += 1
    return judged


def summary(rows: list[dict]) -> dict:
    """{"turns": n, "problems": {problem: count}, "pushback": n}. Problems only counts what went wrong."""
    problems = {}
    for r in rows:
        if r["problem"] != "fine":
            problems[r["problem"]] = problems.get(r["problem"], 0) + 1
    return {"turns": len(rows), "problems": problems,
            "pushback": sum(1 for r in rows if r["pushback"] >= jev.PUSHBACK_MIN)}


def failing(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["problem"] != "fine" or r["pushback"] >= jev.PUSHBACK_MIN]


def score(rows: list[dict]) -> float:
    """Share of judged turns that went wrong (a problem, or a pushback next). Lower is better."""
    return len(failing(rows)) / len(rows) if rows else 0.0


def review(conn, mode_id: str) -> dict | None:
    """Judge the live version if the evolver made it: keep it, or roll back to what it came from.

    Runs after every evidence pass. Nothing happens until JUDGE_AFTER is met; a verdict is given once.
    """
    live = store.current(conn, mode_id)
    v = conn.execute("SELECT id, n, source, parent_id, verdict FROM versions WHERE id = ?",
                     (live["version_id"],)).fetchone()
    if v["source"] != "evolver" or v["verdict"] or not v["parent_id"]:
        return None
    by = store.evidence_by_version(conn, mode_id)
    mine, theirs = by.get(v["id"], []), by.get(v["parent_id"], [])
    sessions = len({r["session_id"] for r in mine})
    if len(mine) < JUDGE_AFTER["turns"] or sessions < JUDGE_AFTER["sessions"]:
        return None
    parent_n = conn.execute("SELECT n FROM versions WHERE id = ?", (v["parent_id"],)).fetchone()["n"]
    a = score(mine)
    if not theirs:
        verdict, why = "kept", f"kept after {len(mine)} turns in {sessions} calls: {a:.0%} went wrong, nothing judged on v{parent_n} to compare"
    else:
        b = score(theirs)
        numbers = f"{a:.0%} of {len(mine)} turns went wrong, against {b:.0%} of {len(theirs)} on v{parent_n}"
        if a > b:
            router.promote(conn, v["parent_id"])
            verdict, why = "rolled_back", f"rolled back to v{parent_n} after {sessions} calls: {numbers}"
        else:
            verdict, why = "kept", f"kept after {sessions} calls: {numbers}"
    store.set_verdict(conn, v["id"], verdict, why)
    print(f"evidence review {mode_id} v{v['n']}: {why}")
    return {"version_id": v["id"], "verdict": verdict, "why": why}


def headline(rows: list[dict]) -> str:
    """One line for a rationale or the evolution page: '4 of 12 turns cut off, 2 pushed back'."""
    s = summary(rows)
    if not s["turns"]:
        return "no judged turns"
    parts = [f"{n} of {s['turns']} turns {p.replace('_', ' ')}" for p, n in sorted(s["problems"].items(), key=lambda kv: -kv[1])]
    if s["pushback"]:
        parts.append(f"{s['pushback']} pushed back")
    return ", ".join(parts) if parts else f"all {s['turns']} turns fine"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Which mode? e.g. app/evidence.py gym [judge]")
    mode_id = sys.argv[1]
    config.load_env()
    with closing(store.connect()) as conn:
        if "judge" in sys.argv[2:]:
            with httpx.Client(headers=config.headers(), timeout=30) as client:
                n = judge_new(conn, mode_id, evolver.conversations(conn, client, mode_id))
            print(f"judged {n} new turn(s)\n")
        rows = store.evidence_for(conn, mode_id)
        print(f"{mode_id}: {headline(rows)}\n")
        for vid, mine in sorted(store.evidence_by_version(conn, mode_id).items()):
            n = conn.execute("SELECT n, verdict FROM versions WHERE id = ?", (vid,)).fetchone()
            tag = f"  [{n['verdict']}]" if n["verdict"] else ""
            print(f"   v{n['n']}: {len(mine)} turns, {score(mine):.0%} went wrong{tag}")
        print()
        for r in rows:
            mark = "" if r["problem"] == "fine" else f"  <- {r['problem']} ({r['confidence']:.0%})"
            if r["pushback"] >= jev.PUSHBACK_MIN:
                mark += f"  pushback {r['pushback']:.2f}"
            print(f"   you    {r['user_text']}")
            print(f"   agent  {r['agent_text']}{mark}\n")


if __name__ == "__main__":
    main()
