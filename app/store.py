"""The mode store: every version of every mode, and what happened to each.

    .venv/bin/python app/store.py             every mode with its live version
    .venv/bin/python app/store.py seed        load app/prompts/modes/*.md as v1
    .venv/bin/python app/store.py show gym    print the live version of one mode
    .venv/bin/python app/store.py switches    every logged mode swap, per session
    .venv/bin/python app/store.py history gym every version of one mode, live one marked
    .venv/bin/python app/store.py diff 1 3    what changed between two version ids
    .venv/bin/python app/store.py promote 3   make a version live (rollback = promote an older id)

One SQLite file. Each mode has saved versions of its text and settings.
A mode also keeps a pointer to the version that is currently active.
Old versions are never changed. A new version is added, then the pointer is moved.
That makes rollback easy: only one value needs to be updated.
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent / "data" / "brainstormer.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS modes (
    id                 TEXT PRIMARY KEY,      -- 'gym', the file stem, used in URLs
    name               TEXT NOT NULL,         -- 'Gym', shown to a person
    status             TEXT NOT NULL,         -- provisional | established | archived
    current_version_id INTEGER REFERENCES versions(id),
    created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS versions (
    id            INTEGER PRIMARY KEY,
    mode_id       TEXT NOT NULL REFERENCES modes(id),
    n             INTEGER NOT NULL,           -- 1, 2, 3 ... per mode, what a person calls it
    settings_json TEXT NOT NULL,              -- the --- block of the mode file, as JSON text
    prompt        TEXT NOT NULL,              -- the L2 overlay text
    rationale     TEXT NOT NULL,              -- why this version exists, in prose
    source        TEXT NOT NULL,              -- seed | manual | evolver
    created_at    TEXT NOT NULL,
    UNIQUE (mode_id, n)
);

CREATE TABLE IF NOT EXISTS switches (
    id         INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,                 -- AssemblyAI's sess_... id
    ts_ms      INTEGER NOT NULL,              -- unix ms, same clock as the timeline
    mode_id    TEXT NOT NULL REFERENCES modes(id),
    version_id INTEGER NOT NULL REFERENCES versions(id),
    source     TEXT NOT NULL                  -- manual | router
);

CREATE TABLE IF NOT EXISTS fingerprints (
    mode_id    TEXT PRIMARY KEY REFERENCES modes(id),
    version_id INTEGER NOT NULL REFERENCES versions(id),  -- which version the vector was made from
    model      TEXT NOT NULL,                 -- embedding model id; vectors from different models never compare
    vector     TEXT NOT NULL,                 -- JSON list of floats
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS routes (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT NOT NULL,
    ts_ms       INTEGER NOT NULL,
    text        TEXT NOT NULL,                -- the partial transcript that was routed
    live_mode   TEXT,                         -- mode at the time of asking
    mode        TEXT,                         -- what the router decided
    switch      INTEGER NOT NULL,             -- 1 if the page was told to change mode
    signal      TEXT NOT NULL,                -- too_short | keyterm | embedding
    scores_json TEXT NOT NULL,                -- cosine per mode, {} for keyterm decisions
    terms       TEXT NOT NULL,                -- keyterms that fired, comma separated
    embed_ms    INTEGER                       -- null when no embedding call was made
);

CREATE TABLE IF NOT EXISTS anchors (
    name       TEXT PRIMARY KEY,               -- 'filler': a vector that competes with modes but is not one
    model      TEXT NOT NULL,
    vector     TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id         INTEGER PRIMARY KEY,
    mode_id    TEXT NOT NULL REFERENCES modes(id),
    session_id TEXT,
    kind       TEXT NOT NULL,                 -- unknown_topic | observation | correction
    text       TEXT NOT NULL,
    vector     TEXT,                          -- the utterance's embedding, so candidates can be clustered
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the database, creating the file and the tables on first use."""
    path = path or DB_FILE  # looked up at call time, so tests can point DB_FILE at a temp file
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row          # rows behave like dicts: row["prompt"]
    conn.execute("PRAGMA foreign_keys = ON")  # off by default in SQLite; on, a bad mode_id is an error
    conn.executescript(SCHEMA)
    if "vector" not in {r["name"] for r in conn.execute("PRAGMA table_info(candidates)")}:
        conn.execute("ALTER TABLE candidates ADD COLUMN vector TEXT")  # databases created before 13 Sep
    return conn


def add_version(conn: sqlite3.Connection, mode_id: str, settings: dict,
                prompt: str, rationale: str, source: str) -> int:
    """Append a version. Does NOT make it live — see promote()."""
    row = conn.execute("SELECT COALESCE(MAX(n), 0) + 1 AS n FROM versions WHERE mode_id = ?",
                       (mode_id,)).fetchone()
    cursor = conn.execute(
        "INSERT INTO versions (mode_id, n, settings_json, prompt, rationale, source, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (mode_id, row["n"], json.dumps(settings, indent=2), prompt, rationale, source, now()),
    )
    conn.commit()
    return cursor.lastrowid


def promote(conn: sqlite3.Connection, version_id: int) -> None:
    """Make a version the live one for its mode. Rollback is the same call with an older id."""
    row = conn.execute("SELECT mode_id FROM versions WHERE id = ?", (version_id,)).fetchone()
    if row is None:
        raise SystemExit(f"No version with id {version_id}")
    conn.execute("UPDATE modes SET current_version_id = ? WHERE id = ?", (version_id, row["mode_id"]))
    conn.commit()


def add_mode(conn: sqlite3.Connection, mode_id: str, name: str, settings: dict,
             prompt: str, rationale: str, source: str, status: str = "established") -> int:
    """Create a mode and its v1 in one go, and make v1 live."""
    conn.execute("INSERT INTO modes (id, name, status, created_at) VALUES (?, ?, ?, ?)",
                 (mode_id, name, status, now()))
    version_id = add_version(conn, mode_id, settings, prompt, rationale, source)
    promote(conn, version_id)
    return version_id


def list_modes(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT m.id, m.name, m.status, v.n, v.source, v.created_at "
        "FROM modes m JOIN versions v ON v.id = m.current_version_id "
        "WHERE m.status != 'archived' "
        "ORDER BY (m.id != 'general'), m.id"  # general first: it is the mode a call starts in
    ).fetchall()


def current(conn: sqlite3.Connection, mode_id: str) -> dict:
    """The live version of a mode, in the shape profiles.assemble() needs."""
    row = conn.execute(
        "SELECT m.name, v.id AS version_id, v.n, v.settings_json, v.prompt "
        "FROM modes m JOIN versions v ON v.id = m.current_version_id "
        "WHERE m.id = ?", (mode_id,)
    ).fetchone()
    if row is None:
        have = ", ".join(m["id"] for m in list_modes(conn))
        raise SystemExit(f"No mode '{mode_id}'. Have: {have}")
    return {
        "mode": mode_id,
        "name": row["name"],
        "version_id": row["version_id"],
        "n": row["n"],
        "settings": json.loads(row["settings_json"]),
        "prompt": row["prompt"],
    }


def version(conn: sqlite3.Connection, version_id: int) -> dict:
    row = conn.execute("SELECT * FROM versions WHERE id = ?", (version_id,)).fetchone()
    if row is None:
        raise SystemExit(f"No version with id {version_id}")
    out = dict(row)
    out["settings"] = json.loads(out.pop("settings_json"))
    return out


def diff(conn: sqlite3.Connection, old_id: int, new_id: int) -> str:
    """Unified diff of two versions, settings first then prompt. Empty string means identical."""
    import difflib
    a, b = version(conn, old_id), version(conn, new_id)

    def as_lines(v: dict) -> list[str]:
        return (json.dumps(v["settings"], indent=2) + "\n\n" + v["prompt"]).splitlines(keepends=True)
    return "".join(difflib.unified_diff(
        as_lines(a), as_lines(b),
        fromfile=f"{a['mode_id']} v{a['n']} ({a['source']})",
        tofile=f"{b['mode_id']} v{b['n']} ({b['source']})", n=2))


def log_switch(conn: sqlite3.Connection, session_id: str, ts_ms: int,
               mode_id: str, version_id: int, source: str) -> int:
    """Record that a session switched to a mode version at ts_ms. One row per swap."""
    cursor = conn.execute(
        "INSERT INTO switches (session_id, ts_ms, mode_id, version_id, source) VALUES (?, ?, ?, ?, ?)",
        (session_id, ts_ms, mode_id, version_id, source),
    )
    conn.commit()
    return cursor.lastrowid


def log_route(conn: sqlite3.Connection, session_id: str, ts_ms: int, text: str,
              live_mode: str | None, decision: dict) -> int:
    """Keep every router decision with its scores. This is the evaluation set, built from use."""
    cursor = conn.execute(
        "INSERT INTO routes (session_id, ts_ms, text, live_mode, mode, switch, signal, scores_json, terms, embed_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, ts_ms, text, live_mode, decision["mode"], int(bool(decision["switch"])),
         decision["signal"], json.dumps(decision.get("scores", {})), ",".join(decision.get("terms", [])),
         decision.get("embed_ms")),
    )
    conn.commit()
    return cursor.lastrowid


def add_candidate(conn: sqlite3.Connection, mode_id: str, session_id: str | None, kind: str, text: str,
                  vector: list[float] | None = None) -> int:
    """Remember something that may become a mode or an L1 fact later. Never acted on by itself."""
    cursor = conn.execute(
        "INSERT INTO candidates (mode_id, session_id, kind, text, vector, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (mode_id, session_id, kind, text, json.dumps(vector) if vector else None, now()))
    conn.commit()
    return cursor.lastrowid


def session_candidates(conn: sqlite3.Connection, session_id: str, kind: str = "unknown_topic") -> list[dict]:
    """This call's unclaimed candidates of one kind, oldest first, vectors decoded."""
    rows = conn.execute(
        "SELECT id, text, vector FROM candidates WHERE session_id = ? AND kind = ? AND mode_id = 'general' "
        "AND vector IS NOT NULL ORDER BY id", (session_id, kind)).fetchall()
    return [{"id": r["id"], "text": r["text"], "vector": json.loads(r["vector"])} for r in rows]


def claim_candidates(conn: sqlite3.Connection, ids: list[int], mode_id: str) -> None:
    """Hand candidates to the mode they spawned; they stop counting toward another spawn."""
    conn.executemany("UPDATE candidates SET mode_id = ? WHERE id = ?", [(mode_id, i) for i in ids])
    conn.commit()


def set_status(conn: sqlite3.Connection, mode_id: str, status: str) -> None:
    if status not in ("provisional", "established", "archived"):
        raise SystemExit(f"Unknown status {status!r}")
    conn.execute("UPDATE modes SET status = ? WHERE id = ?", (status, mode_id))
    conn.commit()


def switches_for(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    """A session's swaps in time order. The evolver joins turn timestamps against this."""
    return conn.execute(
        "SELECT ts_ms, mode_id, version_id, source FROM switches WHERE session_id = ? ORDER BY ts_ms",
        (session_id,),
    ).fetchall()


def seed(conn: sqlite3.Connection, modes_dir: Path) -> None:
    """Load each mode file as v1. Skips modes that already exist, so it is safe to rerun."""
    import profiles  # here, not at the top: profiles will import store, and a top-level import would loop

    for path in sorted(modes_dir.glob("*.md")):
        mode_id = path.stem
        if conn.execute("SELECT 1 FROM modes WHERE id = ?", (mode_id,)).fetchone():
            print(f"{mode_id:<12} already in the store, skipped")
            continue
        settings, prompt = profiles.parse_mode(path)
        add_mode(conn, mode_id, settings.get("name", mode_id), settings, prompt,
                 rationale=f"Hand-written seed, loaded from {path.name}", source="seed")
        print(f"{mode_id:<12} seeded as v1 from {path.name}")


def main() -> None:
    conn = connect()
    argument = sys.argv[1] if len(sys.argv) > 1 else ""

    if argument == "seed":
        seed(conn, Path(__file__).resolve().parent / "prompts" / "modes")
        return

    if argument == "show":
        if len(sys.argv) < 3:
            raise SystemExit("Which mode? e.g. app/store.py show gym")
        live = current(conn, sys.argv[2])
        print(f"{live['name']}  v{live['n']}  (version id {live['version_id']})\n")
        print(json.dumps(live["settings"], indent=2))
        print()
        print(live["prompt"])
        return

    if argument == "promote":
        if len(sys.argv) < 3:
            raise SystemExit("Which version id? e.g. app/store.py promote 3")
        promote(conn, int(sys.argv[2]))
        row = conn.execute("SELECT mode_id, n FROM versions WHERE id = ?", (int(sys.argv[2]),)).fetchone()
        print(f"{row['mode_id']} is now live on v{row['n']}")
        return

    if argument == "diff":
        if len(sys.argv) < 4:
            raise SystemExit("Which two version ids? e.g. app/store.py diff 1 3")
        print(diff(conn, int(sys.argv[2]), int(sys.argv[3])))
        return

    if argument == "history":
        if len(sys.argv) < 3:
            raise SystemExit("Which mode? e.g. app/store.py history gym")
        live_id = conn.execute("SELECT current_version_id FROM modes WHERE id = ?", (sys.argv[2],)).fetchone()
        for v in conn.execute("SELECT id, n, source, created_at, rationale FROM versions "
                              "WHERE mode_id = ? ORDER BY n", (sys.argv[2],)):
            mark = "LIVE" if live_id and v["id"] == live_id[0] else "    "
            first_line = v["rationale"].strip().splitlines()[0] if v["rationale"].strip() else ""
            print(f"{mark} v{v['n']:<3} id {v['id']:<4} {v['source']:<8} "
                  f"{v['created_at'][:16].replace('T', ' ')}  {first_line[:80]}")
        return

    if argument == "switches":
        rows = conn.execute("SELECT session_id, ts_ms, mode_id, version_id, source FROM switches "
                            "ORDER BY session_id, ts_ms").fetchall()
        if not rows:
            print("No switches logged yet. Make a call and change mode.")
            return
        for r in rows:
            when = datetime.fromtimestamp(r["ts_ms"] / 1000, timezone.utc).strftime("%H:%M:%S")
            print(f"{r['session_id']}  {when}  {r['mode_id']:<10} v-id {r['version_id']}  {r['source']}")
        return

    rows = list_modes(conn)
    if not rows:
        print("Store is empty. Run: app/store.py seed")
        return
    print(f"{'mode':<12} {'name':<12} {'status':<12} live  source   since")
    for m in rows:
        print(f"{m['id']:<12} {m['name']:<12} {m['status']:<12} v{m['n']:<4} "
              f"{m['source']:<8} {m['created_at'][:19].replace('T', ' ')}")


if __name__ == "__main__":
    main()
