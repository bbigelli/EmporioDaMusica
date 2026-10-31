"""Conversation persistence.

The interactions API already keeps history server-side, so why store anything
locally? Three reasons, and the first is the one that matters:

1. **Retention is undocumented.** Google does not publish how long an
   interaction id stays resolvable. Building a resume feature on top of an
   unstated guarantee means it works until it silently does not.
2. **A server-side id is not a transcript.** It lets the *model* continue; it
   does not let a human reopen yesterday's conversation and read it, which is
   what a store employee picking up a thread actually needs.
3. **Auditability.** Which tools ran, with which arguments, and what they
   returned is exactly what you want when a customer disputes what they were
   told.

So both are kept: the server-side id for model continuity, and a local SQLite
transcript for everything a human needs. When the id has expired, the
conversation continues as a fresh thread rather than failing — see
`emporio.agent.Conversation.send`.

The sessions database is written at runtime and is not committed; it lives
beside the generated catalog but is listed in .gitignore.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from emporio.config import ROOT

DEFAULT_PATH = ROOT / "data" / "sessions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id            TEXT PRIMARY KEY,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    language              TEXT,
    last_interaction_id   TEXT,
    turn_count            INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    position       INTEGER NOT NULL,
    created_at     TEXT NOT NULL,
    user_message   TEXT NOT NULL,
    reply          TEXT NOT NULL,
    interaction_id TEXT,
    invocations    TEXT NOT NULL,
    UNIQUE (session_id, position)
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, position);
"""


def _now() -> str:
    """UTC timestamp with microseconds.

    Microseconds, not seconds: `list_recent` and `resolve("last")` order by
    this string, and two conversations started within the same second would
    tie — making "resume the last one" reopen an arbitrary one of them.
    Lexicographic order on a fixed-width ISO string is chronological order,
    which is why the timespec is pinned rather than left to isoformat's
    default of dropping a zero microsecond field.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    created_at: str
    updated_at: str
    language: str | None
    turn_count: int
    preview: str

    @property
    def short_id(self) -> str:
        """First segment of the uuid — enough to type, enough to disambiguate."""
        return self.session_id.split("-")[0]


class SessionStore:
    """SQLite-backed conversation history."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        # Declared ON DELETE CASCADE only takes effect with this set, and it
        # is per connection.
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- writing ----------------------------------------------------------

    def create(self, language: str | None = None) -> str:
        session_id = str(uuid.uuid4())
        stamp = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at, language)"
                " VALUES (?, ?, ?, ?)",
                (session_id, stamp, stamp, language),
            )
        return session_id

    def save_turn(
        self,
        session_id: str,
        user_message: str,
        reply: str,
        invocations: list[dict[str, Any]],
        interaction_id: str | None,
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT turn_count FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown session {session_id!r}")
            position = int(row["turn_count"])
            conn.execute(
                "INSERT INTO turns (session_id, position, created_at, user_message,"
                " reply, interaction_id, invocations) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    position,
                    _now(),
                    user_message,
                    reply,
                    interaction_id,
                    json.dumps(invocations, ensure_ascii=False, default=str),
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ?, turn_count = ?,"
                " last_interaction_id = ? WHERE session_id = ?",
                (_now(), position + 1, interaction_id, session_id),
            )

    # -- reading ----------------------------------------------------------

    def exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                is not None
            )

    def resolve(self, prefix: str) -> str | None:
        """Find a session by id or unique short prefix.

        An ambiguous prefix returns None rather than picking one, so
        `--resume ab` can never silently reopen the wrong conversation.
        """
        prefix = prefix.strip()
        if not prefix:
            return None
        if prefix == "last":
            recent = self.list_recent(limit=1)
            return recent[0].session_id if recent else None
        with self._connect() as conn:
            matches = [
                row["session_id"]
                for row in conn.execute(
                    "SELECT session_id FROM sessions WHERE session_id LIKE ? || '%'",
                    (prefix,),
                )
            ]
        return matches[0] if len(matches) == 1 else None

    def load(self, session_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            session = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise KeyError(f"unknown session {session_id!r}")
            turns = [
                {
                    "user_message": row["user_message"],
                    "reply": row["reply"],
                    "interaction_id": row["interaction_id"],
                    "invocations": json.loads(row["invocations"]),
                    "created_at": row["created_at"],
                }
                for row in conn.execute(
                    "SELECT * FROM turns WHERE session_id = ? ORDER BY position",
                    (session_id,),
                )
            ]
        return {
            "session_id": session["session_id"],
            "language": session["language"],
            "last_interaction_id": session["last_interaction_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "turns": turns,
        }

    def list_recent(self, limit: int = 10) -> list[SessionSummary]:
        """Most recently updated sessions, with the opening question as label.

        Empty sessions are skipped: a conversation nobody spoke in is noise in
        a resume list.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.*, ("
                "  SELECT user_message FROM turns t WHERE t.session_id = s.session_id"
                "  ORDER BY position LIMIT 1"
                ") AS preview"
                " FROM sessions s WHERE s.turn_count > 0"
                " ORDER BY s.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            SessionSummary(
                session_id=row["session_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                language=row["language"],
                turn_count=int(row["turn_count"]),
                preview=(row["preview"] or "").strip(),
            )
            for row in rows
        ]

    def delete(self, session_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            return cursor.rowcount > 0
