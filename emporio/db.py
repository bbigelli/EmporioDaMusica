"""Read-only access to the operational database."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from emporio.config import get_settings


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open the store database read-only.

    The agent answers questions; nothing in the request path has any business
    writing to the catalog, so the connection is opened in read-only mode and a
    stray INSERT fails loudly instead of mutating store data.
    """
    settings = get_settings()
    if not settings.database_path.exists():
        raise RuntimeError(
            f"{settings.database_path} is missing. Run: python scripts/build_data.py"
        )

    uri = f"file:{settings.database_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def query(sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> list[dict[str, Any]]:
    """Run a read query and return plain dicts.

    Tools return their results to the model as JSON, so sqlite3.Row is
    converted here rather than at every call site.
    """
    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, params)]


def query_one(sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None
