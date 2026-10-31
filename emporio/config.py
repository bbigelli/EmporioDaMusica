"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# The dataset's last order is 2026-03-22. Every deadline in the policy manual
# runs from a purchase or delivery date, so a real clock months later expires
# all of them and the returns policy can never be exercised. See the README.
DEFAULT_REFERENCE_DATE = date(2026, 3, 25)

DEFAULT_MODEL = "gemini-3.8-flash"


@dataclass(frozen=True)
class Settings:
    database_path: Path
    policy_sections_path: Path
    policy_markdown_path: Path
    model: str
    reference_date: date
    api_key: str | None

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add a key "
                "from https://aistudio.google.com/apikey"
            )
        return self.api_key


def _parse_reference_date(raw: str | None) -> date:
    """Resolve the agent's notion of "today".

    An explicit empty value means "use the real clock", which is distinct from
    the variable being absent (use the dataset-aligned default).
    """
    if raw is None:
        return DEFAULT_REFERENCE_DATE
    raw = raw.strip()
    if not raw:
        return date.today()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"EMPORIO_REFERENCE_DATE must be an ISO date (YYYY-MM-DD), got {raw!r}"
        ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    processed = ROOT / "data" / "processed"
    return Settings(
        database_path=processed / "store.db",
        policy_sections_path=processed / "policy_sections.json",
        policy_markdown_path=processed / "store_policies.md",
        model=os.getenv("EMPORIO_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        reference_date=_parse_reference_date(os.getenv("EMPORIO_REFERENCE_DATE")),
        api_key=os.getenv("GOOGLE_API_KEY") or None,
    )
