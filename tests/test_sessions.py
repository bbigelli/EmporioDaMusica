"""Tests for conversation persistence.

No API key and no network: the store is exercised directly, and the agent
integration runs against the same stub client as test_agent.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from emporio.config import get_settings
from emporio.sessions import SessionStore


def setUpModule() -> None:
    os.environ["EMPORIO_REFERENCE_DATE"] = "2026-03-25"
    get_settings.cache_clear()


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = SessionStore(Path(self._dir.name) / "sessions.db")

    def test_roundtrip_preserves_turns_in_order(self) -> None:
        session = self.store.create(language="pt")
        self.store.save_turn(session, "primeira", "um", [], "i1")
        self.store.save_turn(session, "segunda", "dois", [], "i2")

        record = self.store.load(session)
        self.assertEqual(["primeira", "segunda"], [t["user_message"] for t in record["turns"]])
        self.assertEqual("i2", record["last_interaction_id"])
        self.assertEqual("pt", record["language"])

    def test_invocations_survive_the_roundtrip(self) -> None:
        """The tool trail is the audit record; losing it defeats the point."""
        session = self.store.create()
        invocations = [
            {
                "name": "get_order_status",
                "arguments": {"order_id": 7},
                "result": {"status": "ok", "order_status_label": "delivered"},
                "is_error": False,
            }
        ]
        self.store.save_turn(session, "pedido 7", "entregue", invocations, "i1")

        restored = self.store.load(session)["turns"][0]["invocations"]
        self.assertEqual(invocations, restored)

    def test_accents_survive_the_roundtrip(self) -> None:
        session = self.store.create()
        self.store.save_turn(session, "qual o horário?", "Das 09:00 às 18:00. Não fechamos.", [], None)
        turn = self.store.load(session)["turns"][0]
        self.assertEqual("qual o horário?", turn["user_message"])
        self.assertIn("às 18:00", turn["reply"])
        self.assertIn("Não", turn["reply"])

    def test_resolve_by_short_prefix(self) -> None:
        session = self.store.create()
        self.store.save_turn(session, "oi", "olá", [], "i1")
        self.assertEqual(session, self.store.resolve(session[:8]))
        self.assertEqual(session, self.store.resolve(session))

    def test_ambiguous_prefix_resolves_to_nothing(self) -> None:
        """Reopening the wrong conversation silently is worse than failing.

        A uuid4 starts with one of 16 hex digits, so creating 17 sessions
        guarantees a collision by the pigeonhole principle — no waiting on
        luck for the case to be exercised.
        """
        from collections import Counter

        ids = [self.store.create() for _ in range(17)]
        first_chars = Counter(session[0] for session in ids)
        shared, count = first_chars.most_common(1)[0]
        self.assertGreater(count, 1, "pigeonhole guarantees a repeat")
        self.assertIsNone(self.store.resolve(shared))

    def test_resolve_last_picks_the_most_recent(self) -> None:
        older = self.store.create()
        self.store.save_turn(older, "antiga", "r", [], "i1")
        newer = self.store.create()
        self.store.save_turn(newer, "nova", "r", [], "i2")
        self.assertEqual(newer, self.store.resolve("last"))

    def test_unknown_prefix_resolves_to_nothing(self) -> None:
        self.assertIsNone(self.store.resolve("zzzzzzzz"))
        self.assertIsNone(self.store.resolve(""))
        self.assertIsNone(self.store.resolve("last"))  # nothing recorded yet

    def test_empty_sessions_are_not_listed(self) -> None:
        """A conversation nobody spoke in is noise in a resume list."""
        self.store.create()
        spoken = self.store.create()
        self.store.save_turn(spoken, "oi", "olá", [], "i1")

        listed = self.store.list_recent()
        self.assertEqual([spoken], [item.session_id for item in listed])
        self.assertEqual("oi", listed[0].preview)

    def test_saving_to_an_unknown_session_fails_loudly(self) -> None:
        with self.assertRaises(KeyError):
            self.store.save_turn("nope", "oi", "olá", [], None)

    def test_delete_removes_the_turns_too(self) -> None:
        session = self.store.create()
        self.store.save_turn(session, "oi", "olá", [], "i1")
        self.assertTrue(self.store.delete(session))
        self.assertFalse(self.store.exists(session))
        with self.assertRaises(KeyError):
            self.store.load(session)

    def test_reopening_the_same_file_sees_prior_sessions(self) -> None:
        """Persistence across processes is the whole feature."""
        path = Path(self._dir.name) / "sessions.db"
        session = self.store.create()
        self.store.save_turn(session, "oi", "olá", [], "i1")

        reopened = SessionStore(path)
        self.assertEqual(1, len(reopened.list_recent()))
        self.assertEqual("oi", reopened.load(session)["turns"][0]["user_message"])


# -- agent integration -------------------------------------------------------


@dataclass
class StubInteraction:
    id: str
    status: str = "completed"
    output_text: str | None = "ok"
    steps: list[Any] = field(default_factory=list)


class StubInteractions:
    def __init__(self, scripted: list[StubInteraction], fail_with: Exception | None = None):
        self.scripted = scripted
        self.fail_with = fail_with
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> StubInteraction:
        self.requests.append(kwargs)
        if self.fail_with is not None and kwargs.get("previous_interaction_id"):
            raise self.fail_with
        return self.scripted.pop(0)


class StubClient:
    def __init__(self, scripted: list[StubInteraction], fail_with: Exception | None = None):
        self.interactions = StubInteractions(scripted, fail_with)


class NotFound(Exception):
    status_code = 404


class AgentPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = SessionStore(Path(self._dir.name) / "sessions.db")

    def test_turns_are_persisted_as_they_happen(self) -> None:
        from emporio.agent import Conversation

        client = StubClient([StubInteraction(id="i1", output_text="olá")])
        convo = Conversation(client=client, store=self.store, language="pt")
        convo.send("oi")

        record = self.store.load(convo.session_id)
        self.assertEqual("oi", record["turns"][0]["user_message"])
        self.assertEqual("olá", record["turns"][0]["reply"])
        self.assertEqual("i1", record["last_interaction_id"])

    def test_resume_restores_transcript_and_server_chain(self) -> None:
        from emporio.agent import Conversation

        first = Conversation(
            client=StubClient([StubInteraction(id="i1", output_text="um")]),
            store=self.store,
        )
        first.send("primeira")

        second = Conversation.resume(
            self.store,
            first.session_id,
            client=StubClient([StubInteraction(id="i2", output_text="dois")]),
        )
        self.assertEqual(1, len(second.transcript))
        self.assertEqual("primeira", second.transcript[0].user_message)

        second.send("segunda")
        request = second.client.interactions.requests[0]
        self.assertEqual("i1", request["previous_interaction_id"])
        self.assertEqual(2, len(self.store.load(first.session_id)["turns"]))

    def test_expired_server_history_falls_back_to_a_new_thread(self) -> None:
        """Retention is undocumented, so an expired id is expected eventually.
        Losing the model's memory beats refusing to continue."""
        from emporio import agent
        from emporio.agent import Conversation

        client = StubClient([StubInteraction(id="i9", output_text="recomeçado")], NotFound())
        convo = Conversation(
            client=client, store=self.store, previous_interaction_id="stale-id"
        )
        with self.assertLogs(agent.logger, level="WARNING"):
            turn = convo.send("continuando")

        self.assertEqual("recomeçado", turn.reply)
        self.assertEqual(
            ["stale-id", None],
            [r["previous_interaction_id"] for r in client.interactions.requests],
        )

    def test_reset_opens_a_new_session_without_erasing_the_old_one(self) -> None:
        from emporio.agent import Conversation

        convo = Conversation(
            client=StubClient(
                [StubInteraction(id="i1", output_text="um"), StubInteraction(id="i2", output_text="dois")]
            ),
            store=self.store,
        )
        convo.send("primeira")
        first_session = convo.session_id

        convo.reset()
        self.assertNotEqual(first_session, convo.session_id)
        convo.send("segunda")

        self.assertEqual(1, len(self.store.load(first_session)["turns"]))
        self.assertEqual(1, len(self.store.load(convo.session_id)["turns"]))

    def test_a_failing_store_does_not_lose_the_customer_their_answer(self) -> None:
        """Filing the transcript is an operations concern; the reply is not."""
        from emporio import agent
        from emporio.agent import Conversation

        class BrokenStore:
            def create(self, language=None):
                return "session-1"

            def save_turn(self, **kwargs):
                raise RuntimeError("disk full")

        convo = Conversation(
            client=StubClient([StubInteraction(id="i1", output_text="resposta")]),
            store=BrokenStore(),
        )
        with self.assertLogs(agent.logger, level="ERROR"):
            turn = convo.send("oi")
        self.assertEqual("resposta", turn.reply)

    def test_no_store_means_no_session(self) -> None:
        from emporio.agent import Conversation

        convo = Conversation(client=StubClient([StubInteraction(id="i1")]))
        convo.send("oi")
        self.assertIsNone(convo.session_id)


if __name__ == "__main__":
    unittest.main()
