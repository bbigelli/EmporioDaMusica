"""Tests for the agent loop, driven by a stub interactions client.

These run with no API key and no network. The loop's job is to decide when to
stop, what to do with a failed tool and how to thread conversation state — all
of which can and should be tested without a model.
"""

from __future__ import annotations

import json
import os
import unittest
from dataclasses import dataclass, field
from typing import Any

from emporio.config import get_settings


def setUpModule() -> None:
    os.environ["EMPORIO_REFERENCE_DATE"] = "2026-03-25"
    get_settings.cache_clear()


@dataclass
class StubCall:
    """Stands in for a FunctionCallStep."""

    name: str
    arguments: dict[str, Any]
    id: str = "call-1"
    type: str = "function_call"


@dataclass
class StubInteraction:
    """Stands in for an Interaction."""

    id: str
    status: str = "completed"
    output_text: str | None = "ok"
    steps: list[Any] = field(default_factory=list)


class StubInteractions:
    def __init__(self, scripted: list[StubInteraction]) -> None:
        self.scripted = scripted
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> StubInteraction:
        self.requests.append(kwargs)
        if not self.scripted:
            raise AssertionError("the loop made more requests than the script allows")
        return self.scripted.pop(0)


class StubClient:
    def __init__(self, scripted: list[StubInteraction]) -> None:
        self.interactions = StubInteractions(scripted)


def conversation(scripted: list[StubInteraction]):
    from emporio.agent import Conversation

    client = StubClient(scripted)
    return Conversation(client=client), client


class DirectAnswerTests(unittest.TestCase):
    def test_reply_without_tools(self) -> None:
        convo, client = conversation(
            [StubInteraction(id="i1", status="completed", output_text="Olá! Como posso ajudar?")]
        )
        turn = convo.send("oi")
        self.assertEqual("Olá! Como posso ajudar?", turn.reply)
        self.assertEqual([], turn.invocations)
        self.assertEqual(1, len(client.interactions.requests))

    def test_first_request_carries_prompt_and_tools(self) -> None:
        convo, client = conversation([StubInteraction(id="i1")])
        convo.send("oi")
        request = client.interactions.requests[0]
        self.assertEqual("oi", request["input"])
        self.assertIsNone(request["previous_interaction_id"])
        self.assertTrue(request["tools"])
        self.assertIn("Empório da Música", request["system_instruction"])

    def test_blank_reply_falls_back_to_an_honest_handoff(self) -> None:
        """A model that ends a turn with no text would otherwise render an
        empty message bubble."""
        convo, _ = conversation([StubInteraction(id="i1", output_text="")])
        turn = convo.send("oi")
        self.assertIn("equipe", turn.reply)


class ToolLoopTests(unittest.TestCase):
    def test_single_tool_round(self) -> None:
        convo, client = conversation(
            [
                StubInteraction(
                    id="i1",
                    status="requires_action",
                    output_text=None,
                    steps=[StubCall(name="get_active_promotions", arguments={})],
                ),
                StubInteraction(id="i2", status="completed", output_text="Temos 4 promoções."),
            ]
        )
        turn = convo.send("tem promoção?")

        self.assertEqual("Temos 4 promoções.", turn.reply)
        self.assertEqual(1, len(turn.invocations))
        self.assertEqual("get_active_promotions", turn.invocations[0].name)
        self.assertEqual(4, turn.invocations[0].result["count"])

        # The second request carries the tool result and chains to the first.
        second = client.interactions.requests[1]
        self.assertEqual("i1", second["previous_interaction_id"])
        payload = second["input"][0]
        self.assertEqual("function_result", payload["type"])
        self.assertEqual("call-1", payload["call_id"])
        self.assertFalse(payload["is_error"])

    def test_parallel_calls_in_one_round_are_all_executed(self) -> None:
        convo, client = conversation(
            [
                StubInteraction(
                    id="i1",
                    status="requires_action",
                    output_text=None,
                    steps=[
                        StubCall(name="get_order_status", arguments={"order_id": 7}, id="c1"),
                        StubCall(
                            name="check_return_eligibility", arguments={"order_id": 7}, id="c2"
                        ),
                    ],
                ),
                StubInteraction(id="i2", output_text="Seu pedido foi entregue."),
            ]
        )
        turn = convo.send("posso devolver o pedido 7?")
        self.assertEqual(
            ["get_order_status", "check_return_eligibility"],
            [call.name for call in turn.invocations],
        )
        # Both results go back in a single request.
        self.assertEqual(2, len(client.interactions.requests[1]["input"]))

    def test_results_are_json_with_accents_preserved(self) -> None:
        convo, client = conversation(
            [
                StubInteraction(
                    id="i1",
                    status="requires_action",
                    output_text=None,
                    steps=[StubCall(name="get_order_status", arguments={"order_id": 7})],
                ),
                StubInteraction(id="i2", output_text="pronto"),
            ]
        )
        convo.send("pedido 7")
        raw = client.interactions.requests[1]["input"][0]["result"]
        self.assertIsInstance(raw, str)
        self.assertIn("entregue", raw)
        # ensure_ascii=False, so accented characters are not escaped.
        self.assertNotIn("\\u00e", raw)
        self.assertEqual("ok", json.loads(raw)["status"])

    def test_chained_tool_rounds(self) -> None:
        convo, client = conversation(
            [
                StubInteraction(
                    id="i1",
                    status="requires_action",
                    output_text=None,
                    steps=[StubCall(name="get_order_status", arguments={"order_id": 7}, id="c1")],
                ),
                StubInteraction(
                    id="i2",
                    status="requires_action",
                    output_text=None,
                    steps=[
                        StubCall(
                            name="search_store_policies",
                            arguments={"question": "devolução"},
                            id="c2",
                        )
                    ],
                ),
                StubInteraction(id="i3", output_text="final"),
            ]
        )
        turn = convo.send("posso devolver?")
        self.assertEqual(2, len(turn.invocations))
        self.assertEqual("i3", turn.interaction_id)
        self.assertEqual(
            [None, "i1", "i2"],
            [request["previous_interaction_id"] for request in client.interactions.requests],
        )

    def test_runaway_tool_loop_is_bounded(self) -> None:
        """A model that keeps asking for tools must not spin forever."""
        from emporio.agent import MAX_TOOL_ROUNDS

        scripted = [
            StubInteraction(
                id=f"i{n}",
                status="requires_action",
                output_text="parcial",
                steps=[StubCall(name="get_active_promotions", arguments={}, id=f"c{n}")],
            )
            for n in range(MAX_TOOL_ROUNDS + 3)
        ]
        convo, client = conversation(scripted)
        turn = convo.send("loop")
        self.assertEqual(MAX_TOOL_ROUNDS, len(turn.invocations))
        self.assertEqual(MAX_TOOL_ROUNDS + 1, len(client.interactions.requests))
        self.assertEqual("parcial", turn.reply)


class ToolFailureTests(unittest.TestCase):
    def test_not_found_is_flagged_as_an_error_result(self) -> None:
        convo, client = conversation(
            [
                StubInteraction(
                    id="i1",
                    status="requires_action",
                    output_text=None,
                    steps=[StubCall(name="get_order_status", arguments={"order_id": 9999})],
                ),
                StubInteraction(id="i2", output_text="Não encontrei esse pedido."),
            ]
        )
        turn = convo.send("pedido 9999")
        self.assertTrue(turn.invocations[0].is_error)
        self.assertTrue(client.interactions.requests[1]["input"][0]["is_error"])

    def test_raising_tool_is_reported_not_propagated(self) -> None:
        """A crashing tool must not take the turn down with it: the model gets
        the error as data and can tell the customer honestly."""
        from emporio import agent, tools

        original = tools.call

        def exploding(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("database on fire")

        tools.call = exploding
        try:
            convo, client = conversation(
                [
                    StubInteraction(
                        id="i1",
                        status="requires_action",
                        output_text=None,
                        steps=[StubCall(name="get_active_promotions", arguments={})],
                    ),
                    StubInteraction(id="i2", output_text="Vou confirmar com a equipe."),
                ]
            )
            with self.assertLogs(agent.logger, level="ERROR"):
                turn = convo.send("promoções?")
        finally:
            tools.call = original

        self.assertTrue(turn.invocations[0].is_error)
        self.assertEqual("error", turn.invocations[0].result["status"])
        self.assertEqual("Vou confirmar com a equipe.", turn.reply)

    def test_unknown_tool_name_is_handled(self) -> None:
        convo, _ = conversation(
            [
                StubInteraction(
                    id="i1",
                    status="requires_action",
                    output_text=None,
                    steps=[StubCall(name="launch_missiles", arguments={})],
                ),
                StubInteraction(id="i2", output_text="desculpe"),
            ]
        )
        turn = convo.send("?")
        self.assertEqual("unknown_tool", turn.invocations[0].result["status"])


class ApiError(Exception):
    """Stands in for an SDK error carrying an HTTP status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FlakyInteractions(StubInteractions):
    """Raises a scripted sequence of errors before each successful response."""

    def __init__(self, errors: list[Exception], scripted: list[StubInteraction]) -> None:
        super().__init__(scripted)
        self.errors = errors
        self.attempts = 0

    def create(self, **kwargs):
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return super().create(**kwargs)


class RetryTests(unittest.TestCase):
    """The free tier returns 503 'experiencing high demand' under load, and it
    clears on its own. Retrying turns a dead conversation into a slow one."""

    def setUp(self) -> None:
        from emporio import agent

        self.agent = agent
        self.slept: list[float] = []
        self._sleep = agent.time.sleep
        agent.time.sleep = self.slept.append

    def tearDown(self) -> None:
        self.agent.time.sleep = self._sleep

    def _conversation(self, errors, scripted):
        client = StubClient([])
        client.interactions = FlakyInteractions(errors, scripted)
        return self.agent.Conversation(client=client), client

    def test_overloaded_service_is_retried(self) -> None:
        convo, client = self._conversation(
            [ApiError(503), ApiError(503)], [StubInteraction(id="i1", output_text="pronto")]
        )
        with self.assertLogs(self.agent.logger, level="WARNING"):
            turn = convo.send("oi")
        self.assertEqual("pronto", turn.reply)
        self.assertEqual(3, client.interactions.attempts)
        self.assertEqual([2.0, 4.0], self.slept)  # exponential backoff

    def test_rate_limit_and_gateway_errors_are_retried(self) -> None:
        for status in (429, 500, 502, 504):
            convo, client = self._conversation(
                [ApiError(status)], [StubInteraction(id="i1", output_text="ok")]
            )
            with self.assertLogs(self.agent.logger, level="WARNING"):
                convo.send("oi")
            self.assertEqual(2, client.interactions.attempts, f"status {status}")

    def test_client_errors_are_not_retried(self) -> None:
        """A bad key or a malformed request will fail identically every time;
        retrying only makes the customer wait longer for the same failure."""
        for status in (400, 401, 403, 404):
            convo, client = self._conversation([ApiError(status)], [])
            with self.assertRaises(ApiError):
                convo.send("oi")
            self.assertEqual(1, client.interactions.attempts, f"status {status}")

    def test_timeouts_are_retried_without_a_status_code(self) -> None:
        class APITimeoutError(Exception):
            pass

        convo, client = self._conversation(
            [APITimeoutError("timed out")], [StubInteraction(id="i1", output_text="ok")]
        )
        with self.assertLogs(self.agent.logger, level="WARNING"):
            convo.send("oi")
        self.assertEqual(2, client.interactions.attempts)

    def test_persistent_failure_eventually_raises(self) -> None:
        errors = [ApiError(503) for _ in range(self.agent.MAX_REQUEST_ATTEMPTS)]
        convo, client = self._conversation(errors, [])
        with self.assertLogs(self.agent.logger, level="WARNING"), self.assertRaises(ApiError):
            convo.send("oi")
        self.assertEqual(self.agent.MAX_REQUEST_ATTEMPTS, client.interactions.attempts)

    def test_requests_are_bounded_by_a_timeout(self) -> None:
        convo, client = self._conversation([], [StubInteraction(id="i1", output_text="ok")])
        convo.send("oi")
        self.assertEqual(
            self.agent.REQUEST_TIMEOUT_SECONDS, client.interactions.requests[0]["timeout"]
        )


class ConversationStateTests(unittest.TestCase):
    def test_multi_turn_chains_through_the_server(self) -> None:
        convo, client = conversation(
            [StubInteraction(id="i1", output_text="um"), StubInteraction(id="i2", output_text="dois")]
        )
        convo.send("primeira")
        convo.send("segunda")
        self.assertEqual("i1", client.interactions.requests[1]["previous_interaction_id"])
        self.assertEqual(2, len(convo.transcript))
        self.assertEqual(["primeira", "segunda"], [t.user_message for t in convo.transcript])

    def test_reset_drops_history(self) -> None:
        convo, client = conversation(
            [StubInteraction(id="i1", output_text="um"), StubInteraction(id="i2", output_text="dois")]
        )
        convo.send("primeira")
        convo.reset()
        convo.send("segunda")
        self.assertIsNone(client.interactions.requests[1]["previous_interaction_id"])
        self.assertEqual(1, len(convo.transcript))


class SystemInstructionTests(unittest.TestCase):
    def test_instruction_states_the_reference_date(self) -> None:
        """A model assuming its own training-time date would compute every
        return window wrongly."""
        from emporio.prompts import build_system_instruction

        self.assertIn("2026-03-25", build_system_instruction())

    def test_instruction_carries_the_documentation_corrections(self) -> None:
        from emporio.prompts import build_system_instruction

        instruction = build_system_instruction()
        self.assertIn("contato@emporiodamusica.com.br", instruction)
        self.assertIn("Tamanho do catálogo", instruction)

    def test_instruction_forbids_answering_from_memory(self) -> None:
        from emporio.prompts import build_system_instruction

        instruction = build_system_instruction()
        self.assertIn("must not state a price", instruction)
        self.assertIn("accessories", instruction.lower())

    def test_instruction_specifies_language_mirroring(self) -> None:
        from emporio.prompts import build_system_instruction

        self.assertIn(
            "Match the language of the customer's most recent message",
            build_system_instruction(),
        )

    def test_language_rule_appears_early_and_last(self) -> None:
        """Measured: with the rule buried mid-prompt, the model answered an
        English-speaking customer in Portuguese for all three turns of
        scenario 05. The surrounding Portuguese context outweighed a single
        mention, so the rule is placed for both primacy and recency."""
        from emporio.prompts import build_system_instruction

        instruction = build_system_instruction()
        blocks = instruction.split("\n\n---\n\n")
        self.assertIn("# Language", blocks[1])
        self.assertIn("same language as the", blocks[-1])


class ScenarioTests(unittest.TestCase):
    """The scenarios make specific factual claims in their rationale. If the
    data stops supporting a claim, the example silently stops demonstrating
    what it says it demonstrates."""

    def test_slugs_and_titles_are_unique(self) -> None:
        from emporio.scenarios import SCENARIOS

        self.assertEqual(len({s.slug for s in SCENARIOS}), len(SCENARIOS))
        self.assertEqual(len({s.title for s in SCENARIOS}), len(SCENARIOS))

    def test_every_scenario_is_multi_turn(self) -> None:
        from emporio.scenarios import SCENARIOS

        for scenario in SCENARIOS:
            self.assertGreaterEqual(len(scenario.messages), 2, scenario.slug)

    def test_challenge_asks_for_three_to_five_examples(self) -> None:
        from emporio.scenarios import SCENARIOS

        self.assertTrue(3 <= len(SCENARIOS) <= 5)

    def test_products_named_in_scenarios_exist_with_the_claimed_discounts(self) -> None:
        from emporio import tools

        crafter = tools.get_product_details(name="Crafter HT-100")["product"]
        self.assertEqual(18, crafter["discount_percent"])

        taylor = tools.get_product_details(name="Taylor 110e")["product"]
        self.assertEqual(8, taylor["discount_percent"])

        yamaha = tools.get_product_details(name="Yamaha F310")["product"]
        self.assertEqual("available", yamaha["availability"])
        self.assertLessEqual(yamaha["price_brl"], 1000)

    def test_scenario_three_still_rests_on_an_expired_window(self) -> None:
        from emporio import tools

        result = tools.check_return_eligibility(7)
        self.assertEqual(["legal_warranty"], result["available_routes"])

    def test_scenario_five_destination_is_outside_the_metro_region(self) -> None:
        from emporio import tools

        quote = tools.quote_shipping(5519.08, city="São Paulo")
        self.assertIsNone(quote["fee_brl"])


if __name__ == "__main__":
    unittest.main()
