"""The agent loop: an explicit tool-use cycle over the Gemini interactions API.

No agent framework. The loop is about forty lines and every decision in it —
when to stop, what to do with a failed tool, how history is kept — is visible
here rather than configured somewhere inside a library. For a prototype whose
purpose is to show reasoning, that visibility is the feature; LangGraph or
Pydantic AI would add abstraction over a control flow that is already simple.

## Conversation state

The interactions API keeps history server-side: each turn passes the previous
interaction's id and the service rebuilds the context, including the model's own
reasoning steps with their signatures intact. The alternative, `store=False`,
requires replaying every model step by hand and getting the signatures right.

Server-side state is used for the conversation, and a local transcript is kept
alongside it for two things the server cannot provide: a readable log for the
interaction examples, and resumability that does not depend on an undocumented
retention window. See `Conversation.transcript`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from google import genai

from emporio import tools
from emporio.config import get_settings
from emporio.prompts import build_system_instruction

logger = logging.getLogger(__name__)

# A turn that still wants tools after this many rounds is looping. Six is well
# clear of the deepest legitimate chain observed (order -> product -> policy)
# while still bounding a runaway.
MAX_TOOL_ROUNDS = 6

# Status the service returns when it is waiting for tool results.
REQUIRES_ACTION = "requires_action"


@dataclass
class ToolInvocation:
    """One tool call and its result, surfaced for display and for tests."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    is_error: bool = False

    def summary(self) -> str:
        shown = ", ".join(f"{key}={value!r}" for key, value in self.arguments.items())
        return f"{self.name}({shown})"


@dataclass
class Turn:
    """The agent's answer to one customer message."""

    user_message: str
    reply: str
    invocations: list[ToolInvocation] = field(default_factory=list)
    interaction_id: str | None = None


@dataclass
class Conversation:
    """A multi-turn conversation with one customer."""

    client: Any = None
    system_instruction: str = field(default_factory=build_system_instruction)
    previous_interaction_id: str | None = None
    transcript: list[Turn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.client is None:
            settings = get_settings()
            self.client = genai.Client(api_key=settings.require_api_key())

    # -- internals --------------------------------------------------------

    def _create(self, payload: Any) -> Any:
        settings = get_settings()
        return self.client.interactions.create(
            model=settings.model,
            input=payload,
            tools=tools.DECLARATIONS,
            system_instruction=self.system_instruction,
            previous_interaction_id=self.previous_interaction_id,
        )

    @staticmethod
    def _function_calls(interaction: Any) -> list[Any]:
        return [
            step
            for step in (interaction.steps or [])
            if getattr(step, "type", None) == "function_call"
        ]

    def _run_tools(self, calls: list[Any]) -> tuple[list[dict[str, Any]], list[ToolInvocation]]:
        """Execute requested tools and shape the results for the next request.

        A tool that raises is reported back with `is_error` rather than
        propagated: the model gets a chance to recover or to tell the customer
        the detail could not be confirmed, which beats dropping the turn.
        """
        results: list[dict[str, Any]] = []
        invocations: list[ToolInvocation] = []

        for call in calls:
            arguments = dict(call.arguments or {})
            try:
                output = tools.call(call.name, arguments)
                failed = output.get("status") in {
                    "unknown_tool",
                    "invalid_arguments",
                    "not_found",
                }
            except Exception as exc:  # noqa: BLE001 - surfaced to the model as data
                logger.exception("tool %s raised", call.name)
                output = {"status": "error", "detail": str(exc)}
                failed = True

            invocations.append(
                ToolInvocation(
                    name=call.name, arguments=arguments, result=output, is_error=failed
                )
            )
            results.append(
                {
                    "type": "function_result",
                    "call_id": call.id,
                    "name": call.name,
                    # The API accepts a plain string for `result`; JSON keeps
                    # the structure legible to the model. ensure_ascii=False so
                    # Portuguese reaches it with its accents intact.
                    "result": json.dumps(output, ensure_ascii=False, default=str),
                    "is_error": failed,
                }
            )

        return results, invocations

    # -- public API -------------------------------------------------------

    def send(self, message: str) -> Turn:
        """Answer one customer message, running tools until the model is done."""
        interaction = self._create(message)
        invocations: list[ToolInvocation] = []

        for round_number in range(MAX_TOOL_ROUNDS):
            calls = self._function_calls(interaction)
            if interaction.status != REQUIRES_ACTION or not calls:
                break

            results, executed = self._run_tools(calls)
            invocations.extend(executed)
            self.previous_interaction_id = interaction.id
            interaction = self._create(results)
        else:
            logger.warning("stopped after %d tool rounds", MAX_TOOL_ROUNDS)

        self.previous_interaction_id = interaction.id
        reply = (interaction.output_text or "").strip()
        if not reply:
            # Reaching here means the model ended a turn without text — a
            # blank bubble is worse for the customer than an honest handoff.
            reply = (
                "Desculpe, não consegui concluir essa consulta agora. "
                "Pode repetir a pergunta, ou prefere que eu encaminhe para a equipe?"
            )

        turn = Turn(
            user_message=message,
            reply=reply,
            invocations=invocations,
            interaction_id=interaction.id,
        )
        self.transcript.append(turn)
        return turn

    def reset(self) -> None:
        """Start a fresh conversation, dropping server and local history."""
        self.previous_interaction_id = None
        self.transcript.clear()
