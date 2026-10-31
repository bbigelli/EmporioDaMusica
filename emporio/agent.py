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

Server-side state drives the model; a local SQLite transcript
(`emporio.sessions`) records what was said and which tools ran. The server
cannot give the second: retention is undocumented, an interaction id is not
something a human can read back, and a disputed answer needs an audit trail.

Pass a `SessionStore` to persist, or leave `store` unset for an ephemeral
conversation. `Conversation.resume` reopens a stored session; if the
server-side id has aged out by then, the next turn silently starts a new
server thread rather than failing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from google import genai

from emporio import tools
from emporio.config import REQUEST_TIMEOUT_SECONDS, get_settings
from emporio.prompts import build_system_instruction

logger = logging.getLogger(__name__)

# The free tier returns 503 under load and it clears on its own.
MAX_REQUEST_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0

# HTTP statuses worth another try: overloaded, rate limited, gateway hiccups.
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _is_stale_interaction(error: Exception) -> bool:
    """Whether a failure means the referenced interaction no longer exists.

    A 404 on a request that carried previous_interaction_id is the signal that
    server-side history has aged out. Matched on the status code, with the
    message checked only as a secondary hint, since the wording is not part of
    any contract.
    """
    status = getattr(error, "status_code", None)
    if status == 404:
        return True
    text = str(error).lower()
    return "previous_interaction_id" in text and "not found" in text


def _is_transient(error: Exception) -> bool:
    """Whether a failure is worth retrying.

    Checks the structured status code first and only falls back to the message
    when the exception does not carry one — string matching on error text is
    brittle, but a timeout class name is a reliable enough last resort.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in TRANSIENT_STATUS_CODES
    return "Timeout" in type(error).__name__


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
    language: str | None = None
    system_instruction: str | None = None
    previous_interaction_id: str | None = None
    transcript: list[Turn] = field(default_factory=list)
    timeout: float = REQUEST_TIMEOUT_SECONDS
    store: Any = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.system_instruction is None:
            self.system_instruction = build_system_instruction(self.language)
        if self.client is None:
            settings = get_settings()
            self.client = genai.Client(api_key=settings.require_api_key())
        if self.store is not None and self.session_id is None:
            self.session_id = self.store.create(language=self.language)

    @classmethod
    def resume(cls, store: Any, session_id: str, **kwargs: Any) -> Conversation:
        """Reopen a stored conversation.

        The local transcript is restored for display and the server-side id for
        model continuity. If that id has since expired, the next `send` starts
        a fresh server thread rather than failing — the customer loses the
        model's memory of earlier turns, which is bad, but far less bad than a
        conversation that cannot be continued at all.
        """
        record = store.load(session_id)
        conversation = cls(
            store=store,
            session_id=session_id,
            language=kwargs.pop("language", record["language"]),
            previous_interaction_id=record["last_interaction_id"],
            **kwargs,
        )
        conversation.transcript = [
            Turn(
                user_message=turn["user_message"],
                reply=turn["reply"],
                invocations=[ToolInvocation(**call) for call in turn["invocations"]],
                interaction_id=turn["interaction_id"],
            )
            for turn in record["turns"]
        ]
        return conversation

    # -- internals --------------------------------------------------------

    def _create(self, payload: Any) -> Any:
        """Send one request, retrying the transient failures.

        The free tier returns 503 "experiencing high demand" under load, which
        clears on its own. Retrying with backoff turns a dead conversation into
        a slow one. Everything else — a bad key, a malformed request — is
        raised immediately, because retrying it only wastes the customer's
        time.
        """
        settings = get_settings()
        last_error: Exception | None = None

        for attempt in range(MAX_REQUEST_ATTEMPTS):
            try:
                return self.client.interactions.create(
                    model=settings.model,
                    input=payload,
                    tools=tools.DECLARATIONS,
                    system_instruction=self.system_instruction,
                    previous_interaction_id=self.previous_interaction_id,
                    timeout=self.timeout,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised below unless transient
                if not _is_transient(exc) or attempt == MAX_REQUEST_ATTEMPTS - 1:
                    raise
                last_error = exc
                delay = RETRY_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    "transient API failure (%s), retrying in %.1fs",
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)

        raise last_error  # unreachable; the loop either returns or raises

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
        try:
            interaction = self._create(message)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is a stale id
            if not _is_stale_interaction(exc) or self.previous_interaction_id is None:
                raise
            # The server-side history expired. Retention is undocumented, so
            # this is expected eventually rather than exceptional: drop the
            # chain and continue as a new thread instead of failing the turn.
            logger.warning(
                "interaction %s no longer resolvable; starting a new server thread",
                self.previous_interaction_id,
            )
            self.previous_interaction_id = None
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

        if self.store is not None and self.session_id is not None:
            try:
                self.store.save_turn(
                    session_id=self.session_id,
                    user_message=turn.user_message,
                    reply=turn.reply,
                    invocations=[asdict(call) for call in turn.invocations],
                    interaction_id=turn.interaction_id,
                )
            except Exception:  # noqa: BLE001 - logging must not lose an answer
                # The customer already has their reply; failing to file it is
                # an operations problem, not theirs.
                logger.exception("could not persist turn for session %s", self.session_id)

        return turn

    def reset(self) -> None:
        """Start a fresh conversation, dropping server and local history.

        When persistence is on this opens a *new* session rather than erasing
        the old one: the previous conversation stays readable, which is the
        point of keeping a transcript at all.
        """
        self.previous_interaction_id = None
        self.transcript.clear()
        if self.store is not None:
            self.session_id = self.store.create(language=self.language)
