"""Scripted conversations used to produce the interaction examples.

Running them from code rather than pasting a terminal session means the
transcripts in examples/ are reproducible: anyone with a key can regenerate
them and see whether the agent still behaves the same way.

Each scenario is multi-turn on purpose. A single question only shows retrieval;
a follow-up shows whether the agent kept context — whether "e o frete?" is
understood as being about the instrument just discussed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from emporio.config import get_settings


@dataclass(frozen=True)
class Scenario:
    slug: str
    title: str
    why: str
    messages: tuple[str, ...]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        slug="01-catalog-on-a-budget",
        title="Browsing the catalog on a budget",
        why=(
            "Structured catalog query with a price ceiling, then a follow-up that "
            "depends on remembering which instrument was being discussed. Also "
            "shows how unavailable matches are handled: two of the fourteen "
            "acoustic guitars under BRL 1000 are out of stock or discontinued, and "
            "the agent should not present them as buyable."
        ),
        messages=(
            "oi! quais opções de violões vocês têm disponíveis até R$ 1000?",
            "gostei do Yamaha F310. ele é bom pra quem está começando?",
            "e se eu quiser pagar no pix, quanto fica?",
        ),
    ),
    Scenario(
        slug="02-promotional-price-and-pix",
        title="A promotional price meets the PIX discount",
        why=(
            "The non-trivial pricing case. The Crafter HT-100 carries an active 18% "
            "discount, and section 6.2 says promotions are not cumulative — so the "
            "5% PIX discount does *not* stack on top of it. Answering this correctly "
            "requires reading the live promotion and applying the rule; a plausible "
            "wrong answer (R$ 1.868,82) undercharges the store."
        ),
        messages=(
            "quanto custa o Crafter HT-100?",
            "vi que tem desconto. se eu pagar no pix ganho os 5% também?",
            "beleza. e em quantas vezes posso parcelar?",
        ),
    ),
    Scenario(
        slug="03-return-window-expired",
        title="The withdrawal window closed, but the warranty has not",
        why=(
            "The non-trivial policy case, and the reason the rules layer reports all "
            "four return routes instead of one verdict. Order 7 was delivered "
            "2026-02-17, so against the reference date of 2026-03-25 the 7-day "
            "withdrawal, the 30-day defect exchange and the 7-day preference "
            "exchange have all expired — but the 90-day legal warranty runs to "
            "2026-05-18. A flat 'no' would be both unhelpful and wrong."
        ),
        messages=(
            "comprei um instrumento no pedido 7 e me arrependi. consigo devolver?",
            "e se ele tiver um defeito de fábrica, eu fico sem nada?",
        ),
    ),
    Scenario(
        slug="04-out-of-scope",
        title="Two kinds of out of scope",
        why=(
            "Section 7.1 draws a line the agent has to hold: the store sells "
            "instruments and explicitly not accessories, so a request for strings "
            "gets a warm redirect rather than a catalog search. The second question "
            "is outside the store entirely and should get one light sentence, not a "
            "lecture about limitations. The third checks the agent comes back to "
            "work afterwards."
        ),
        messages=(
            "vocês vendem cordas de nylon e palheta?",
            "entendi. e qual a previsão do tempo em campo grande amanhã?",
            "haha tudo bem. me fala o horário de funcionamento então",
        ),
    ),
    Scenario(
        slug="05-english-customer-shipping",
        title="An English-speaking customer, shipping outside the region",
        why=(
            "Checks the two things that interact badly if either is wrong. The agent "
            "must mirror the customer's language, and it must not invent a shipping "
            "fee: São Paulo is outside metropolitan Campo Grande, where the cost "
            "depends on postcode, weight and dimensions. Carrier names and estimates "
            "are available; a number is not. The Taylor 110e also carries an active "
            "8% discount, which should be presented with its original price."
        ),
        messages=(
            "Hi! I'm interested in the Taylor 110e. How much is it?",
            "Great. Do you ship to São Paulo, and how much would that cost?",
            "What's your return policy if I don't like it?",
        ),
    ),
)


def _render(scenario: Scenario, turns: list, settings) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {scenario.title}",
        "",
        f"**Why this scenario.** {scenario.why}",
        "",
        f"<sub>Generated by `python -m emporio.cli demo` on {stamp} · "
        f"model `{settings.model}` · reference date `{settings.reference_date}`</sub>",
        "",
        "---",
        "",
    ]
    for turn in turns:
        lines += [f"**Cliente**", "", turn.user_message, ""]
        if turn.invocations:
            lines.append("<details><summary>Tool calls</summary>")
            lines.append("")
            for call in turn.invocations:
                marker = " *(error)*" if call.is_error else ""
                lines += [
                    f"`{call.summary()}`{marker}",
                    "",
                    "```json",
                    json.dumps(call.result, ensure_ascii=False, indent=2, default=str),
                    "```",
                    "",
                ]
            lines += ["</details>", ""]
        else:
            lines += ["<sub>No tool call — answered from the system instruction.</sub>", ""]
        lines += ["**Atendente**", "", turn.reply, "", "---", ""]
    return "\n".join(lines).rstrip() + "\n"


def _render_index(results: list[tuple[Scenario, Path]]) -> str:
    lines = [
        "# Interaction examples",
        "",
        "Generated with `python -m emporio.cli demo`, so they can be reproduced "
        "rather than taken on trust.",
        "",
        "Tool calls and their raw results are folded into each transcript. In a "
        "customer-facing product they would be hidden; here they are the point — "
        "they show whether the agent consulted the catalog, the policy manual, "
        "both, or neither.",
        "",
    ]
    for scenario, path in results:
        lines += [f"- [{scenario.title}]({path.name})  ", f"  {scenario.why}", ""]
    return "\n".join(lines).rstrip() + "\n"


def run_all(output: Path, console) -> list[tuple[Scenario, Path]]:
    """Run every scenario and write its transcript."""
    from emporio.agent import Conversation

    settings = get_settings()
    output.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Scenario, Path]] = []

    for scenario in SCENARIOS:
        console.print(f"\n[bold cyan]{scenario.title}[/bold cyan]")
        conversation = Conversation()
        turns = []
        for message in scenario.messages:
            console.print(f"  [dim]você ›[/dim] {message}")
            turn = conversation.send(message)
            for call in turn.invocations:
                console.print(f"    [yellow]⚙[/yellow] [dim]{call.summary()}[/dim]")
            first_line = turn.reply.splitlines()[0] if turn.reply else ""
            console.print(f"  [cyan]atendente ›[/cyan] {first_line[:88]}")
            turns.append(turn)

        path = output / f"{scenario.slug}.md"
        path.write_text(_render(scenario, turns, settings), encoding="utf-8", newline="\n")
        results.append((scenario, path))
        console.print(f"  [green]→[/green] {path.relative_to(output.parent)}")

    index = output / "README.md"
    index.write_text(_render_index(results), encoding="utf-8", newline="\n")
    console.print(f"\n[green]→[/green] {index.relative_to(output.parent)}")
    return results
