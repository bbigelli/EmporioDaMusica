"""Command-line interface.

Four commands:

    python -m emporio.cli              interactive chat
    python -m emporio.cli ask "..."    one question, one answer
    python -m emporio.cli tools        inspect the data and policy layers
    python -m emporio.cli demo         run the scripted scenarios to examples/

The interface chrome is in English, like the rest of the repository. What the
agent says is a separate matter: it mirrors the customer's language.

Tool calls are shown inline by default. For a customer-facing product they
would be hidden, but the point of this prototype is to make the agent's
decisions legible — whether it reached for the catalog or the policy manual is
the interesting part. `--quiet` hides them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from emporio.config import ROOT, get_settings

app = typer.Typer(
    add_completion=False,
    help="Customer service agent for Empório da Música.",
    invoke_without_command=True,
)


def _force_utf8() -> None:
    """Make stdout UTF-8 regardless of the console's code page.

    Windows consoles default to a legacy code page (1252 here), which turns
    every accented character in the catalog and the policy manual into mojibake
    or a UnicodeEncodeError. Dropping accents from customer-facing text is a
    product defect, so this is not optional.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # already detached or redirected
                pass


_force_utf8()
console = Console()

# Both languages are accepted, since the person at the keyboard may be either
# the store's team or an English-speaking reviewer.
EXIT_WORDS = {"/exit", "/quit", "/sair", "exit", "quit", "sair"}
RESET_WORDS = {"/reset", "/clear", "/limpar"}


def _banner() -> None:
    settings = get_settings()
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("Empório da Música", "bold"),
                ("  ·  ", "dim"),
                ("Customer service", "cyan"),
                ("\nSua música começa aqui.", "italic dim"),
            ),
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print(
        f"  [dim]model[/dim] {settings.model}   "
        f"[dim]reference date[/dim] {settings.reference_date}",
    )
    console.print(
        "  [dim]the agent replies in whichever language you write in[/dim]"
    )
    console.print("  [dim]/reset to start over · /exit to quit[/dim]\n")


def _show_invocations(turn, quiet: bool) -> None:
    if quiet or not turn.invocations:
        return
    for call in turn.invocations:
        style = "red" if call.is_error else "yellow"
        marker = "✗" if call.is_error else "⚙"
        console.print(f"  [{style}]{marker}[/{style}] [dim]{call.summary()}[/dim]")


def _show_reply(turn) -> None:
    console.print()
    console.print("  [bold cyan]Agent[/bold cyan]")
    for line in turn.reply.splitlines():
        console.print(f"  {line}" if line.strip() else "")
    console.print()


def _require_key() -> None:
    settings = get_settings()
    if settings.api_key:
        return
    console.print()
    console.print(
        Panel(
            "No API key found.\n\n"
            "1. Create a key at [link]https://aistudio.google.com/apikey[/link]\n"
            "2. Copy [bold].env.example[/bold] to [bold].env[/bold]\n"
            "3. Fill in [bold]GEMINI_API_KEY[/bold]\n\n"
            "The catalog and policy layers work without a key:\n"
            "[dim]python -m emporio.cli tools[/dim]",
            title="Configuration needed",
            border_style="red",
            padding=(1, 2),
        )
    )
    raise typer.Exit(code=1)


@app.command()
def chat(quiet: bool = typer.Option(False, "--quiet", help="Hide tool calls.")) -> None:
    """Interactive conversation with the agent."""
    _require_key()
    from emporio.agent import Conversation

    _banner()
    conversation = Conversation()

    while True:
        try:
            message = console.input("[bold]you[/bold] [dim]›[/dim] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]goodbye![/dim]\n")
            return

        if not message:
            continue
        if message.lower() in EXIT_WORDS:
            console.print("\n  [dim]goodbye![/dim]\n")
            return
        if message.lower() in RESET_WORDS:
            conversation.reset()
            console.print("  [dim]conversation reset[/dim]\n")
            continue

        try:
            with console.status("[dim]consulting...[/dim]", spinner="dots"):
                turn = conversation.send(message)
        except Exception as exc:  # noqa: BLE001 - a CLI should not traceback at a user
            console.print(f"\n  [red]error:[/red] {exc}\n")
            continue

        _show_invocations(turn, quiet)
        _show_reply(turn)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The customer's question."),
    quiet: bool = typer.Option(False, "--quiet", help="Hide tool calls."),
) -> None:
    """Ask a single question and print the answer."""
    _require_key()
    from emporio.agent import Conversation

    turn = Conversation().send(question)
    _show_invocations(turn, quiet)
    _show_reply(turn)


@app.command()
def tools(
    question: str = typer.Option(None, "--policy", help="Policy question to look up.")
) -> None:
    """Inspect the data and policy layers directly, with no model or API key.

    Useful for checking that the pipelines produced what the agent will see.
    """
    from emporio import policies
    from emporio import tools as toolset

    settings = get_settings()
    console.print(Rule("[bold]data layer[/bold]"))

    promotions = toolset.get_active_promotions()
    table = Table(box=None, pad_edge=False, header_style="dim")
    table.add_column("active promotion")
    table.add_column("price", justify="right")
    for item in promotions["promotions"]:
        table.add_row(item["product_name"], item["price_presentation"])
    console.print(table)

    console.print()
    console.print(f"[dim]reference date:[/dim] {settings.reference_date}")
    eligibility = toolset.check_return_eligibility(7)
    console.print(f"[dim]order 7 return routes:[/dim] {eligibility['summary']}")

    if question:
        console.print()
        console.print(Rule("[bold]policy layer[/bold]"))
        result = toolset.search_store_policies(question)
        for section in result["sections"]:
            console.print(
                f"\n[bold]{section['title']}[/bold] [dim](score {section['score']})[/dim]"
            )
            console.print(Markdown(section["text"][:400]))
    else:
        console.print()
        console.print(
            f"[dim]{len(policies.load_sections())} policy sections indexed · "
            'use --policy "your question" to search[/dim]'
        )


@app.command()
def demo(
    output: Path = typer.Option(
        ROOT / "examples", "--output", help="Directory for the generated transcripts."
    ),
) -> None:
    """Run the scripted scenarios and write them to examples/."""
    _require_key()
    from emporio.scenarios import run_all

    run_all(output, console)


@app.callback()
def main(ctx: typer.Context) -> None:
    """Default to interactive chat when no command is given."""
    if ctx.invoked_subcommand is None:
        chat(quiet=False)


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
