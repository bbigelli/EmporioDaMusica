"""Command-line interface.

Four commands:

    python -m emporio.cli              interactive chat
    python -m emporio.cli --lang pt    interactive chat, opening in Portuguese
    python -m emporio.cli ask "..."    one question, one answer
    python -m emporio.cli --resume last  continue the previous conversation
    python -m emporio.cli sessions     list recorded conversations
    python -m emporio.cli tools        inspect the data and policy layers
    python -m emporio.cli demo         run the scripted scenarios to examples/

Conversations are recorded to data/sessions.db so they can be reopened later;
`--ephemeral` skips that.

`--lang en|pt` sets the language the agent opens in; it still mirrors whatever
the customer actually writes. The CLI's own labels stay in English, since they
are developer surface — the Streamlit app has the full 🇧🇷 / 🇺🇸 switch.

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
from emporio.i18n import DEFAULT_LANGUAGE, FLAGS
from emporio.sessions import SessionStore

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


def _validate_language(lang: str) -> str:
    """Accept only the two supported codes, and say so rather than guessing."""
    lang = lang.strip().lower()
    if lang not in FLAGS:
        console.print(f"\n  [red]unknown language {lang!r}[/red]; use en or pt\n")
        raise typer.Exit(code=2)
    return lang


def _banner(language: str = DEFAULT_LANGUAGE) -> None:
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
        f"  [dim]opening in[/dim] {FLAGS[language]} "
        "[dim]· the agent replies in whichever language you write in[/dim]"
    )
    console.print("  [dim]/reset to start over · /exit to quit[/dim]\n")


def _replay(conversation) -> None:
    """Print a resumed conversation so the thread reads continuously."""
    if not conversation.transcript:
        return
    count = len(conversation.transcript)
    console.print(f"  [dim]— {count} earlier turn(s) —[/dim]\n")
    for turn in conversation.transcript:
        console.print(f"  [bold]you[/bold] [dim]›[/dim] {turn.user_message}")
        _show_reply(turn)


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
def chat(
    quiet: bool = typer.Option(False, "--quiet", help="Hide tool calls."),
    lang: str = typer.Option(
        DEFAULT_LANGUAGE, "--lang", help="Opening language: en or pt."
    ),
    resume: str = typer.Option(
        None, "--resume", help="Session id, short prefix, or 'last'."
    ),
    ephemeral: bool = typer.Option(
        False, "--ephemeral", help="Do not record this conversation."
    ),
) -> None:
    """Interactive conversation with the agent."""
    _require_key()
    from emporio.agent import Conversation

    lang = _validate_language(lang)
    store = None if ephemeral else SessionStore()

    if resume:
        if store is None:
            console.print("\n  [red]--resume cannot be combined with --ephemeral[/red]\n")
            raise typer.Exit(code=2)
        session_id = store.resolve(resume)
        if session_id is None:
            console.print(
                f"\n  [red]no single session matches {resume!r}[/red]"
                "  ·  list them with: python -m emporio.cli sessions\n"
            )
            raise typer.Exit(code=2)
        conversation = Conversation.resume(store, session_id, language=lang)
        _banner(lang)
        _replay(conversation)
    else:
        conversation = Conversation(language=lang, store=store)
        _banner(lang)

    if conversation.session_id:
        console.print(
            f"  [dim]session[/dim] {conversation.session_id.split('-')[0]}"
            "  [dim]· resume later with --resume last[/dim]\n"
        )

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
    lang: str = typer.Option(
        DEFAULT_LANGUAGE, "--lang", help="Opening language: en or pt."
    ),
) -> None:
    """Ask a single question and print the answer."""
    _require_key()
    from emporio.agent import Conversation

    turn = Conversation(language=_validate_language(lang)).send(question)
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
def sessions(limit: int = typer.Option(10, "--limit", help="How many to show.")) -> None:
    """List recorded conversations, most recent first."""
    recorded = SessionStore().list_recent(limit=limit)
    if not recorded:
        console.print("\n  [dim]no conversations recorded yet[/dim]\n")
        return

    table = Table(box=None, pad_edge=False, header_style="dim")
    table.add_column("id")
    table.add_column("turns", justify="right")
    table.add_column("updated")
    table.add_column("opened with")
    for item in recorded:
        preview = item.preview[:52] + ("…" if len(item.preview) > 52 else "")
        table.add_row(
            item.short_id,
            str(item.turn_count),
            item.updated_at.replace("T", " ")[:16],
            preview,
        )
    console.print()
    console.print(table)
    console.print("\n  [dim]resume with: python -m emporio.cli --resume <id>[/dim]\n")


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
        chat(quiet=False, lang=DEFAULT_LANGUAGE, resume=None, ephemeral=False)


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
