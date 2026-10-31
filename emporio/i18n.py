"""Interface strings in both languages.

The agent itself is bilingual by design — it mirrors whatever the customer
writes. This module is for the *chrome* around it: sidebar labels, buttons,
placeholders, the opening greeting. A reviewer reading English and a store
employee reading Portuguese want different ones, and neither should have to
read the other.

The selected language also seeds the agent: it opens in that language and
keeps mirroring the customer from there, so switching to 🇧🇷 and then writing
in English still gets an English reply.

Only interface copy lives here. Catalog data, the policy manual and the store's
own name stay as they are.
"""

from __future__ import annotations

from typing import Literal

Language = Literal["pt", "en"]

DEFAULT_LANGUAGE: Language = "en"

FLAGS: dict[Language, str] = {"pt": "🇧🇷", "en": "🇺🇸"}
NAMES: dict[Language, str] = {"pt": "Português", "en": "English"}


def option_label(language: Language) -> str:
    """'🇧🇷 Português' — used where only text can be rendered, like the CLI."""
    return f"{FLAGS[language]} {NAMES[language]}"


# Inline SVG flags for the web interface.
#
# Flag emoji are regional indicator pairs, and Windows has no glyph for them:
# Segoe UI Emoji falls back to drawing the two letters, so 🇧🇷 renders as a
# small "BR". That is fine in a terminal and wrong in a product, so the web UI
# draws the flags itself. Inline SVG also means no network fetch and no image
# files to ship.
#
# These are simplified: the US canton has no stars and the Brazilian globe has
# no banner. At 28x20 neither would be legible anyway.
_US_STRIPES = "".join(
    f'<rect y="{i * 20 / 13:.3f}" width="28" height="{20 / 13:.3f}" fill="#b22234"/>'
    for i in range(0, 13, 2)
)

FLAG_SVG: dict[Language, str] = {
    "pt": (
        '<svg width="28" height="20" viewBox="0 0 28 20" role="img" '
        'aria-label="Bandeira do Brasil">'
        '<rect width="28" height="20" fill="#009b3a"/>'
        '<polygon points="14,2.2 25.6,10 14,17.8 2.4,10" fill="#fedf00"/>'
        '<circle cx="14" cy="10" r="4.3" fill="#002776"/>'
        "</svg>"
    ),
    "en": (
        '<svg width="28" height="20" viewBox="0 0 28 20" role="img" '
        'aria-label="Flag of the United States">'
        '<rect width="28" height="20" fill="#ffffff"/>'
        f"{_US_STRIPES}"
        '<rect width="11.9" height="10.77" fill="#3c3b6e"/>'
        "</svg>"
    ),
}


STRINGS: dict[str, dict[Language, str]] = {
    "page_title": {
        "pt": "Empório da Música · Atendimento",
        "en": "Empório da Música · Customer service",
    },
    "tagline": {
        "pt": "*Sua música começa aqui.* · Atendimento virtual",
        "en": "*Sua música começa aqui.* · Virtual customer service agent",
    },
    "greeting": {
        "pt": (
            "Oi! Aqui é o atendimento da Empório da Música. 🎸\n\n"
            "Posso ajudar com preços, disponibilidade, status de pedido, trocas, "
            "frete e condições de pagamento. O que você precisa?"
        ),
        "en": (
            "Hi! This is Empório da Música customer service. 🎸\n\n"
            "I can help with prices, availability, order status, returns, shipping "
            "and payment terms. What do you need?"
        ),
    },
    "settings": {"pt": "Configuração", "en": "Configuration"},
    "language": {"pt": "Idioma", "en": "Language"},
    "model": {"pt": "Modelo", "en": "Model"},
    "reference_date": {"pt": "Data de referência", "en": "Reference date"},
    "key_missing": {"pt": "Chave de API não configurada", "en": "No API key configured"},
    "key_ok": {"pt": "Chave de API configurada", "en": "API key configured"},
    "key_steps": {
        "pt": (
            "1. Gere uma chave no "
            "[Google AI Studio](https://aistudio.google.com/apikey)\n"
            "2. Copie `.env.example` para `.env`\n"
            "3. Preencha `GEMINI_API_KEY`\n"
            "4. Reinicie o app"
        ),
        "en": (
            "1. Create a key at "
            "[Google AI Studio](https://aistudio.google.com/apikey)\n"
            "2. Copy `.env.example` to `.env`\n"
            "3. Fill in `GEMINI_API_KEY`\n"
            "4. Restart the app"
        ),
    },
    "new_conversation": {"pt": "Nova conversa", "en": "New conversation"},
    "tool_calls_note": {
        "pt": (
            "As chamadas de ferramenta aparecem em cada resposta. Um produto real "
            "as esconderia; aqui são o ponto interessante — mostram se o agente "
            "consultou o catálogo, o manual de políticas, ambos ou nenhum."
        ),
        "en": (
            "Tool calls are shown with every reply. A real product would hide "
            "them; here they are the interesting part — they show whether the "
            "agent consulted the catalog, the policy manual, both, or neither."
        ),
    },
    "bilingual_note": {
        "pt": (
            "O agente responde no idioma em que você escrever, independente desta "
            "seleção."
        ),
        "en": (
            "The agent replies in whichever language you write in, regardless of "
            "this setting."
        ),
    },
    "session": {"pt": "Sessão", "en": "Session"},
    "persistence_note": {
        "pt": (
            "A conversa é gravada e pode ser retomada depois pela linha de "
            "comando: `python -m emporio.cli --resume last`."
        ),
        "en": (
            "This conversation is recorded and can be reopened later from the "
            "command line: `python -m emporio.cli --resume last`."
        ),
    },
    "no_tool_call": {
        "pt": "_respondido sem consultar o sistema_",
        "en": "_answered without consulting the system_",
    },
    "input_placeholder": {"pt": "Digite sua mensagem...", "en": "Type your message..."},
    "input_disabled": {
        "pt": "Configure uma chave de API para conversar",
        "en": "Configure an API key to start chatting",
    },
    "thinking": {"pt": "Consultando...", "en": "Consulting..."},
    "request_failed": {
        "pt": "Não consegui concluir a consulta:",
        "en": "Could not complete the request:",
    },
}


def t(key: str, language: Language) -> str:
    """Look up an interface string."""
    try:
        return STRINGS[key][language]
    except KeyError as exc:  # a missing key is a bug, not a runtime condition
        raise KeyError(f"no {language!r} string for {key!r}") from exc
