"""Streamlit chat interface.

    streamlit run app.py

The CLI is the primary interface; this exists so the agent's decisions are
visible while using it. Every reply shows which tools ran and what they
returned, which is the part worth watching: whether the agent reached for the
catalog, the policy manual, both, or neither.

The 🇧🇷 / 🇺🇸 selector in the sidebar switches the interface copy and seeds the
agent's opening language. It does not lock the agent: it still mirrors
whatever language the customer actually writes in.
"""

from __future__ import annotations

import json

import streamlit as st

from emporio.config import get_settings
from emporio.i18n import DEFAULT_LANGUAGE, Language, option_label, t

LANGUAGE_OPTIONS: list[Language] = ["en", "pt"]

st.set_page_config(
    page_title="Empório da Música · Customer service", page_icon="🎸", layout="centered"
)


def md(text: str) -> str:
    """Escape dollar signs before handing text to st.markdown.

    Streamlit renders `$...$` as LaTeX. Every reply here is full of prices, so
    "R$ 1.967,18 (de R$ 2.399,00)" has its middle swallowed into a maths span
    and the currency symbols vanish. Escaping is the documented way out.
    """
    return text.replace("$", r"\$")


def language_selector() -> Language:
    """The 🇧🇷 / 🇺🇸 switch. Changing it restarts the conversation.

    The selected language is baked into the agent's system instruction, so an
    in-flight conversation cannot adopt it retroactively — clearing the thread
    is more honest than leaving the greeting in one language and the rest in
    another.
    """
    previous = st.session_state.get("language", DEFAULT_LANGUAGE)
    chosen = st.radio(
        t("language", previous),
        options=LANGUAGE_OPTIONS,
        format_func=option_label,
        index=LANGUAGE_OPTIONS.index(previous),
        horizontal=True,
        label_visibility="collapsed",
    )
    if chosen != previous:
        st.session_state.language = chosen
        st.session_state.pop("conversation", None)
        st.session_state.pop("messages", None)
        st.rerun()

    st.session_state.language = chosen
    return chosen


def sidebar() -> tuple[Language, bool]:
    """Render the sidebar. Returns the language and whether the agent can run."""
    settings = get_settings()
    with st.sidebar:
        language = language_selector()

        st.subheader(t("settings", language))
        st.caption(f"**{t('model', language)}**  \n`{settings.model}`")
        st.caption(f"**{t('reference_date', language)}**  \n`{settings.reference_date}`")

        if not settings.api_key:
            st.error(t("key_missing", language))
            st.markdown(t("key_steps", language))
            return language, False

        st.success(t("key_ok", language))
        if st.button(t("new_conversation", language), use_container_width=True):
            st.session_state.pop("conversation", None)
            st.session_state.pop("messages", None)
            st.rerun()

        st.divider()
        st.caption(t("tool_calls_note", language))
        st.caption(t("bilingual_note", language))
        return language, True


def render_invocations(invocations, language: Language) -> None:
    if not invocations:
        st.caption(t("no_tool_call", language))
        return
    for call in invocations:
        icon = "🔴" if call.is_error else "⚙️"
        with st.expander(f"{icon} `{call.summary()}`", expanded=False):
            st.json(json.loads(json.dumps(call.result, ensure_ascii=False, default=str)))


def main() -> None:
    language, ready = sidebar()

    st.title("Empório da Música")
    st.caption(t("tagline", language))

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": t("greeting", language), "invocations": []}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(md(message["content"]))
            if message["role"] == "assistant" and message.get("invocations"):
                render_invocations(message["invocations"], language)

    if not ready:
        st.chat_input(t("input_disabled", language), disabled=True)
        return

    prompt = st.chat_input(t("input_placeholder", language))
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt, "invocations": []})
    with st.chat_message("user"):
        st.markdown(md(prompt))

    if "conversation" not in st.session_state:
        from emporio.agent import Conversation

        st.session_state.conversation = Conversation(language=language)

    with st.chat_message("assistant"):
        with st.spinner(t("thinking", language)):
            try:
                turn = st.session_state.conversation.send(prompt)
            except Exception as exc:  # noqa: BLE001 - shown to the user, not raised
                st.error(f"{t('request_failed', language)} {exc}")
                return
        st.markdown(md(turn.reply))
        render_invocations(turn.invocations, language)

    st.session_state.messages.append(
        {"role": "assistant", "content": turn.reply, "invocations": turn.invocations}
    )


main()
