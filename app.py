"""Streamlit chat interface.

    streamlit run app.py

The CLI is the primary interface; this exists so the agent's decisions are
visible while using it. Every reply shows which tools ran and what they
returned, which is the part worth watching: whether the agent reached for the
catalog, the policy manual, both, or neither.

The application chrome is in English, like the rest of the repository. What
the *agent* says is a separate matter: it mirrors the customer's language, so
a Portuguese message gets a Portuguese reply. The opening greeting is in
Portuguese because that is the store's home language and the agent cannot yet
know what the customer speaks.
"""

from __future__ import annotations

import json

import streamlit as st

from emporio.config import get_settings

st.set_page_config(
    page_title="Empório da Música · Customer service", page_icon="🎸", layout="centered"
)

GREETING = (
    "Oi! Aqui é o atendimento da Empório da Música. 🎸\n\n"
    "Posso ajudar com preços, disponibilidade, status de pedido, trocas, frete "
    "e as condições de pagamento. O que você precisa?\n\n"
    "_Feel free to write in English — I'll reply in whichever language you use._"
)


def md(text: str) -> str:
    """Escape dollar signs before handing text to st.markdown.

    Streamlit renders `$...$` as LaTeX. Every reply here is full of prices, so
    "R$ 1.967,18 (de R$ 2.399,00)" has its middle swallowed into a maths span
    and the currency symbols vanish. Escaping is the documented way out.
    """
    return text.replace("$", r"\$")


def sidebar() -> bool:
    """Render the sidebar. Returns whether the agent can run."""
    settings = get_settings()
    with st.sidebar:
        st.subheader("Configuration")
        st.caption(f"**Model**  \n`{settings.model}`")
        st.caption(f"**Reference date**  \n`{settings.reference_date}`")

        if not settings.api_key:
            st.error("No API key configured")
            st.markdown(
                "1. Create a key at "
                "[Google AI Studio](https://aistudio.google.com/apikey)\n"
                "2. Copy `.env.example` to `.env`\n"
                "3. Fill in `GEMINI_API_KEY`\n"
                "4. Restart the app"
            )
            return False

        st.success("API key configured")
        if st.button("New conversation", use_container_width=True):
            st.session_state.pop("conversation", None)
            st.session_state.pop("messages", None)
            st.rerun()

        st.divider()
        st.caption(
            "Tool calls are shown with every reply. A real product would hide "
            "them; here they are the interesting part — they show whether the "
            "agent consulted the catalog, the policy manual, both, or neither."
        )
        st.caption(
            "The agent answers in the customer's language. Try it in English "
            "or Portuguese."
        )
        return True


def render_invocations(invocations) -> None:
    if not invocations:
        st.caption("_answered without consulting the system_")
        return
    for call in invocations:
        icon = "🔴" if call.is_error else "⚙️"
        with st.expander(f"{icon} `{call.summary()}`", expanded=False):
            st.json(json.loads(json.dumps(call.result, ensure_ascii=False, default=str)))


def main() -> None:
    st.title("Empório da Música")
    st.caption("*Sua música começa aqui.* · Virtual customer service agent")

    ready = sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": GREETING, "invocations": []}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(md(message["content"]))
            if message["role"] == "assistant" and message.get("invocations"):
                render_invocations(message["invocations"])

    if not ready:
        st.chat_input("Configure an API key to start chatting", disabled=True)
        return

    prompt = st.chat_input("Type your message...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt, "invocations": []})
    with st.chat_message("user"):
        st.markdown(md(prompt))

    if "conversation" not in st.session_state:
        from emporio.agent import Conversation

        st.session_state.conversation = Conversation()

    with st.chat_message("assistant"):
        with st.spinner("Consulting..."):
            try:
                turn = st.session_state.conversation.send(prompt)
            except Exception as exc:  # noqa: BLE001 - shown to the user, not raised
                st.error(f"Could not complete the request: {exc}")
                return
        st.markdown(md(turn.reply))
        render_invocations(turn.invocations)

    st.session_state.messages.append(
        {"role": "assistant", "content": turn.reply, "invocations": turn.invocations}
    )


main()
