"""Streamlit chat interface.

    streamlit run app.py

The CLI is the primary interface; this exists so the agent's decisions are
visible while using it. Every reply shows which tools ran and what they
returned, which is the part worth watching: whether the agent reached for the
catalog, the policy manual, both, or neither.
"""

from __future__ import annotations

import json

import streamlit as st

from emporio.config import get_settings

st.set_page_config(page_title="Empório da Música · Atendimento", page_icon="🎸", layout="centered")

GREETING = (
    "Oi! Aqui é o atendimento da Empório da Música. 🎸\n\n"
    "Posso ajudar com preços, disponibilidade, status de pedido, trocas, frete "
    "e as condições de pagamento. O que você precisa?"
)


def sidebar() -> bool:
    """Render the sidebar. Returns whether the agent can run."""
    settings = get_settings()
    with st.sidebar:
        st.subheader("Configuração")
        st.caption(f"**Modelo**  \n`{settings.model}`")
        st.caption(f"**Data de referência**  \n`{settings.reference_date}`")

        if not settings.api_key:
            st.error("Chave de API não configurada")
            st.markdown(
                "1. Gere uma chave no "
                "[Google AI Studio](https://aistudio.google.com/apikey)\n"
                "2. Copie `.env.example` para `.env`\n"
                "3. Preencha `GEMINI_API_KEY`\n"
                "4. Reinicie o app"
            )
            return False

        st.success("Chave de API configurada")
        if st.button("Nova conversa", use_container_width=True):
            st.session_state.pop("conversation", None)
            st.session_state.pop("messages", None)
            st.rerun()

        st.divider()
        st.caption(
            "As chamadas de ferramenta aparecem em cada resposta. Num produto "
            "real ficariam escondidas; aqui são o ponto interessante."
        )
        return True


def render_invocations(invocations) -> None:
    if not invocations:
        st.caption("_respondido sem consultar o sistema_")
        return
    for call in invocations:
        icon = "🔴" if call.is_error else "⚙️"
        with st.expander(f"{icon} `{call.summary()}`", expanded=False):
            st.json(json.loads(json.dumps(call.result, ensure_ascii=False, default=str)))


def main() -> None:
    st.title("Empório da Música")
    st.caption("Sua música começa aqui. · Atendimento virtual")

    ready = sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": GREETING, "invocations": []}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("invocations") is not None:
                if message["invocations"]:
                    render_invocations(message["invocations"])

    if not ready:
        st.chat_input("Configure a chave de API para conversar", disabled=True)
        return

    prompt = st.chat_input("Digite sua mensagem...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt, "invocations": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "conversation" not in st.session_state:
        from emporio.agent import Conversation

        st.session_state.conversation = Conversation()

    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            try:
                turn = st.session_state.conversation.send(prompt)
            except Exception as exc:  # noqa: BLE001 - shown to the user, not raised
                st.error(f"Não consegui concluir a consulta: {exc}")
                return
        st.markdown(turn.reply)
        render_invocations(turn.invocations)

    st.session_state.messages.append(
        {"role": "assistant", "content": turn.reply, "invocations": turn.invocations}
    )


main()
