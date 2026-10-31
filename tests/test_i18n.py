"""Tests for the interface language switch."""

from __future__ import annotations

import re
import unittest

from emporio import i18n


class StringTableTests(unittest.TestCase):
    def test_every_key_exists_in_both_languages(self) -> None:
        """A half-translated table shows a Portuguese label in an English UI,
        which is exactly the bug the switch exists to remove."""
        for key, translations in i18n.STRINGS.items():
            self.assertEqual({"pt", "en"}, set(translations), f"{key} is incomplete")
            for language, text in translations.items():
                self.assertTrue(text.strip(), f"{key}/{language} is empty")

    def test_the_two_languages_actually_differ(self) -> None:
        """Catches a copy-paste that leaves one language holding the other's
        text. The tagline is exempt: it embeds the store's Portuguese slogan."""
        exempt = {"tagline"}
        for key, translations in i18n.STRINGS.items():
            if key in exempt:
                continue
            self.assertNotEqual(
                translations["pt"], translations["en"], f"{key} is identical in both"
            )

    def test_portuguese_strings_keep_their_accents(self) -> None:
        """Dropping accents from customer-facing text is a product defect."""
        joined = " ".join(value["pt"] for value in i18n.STRINGS.values())
        self.assertTrue(re.search(r"[áàâãéêíóôõúç]", joined))
        for wrong in ("Configuracao", "nao ", "Portugues", "Idioma de resposta"):
            self.assertNotIn(wrong, joined)

    def test_english_strings_carry_no_stray_portuguese(self) -> None:
        """The slogan in the tagline is deliberate; nothing else should be."""
        for key, translations in i18n.STRINGS.items():
            if key == "tagline":
                continue
            text = translations["en"]
            self.assertFalse(
                re.search(r"\b(você|não|chave|idioma|conversa|mensagem)\b", text.lower()),
                f"{key}/en still reads Portuguese: {text!r}",
            )

    def test_selector_labels_carry_the_flags(self) -> None:
        self.assertEqual("🇧🇷 Português", i18n.option_label("pt"))
        self.assertEqual("🇺🇸 English", i18n.option_label("en"))

    def test_lookup_rejects_an_unknown_key(self) -> None:
        with self.assertRaises(KeyError):
            i18n.t("no_such_key", "en")

    def test_greetings_introduce_the_same_capabilities(self) -> None:
        """Both greetings should set the same expectations, not a richer offer
        in one language than the other."""
        for language, expected in (
            ("pt", ("preços", "pedido", "frete", "pagamento")),
            ("en", ("prices", "order", "shipping", "payment")),
        ):
            greeting = i18n.t("greeting", language).lower()
            for term in expected:
                self.assertIn(term, greeting, f"{language} greeting lacks {term}")


class SystemInstructionLanguageTests(unittest.TestCase):
    def test_selected_language_is_passed_to_the_model(self) -> None:
        from emporio.prompts import build_system_instruction

        self.assertIn("Brazilian Portuguese, so open", build_system_instruction("pt"))
        self.assertIn("English, so open", build_system_instruction("en"))

    def test_no_selection_leaves_pure_mirroring(self) -> None:
        from emporio.prompts import build_system_instruction

        self.assertNotIn("# Selected language", build_system_instruction(None))

    def test_selection_does_not_override_mirroring(self) -> None:
        """Someone who sets the UI to Portuguese and then writes in English
        still wants an English answer."""
        from emporio.prompts import build_system_instruction

        instruction = build_system_instruction("pt")
        self.assertIn("does not override the rule above", instruction)
        self.assertIn("Match the language of the customer's most recent message", instruction)

    def test_conversation_accepts_a_language(self) -> None:
        from emporio.agent import Conversation

        class _StubInteractions:
            def create(self, **kwargs):  # pragma: no cover - never called here
                raise AssertionError

        class _StubClient:
            interactions = _StubInteractions()

        convo = Conversation(client=_StubClient(), language="en")
        self.assertIn("English, so open", convo.system_instruction)


if __name__ == "__main__":
    unittest.main()
