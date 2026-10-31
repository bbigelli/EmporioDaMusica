"""Tests for the agent's tool surface and policy retrieval.

The reference date is pinned here so a developer's local .env cannot change
what these tests assert.
"""

from __future__ import annotations

import os
import unittest

from emporio.config import get_settings


def setUpModule() -> None:
    os.environ["EMPORIO_REFERENCE_DATE"] = "2026-03-25"
    get_settings.cache_clear()


class MoneyTests(unittest.TestCase):
    def test_brazilian_currency_format(self) -> None:
        """R$ 1.967,18 and R$ 1,967.18 differ by a factor of a thousand to a
        Brazilian reader, so this is not cosmetic."""
        from emporio.money import brl

        self.assertEqual("R$ 1.967,18", brl(1967.18))
        self.assertEqual("R$ 429,90", brl(429.9))
        self.assertEqual("R$ 12.345,67", brl(12345.67))
        self.assertEqual("R$ 1.234.567,89", brl(1234567.89))
        self.assertEqual("R$ 0,00", brl(0.0))
        self.assertIsNone(brl(None))

    def test_money_rounds_half_up_not_to_even(self) -> None:
        """A 5% PIX discount on R$ 699,90 is 664.905. Python's round() gives
        664.90; every Brazilian till gives 664.91."""
        from emporio.money import round_brl

        self.assertEqual(664.91, round_brl(664.905))
        self.assertEqual(0.01, round_brl(0.005))
        self.assertEqual(2.68, round_brl(2.675))
        self.assertEqual(459.92, round_brl(459.9233))

    def test_pix_discount_uses_commercial_rounding(self) -> None:
        from emporio import tools

        product = tools.get_product_details(name="Yamaha F310")["product"]
        self.assertEqual("R$ 664,91", product["payment"]["pix_total_display"])


class CatalogToolTests(unittest.TestCase):
    def test_price_filter_applies_to_the_promotional_price(self) -> None:
        """A discounted item that falls under the customer's ceiling must be
        found, otherwise the promotion is invisible to the people it targets."""
        from emporio import tools

        result = tools.search_products(query_text="Ohana CK-20", max_price_brl=500, limit=5)
        names = [item["name"] for item in result["results"]]
        self.assertTrue(
            any("Ohana CK-20" in name for name in names),
            "list price is 549.00; with the active 20% discount it is 439.20",
        )

    def test_unavailable_matches_are_withheld_but_reported(self) -> None:
        """Section 7.3 wants out-of-stock and discontinued handled distinctly,
        so they are filtered out of results yet still surfaced, rather than
        vanishing as though the catalog were smaller."""
        from emporio import tools

        result = tools.search_products(category="Violões", max_price_brl=1000, limit=25)
        self.assertTrue(all(item["availability"] == "available" for item in result["results"]))
        withheld = result["withheld"]
        self.assertEqual(2, withheld["count"])
        reasons = {item["availability"] for item in withheld["items"]}
        self.assertEqual({"out_of_stock", "discontinued"}, reasons)

    def test_accent_insensitive_text_search(self) -> None:
        from emporio import tools

        with_accent = tools.search_products(query_text="violão nylon", limit=25)
        without = tools.search_products(query_text="violao nylon", limit=25)
        self.assertEqual(with_accent["total_matches"], without["total_matches"])
        self.assertGreater(without["total_matches"], 0)

    def test_unknown_category_lists_the_real_ones(self) -> None:
        from emporio import tools

        result = tools.search_products(category="Pedais")
        self.assertEqual("unknown_category", result["status"])
        self.assertIn("Violões", result["known_categories"])

    def test_ambiguous_product_name_returns_candidates(self) -> None:
        """Silently picking one match is how a customer gets quoted the price
        of a different instrument."""
        from emporio import tools

        result = tools.get_product_details(name="Takamine")
        self.assertEqual("ambiguous", result["status"])
        self.assertEqual(4, len(result["candidates"]))

    def test_exact_product_name_resolves(self) -> None:
        from emporio import tools

        product = tools.get_product_details(name="Takamine GD20")["product"]
        self.assertEqual("Takamine GD20 Dreadnought Natural", product["name"])
        self.assertEqual("R$ 2.199,00", product["price_display"])

    def test_missing_product_is_not_invented(self) -> None:
        from emporio import tools

        # Roland and Stradivarius are absent from the catalog; Fender, Gibson
        # and Taylor are present, so a plausible-sounding brand is not enough
        # to make this test meaningful.
        self.assertEqual("not_found", tools.get_product_details(name="Roland Jupiter-8")["status"])
        self.assertEqual("not_found", tools.get_product_details(product_id=99999)["status"])

    def test_promotional_product_shows_original_price_and_percentage(self) -> None:
        from emporio import tools

        product = tools.get_product_details(name="Crafter HT-100")["product"]
        self.assertTrue(product["on_promotion"])
        self.assertIn("R$ 2.399,00", product["price_presentation"])
        self.assertIn("18%", product["price_presentation"])
        self.assertFalse(product["payment"]["pix_discount_applies"])
        self.assertEqual("R$ 1.967,18", product["payment"]["pix_total_display"])

    def test_only_active_promotions_are_listed(self) -> None:
        from emporio import tools

        result = tools.get_active_promotions()
        self.assertEqual(4, result["count"])
        self.assertTrue(
            all("de R$" in item["price_presentation"] for item in result["promotions"])
        )


class OrderToolTests(unittest.TestCase):
    def test_undispatched_order_explains_the_absent_tracking(self) -> None:
        from emporio import tools

        order = tools.get_order_status(13)
        self.assertFalse(order["dispatched"])
        self.assertIsNone(order["tracking_code"])
        self.assertIn("ainda não foi despachado", order["tracking_note"])

    def test_dispatched_order_has_tracking(self) -> None:
        from emporio import tools

        order = tools.get_order_status(8)
        self.assertTrue(order["dispatched"])
        self.assertTrue(order["tracking_code"].startswith("BR"))

    def test_order_includes_its_items(self) -> None:
        from emporio import tools

        order = tools.get_order_status(7)
        self.assertTrue(order["items"])
        self.assertIn("unit_price_display", order["items"][0])

    def test_missing_order_is_not_invented(self) -> None:
        from emporio import tools

        self.assertEqual("not_found", tools.get_order_status(9999)["status"])

    def test_delivered_order_past_its_windows_still_has_warranty(self) -> None:
        """The non-trivial case end to end: order 7 is past withdrawal, defect
        and preference deadlines but inside the 90-day legal warranty."""
        from emporio import tools

        result = tools.check_return_eligibility(7)
        self.assertEqual(["legal_warranty"], result["available_routes"])
        self.assertEqual(4, len(result["windows"]))


class ShippingToolTests(unittest.TestCase):
    def test_metropolitan_free_shipping(self) -> None:
        from emporio import tools

        quote = tools.quote_shipping(600.0, city="Campo Grande")
        self.assertEqual("R$ 0,00", quote["fee_display"])

    def test_other_cities_get_carriers_and_no_amount(self) -> None:
        from emporio import tools

        quote = tools.quote_shipping(600.0, city="Dourados")
        self.assertIsNone(quote["fee_brl"])
        self.assertNotIn("fee_display", quote)
        self.assertEqual(3, len(quote["carriers"]))


class PolicyRetrievalTests(unittest.TestCase):
    """Retrieval is graded on recall at 3, because the agent sees three
    sections and chooses. Top-1 precision is reported in the README."""

    CASES = (
        ("qual o horário de funcionamento?", "2"),
        ("vocês abrem no domingo?", "2"),
        ("what are your opening hours?", "2"),
        ("me arrependi da compra, posso devolver?", "4.1"),
        ("can I return my order?", "4.1"),
        ("o produto chegou com defeito", "4.2"),
        ("meu violão quebrou, tem garantia?", "8.1"),
        ("quanto custa o frete para Dourados?", "5.2"),
        ("frete grátis a partir de quanto?", "5.1"),
        ("aceita boleto?", "3"),
        ("posso parcelar em 10x?", "3.1"),
        ("qual o endereço da loja?", "1.2"),
        ("como rastreio meu pedido?", "5.3"),
        ("vendem cordas de violão?", "1"),
        ("do you accept credit card instalments?", "3.1"),
    )

    def test_recall_at_three(self) -> None:
        from emporio import policies

        for question, expected in self.CASES:
            found = [hit.section.id for hit in policies.search(question, limit=3)]
            self.assertIn(expected, found, f"{question!r} returned {found}")

    def test_portuguese_morphology_is_bridged_by_prefix_match(self) -> None:
        """"parcelar" appears nowhere in the manual, which says "Parcelamento"
        and "parcela". Before prefix matching this query scored zero hits."""
        from emporio import policies

        hits = policies.search("posso parcelar em 10x?", limit=3)
        self.assertTrue(hits)
        self.assertEqual("3.1", hits[0].section.id)

    def test_ingestion_notes_are_excluded_from_search(self) -> None:
        """They are instructions for the agent, not an answer to a customer,
        and their brevity made BM25 rank them above real sections."""
        from emporio import policies

        for question, _ in self.CASES:
            ids = [hit.section.id for hit in policies.search(question, limit=5)]
            self.assertNotIn("notes", ids, question)
        self.assertIn("Tamanho do catálogo", policies.pipeline_notes())

    def test_exact_term_outranks_a_prefix_match(self) -> None:
        from emporio import policies

        terms = policies.build_query_terms("garantia")
        self.assertEqual(policies.WEIGHT_VERBATIM, terms["garantia"])

    def test_empty_query_returns_nothing_rather_than_everything(self) -> None:
        from emporio import policies

        self.assertEqual([], policies.search("de a o"))


class DispatchTests(unittest.TestCase):
    def test_unknown_tool_is_reported_as_data(self) -> None:
        from emporio import tools

        result = tools.call("delete_everything", {})
        self.assertEqual("unknown_tool", result["status"])
        self.assertIn("search_products", result["available"])

    def test_bad_arguments_are_reported_not_raised(self) -> None:
        """A malformed call is the model's mistake to correct on the next turn,
        so it comes back as a tool result instead of killing the loop."""
        from emporio import tools

        result = tools.call("get_order_status", {"nonsense": 1})
        self.assertEqual("invalid_arguments", result["status"])

    def test_every_declaration_has_an_implementation(self) -> None:
        from emporio import tools

        declared = {item["name"] for item in tools.DECLARATIONS}
        self.assertEqual(set(tools.REGISTRY), declared)

    def test_declarations_describe_their_parameters(self) -> None:
        """An undocumented parameter is one the model will either ignore or
        misuse."""
        from emporio import tools

        for declaration in tools.DECLARATIONS:
            self.assertTrue(declaration["description"].strip(), declaration["name"])
            properties = declaration["parameters"].get("properties", {})
            for name, schema in properties.items():
                self.assertIn("type", schema, f"{declaration['name']}.{name}")
                self.assertTrue(
                    schema.get("description", "").strip(),
                    f"{declaration['name']}.{name} has no description",
                )


if __name__ == "__main__":
    unittest.main()
