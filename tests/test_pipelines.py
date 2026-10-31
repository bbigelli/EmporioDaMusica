"""Tests for the data and policy build pipelines.

Uses unittest from the standard library rather than pytest: the pipelines have
no test-only dependencies and `python -m unittest` works on a bare checkout.

Run from the repository root, after both build scripts:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    """Import a file from scripts/ as a module.

    The module must be registered in sys.modules before execution, otherwise
    @dataclass cannot resolve its own module and raises.
    """
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PolicyExtractionTests(unittest.TestCase):
    """The policy document feeds customer-facing answers, so its content is
    pinned by value, not merely checked for existence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.build_policies = load_script("build_policies")
        cls.source = cls.build_policies.extract_text()
        cls.markdown = (ROOT / "data" / "processed" / "store_policies.md").read_text(
            encoding="utf-8"
        )
        cls.sections = json.loads(
            (ROOT / "data" / "processed" / "policy_sections.json").read_text(encoding="utf-8")
        )

    def test_curated_tables_match_the_pdf(self) -> None:
        self.assertEqual([], self.build_policies.verify_curated_tables(self.source))

    def test_changed_pdf_fails_verification(self) -> None:
        """A silent mismatch is the failure mode that matters. If the source
        document changes and verification still passes, the build would ship a
        stale policy table."""
        tampered = self.source.replace("09:00 às 18:00", "10:00 às 19:00")
        self.assertNotEqual(self.source, tampered, "fixture no longer present in the PDF")
        self.assertTrue(self.build_policies.verify_curated_tables(tampered))

    def test_opening_hours_are_not_scrambled(self) -> None:
        """Regression lock on the extraction bug that motivated curation:
        pdftotext -layout pairs weekdays with "Fechado", which would have the
        agent tell customers the store is closed Monday to Friday."""
        self.assertIn("| Segunda a Sexta-feira | 09:00 às 18:00 |", self.markdown)
        self.assertIn("| Sábado | 09:00 às 13:00 |", self.markdown)
        self.assertIn("| Domingo e Feriados | Fechado |", self.markdown)

    def test_accents_survive_extraction(self) -> None:
        """Dropping an accent from customer-facing text is a product
        regression, and Windows code pages make it a real risk."""
        for term in ("Horário", "Devoluções", "Promoções", "Garantia", "Músico", "não"):
            self.assertIn(term, self.markdown)
        self.assertNotIn("Horario de Funcionamento", self.markdown)

    def test_deadline_rules_are_present_verbatim(self) -> None:
        """The numbers the returns tool depends on."""
        self.assertIn("7 (sete) dias corridos", self.markdown)
        self.assertIn("30 (trinta) dias corridos", self.markdown)
        self.assertIn("90 (noventa) dias", self.markdown)
        self.assertIn("Frete grátis para pedidos acima de R$ 500,00", self.markdown)
        self.assertIn("não são cumulativas", self.markdown)

    def test_no_orphaned_table_cell_fragments(self) -> None:
        """A wrapped table cell once leaked into section 3 as a lone paragraph
        reading "tabela"."""
        for line in self.markdown.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "|", "-", ">")):
                self.assertGreater(
                    len(stripped.split()),
                    2,
                    f"suspiciously short paragraph, likely a table fragment: {stripped!r}",
                )

    def test_every_content_word_survives(self) -> None:
        """Only cover-page and footer wording may be missing."""

        def words(text: str) -> set[str]:
            return set(re.findall(r"[0-9a-zà-ÿ]+", text.lower()))

        allowed = {
            "aqui",
            "atualização",
            "começa",
            "emporiodamusica",  # unaccented cover spelling; see pipeline notes
            "página",
            "restrito",
            "versão",
            "última",
        }
        missing = words(self.source) - words(self.markdown)
        self.assertEqual(set(), missing - allowed)

    def test_retrieval_units_are_never_empty(self) -> None:
        for unit in self.sections:
            self.assertTrue(unit["text"].strip(), f"section {unit['id']} has no text")

    def test_every_heading_is_reachable_by_retrieval(self) -> None:
        """A heading may be absent from the index only when it has no body of
        its own and its sub-sections are indexed — otherwise content is
        silently unreachable."""
        indexed = {unit["id"] for unit in self.sections}
        headings = re.findall(r"^#{2,3} (\d+(?:\.\d+)?)[. ]", self.markdown, re.M)
        self.assertEqual(10, len({h for h in headings if "." not in h}))
        for heading in headings:
            if heading in indexed:
                continue
            children = [i for i in indexed if i.startswith(f"{heading}.")]
            self.assertTrue(
                children, f"section {heading} is neither indexed nor covered by sub-sections"
            )
        self.assertIn("notes", indexed)

    def test_table_content_is_searchable_in_retrieval_units(self) -> None:
        """Tables are flattened to "label: value" lines so a lexical query for
        "sábado" or "boleto" can hit a cell."""
        hours = next(unit for unit in self.sections if unit["id"] == "2")
        self.assertIn("Dia: Sábado; Horário: 09:00 às 13:00", hours["text"])


class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        db = ROOT / "data" / "processed" / "store.db"
        if not db.exists():
            raise unittest.SkipTest("run scripts/build_data.py first")
        cls.conn = sqlite3.connect(db)
        cls.conn.execute("PRAGMA foreign_keys = ON")
        cls.conn.row_factory = sqlite3.Row

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn.close()

    def test_referential_integrity(self) -> None:
        self.assertEqual([], self.conn.execute("PRAGMA foreign_key_check").fetchall())

    def test_foreign_keys_are_enforced_not_decorative(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO order_items (order_id, product_id, quantity) VALUES (9999, 9999, 1)"
            )
        self.conn.rollback()

    def test_effective_price_applies_active_discount(self) -> None:
        row = self.conn.execute(
            "SELECT list_price_brl, discount_percent, effective_price_brl"
            " FROM product_pricing WHERE name LIKE 'Crafter HT-100%'"
        ).fetchone()
        self.assertEqual(18, row["discount_percent"])
        expected = round(row["list_price_brl"] * 0.82, 2)
        self.assertAlmostEqual(expected, row["effective_price_brl"], places=2)

    def test_inactive_promotions_never_reach_pricing(self) -> None:
        """21 of the 25 promotions are expired. Section 7.3 of the manual
        forbids promising a discount that is no longer in force."""
        promoted = {
            row["product_id"]
            for row in self.conn.execute(
                "SELECT product_id FROM product_pricing WHERE on_promotion = 1"
            )
        }
        active = {
            row["product_id"]
            for row in self.conn.execute(
                "SELECT product_id FROM promotions WHERE is_active = 1"
            )
        }
        self.assertEqual(active, promoted)

    def test_discounts_are_never_stacked(self) -> None:
        """Section 6.2: promotions are not cumulative. Verified against a
        synthetic second active promotion, since the shipped data has none."""
        self.conn.execute(
            "INSERT INTO promotions VALUES (9001, 90, 30, 'synthetic stacking test', 1)"
        )
        row = self.conn.execute(
            "SELECT discount_percent, list_price_brl, effective_price_brl"
            " FROM product_pricing WHERE product_id = 90"
        ).fetchone()
        self.conn.rollback()
        # The best single discount, not 18 + 30 compounded.
        self.assertEqual(30, row["discount_percent"])
        self.assertAlmostEqual(
            round(row["list_price_brl"] * 0.70, 2), row["effective_price_brl"], places=2
        )

    def test_availability_separates_the_four_cases(self) -> None:
        """status alone cannot answer "can the customer buy this today?"."""
        by_product = {
            row["product_id"]: row["availability"]
            for row in self.conn.execute(
                "SELECT product_id, availability FROM product_pricing"
            )
        }
        self.assertEqual("out_of_stock", by_product[96])  # active, zero stock
        self.assertEqual("discontinued", by_product[113])  # discontinued, zero stock
        self.assertEqual("discontinued", by_product[136])  # discontinued, stock remains
        self.assertEqual("coming_soon", by_product[130])  # not for sale yet
        self.assertEqual("available", by_product[81])

    def test_unshipped_orders_have_no_tracking(self) -> None:
        """The NULLs line up with orders that have not shipped, so they are
        meaningful absence the order tool must phrase, not missing data."""
        rows = self.conn.execute(
            "SELECT status, tracking_code, estimated_delivery FROM orders"
        ).fetchall()
        for row in rows:
            if row["status"] in ("delivered", "shipped"):
                self.assertIsNotNone(row["tracking_code"], row["status"])
                self.assertIsNotNone(row["estimated_delivery"], row["status"])
            else:
                self.assertIsNone(row["tracking_code"], row["status"])

    def test_specs_json_is_parsed_and_preserved(self) -> None:
        row = self.conn.execute(
            "SELECT specs_json, spec_top FROM products WHERE name LIKE 'Yamaha C70%'"
        ).fetchone()
        self.assertEqual("Spruce Sólido", row["spec_top"])
        self.assertEqual("Spruce Sólido", json.loads(row["specs_json"])["top"])

    def test_accents_survive_the_database_roundtrip(self) -> None:
        names = [
            row[0] for row in self.conn.execute("SELECT name FROM categories ORDER BY category_id")
        ]
        self.assertIn("Violões", names)
        self.assertIn("Baterias e Percussão", names)
        self.assertIn("Instrumentos de Sopro (Madeiras)", names)


if __name__ == "__main__":
    unittest.main()
