"""Tests for the deterministic policy rules.

Every case passes an explicit reference date, so these tests do not depend on
the environment or on the real clock.
"""

from __future__ import annotations

import unittest
from datetime import date

from emporio import rules

REFERENCE = date(2026, 3, 25)


class PaymentTests(unittest.TestCase):
    def test_pix_discount_applies_off_list_price(self) -> None:
        quote = rules.quote_payment(list_price_brl=2199.0)
        self.assertTrue(quote.pix_discount_applies)
        self.assertAlmostEqual(2089.05, quote.pix_total_brl, places=2)

    def test_pix_discount_does_not_stack_on_a_promotional_price(self) -> None:
        """Section 6.2: promotions are not cumulative and the 5% PIX discount
        does not apply over an already promotional price. Getting this wrong
        undercharges the store on every promoted item."""
        quote = rules.quote_payment(
            list_price_brl=2399.0, effective_price_brl=1967.18, discount_percent=18
        )
        self.assertFalse(quote.pix_discount_applies)
        self.assertAlmostEqual(1967.18, quote.pix_total_brl, places=2)
        self.assertTrue(any("not cumulative" in note for note in quote.notes))

    def test_promotional_price_is_presented_with_its_original(self) -> None:
        """Section 6.2 requires the original price and the percentage."""
        quote = rules.quote_payment(
            list_price_brl=2399.0, effective_price_brl=1967.18, discount_percent=18
        )
        note = " ".join(quote.notes)
        self.assertIn("R$ 2.399,00", note)
        self.assertIn("18%", note)

    def test_instalment_tier_minimums(self) -> None:
        """Section 3.1: no tier minimum up to 3x (but no instalment below 50),
        80 from 4x to 6x, 100 from 7x to 12x."""
        plans = {int(plan["instalments"]): plan["instalment_brl"] for plan in rules.instalment_plans(1200.0)}
        self.assertEqual(12, max(plans))  # 1200/12 = 100, exactly the floor

        # 1199 / 12 = 99.92, just below the 7-12x floor of 100.
        self.assertNotIn(12, {int(p["instalments"]) for p in rules.instalment_plans(1199.0)})

        # 480 / 6 = 80 exactly; 7x would be 68.57, below the 100 floor.
        self.assertEqual(6, max(int(p["instalments"]) for p in rules.instalment_plans(480.0)))

        # 149 / 3 = 49.67, below the 50 floor that applies to the 1-3x tier.
        self.assertEqual(2, max(int(p["instalments"]) for p in rules.instalment_plans(149.0)))

    def test_cheap_item_is_single_payment_only(self) -> None:
        plans = rules.instalment_plans(45.0)
        self.assertEqual([1], [int(plan["instalments"]) for plan in plans])

    def test_mixed_payment_threshold(self) -> None:
        self.assertFalse(rules.quote_payment(list_price_brl=2000.0).mixed_payment_allowed)
        self.assertTrue(rules.quote_payment(list_price_brl=2000.01).mixed_payment_allowed)


class ShippingTests(unittest.TestCase):
    def test_free_shipping_above_threshold(self) -> None:
        quote = rules.quote_shipping(500.01)
        self.assertEqual(0.0, quote["fee_brl"])
        self.assertTrue(quote["free_shipping"])

    def test_flat_fee_at_and_below_threshold(self) -> None:
        """Section 5.1 reads "acima de R$ 500,00", so exactly 500 pays."""
        quote = rules.quote_shipping(500.0)
        self.assertEqual(35.0, quote["fee_brl"])
        self.assertFalse(quote["free_shipping"])

    def test_outside_metropolitan_region_quotes_no_amount(self) -> None:
        """The fee depends on postcode, weight and dimensions, which the system
        cannot compute. A number here would be invented."""
        quote = rules.quote_shipping(1500.0, metropolitan=False)
        self.assertIsNone(quote["fee_brl"])
        self.assertEqual(3, len(quote["carriers"]))


class ReturnWindowTests(unittest.TestCase):
    def test_all_four_routes_are_assessed(self) -> None:
        assessment = rules.evaluate_return_request(
            order_status="delivered",
            order_date=date(2026, 3, 20),
            delivery_date=date(2026, 3, 23),
            reference=REFERENCE,
        )
        self.assertEqual(
            ["withdrawal", "defect_exchange", "preference_exchange", "legal_warranty"],
            [window["kind"] for window in assessment["windows"]],
        )
        self.assertTrue(all(window["allowed"] for window in assessment["windows"]))

    def test_withdrawal_expires_seven_days_after_delivery(self) -> None:
        on_time = rules.evaluate_return_request(
            order_status="delivered",
            order_date=date(2026, 3, 10),
            delivery_date=date(2026, 3, 18),  # deadline 2026-03-25, the reference date
            reference=REFERENCE,
        )
        self.assertIn("withdrawal", on_time["available_routes"])

        one_day_late = rules.evaluate_return_request(
            order_status="delivered",
            order_date=date(2026, 3, 10),
            delivery_date=date(2026, 3, 17),  # deadline 2026-03-24
            reference=REFERENCE,
        )
        self.assertNotIn("withdrawal", one_day_late["available_routes"])

    def test_expired_withdrawal_still_reports_open_warranty(self) -> None:
        """The case that makes the four-route split worth the code: order 7 in
        the dataset is past every exchange deadline but inside the 90-day legal
        warranty, so the honest answer is not a flat no."""
        assessment = rules.evaluate_return_request(
            order_status="delivered",
            order_date=date(2026, 2, 3),
            delivery_date=date(2026, 2, 17),
            reference=REFERENCE,
        )
        self.assertEqual(["legal_warranty"], assessment["available_routes"])
        withdrawal = next(w for w in assessment["windows"] if w["kind"] == "withdrawal")
        self.assertFalse(withdrawal["allowed"])
        self.assertIn("closed on 2026-02-24", withdrawal["reason"])

    def test_defect_window_runs_from_purchase_not_delivery(self) -> None:
        """Section 4.2 counts 30 days from the purchase; 4.1 counts 7 from
        delivery. Conflating them would silently extend one of the windows."""
        assessment = rules.evaluate_return_request(
            order_status="delivered",
            order_date=date(2026, 2, 20),  # deadline 2026-03-22, expired
            delivery_date=date(2026, 3, 24),  # withdrawal still open
            reference=REFERENCE,
        )
        self.assertIn("withdrawal", assessment["available_routes"])
        self.assertNotIn("defect_exchange", assessment["available_routes"])

    def test_undelivered_orders_have_not_started_counting(self) -> None:
        for status in ("pending", "confirmed", "shipped"):
            assessment = rules.evaluate_return_request(
                order_status=status,
                order_date=date(2026, 3, 20),
                delivery_date=None,
                reference=REFERENCE,
            )
            self.assertFalse(assessment["applicable"], status)
            self.assertEqual([], assessment["windows"])

    def test_dispatched_order_is_not_offered_cancellation(self) -> None:
        """Telling a customer whose order is in transit that it "can be
        cancelled before dispatch" contradicts the status in the same
        sentence."""
        shipped = rules.evaluate_return_request(
            order_status="shipped",
            order_date=date(2026, 3, 1),
            delivery_date=date(2026, 3, 28),
            reference=REFERENCE,
        )
        self.assertNotIn("cancelled before shipping", shipped["summary"])
        self.assertIn("in transit", shipped["summary"])

        confirmed = rules.evaluate_return_request(
            order_status="confirmed",
            order_date=date(2026, 3, 20),
            delivery_date=None,
            reference=REFERENCE,
        )
        self.assertIn("cancelled before shipping", confirmed["summary"])

    def test_cancelled_order_has_nothing_to_return(self) -> None:
        assessment = rules.evaluate_return_request(
            order_status="cancelled",
            order_date=date(2026, 1, 28),
            delivery_date=None,
            reference=REFERENCE,
        )
        self.assertFalse(assessment["applicable"])
        self.assertIn("cancelled", assessment["summary"])

    def test_missing_delivery_date_is_reported_not_assumed(self) -> None:
        assessment = rules.evaluate_return_request(
            order_status="delivered",
            order_date=date(2026, 3, 1),
            delivery_date=None,
            reference=REFERENCE,
        )
        withdrawal = next(w for w in assessment["windows"] if w["kind"] == "withdrawal")
        self.assertFalse(withdrawal["allowed"])
        self.assertIsNone(withdrawal["deadline"])
        self.assertIn("no delivery date is recorded", withdrawal["reason"].lower())


if __name__ == "__main__":
    unittest.main()
