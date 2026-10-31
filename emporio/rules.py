"""The store's written policies, encoded as functions.

Every rule here comes from `data/processed/store_policies.md`; the section that
states each one is cited in the constant or docstring. The agent does not
compute deadlines, discounts, instalments or shipping fees itself — it calls
these functions and reports what they return.

That split is deliberate. The policy manual says, in section 7.1, never to give
a customer a price or a deadline without checking the system, and arithmetic is
exactly where a language model is least reliable and least auditable. A wrong
refund window is a legal problem, not a style problem.

Two places where the manual is ambiguous or self-contradictory, resolved here
and flagged in the README:

1. Section 3's table says credit card instalments carry a BRL 100 minimum,
   while section 3.1 gives per-tier minimums of none/80/100. The more specific
   rule wins, so 3.1 is implemented.
2. Section 3.1's "sem valor mínimo de parcela (exceto abaixo de R$ 50,00)" is
   read as: instalments of 1-3x have no tier minimum, but an individual
   instalment may not fall below BRL 50.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from emporio.config import get_settings
from emporio.money import brl, round_brl

# --- Payment -------------------------------------------------------------
# Section 3: PIX pays in full with a 5% discount off the list price.
PIX_DISCOUNT_PERCENT = 5

# Section 3.1, as (max_instalments, minimum_instalment_brl).
INSTALMENT_TIERS: tuple[tuple[int, float], ...] = ((3, 50.0), (6, 80.0), (12, 100.0))
MAX_INSTALMENTS = 12

# Section 3.1: PIX and card may be combined above this amount.
MIXED_PAYMENT_MIN_BRL = 2000.0

# --- Shipping ------------------------------------------------------------
# Section 5.1, metropolitan Campo Grande only.
FREE_SHIPPING_THRESHOLD_BRL = 500.0
FLAT_SHIPPING_FEE_BRL = 35.0
METRO_DELIVERY_DAYS = "1 to 3 business days"

# Section 5.2. Carriers outside the metropolitan region; the actual fee is
# quoted from the destination postcode, weight and dimensions, so no amount is
# stated here — the agent must not invent one. Carrier names are proper nouns
# and stay as the Brazilian carriers are actually called.
CARRIERS: tuple[dict[str, str], ...] = (
    {"name": "PAC (Correios)", "estimate": "5 to 12 business days"},
    {"name": "SEDEX (Correios)", "estimate": "2 to 5 business days"},
    {"name": "Jadlog (.package)", "estimate": "3 to 8 business days"},
)
BULKY_CATEGORIES_NOTE = (
    "Large instruments (acoustic drum kits, digital pianos, double basses) may "
    "require special freight quoted individually."
)

# --- Returns, exchanges and warranty ------------------------------------
WITHDRAWAL_DAYS = 7  # Section 4.1, from delivery, no justification needed
DEFECT_EXCHANGE_DAYS = 30  # Section 4.2, from purchase
PREFERENCE_EXCHANGE_DAYS = 7  # Section 4.3, from delivery
LEGAL_WARRANTY_DAYS = 90  # Section 8.1, from delivery
REFUND_BUSINESS_DAYS = 10  # Section 4.1

INELIGIBLE_FOR_EXCHANGE = (
    "Instruments customised or set up to order (special setup or adjustment).",
    'Items bought in clearance promotions explicitly marked "final sale".',
    "Wind instrument mouthpieces, for hygiene reasons.",
)

OrderStatus = Literal["pending", "confirmed", "shipped", "delivered", "cancelled"]


def today() -> date:
    """The agent's reference date. See config.DEFAULT_REFERENCE_DATE."""
    return get_settings().reference_date


# --- Payment quoting -----------------------------------------------------


@dataclass
class PaymentOption:
    method: str
    total_brl: float
    detail: str


@dataclass
class PaymentQuote:
    list_price_brl: float
    effective_price_brl: float
    on_promotion: bool
    discount_percent: int | None
    pix_total_brl: float
    pix_discount_applies: bool
    options: list[PaymentOption] = field(default_factory=list)
    max_instalments: int = 0
    mixed_payment_allowed: bool = False
    notes: list[str] = field(default_factory=list)


def instalment_plans(total: float) -> list[dict[str, object]]:
    """Every instalment count allowed for an amount, per section 3.1.

    A single payment is always available: 1x on a credit card is paying in
    full, not instalment credit, so the per-instalment floors in section 3.1
    do not gate it. Applying the BRL 50 floor to 1x would have made any item
    under BRL 50 unpayable by card.
    """
    plans: list[dict[str, object]] = []
    for count in range(1, MAX_INSTALMENTS + 1):
        minimum = next(floor for cap, floor in INSTALMENT_TIERS if count <= cap)
        amount = round_brl(total / count)
        if count == 1 or amount >= minimum:
            plans.append(
                {
                    "instalments": count,
                    "instalment_brl": amount,
                    "minimum_instalment_brl": 0.0 if count == 1 else minimum,
                }
            )
    return plans


def quote_payment(
    list_price_brl: float,
    effective_price_brl: float | None = None,
    discount_percent: int | None = None,
) -> PaymentQuote:
    """Price a product across every payment method the store accepts.

    The one rule worth stating twice: section 6.2 says the 5% PIX discount does
    **not** apply on top of an already promotional price. Promotions are not
    cumulative, so a product on promotion pays its promotional price via PIX —
    not that price minus another 5%.
    """
    effective = list_price_brl if effective_price_brl is None else effective_price_brl
    on_promotion = bool(discount_percent)

    pix_discount_applies = not on_promotion
    pix_total = (
        round_brl(effective * (1 - PIX_DISCOUNT_PERCENT / 100))
        if pix_discount_applies
        else round_brl(effective)
    )

    plans = instalment_plans(effective)
    max_instalments = max((int(plan["instalments"]) for plan in plans), default=0)

    # PIX and boleto are Brazilian payment instruments with no English
    # equivalent, so the names stay; only the descriptions are translated.
    options = [
        PaymentOption(
            method="PIX",
            total_brl=pix_total,
            detail=(
                f"paid in full, with a {PIX_DISCOUNT_PERCENT}% discount"
                if pix_discount_applies
                else "paid in full; the PIX discount does not stack with a promotional price"
            ),
        ),
        PaymentOption(
            method="Debit card",
            total_brl=round_brl(effective),
            detail="paid in full, all card networks accepted",
        ),
        PaymentOption(
            method="Boleto (bank slip)",
            total_brl=round_brl(effective),
            detail="paid in full, clears within 3 business days",
        ),
    ]
    if max_instalments >= 1:
        best = plans[-1]
        options.append(
            PaymentOption(
                method="Credit card",
                total_brl=round_brl(effective),
                detail=(
                    f"up to {best['instalments']} interest-free instalments of "
                    f"{brl(float(best['instalment_brl']))}"
                ),
            )
        )

    notes: list[str] = []
    if on_promotion:
        notes.append(
            f"List price {brl(list_price_brl)}, with a {discount_percent}% promotional "
            "discount. Promotions are not cumulative."
        )
    mixed_allowed = effective > MIXED_PAYMENT_MIN_BRL
    if mixed_allowed:
        notes.append(
            f"Purchases above {brl(MIXED_PAYMENT_MIN_BRL)} may combine payment methods "
            "(for example PIX plus card)."
        )

    return PaymentQuote(
        list_price_brl=round_brl(list_price_brl),
        effective_price_brl=round_brl(effective),
        on_promotion=on_promotion,
        discount_percent=discount_percent,
        pix_total_brl=pix_total,
        pix_discount_applies=pix_discount_applies,
        options=options,
        max_instalments=max_instalments,
        mixed_payment_allowed=mixed_allowed,
        notes=notes,
    )


# --- Shipping quoting ----------------------------------------------------


def quote_shipping(order_total_brl: float, metropolitan: bool = True) -> dict[str, object]:
    """Shipping terms for an order value, per sections 5.1 and 5.2.

    Outside metropolitan Campo Grande the fee depends on postcode, weight and
    dimensions, so this returns the carriers and their estimates with no
    amount: a quoted number the system cannot compute would be exactly the
    "propaganda enganosa" section 7.1 warns about.
    """
    if not metropolitan:
        return {
            "region": "outside_metropolitan_campo_grande",
            "fee_brl": None,
            "fee_note": (
                "The shipping cost is calculated from the destination postcode, "
                "weight and dimensions of the product."
            ),
            "carriers": [dict(carrier) for carrier in CARRIERS],
            "insurance": "Every shipment includes insurance against loss and damage.",
            "bulky_note": BULKY_CATEGORIES_NOTE,
        }

    free = order_total_brl > FREE_SHIPPING_THRESHOLD_BRL
    return {
        "region": "metropolitan_campo_grande",
        "fee_brl": 0.0 if free else FLAT_SHIPPING_FEE_BRL,
        "free_shipping": free,
        "fee_note": (
            f"Free shipping on orders above {brl(FREE_SHIPPING_THRESHOLD_BRL)}."
            if free
            else (
                f"Orders up to {brl(FREE_SHIPPING_THRESHOLD_BRL)} pay a flat fee of "
                f"{brl(FLAT_SHIPPING_FEE_BRL)}."
            )
        ),
        "delivery_estimate": METRO_DELIVERY_DAYS,
        "carrier": "Own courier; the customer is phoned before delivery.",
    }


# --- Returns, exchanges and warranty ------------------------------------


@dataclass
class ReturnWindow:
    kind: str
    policy_section: str
    allowed: bool
    deadline: date | None
    days_remaining: int | None
    reason: str
    conditions: list[str] = field(default_factory=list)


def _window(
    kind: str,
    section: str,
    start: date | None,
    span_days: int,
    start_label: str,
    conditions: list[str],
    reference: date,
) -> ReturnWindow:
    if start is None:
        return ReturnWindow(
            kind=kind,
            policy_section=section,
            allowed=False,
            deadline=None,
            days_remaining=None,
            reason=f"No {start_label} date is recorded for this order.",
            conditions=conditions,
        )
    deadline = start + timedelta(days=span_days)
    remaining = (deadline - reference).days
    if remaining >= 0:
        reason = (
            f"{span_days} calendar days from {start_label} ({start.isoformat()}); "
            f"closes on {deadline.isoformat()}, {remaining} days from now."
        )
    else:
        reason = (
            f"{span_days} calendar days from {start_label} ({start.isoformat()}); "
            f"closed on {deadline.isoformat()}, {abs(remaining)} days ago."
        )
    return ReturnWindow(
        kind=kind,
        policy_section=section,
        allowed=remaining >= 0,
        deadline=deadline,
        days_remaining=remaining,
        reason=reason,
        conditions=conditions,
    )


def evaluate_return_request(
    order_status: str,
    order_date: date,
    delivery_date: date | None,
    reference: date | None = None,
) -> dict[str, object]:
    """Assess every route by which an order could be returned or exchanged.

    Four routes exist and they start from different dates: withdrawal and
    preference exchange run from delivery, defect exchange runs from purchase,
    and the legal warranty runs from delivery. Reporting them together is what
    lets the agent say "the 7 days are gone, but you still have warranty
    cover" instead of a flat no.

    Known limitation: the dataset records `estimated_delivery`, not the date
    the customer actually took possession. For delivered orders it is used as
    the best available proxy, and the caller is told so via
    `delivery_date_is_estimated`.
    """
    reference = reference or today()

    if order_status == "cancelled":
        return {
            "order_status": order_status,
            "applicable": False,
            "summary": (
                "Order cancelled. There is nothing to return; if it was charged, the "
                "refund follows the original payment method."
            ),
            "windows": [],
            "ineligible_items": list(INELIGIBLE_FOR_EXCHANGE),
        }

    if order_status in ("pending", "confirmed", "shipped"):
        status_label = {
            "pending": "awaiting payment confirmation",
            "confirmed": "confirmed and being prepared",
            "shipped": "in transit",
        }[order_status]
        # The cancel-before-dispatch route is only open while the order is
        # still in the store's hands; suggesting it for an order already in
        # transit contradicts the status in the same sentence.
        next_step = (
            "The order has not been dispatched yet, so it can still be cancelled "
            "before shipping."
            if order_status in ("pending", "confirmed")
            else (
                "The order is already in transit. Once it arrives, the "
                f"{WITHDRAWAL_DAYS}-day withdrawal window opens. If the package "
                "arrives visibly damaged, the customer should refuse delivery and "
                "contact the store immediately."
            )
        )
        return {
            "order_status": order_status,
            "applicable": False,
            "summary": (
                f"Order is {status_label}; the product has not been received, so no "
                f"return deadline has started counting. {next_step}"
            ),
            "windows": [],
            "ineligible_items": list(INELIGIBLE_FOR_EXCHANGE),
        }

    windows = [
        _window(
            kind="withdrawal",
            section="4.1 Direito de Arrependimento",
            start=delivery_date,
            span_days=WITHDRAWAL_DAYS,
            start_label="delivery",
            conditions=[
                "Original packaging, unused, with all accessories and manuals.",
                f"Refunded to the original payment method within {REFUND_BUSINESS_DAYS} "
                "business days.",
                "Return shipping is paid by the store.",
            ],
            reference=reference,
        ),
        _window(
            kind="defect_exchange",
            section="4.2 Trocas por Defeito",
            start=order_date,
            span_days=DEFECT_EXCHANGE_DAYS,
            start_label="purchase",
            conditions=[
                "Applies to manufacturing defects.",
                "After the deadline the manufacturer's warranty applies; the store can "
                "act as intermediary.",
                "Does not cover misuse, drops, excessive humidity or unauthorised "
                "modifications.",
            ],
            reference=reference,
        ),
        _window(
            kind="preference_exchange",
            section="4.3 Trocas por Preferência",
            start=delivery_date,
            span_days=PREFERENCE_EXCHANGE_DAYS,
            start_label="delivery",
            conditions=[
                "Subject to availability of the desired product.",
                "Product in perfect condition and in its original packaging.",
                "Any price difference is charged or refunded as applicable.",
            ],
            reference=reference,
        ),
        _window(
            kind="legal_warranty",
            section="8.1 Garantia Legal",
            start=delivery_date,
            span_days=LEGAL_WARRANTY_DAYS,
            start_label="delivery",
            conditions=[
                "Statutory 90-day warranty against manufacturing defects.",
                "The manufacturer's own warranty may be longer, from 6 months to 2 years.",
            ],
            reference=reference,
        ),
    ]

    available = [window.kind for window in windows if window.allowed]
    return {
        "order_status": order_status,
        "applicable": True,
        "reference_date": reference.isoformat(),
        "delivery_date": delivery_date.isoformat() if delivery_date else None,
        "delivery_date_is_estimated": delivery_date is not None,
        "available_routes": available,
        "summary": (
            "No return or exchange window is still open for this order."
            if not available
            else f"Routes still available: {', '.join(available)}."
        ),
        "windows": [
            {
                "kind": window.kind,
                "policy_section": window.policy_section,
                "allowed": window.allowed,
                "deadline": window.deadline.isoformat() if window.deadline else None,
                "days_remaining": window.days_remaining,
                "reason": window.reason,
                "conditions": window.conditions,
            }
            for window in windows
        ],
        "ineligible_items": list(INELIGIBLE_FOR_EXCHANGE),
    }
