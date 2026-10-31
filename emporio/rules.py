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
from emporio.money import brl

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
METRO_DELIVERY_DAYS = "1 a 3 dias úteis"

# Section 5.2. Carriers outside the metropolitan region; the actual fee is
# quoted from the destination postcode, weight and dimensions, so no amount is
# stated here — the agent must not invent one.
CARRIERS: tuple[dict[str, str], ...] = (
    {"name": "PAC (Correios)", "estimate": "5 a 12 dias úteis"},
    {"name": "SEDEX (Correios)", "estimate": "2 a 5 dias úteis"},
    {"name": "Jadlog (.package)", "estimate": "3 a 8 dias úteis"},
)
BULKY_CATEGORIES_NOTE = (
    "Instrumentos de grande porte (baterias acústicas, pianos digitais, "
    "contrabaixos) podem exigir frete especial com cotação individual."
)

# --- Returns, exchanges and warranty ------------------------------------
WITHDRAWAL_DAYS = 7  # Section 4.1, from delivery, no justification needed
DEFECT_EXCHANGE_DAYS = 30  # Section 4.2, from purchase
PREFERENCE_EXCHANGE_DAYS = 7  # Section 4.3, from delivery
LEGAL_WARRANTY_DAYS = 90  # Section 8.1, from delivery
REFUND_BUSINESS_DAYS = 10  # Section 4.1

INELIGIBLE_FOR_EXCHANGE = (
    "Instrumentos com personalização ou ajustes sob encomenda (setup, regulagem especial).",
    'Produtos adquiridos em promoções de liquidação com aviso explícito de "venda final".',
    "Boquilhas de instrumentos de sopro, por questões de higiene.",
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
        amount = round(total / count, 2)
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
        round(effective * (1 - PIX_DISCOUNT_PERCENT / 100), 2)
        if pix_discount_applies
        else round(effective, 2)
    )

    plans = instalment_plans(effective)
    max_instalments = max((int(plan["instalments"]) for plan in plans), default=0)

    options = [
        PaymentOption(
            method="PIX",
            total_brl=pix_total,
            detail=(
                f"à vista com {PIX_DISCOUNT_PERCENT}% de desconto"
                if pix_discount_applies
                else "à vista; o desconto de PIX não se acumula com o preço promocional"
            ),
        ),
        PaymentOption(
            method="Cartão de débito",
            total_brl=round(effective, 2),
            detail="à vista, todas as bandeiras",
        ),
        PaymentOption(
            method="Boleto bancário",
            total_brl=round(effective, 2),
            detail="à vista, compensação em até 3 dias úteis",
        ),
    ]
    if max_instalments >= 1:
        best = plans[-1]
        options.append(
            PaymentOption(
                method="Cartão de crédito",
                total_brl=round(effective, 2),
                detail=(
                    f"até {best['instalments']}x sem juros de "
                    f"{brl(float(best['instalment_brl']))}"
                ),
            )
        )

    notes: list[str] = []
    if on_promotion:
        notes.append(
            f"Preço de tabela {brl(list_price_brl)}, com {discount_percent}% de desconto "
            f"promocional. Promoções não são cumulativas."
        )
    mixed_allowed = effective > MIXED_PAYMENT_MIN_BRL
    if mixed_allowed:
        notes.append(
            f"Compras acima de {brl(MIXED_PAYMENT_MIN_BRL)} podem combinar formas de "
            "pagamento (ex.: PIX + cartão)."
        )

    return PaymentQuote(
        list_price_brl=round(list_price_brl, 2),
        effective_price_brl=round(effective, 2),
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
                "O valor do frete é calculado com base no CEP de destino, peso e "
                "dimensões do produto."
            ),
            "carriers": [dict(carrier) for carrier in CARRIERS],
            "insurance": "Todos os envios incluem seguro contra extravios e danos.",
            "bulky_note": BULKY_CATEGORIES_NOTE,
        }

    free = order_total_brl > FREE_SHIPPING_THRESHOLD_BRL
    return {
        "region": "metropolitan_campo_grande",
        "fee_brl": 0.0 if free else FLAT_SHIPPING_FEE_BRL,
        "free_shipping": free,
        "fee_note": (
            f"Frete grátis para pedidos acima de {brl(FREE_SHIPPING_THRESHOLD_BRL)}."
            if free
            else (
                f"Pedidos de até {brl(FREE_SHIPPING_THRESHOLD_BRL)} pagam taxa fixa "
                f"de {brl(FLAT_SHIPPING_FEE_BRL)}."
            )
        ),
        "delivery_estimate": METRO_DELIVERY_DAYS,
        "carrier": "Motoboy próprio; o cliente é contactado por telefone antes da entrega.",
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
            reason=f"Não há {start_label} registrada para este pedido.",
            conditions=conditions,
        )
    deadline = start + timedelta(days=span_days)
    remaining = (deadline - reference).days
    if remaining >= 0:
        reason = (
            f"Prazo de {span_days} dias corridos a partir d{start_label} "
            f"({start.isoformat()}); vence em {deadline.isoformat()}."
        )
    else:
        reason = (
            f"Prazo de {span_days} dias corridos a partir d{start_label} "
            f"({start.isoformat()}) venceu em {deadline.isoformat()}, "
            f"há {abs(remaining)} dias."
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
                "Pedido cancelado. Não há devolução a processar; se houve cobrança, "
                "o estorno segue a forma de pagamento original."
            ),
            "windows": [],
            "ineligible_items": list(INELIGIBLE_FOR_EXCHANGE),
        }

    if order_status in ("pending", "confirmed", "shipped"):
        status_label = {
            "pending": "aguardando confirmação de pagamento",
            "confirmed": "confirmado e em preparação",
            "shipped": "em trânsito",
        }[order_status]
        # The cancel-before-dispatch route is only open while the order is
        # still in the store's hands; suggesting it for an order already in
        # transit contradicts the status in the same sentence.
        next_step = (
            "O pedido ainda não foi despachado, então pode ser cancelado antes do envio."
            if order_status in ("pending", "confirmed")
            else (
                "O pedido já está em trânsito. Após o recebimento, abre-se o prazo de "
                f"{WITHDRAWAL_DAYS} dias para devolução por arrependimento. Em caso de "
                "avaria visível, o cliente deve recusar o recebimento e contatar a loja."
            )
        )
        return {
            "order_status": order_status,
            "applicable": False,
            "summary": (
                f"Pedido {status_label}; o produto ainda não foi recebido, então os "
                f"prazos de devolução não começaram a contar. {next_step}"
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
            start_label="o recebimento",
            conditions=[
                "Embalagem original, sem sinais de uso, com todos os acessórios e manuais.",
                f"Reembolso na forma de pagamento original em até {REFUND_BUSINESS_DAYS} dias úteis.",
                "Frete de devolução por conta da loja.",
            ],
            reference=reference,
        ),
        _window(
            kind="defect_exchange",
            section="4.2 Trocas por Defeito",
            start=order_date,
            span_days=DEFECT_EXCHANGE_DAYS,
            start_label="a compra",
            conditions=[
                "Vale para defeito de fabricação.",
                "Após o prazo, a garantia é acionada com o fabricante; a loja pode intermediar.",
                "Não cobre mau uso, quedas, umidade excessiva ou modificações não autorizadas.",
            ],
            reference=reference,
        ),
        _window(
            kind="preference_exchange",
            section="4.3 Trocas por Preferência",
            start=delivery_date,
            span_days=PREFERENCE_EXCHANGE_DAYS,
            start_label="o recebimento",
            conditions=[
                "Sujeita à disponibilidade do produto desejado.",
                "Produto em perfeito estado e na embalagem original.",
                "Diferenças de valor são cobradas ou reembolsadas conforme o caso.",
            ],
            reference=reference,
        ),
        _window(
            kind="legal_warranty",
            section="8.1 Garantia Legal",
            start=delivery_date,
            span_days=LEGAL_WARRANTY_DAYS,
            start_label="o recebimento",
            conditions=[
                "Garantia legal de 90 dias contra defeitos de fabricação.",
                "A garantia do fabricante pode ser maior, de 6 meses a 2 anos.",
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
            "Nenhum prazo de troca ou devolução segue aberto para este pedido."
            if not available
            else f"Rotas ainda disponíveis: {', '.join(available)}."
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
