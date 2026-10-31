"""The tools the agent may call, and their declarations for the model.

Design choice: every tool has a closed signature over the store's own concepts
— no tool takes SQL, and none takes free-form filters. A text-to-SQL agent
would answer a wider range of unanticipated questions, but it would do so by
letting the model author the query that produces a price or a stock figure.
Section 7.1 of the policy manual forbids quoting price, stock or deadlines
without checking the system, and an invented JOIN is a plausible-looking way to
check the system wrongly. The narrower surface is the point.

Each tool returns plain JSON-serialisable data, including a pre-formatted
Brazilian currency string. Formatting money is not a task to delegate to a
language model when `R$ 1.967,18` and `R$ 1,967.18` differ by a factor of a
thousand to a reader.

## Language inside tool results

Everything written in this repository is English, including the prose fields
the model reads: `availability_label`, `tracking_note`, `summary`,
`conditions`. The agent translates them at reply time, so a Brazilian customer
still gets Portuguese — that round trip is the common case and is exercised by
four of the five transcripts in `examples/`.

Two things stay in Portuguese, because they are *data* rather than code and
rewriting them would mean editing the sources the challenge supplied:

- Catalog values from the CSVs: product names, category names, descriptions.
- Policy text returned by `search_store_policies`, extracted from the store's
  PDF manual. Paraphrasing a refund window into English in the pipeline would
  put a translation of a legal term between the manual and the customer, with
  no way to check it against the source.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from emporio import policies, rules
from emporio.db import query, query_one
from emporio.money import brl
from emporio.policies import normalise

# How many catalog rows a single search may return. Large enough for a real
# browse ("violões até R$ 1000" matches 14), small enough to keep the model's
# context focused.
MAX_SEARCH_RESULTS = 25
DEFAULT_SEARCH_RESULTS = 10

AVAILABILITY_LABELS = {
    "available": "in stock",
    "out_of_stock": "temporarily out of stock",
    "discontinued": "discontinued, no longer in the catalog",
    "coming_soon": "coming soon, not yet available for purchase",
}

ORDER_STATUS_LABELS = {
    "pending": "awaiting payment confirmation",
    "confirmed": "confirmed, being prepared",
    "shipped": "dispatched, in transit",
    "delivered": "delivered",
    "cancelled": "cancelled",
}


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _product_payload(row: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "product_id": row["product_id"],
        "name": row["name"],
        "category": row["category_name"],
        "list_price_brl": row["list_price_brl"],
        "list_price_display": brl(row["list_price_brl"]),
        "availability": row["availability"],
        "availability_label": AVAILABILITY_LABELS[row["availability"]],
        "stock_quantity": row["stock_quantity"],
    }
    if row["on_promotion"]:
        payload |= {
            "on_promotion": True,
            "discount_percent": row["discount_percent"],
            "price_brl": row["effective_price_brl"],
            "price_display": brl(row["effective_price_brl"]),
            # Section 6.2 requires showing the original price and the discount
            # percentage alongside any promotional price.
            "price_presentation": (
                f"{brl(row['effective_price_brl'])} "
                f"(down from {brl(row['list_price_brl'])}, "
                f"{row['discount_percent']}% off)"
            ),
        }
    else:
        payload |= {
            "on_promotion": False,
            "price_brl": row["list_price_brl"],
            "price_display": brl(row["list_price_brl"]),
            "price_presentation": brl(row["list_price_brl"]),
        }

    if full:
        payload["description"] = row["description"]
        payload["specs"] = row["specs_json"]
        quote = rules.quote_payment(
            list_price_brl=row["list_price_brl"],
            effective_price_brl=row["effective_price_brl"],
            discount_percent=row["discount_percent"],
        )
        payload["payment"] = {
            "pix_total_brl": quote.pix_total_brl,
            "pix_total_display": brl(quote.pix_total_brl),
            "pix_discount_applies": quote.pix_discount_applies,
            "max_instalments": quote.max_instalments,
            "options": [
                {
                    "method": option.method,
                    "total_display": brl(option.total_brl),
                    "detail": option.detail,
                }
                for option in quote.options
            ],
            "notes": quote.notes,
        }
    return payload


# --- Catalog -------------------------------------------------------------


def search_products(
    query_text: str | None = None,
    category: str | None = None,
    max_price_brl: float | None = None,
    min_price_brl: float | None = None,
    only_available: bool = True,
    limit: int = DEFAULT_SEARCH_RESULTS,
) -> dict[str, Any]:
    """Search the catalog with structured filters.

    Price filters apply to the *effective* price, so a promotional item that
    falls under the customer's ceiling is found.

    `only_available` defaults to True because section 7.3 requires a distinct
    response for out-of-stock and discontinued products. When it hides results,
    the excluded items are summarised in `withheld` so the agent can offer
    them explicitly instead of pretending the catalog is smaller than it is.
    """
    limit = max(1, min(int(limit), MAX_SEARCH_RESULTS))

    conditions: list[str] = []
    params: dict[str, Any] = {}

    if category:
        wanted = normalise(category)
        matches = [
            row["category_id"]
            for row in query("SELECT category_id, name FROM categories")
            if wanted in normalise(row["name"]) or normalise(row["name"]) in wanted
        ]
        if not matches:
            known = [row["name"] for row in query("SELECT name FROM categories ORDER BY name")]
            return {
                "status": "unknown_category",
                "requested": category,
                "known_categories": known,
                "results": [],
            }
        placeholders = ", ".join(f":cat{i}" for i in range(len(matches)))
        conditions.append(f"category_id IN ({placeholders})")
        params |= {f"cat{i}": value for i, value in enumerate(matches)}

    if max_price_brl is not None:
        conditions.append("effective_price_brl <= :max_price")
        params["max_price"] = float(max_price_brl)
    if min_price_brl is not None:
        conditions.append("effective_price_brl >= :min_price")
        params["min_price"] = float(min_price_brl)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = query(
        f"SELECT * FROM product_pricing {where} ORDER BY effective_price_brl",
        params,
    )

    # Text matching happens here rather than in SQL so it can ignore accents:
    # customers type "violao" and the catalog says "Violões".
    if query_text:
        needle = normalise(query_text)
        words = [word for word in needle.split() if len(word) > 2]
        def matches_text(row: dict[str, Any]) -> bool:
            haystack = normalise(f"{row['name']} {row['description']} {row['specs_json']}")
            if needle in haystack:
                return True
            return bool(words) and all(word in haystack for word in words)

        rows = [row for row in rows if matches_text(row)]

    available = [row for row in rows if row["availability"] == "available"]
    unavailable = [row for row in rows if row["availability"] != "available"]

    selected = available if only_available else rows
    payload: dict[str, Any] = {
        "status": "ok",
        "total_matches": len(selected),
        "returned": min(len(selected), limit),
        "results": [_product_payload(row) for row in selected[:limit]],
    }
    if only_available and unavailable:
        payload["withheld"] = {
            "reason": "matched the filters but are not currently purchasable",
            "count": len(unavailable),
            "items": [
                {
                    "product_id": row["product_id"],
                    "name": row["name"],
                    "price_display": brl(row["effective_price_brl"]),
                    "availability": row["availability"],
                    "availability_label": AVAILABILITY_LABELS[row["availability"]],
                }
                for row in unavailable
            ],
        }
    if len(selected) > limit:
        payload["truncated"] = True
    return payload


def get_product_details(
    product_id: int | None = None, name: str | None = None
) -> dict[str, Any]:
    """Full detail for one product, including payment options.

    When a name matches several products the candidates are returned instead of
    a guess. Picking one silently is how a customer ends up quoted the price of
    a different instrument.
    """
    if product_id is None and not name:
        return {"status": "missing_argument", "detail": "provide product_id or name"}

    if product_id is not None:
        row = query_one("SELECT * FROM product_pricing WHERE product_id = :pid", {"pid": int(product_id)})
        if not row:
            return {"status": "not_found", "product_id": product_id}
        return {"status": "ok", "product": _product_payload(row, full=True)}

    needle = normalise(str(name))
    words = [word for word in needle.split() if len(word) > 1]
    rows = [
        row
        for row in query("SELECT * FROM product_pricing ORDER BY name")
        if needle in normalise(row["name"])
        or (words and all(word in normalise(row["name"]) for word in words))
    ]

    if not rows:
        return {"status": "not_found", "name": name}
    if len(rows) > 1:
        return {
            "status": "ambiguous",
            "name": name,
            "candidates": [
                {
                    "product_id": row["product_id"],
                    "name": row["name"],
                    "price_display": brl(row["effective_price_brl"]),
                    "availability_label": AVAILABILITY_LABELS[row["availability"]],
                }
                for row in rows
            ],
        }
    return {"status": "ok", "product": _product_payload(rows[0], full=True)}


def get_active_promotions() -> dict[str, Any]:
    """Every promotion currently in force.

    Only `is_active` rows exist in this result. Section 7.3 is explicit that an
    expired promotion must never be offered, and 21 of the 25 promotions in the
    dataset are expired.
    """
    rows = query(
        "SELECT pr.promotion_id, pr.discount_percent, pr.description,"
        "       pp.product_id, pp.name, pp.category_name, pp.list_price_brl,"
        "       pp.effective_price_brl, pp.availability"
        " FROM promotions pr"
        " JOIN product_pricing pp ON pp.product_id = pr.product_id"
        " WHERE pr.is_active = 1"
        " ORDER BY pr.discount_percent DESC"
    )
    return {
        "status": "ok",
        "count": len(rows),
        "promotions": [
            {
                "promotion_id": row["promotion_id"],
                "campaign": row["description"],
                "discount_percent": row["discount_percent"],
                "product_id": row["product_id"],
                "product_name": row["name"],
                "category": row["category_name"],
                "list_price_display": brl(row["list_price_brl"]),
                "promotional_price_display": brl(row["effective_price_brl"]),
                "price_presentation": (
                    f"{brl(row['effective_price_brl'])} "
                    f"(down from {brl(row['list_price_brl'])}, "
                    f"{row['discount_percent']}% off)"
                ),
                "availability": row["availability"],
                "availability_label": AVAILABILITY_LABELS[row["availability"]],
            }
            for row in rows
        ],
        "rules": [
            "Promotions are not cumulative.",
            "The 5% PIX discount does not apply on top of an already promotional price.",
            "Promotions are subject to stock availability; no rain checks.",
        ],
    }


# --- Orders --------------------------------------------------------------


def get_order_status(order_id: int) -> dict[str, Any]:
    """Status, items and tracking for one order.

    `tracking_code` and `estimated_delivery` are NULL for every order that has
    not shipped — 12 of 20 in this dataset. They are reported as an explicit
    "not dispatched yet" rather than passed through as nulls for the model to
    narrate around.
    """
    order = query_one(
        "SELECT o.*, c.name AS customer_name, c.city AS customer_city"
        " FROM orders o JOIN customers c ON c.customer_id = o.customer_id"
        " WHERE o.order_id = :oid",
        {"oid": int(order_id)},
    )
    if not order:
        return {"status": "not_found", "order_id": order_id}

    items = query(
        "SELECT i.quantity, p.product_id, p.name, p.price_brl"
        " FROM order_items i JOIN products p ON p.product_id = i.product_id"
        " WHERE i.order_id = :oid",
        {"oid": int(order_id)},
    )

    dispatched = order["tracking_code"] is not None
    return {
        "status": "ok",
        "order_id": order["order_id"],
        "order_status": order["status"],
        "order_status_label": ORDER_STATUS_LABELS[order["status"]],
        "customer_name": order["customer_name"],
        "customer_city": order["customer_city"],
        "order_date": order["order_date"],
        "total_display": brl(order["total_brl"]),
        "payment_method": order["payment_method"],
        "dispatched": dispatched,
        "tracking_code": order["tracking_code"],
        "estimated_delivery": order["estimated_delivery"],
        "tracking_note": (
            "The tracking code is sent by email and WhatsApp as soon as the order "
            "is dispatched; this order has not been dispatched yet."
            if not dispatched
            else "Trackable on the Correios or Jadlog website."
        ),
        "internal_note": order["notes"],
        "items": [
            {
                "product_id": item["product_id"],
                "name": item["name"],
                "quantity": item["quantity"],
                "unit_price_display": brl(item["price_brl"]),
            }
            for item in items
        ],
    }


def check_return_eligibility(order_id: int) -> dict[str, Any]:
    """Whether an order can still be returned or exchanged, and by which route.

    Returns all four routes from sections 4.1, 4.2, 4.3 and 8.1 with their own
    deadlines, because they start from different dates. That is what lets the
    agent say "the 7-day withdrawal window closed, but you are still inside the
    90-day legal warranty" rather than a bare no.
    """
    order = query_one(
        "SELECT order_id, status, order_date, estimated_delivery, total_brl"
        " FROM orders WHERE order_id = :oid",
        {"oid": int(order_id)},
    )
    if not order:
        return {"status": "not_found", "order_id": order_id}

    assessment = rules.evaluate_return_request(
        order_status=order["status"],
        order_date=_parse_date(order["order_date"]),
        delivery_date=_parse_date(order["estimated_delivery"]),
    )
    return {
        "status": "ok",
        "order_id": order["order_id"],
        "order_total_display": brl(order["total_brl"]),
        **assessment,
    }


# --- Shipping ------------------------------------------------------------


def quote_shipping(order_total_brl: float, city: str | None = None) -> dict[str, Any]:
    """Shipping terms for an order value and destination.

    Outside metropolitan Campo Grande the fee depends on postcode, weight and
    dimensions, so no amount is returned — only the carriers and estimates.
    Inventing a number is precisely the "propaganda enganosa" section 7.1
    warns against.
    """
    metro_cities = {"campo grande", "terenos", "sidrolandia", "jaraguari", "ribas do rio pardo"}
    metropolitan = city is None or normalise(city) in metro_cities
    quote = rules.quote_shipping(float(order_total_brl), metropolitan=metropolitan)
    if quote.get("fee_brl") is not None:
        quote["fee_display"] = brl(float(quote["fee_brl"]))
    quote["destination"] = city or "Campo Grande (assumido)"
    quote["free_shipping_threshold_display"] = brl(rules.FREE_SHIPPING_THRESHOLD_BRL)
    return {"status": "ok", **quote}


# --- Policies ------------------------------------------------------------


def search_store_policies(question: str, limit: int = 3) -> dict[str, Any]:
    """Find the policy sections that answer a question about store rules.

    The manual is in Brazilian Portuguese. English queries are translated
    internally for common policy vocabulary, but a Portuguese query retrieves
    more reliably.
    """
    hits = policies.search(question, limit=max(1, min(int(limit), 5)))
    return {
        "status": "ok" if hits else "no_match",
        "question": question,
        "sections": [hit.as_dict() for hit in hits],
    }


# --- Declarations --------------------------------------------------------

REGISTRY = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "get_active_promotions": get_active_promotions,
    "get_order_status": get_order_status,
    "check_return_eligibility": check_return_eligibility,
    "quote_shipping": quote_shipping,
    "search_store_policies": search_store_policies,
}

DECLARATIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_products",
        "description": (
            "Search the store catalog by text, category and price range. Use this for "
            "any question about what the store sells, what is in stock, or what fits a "
            "budget. Prices returned already include any active promotion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": (
                        "Words to match against product name, description and specs, "
                        "e.g. 'violão nylon', 'Takamine', 'baixo 5 cordas'. Accent "
                        "insensitive."
                    ),
                },
                "category": {
                    "type": "string",
                    "description": (
                        "One of: Guitarras, Baixos, Baterias e Percussão, Teclados e "
                        "Pianos, Violões, Instrumentos de Sopro (Madeiras), "
                        "Instrumentos de Sopro (Metais), Cordas Orquestrais, Ukuleles."
                    ),
                },
                "max_price_brl": {"type": "number", "description": "Upper price limit in BRL."},
                "min_price_brl": {"type": "number", "description": "Lower price limit in BRL."},
                "only_available": {
                    "type": "boolean",
                    "description": (
                        "Default true: return only purchasable products. Items that "
                        "matched but are out of stock, discontinued or not yet "
                        "released are summarised separately in 'withheld'."
                    ),
                },
                "limit": {"type": "integer", "description": "Maximum results, up to 25."},
            },
        },
    },
    {
        "type": "function",
        "name": "get_product_details",
        "description": (
            "Full detail for a single product: description, specs, availability and "
            "every payment option with its total. Use this when the customer asks the "
            "price of a specific model or how they can pay for it. Give product_id when "
            "known, otherwise name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "integer", "description": "Catalog id."},
                "name": {
                    "type": "string",
                    "description": (
                        "Product name or part of it, e.g. 'Takamine GD20'. If several "
                        "products match, the candidates are returned for you to "
                        "disambiguate with the customer — never pick one yourself."
                    ),
                },
            },
        },
    },
    {
        "type": "function",
        "name": "get_active_promotions",
        "description": (
            "List every promotion currently in force, with original price, discount "
            "percentage and promotional price. Use this whenever the customer asks "
            "about discounts, offers or a specific campaign. Promotions absent from "
            "this list are not active and must never be offered."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "get_order_status",
        "description": (
            "Status, items, payment method and tracking for one order. Use this for any "
            "question about an existing order. Requires the order number."
        ),
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "integer", "description": "Order number."}},
            "required": ["order_id"],
        },
    },
    {
        "type": "function",
        "name": "check_return_eligibility",
        "description": (
            "Assess whether an order can still be returned or exchanged. Returns all "
            "four routes — right of withdrawal, defect exchange, preference exchange "
            "and legal warranty — each with its own deadline and conditions, because "
            "they run from different dates. Use this before answering any question "
            "about giving a product back, swapping it, or a defect."
        ),
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "integer", "description": "Order number."}},
            "required": ["order_id"],
        },
    },
    {
        "type": "function",
        "name": "quote_shipping",
        "description": (
            "Shipping terms for an order value and destination city: fee or free "
            "shipping inside metropolitan Campo Grande, carriers and estimates "
            "elsewhere. Outside the metropolitan region no amount is returned because "
            "it depends on postcode, weight and dimensions — do not invent one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_total_brl": {
                    "type": "number",
                    "description": "Order value in BRL, which decides free shipping.",
                },
                "city": {
                    "type": "string",
                    "description": "Destination city. Defaults to Campo Grande.",
                },
            },
            "required": ["order_total_brl"],
        },
    },
    {
        "type": "function",
        "name": "search_store_policies",
        "description": (
            "Look up the store's written policies: opening hours, address and contact, "
            "payment terms and instalment rules, returns and exchanges, shipping, "
            "promotions, warranty, privacy. Use this for any question about store rules "
            "rather than answering from memory. Queries in Portuguese retrieve best."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The customer's question, or its key policy terms.",
                },
                "limit": {"type": "integer", "description": "Sections to return, up to 5."},
            },
            "required": ["question"],
        },
    },
]


def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a model-requested tool call."""
    function = REGISTRY.get(name)
    if function is None:
        return {"status": "unknown_tool", "tool": name, "available": sorted(REGISTRY)}
    try:
        return function(**arguments)
    except TypeError as exc:
        # A bad signature is the model's mistake to correct, so it is reported
        # back as data rather than raised into the loop.
        return {"status": "invalid_arguments", "tool": name, "detail": str(exc)}
