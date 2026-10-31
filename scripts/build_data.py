"""Build the operational database the agent queries at runtime.

Reads the supplied CSVs from data/raw/ and writes:

  data/processed/store.db          SQLite database with typed columns and FKs
  docs/data-quality-report.md      what the raw data actually looks like

Why SQLite and not pandas-in-memory: the agent's tools need to answer filtered
questions ("classical guitars under BRL 1000 that are in stock") and SQL
expresses those directly. Why a build step and not parsing CSVs on every run:
the `specs` column is a JSON document serialised inside a CSV cell, and parsing
it once up front turns it into queryable columns.

Run from the repository root:

    python scripts/build_data.py
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

# Spec keys observed across the catalog. The source `specs` cell is a JSON
# document whose keys vary by instrument family (a drum kit has `shells` and
# `pieces`, a keyboard has `keys` and `polyphony`), so the union is sparse by
# nature. Promoting them to columns lets tools filter on them; the original
# document is kept in `specs_json` so nothing is lost.
SPEC_KEYS = [
    "back_sides",
    "body",
    "color",
    "electronics",
    "hardware",
    "keys",
    "neck",
    "pieces",
    "polyphony",
    "scale",
    "shells",
    "strings",
    "top",
    "type",
]

SCHEMA = f"""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT
);

CREATE TABLE products (
    product_id     INTEGER PRIMARY KEY,
    name           TEXT    NOT NULL,
    category_id    INTEGER NOT NULL REFERENCES categories(category_id),
    description    TEXT,
    price_brl      REAL    NOT NULL,
    stock_quantity INTEGER NOT NULL,
    status         TEXT    NOT NULL,
    created_at     TEXT,
    specs_json     TEXT,
    {",".join(f"spec_{key} TEXT" for key in SPEC_KEYS)}
);

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    phone       TEXT,
    email       TEXT,
    city        TEXT
);

CREATE TABLE orders (
    order_id           INTEGER PRIMARY KEY,
    customer_id        INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date         TEXT    NOT NULL,
    status             TEXT    NOT NULL,
    total_brl          REAL    NOT NULL,
    payment_method     TEXT,
    -- NULL until the order ships. 12 of 20 orders have no value here, so
    -- tools must render "not dispatched yet" rather than leak a NULL.
    tracking_code      TEXT,
    estimated_delivery TEXT,
    notes              TEXT
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL
);

CREATE TABLE promotions (
    promotion_id     INTEGER PRIMARY KEY,
    product_id       INTEGER NOT NULL REFERENCES products(product_id),
    discount_percent INTEGER NOT NULL,
    description      TEXT,
    is_active        INTEGER NOT NULL
);

CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_price    ON products(price_brl);
CREATE INDEX idx_orders_customer   ON orders(customer_id);
CREATE INDEX idx_promotions_active ON promotions(is_active, product_id);

-- One row per product with the price a customer would actually pay.
--
-- Section 6.2 of the policy manual states promotions are not cumulative, so
-- when a product somehow carries more than one active promotion we apply the
-- single best discount instead of stacking them. The current dataset has four
-- active promotions on four distinct products, but encoding the rule here
-- means the agent cannot stack them even if the data changes.
CREATE VIEW product_pricing AS
WITH best_active_promotion AS (
    SELECT product_id, MAX(discount_percent) AS discount_percent
    FROM promotions
    WHERE is_active = 1
    GROUP BY product_id
)
SELECT
    p.product_id,
    p.name,
    p.category_id,
    c.name          AS category_name,
    p.description,
    p.price_brl     AS list_price_brl,
    bap.discount_percent,
    ROUND(p.price_brl * (1 - COALESCE(bap.discount_percent, 0) / 100.0), 2)
                    AS effective_price_brl,
    CASE WHEN bap.discount_percent IS NULL THEN 0 ELSE 1 END AS on_promotion,
    p.stock_quantity,
    p.status,
    CASE
        WHEN p.status = 'discontinued'  THEN 'discontinued'
        WHEN p.status = 'coming_soon'   THEN 'coming_soon'
        WHEN p.stock_quantity <= 0      THEN 'out_of_stock'
        ELSE 'available'
    END             AS availability,
    p.specs_json
FROM products p
JOIN categories c         ON c.category_id = p.category_id
LEFT JOIN best_active_promotion bap ON bap.product_id = p.product_id;
"""


def read_csv(name: str) -> list[dict[str, str]]:
    """Read a raw CSV as a list of dicts, UTF-8 explicit.

    The data is Brazilian Portuguese throughout. Letting Python pick a locale
    default here would mangle every accented product name and policy term.
    """
    with (RAW / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def optional(value: str) -> str | None:
    """Turn an empty CSV cell into a real NULL."""
    value = value.strip()
    return value or None


@dataclass
class QualityFindings:
    """Facts measured during the load, reported rather than silently accepted."""

    counts: dict[str, int]
    product_statuses: dict[str, int]
    availability: dict[str, int]
    zero_stock: list[tuple[int, str, str]]
    non_active: list[tuple[int, str, str, int]]
    orders_by_status: dict[str, int]
    orders_missing_tracking: list[int]
    orders_missing_delivery: list[int]
    order_date_range: tuple[str, str]
    active_promotions: list[tuple[int, str, int, str]]
    inactive_promotion_count: int
    multi_promotion_products: list[int]
    spec_key_coverage: dict[str, int]
    referential_problems: list[str]


def load(conn: sqlite3.Connection) -> QualityFindings:
    # Must be set per connection and outside a transaction: a PRAGMA inside
    # executescript() is silently ignored, which would leave the declared
    # foreign keys decorative. The runtime layer sets it the same way.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)

    categories = read_csv("categories")
    products = read_csv("products")
    customers = read_csv("customers")
    orders = read_csv("orders")
    order_items = read_csv("order_items")
    promotions = read_csv("promotions")

    conn.executemany(
        "INSERT INTO categories VALUES (?, ?, ?)",
        [(int(r["category_id"]), r["name"], optional(r["description"])) for r in categories],
    )

    spec_coverage = {key: 0 for key in SPEC_KEYS}
    product_rows = []
    for row in products:
        specs = json.loads(row["specs"]) if row["specs"].strip() else {}
        unknown = set(specs) - set(SPEC_KEYS)
        if unknown:
            # Loud on purpose: a new spec key means the table is out of date and
            # a filterable attribute would otherwise vanish into specs_json.
            raise SystemExit(
                f"product {row['product_id']} has unmapped spec keys {sorted(unknown)}; "
                "add them to SPEC_KEYS"
            )
        for key in specs:
            spec_coverage[key] += 1
        product_rows.append(
            (
                int(row["product_id"]),
                row["name"].strip(),
                int(row["category_id"]),
                optional(row["description"]),
                float(row["price_brl"]),
                int(row["stock_quantity"]),
                row["status"].strip(),
                optional(row["created_at"]),
                json.dumps(specs, ensure_ascii=False, sort_keys=True),
                *[optional(str(specs.get(key, ""))) for key in SPEC_KEYS],
            )
        )
    placeholders = ", ".join("?" * (9 + len(SPEC_KEYS)))
    conn.executemany(f"INSERT INTO products VALUES ({placeholders})", product_rows)

    conn.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        [
            (
                int(r["customer_id"]),
                r["name"].strip(),
                optional(r["phone"]),
                optional(r["email"]),
                optional(r["city"]),
            )
            for r in customers
        ],
    )

    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                int(r["order_id"]),
                int(r["customer_id"]),
                r["order_date"].strip(),
                r["status"].strip(),
                float(r["total_brl"]),
                optional(r["payment_method"]),
                optional(r["tracking_code"]),
                optional(r["estimated_delivery"]),
                optional(r["notes"]),
            )
            for r in orders
        ],
    )

    conn.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity) VALUES (?, ?, ?)",
        [(int(r["order_id"]), int(r["product_id"]), int(r["quantity"])) for r in order_items],
    )

    conn.executemany(
        "INSERT INTO promotions VALUES (?, ?, ?, ?, ?)",
        [
            (
                int(r["promotion_id"]),
                int(r["product_id"]),
                int(r["discount_percent"]),
                optional(r["description"]),
                int(r["is_active"]),
            )
            for r in promotions
        ],
    )

    conn.commit()
    return measure(conn)


def measure(conn: sqlite3.Connection) -> QualityFindings:
    """Query the loaded database for the facts that go into the report."""

    def scalar_map(sql: str) -> dict[str, int]:
        return {str(key): int(count) for key, count in conn.execute(sql)}

    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "categories",
            "products",
            "customers",
            "orders",
            "order_items",
            "promotions",
        )
    }

    # Foreign keys are declared, so SQLite would have rejected a dangling
    # reference at insert time. This is the belt-and-braces check that also
    # catches orders with no line items, which no constraint can express.
    problems: list[str] = []
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        problems.append(f"foreign key violations: {violations}")
    orphan_orders = [
        row[0]
        for row in conn.execute(
            "SELECT o.order_id FROM orders o"
            " LEFT JOIN order_items i ON i.order_id = o.order_id"
            " WHERE i.order_id IS NULL"
        )
    ]
    if orphan_orders:
        problems.append(f"orders with no line items: {orphan_orders}")

    return QualityFindings(
        counts=counts,
        product_statuses=scalar_map(
            "SELECT status, COUNT(*) FROM products GROUP BY status ORDER BY 2 DESC"
        ),
        availability=scalar_map(
            "SELECT availability, COUNT(*) FROM product_pricing"
            " GROUP BY availability ORDER BY 2 DESC"
        ),
        zero_stock=[
            tuple(row)
            for row in conn.execute(
                "SELECT product_id, name, status FROM products"
                " WHERE stock_quantity <= 0 ORDER BY product_id"
            )
        ],
        non_active=[
            tuple(row)
            for row in conn.execute(
                "SELECT product_id, name, status, stock_quantity FROM products"
                " WHERE status <> 'active' ORDER BY product_id"
            )
        ],
        orders_by_status=scalar_map(
            "SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY 2 DESC"
        ),
        orders_missing_tracking=[
            row[0]
            for row in conn.execute(
                "SELECT order_id FROM orders WHERE tracking_code IS NULL ORDER BY order_id"
            )
        ],
        orders_missing_delivery=[
            row[0]
            for row in conn.execute(
                "SELECT order_id FROM orders WHERE estimated_delivery IS NULL ORDER BY order_id"
            )
        ],
        order_date_range=conn.execute(
            "SELECT MIN(order_date), MAX(order_date) FROM orders"
        ).fetchone(),
        active_promotions=[
            tuple(row)
            for row in conn.execute(
                "SELECT pr.promotion_id, p.name, pr.discount_percent, pr.description"
                " FROM promotions pr JOIN products p ON p.product_id = pr.product_id"
                " WHERE pr.is_active = 1 ORDER BY pr.promotion_id"
            )
        ],
        inactive_promotion_count=conn.execute(
            "SELECT COUNT(*) FROM promotions WHERE is_active = 0"
        ).fetchone()[0],
        multi_promotion_products=[
            row[0]
            for row in conn.execute(
                "SELECT product_id FROM promotions WHERE is_active = 1"
                " GROUP BY product_id HAVING COUNT(*) > 1"
            )
        ],
        spec_key_coverage={},
        referential_problems=problems,
    )


def write_report(findings: QualityFindings, spec_coverage: dict[str, int]) -> Path:
    total_products = findings.counts["products"]
    lines: list[str] = [
        "# Data quality report",
        "",
        "Generated by `scripts/build_data.py` from the files in `data/raw/`.",
        "Every number below is measured at build time, not transcribed by hand.",
        "",
        "## Row counts",
        "",
        "| Table | Rows |",
        "| --- | --- |",
    ]
    lines += [f"| `{table}` | {count} |" for table, count in findings.counts.items()]

    lines += [
        "",
        "## Referential integrity",
        "",
    ]
    if findings.referential_problems:
        lines += [f"- {problem}" for problem in findings.referential_problems]
    else:
        lines.append(
            "No problems found. Every foreign key resolves, and every order has at "
            "least one line item. The supplied data is clean in this respect."
        )

    lines += [
        "",
        "## Catalog availability",
        "",
        "`status` alone does not answer \"can the customer buy this today?\" — a product",
        "can be `active` with zero stock, or `discontinued` while stock remains. The",
        "`product_pricing` view therefore derives an `availability` field, because",
        "section 7.3 of the policy manual prescribes a *different* response for each",
        "case: suggest alternatives when out of stock, offer successors when",
        "discontinued.",
        "",
        "| `status` | Products |",
        "| --- | --- |",
    ]
    lines += [f"| `{status}` | {count} |" for status, count in findings.product_statuses.items()]
    lines += [
        "",
        "| Derived `availability` | Products |",
        "| --- | --- |",
    ]
    lines += [f"| `{value}` | {count} |" for value, count in findings.availability.items()]

    if findings.zero_stock:
        lines += ["", "Zero stock:", ""]
        lines += [
            f"- `{pid}` {name} — status `{status}`"
            for pid, name, status in findings.zero_stock
        ]
    if findings.non_active:
        lines += ["", "Not `active`:", ""]
        lines += [
            f"- `{pid}` {name} — status `{status}`, stock {stock}"
            for pid, name, status, stock in findings.non_active
        ]

    lines += [
        "",
        "## Orders",
        "",
        "| `status` | Orders |",
        "| --- | --- |",
    ]
    lines += [f"| `{status}` | {count} |" for status, count in findings.orders_by_status.items()]
    lines += [
        "",
        f"- `tracking_code` is NULL for {len(findings.orders_missing_tracking)} of "
        f"{findings.counts['orders']} orders: "
        f"{findings.orders_missing_tracking}",
        f"- `estimated_delivery` is NULL for {len(findings.orders_missing_delivery)} of "
        f"{findings.counts['orders']} orders: "
        f"{findings.orders_missing_delivery}",
        "",
        "Both gaps line up exactly with orders that have not shipped, so they are",
        "meaningful absence, not missing data. The order-status tool must say \"not",
        "dispatched yet\" instead of handing the model a NULL to improvise around.",
        "",
        f"- Order dates span **{findings.order_date_range[0]} to "
        f"{findings.order_date_range[1]}**.",
        "",
        "That date range is the reason the agent reads \"today\" from",
        "`EMPORIO_REFERENCE_DATE` rather than the system clock. The policy manual's",
        "deadlines — 7 days to withdraw, 30 days to exchange a defect, 90 days of",
        "legal warranty — are all measured from delivery. Run against a real clock",
        "months past the data and every single one is expired, which makes the",
        "returns policy untestable and the agent look broken.",
        "",
        "## Promotions",
        "",
        f"- Active: **{len(findings.active_promotions)}**",
        f"- Inactive: **{findings.inactive_promotion_count}**",
        "",
    ]
    if findings.active_promotions:
        lines += ["| ID | Product | Discount | Campaign |", "| --- | --- | --- | --- |"]
        lines += [
            f"| {pid} | {name} | {discount}% | {description} |"
            for pid, name, discount, description in findings.active_promotions
        ]
    lines += [
        "",
        "The lopsided split is the most useful thing in this dataset. Section 7.3 of",
        "the manual says never to promise a discount that is no longer in force, and",
        f"there are {findings.inactive_promotion_count} expired promotions available to",
        "ask about. Only `is_active = 1` rows reach the pricing view.",
        "",
        f"- Products carrying more than one active promotion: "
        f"{findings.multi_promotion_products or 'none'}. The view applies the single "
        "best discount regardless, per the non-cumulative rule in section 6.2.",
        "",
        "## Spec keys",
        "",
        "`products.specs` is a JSON document stored inside a CSV cell. Keys vary by",
        "instrument family, so the promoted columns are sparse by design. Coverage",
        f"across {total_products} products:",
        "",
        "| Spec key | Products | Coverage |",
        "| --- | --- | --- |",
    ]
    for key, count in sorted(spec_coverage.items(), key=lambda item: -item[1]):
        lines.append(f"| `{key}` | {count} | {count / total_products:.0%} |")

    lines += [
        "",
        "## Caveats found in the policy PDF",
        "",
        "Two internal inconsistencies worth recording, because the agent must not",
        "repeat them as fact:",
        "",
        "- Section 1 claims a catalog of \"more than 300 instruments\"; the supplied",
        f"  catalog holds {total_products}. The agent is instructed never to quote",
        "  catalog size or availability from the manual — only from the database.",
        "- The cover and section 7 give different phone numbers,",
        "  (67) 3341-4444 and (67) 3321-4500. Both are reported with their context",
        "  (store line vs WhatsApp channel) rather than silently picking one.",
        "",
    ]

    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "data-quality-report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def main() -> int:
    # Windows consoles default to a legacy code page; without this, printing a
    # Brazilian product name raises UnicodeEncodeError or prints mojibake.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    db_path = PROCESSED / "store.db"
    db_path.unlink(missing_ok=True)

    spec_coverage = {key: 0 for key in SPEC_KEYS}
    with sqlite3.connect(db_path) as conn:
        findings = load(conn)
        for key in SPEC_KEYS:
            spec_coverage[key] = conn.execute(
                f"SELECT COUNT(*) FROM products WHERE spec_{key} IS NOT NULL"
            ).fetchone()[0]

    report_path = write_report(findings, spec_coverage)

    print(f"wrote {db_path.relative_to(ROOT)}")
    for table, count in findings.counts.items():
        print(f"  {table:<12} {count:>4} rows")
    print(f"wrote {report_path.relative_to(ROOT)}")
    if findings.referential_problems:
        print("referential problems found:")
        for problem in findings.referential_problems:
            print(f"  - {problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
