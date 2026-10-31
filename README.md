# Empório da Música — customer service agent

A prototype text-messaging agent that answers customer questions for Empório da
Música, a (fictional) musical instrument store in Campo Grande, MS, Brazil.

The agent answers from two sources and has to know which one a question needs:

- **Operational data** (`data/raw/*.csv`) — catalog, stock, prices, promotions,
  orders, customers. Loaded into SQLite and exposed through typed tools.
- **The store policy manual** (`data/raw/store_policies.pdf`) — opening hours,
  payment terms, returns, shipping, warranty, service guidelines.

## Status

Work in progress.

- [x] Data pipeline: CSV → SQLite, with a data quality report
- [x] Policy pipeline: PDF → curated markdown and retrieval units
- [ ] Deterministic business rules
- [ ] Agent tools
- [ ] Agent loop and persona
- [ ] CLI
- [ ] Interaction examples

## Running the pipelines

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate elsewhere
pip install -r requirements.txt

python scripts/build_data.py       # -> data/processed/store.db
python scripts/build_policies.py   # -> data/processed/store_policies.md + policy_sections.json

python -m unittest discover -s tests -v
```

Generated artifacts are committed, so the agent runs without re-running either
pipeline. They only need to run when a source file changes.

## Technical decisions so far

### SQLite built offline, not CSVs parsed per request

`products.specs` is a JSON document serialised inside a CSV cell, with keys
that vary by instrument family. Parsing it once at build time turns those keys
into queryable columns, so a tool can filter on `spec_strings = 'nylon'`
instead of the model guessing from free text.

Two business rules live in the schema rather than in the prompt, because a rule
the model has to remember is a rule it can forget:

- The `product_pricing` view applies **only** `is_active` promotions, and takes
  the single best discount rather than stacking them — section 6.2 of the
  manual states promotions are not cumulative. 21 of the 25 promotions in the
  dataset are expired.
- The view derives an `availability` field, because `status` alone cannot
  answer "can the customer buy this today?". A product can be `active` with
  zero stock, or `discontinued` with stock remaining, and section 7.3
  prescribes a different response for each case.

`scripts/build_data.py` also emits
[`docs/data-quality-report.md`](docs/data-quality-report.md), measured at build
time rather than transcribed by hand.

### The policy PDF is converted offline, with its tables pinned

Extraction quality varies sharply by tool, and the failures are silent — the
worst property a policy source can have. Measured on this document:

| Method | Result |
| --- | --- |
| `pdftotext -layout` | reorders table cells: the opening-hours table comes out as "Segunda a Sexta-feira → Fechado" |
| `pdftotext` (default) | encodes to latin-1, not UTF-8, mangling every accent |
| `pypdf` default mode | one word per line; all structure lost |
| `pypdf` layout mode | reconstructs every table in this document correctly |

So the pipeline uses `pypdf` layout mode, and the four tables are additionally
pinned to curated markdown. Every curated cell is asserted against the
extracted text: if the PDF changes, the build fails instead of shipping a stale
policy. A wrong refund window or a wrong closing time is a customer-facing
error, not a formatting nit — `tests/test_pipelines.py` locks that behaviour.

The pipeline also appends a short "notas do pipeline de ingestão" section
recording three contradictions found inside the source document, so the agent
does not repeat them to a customer: a catalog size that does not match the
data, two different phone numbers, and an email address spelled with an accent
in the domain.

### A configurable reference date

The supplied orders span 2026-01 to 2026-03-22. The manual's deadlines — 7 days
to withdraw, 30 days to exchange a defect, 90 days of legal warranty — are all
measured from delivery, so running against a real clock months past the data
expires every one of them and makes the returns policy untestable. "Today"
therefore comes from `EMPORIO_REFERENCE_DATE` (see `.env.example`), defaulting
to a date just after the dataset ends. Unset it to use the system clock.

---

Remaining sections — model and provider choice, retrieval strategy, prompt
design, known limitations, and use of coding assistants — are written as those
parts land.
