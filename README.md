# Empório da Música — customer service agent

A prototype text-messaging agent that answers customer questions for Empório da
Música, a (fictional) musical instrument store in Campo Grande, MS, Brazil.

The agent answers from two sources and has to know which one a question needs:

- **Operational data** (`data/raw/*.csv`) — catalog, stock, prices, promotions,
  orders, customers. Loaded into SQLite and exposed through typed tools.
- **The store policy manual** (`data/raw/store_policies.pdf`) — opening hours,
  payment terms, returns, shipping, warranty, service guidelines.

Full setup instructions and the rationale behind each technical decision are
documented below as the project takes shape.

## Status

Work in progress.

- [x] Data pipeline: CSV → SQLite, with a data quality report
- [x] Policy pipeline: PDF → curated markdown
- [ ] Deterministic business rules
- [ ] Agent tools
- [ ] Agent loop and persona
- [ ] CLI
- [ ] Interaction examples
