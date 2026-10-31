# Source file manifest

Files in this directory are the materials supplied with the challenge, byte-for-byte
unchanged. They were renamed to ASCII snake_case so that paths stay portable across
operating systems and safe in shell tooling — the original names carried spaces, a
vendor prefix, and (on the PDF) a non-ASCII character.

| Original name | Name in this repository |
| --- | --- |
| `desafio_tecnico_ai_eng - products.csv` | `products.csv` |
| `desafio_tecnico_ai_eng - orders.csv` | `orders.csv` |
| `desafio_tecnico_ai_eng - order_items.csv` | `order_items.csv` |
| `desafio_tecnico_ai_eng - customers.csv` | `customers.csv` |
| `desafio_tecnico_ai_eng - categories.csv` | `categories.csv` |
| `desafio_tecnico_ai_eng - promotions.csv` | `promotions.csv` |
| `políticas_da_loja.pdf` | `store_policies.pdf` |

Nothing in this directory is edited by the build scripts. Every derived artifact is
written to `data/processed/`.
