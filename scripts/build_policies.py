"""Turn the store policy PDF into retrievable markdown.

Reads data/raw/store_policies.pdf and writes:

  data/processed/store_policies.md        human-readable, one heading per section
  data/processed/policy_sections.json     the same content as retrieval units

Both artifacts are committed, so running the agent never requires parsing the
PDF. This script only needs to run when the source document changes.

## Why the PDF is not parsed at query time

Extraction quality varies sharply by tool and the failures are silent, which is
the worst property a policy source can have. Measured on this document:

  pdftotext -layout   reorders table cells; the opening-hours table comes out
                      as "Segunda a Sexta-feira -> Fechado", which would have
                      the agent tell customers the store is shut on weekdays
  pdftotext (default) encodes to latin-1, not UTF-8, mangling every accent
  pypdf default mode  emits one word per line, destroying all structure
  pypdf layout mode   reconstructs every table in this document correctly

So this script uses pypdf's layout mode, and the four tables are additionally
pinned to curated markdown (see CURATED_TABLES). Every curated cell is asserted
against the extracted text: if the PDF ever changes, the build fails loudly
instead of shipping a stale policy. A wrong refund window or a wrong closing
time is a customer-facing error, not a formatting nit.

Run from the repository root:

    python scripts/build_policies.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PDF = ROOT / "data" / "raw" / "store_policies.pdf"
PROCESSED = ROOT / "data" / "processed"

# Repeated running header and page footer, dropped before parsing.
BOILERPLATE = (
    re.compile(r"^Empório da Música Manual de Políticas e Procedimentos\s*$"),
    re.compile(r"^\s*Página \d+\s*$"),
    re.compile(r"^\s*Empório da Música\s*[—-]\s*Sua música começa aqui\.\s*$"),
    re.compile(r"^\s*Última atualização:.*$"),
)

# A section heading: "4." or "4.1", at column zero. The column-zero test is
# what separates headings from the numbered list in section 7.2, whose items
# ("1. Saudação: ...") are indented four spaces by the layout extractor.
HEADING = re.compile(r"^(\d+(?:\.\d+)?)\.?\s+(\S.*?)\s*$")

BULLET = re.compile(r"^\s*[•·]\s*(.*)$")
NUMBERED_ITEM = re.compile(r"^\s{2,}(\d+)\.\s+(.*)$")

# Three or more consecutive spaces inside a line mean column separation. Used
# to drop raw table rows from sections whose tables are supplied curated.
TABLE_ROW = re.compile(r"\S {3,}\S")

VERSION = re.compile(r"Versão\s+([\d.]+)\s*[—-]\s*(\w+\s+de\s+\d{4})")

# Marks where a raw table stood, so the curated one renders in place.
TABLE_PLACEHOLDER = "<<<CURATED_TABLE>>>"

# Tables transcribed from the PDF and verified cell by cell against it.
# Keyed by the section number they belong to.
CURATED_TABLES: dict[str, list[list[str]]] = {
    "1.2": [
        ["Campo", "Informação"],
        ["Razão Social", "Empório da Música Instrumentos Musicais Ltda."],
        ["Nome Fantasia", "Empório da Música"],
        ["CNPJ", "12.345.678/0001-90"],
        ["Endereço", "Rua 14 de Maio, 3200 — Centro, Campo Grande - MS, 79202-333"],
        ["Telefone / WhatsApp", "(67) 3341-4444"],
        ["E-mail", "contato@empóriodamusica.com.br"],
        ["Fundação", "2008"],
    ],
    "2": [
        ["Dia", "Horário"],
        ["Segunda a Sexta-feira", "09:00 às 18:00"],
        ["Sábado", "09:00 às 13:00"],
        ["Domingo e Feriados", "Fechado"],
    ],
    "3": [
        ["Forma", "Condições", "Observações"],
        ["PIX", "Pagamento à vista", "5% de desconto sobre o preço de tabela"],
        ["Cartão de Débito", "Pagamento à vista", "Todas as bandeiras aceitas"],
        ["Cartão de Crédito", "Até 12x sem juros", "Parcela mínima de R$ 100,00"],
        ["Boleto Bancário", "Pagamento à vista", "Compensação em até 3 dias úteis"],
    ],
    "5.2": [
        ["Modalidade", "Prazo Estimado", "Rastreamento", "Seguro"],
        ["PAC (Correios)", "5 a 12 dias úteis", "Sim", "Incluído"],
        ["SEDEX (Correios)", "2 a 5 dias úteis", "Sim", "Incluído"],
        ["Jadlog (.package)", "3 a 8 dias úteis", "Sim", "Incluído"],
    ],
}


# Contradictions inside the source document, appended to the generated markdown
# and indexed as its own retrieval unit. The agent reads these alongside the
# policy text so it does not repeat a documented error to a customer. Written
# in Portuguese because it joins the Portuguese policy text in the prompt.
MAINTAINER_NOTES = [
    (
        "Tamanho do catálogo: a seção 1 afirma \"mais de 300 instrumentos\", mas o"
        " catálogo disponível no sistema é menor. Nunca citar quantidade de itens,"
        " disponibilidade ou variedade com base neste manual — sempre consultar o"
        " sistema."
    ),
    (
        "Telefones: o manual traz dois números. (67) 3341-4444 é o telefone da loja"
        " (seção 1.2) e (67) 3321-4500 é o canal de WhatsApp (seção 7). Informar"
        " cada um com seu contexto, em vez de escolher um deles."
    ),
    (
        "E-mail: o documento grafa o endereço de duas formas,"
        " contato@empóriodamusica.com.br (seção 1.2) e"
        " contato@emporiodamusica.com.br (capa). A forma correta é a sem acento —"
        " domínios não aceitam caracteres acentuados sem IDN. Informar"
        " contato@emporiodamusica.com.br."
    ),
    (
        "Este manual é um documento interno da equipe de atendimento. Aplicar as"
        " regras ao atender, sem apresentá-lo ao cliente como fonte nem citar"
        " números de seção."
    ),
]


@dataclass
class Section:
    number: str
    title: str
    blocks: list[str] = field(default_factory=list)

    @property
    def level(self) -> int:
        return 1 + self.number.count(".")

    @property
    def heading(self) -> str:
        return f"{self.number}. {self.title}" if self.level == 1 else f"{self.number} {self.title}"


def extract_text() -> str:
    reader = pypdf.PdfReader(SOURCE_PDF)
    return "\n".join(page.extract_text(extraction_mode="layout") for page in reader.pages)


def normalise(text: str) -> str:
    """Collapse whitespace and unify dash forms for substring comparison.

    Table cells wrap across lines in the PDF ("5% de desconto sobre o preço de"
    / "tabela"), so the curated cells can only be checked against a
    whitespace-flattened copy of the source. Unicode is normalised to NFC
    because an accent composed differently is a false mismatch, not a change.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("—", "-").replace("–", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", text).strip()


def verify_curated_tables(source: str) -> list[str]:
    """Check every curated cell still appears in the PDF. Returns mismatches."""
    flat = normalise(source)
    missing: list[str] = []
    for section, rows in CURATED_TABLES.items():
        for row in rows[1:]:  # header labels are formatting, not content
            for cell in row:
                if normalise(cell) not in flat:
                    missing.append(f"section {section}: {cell!r}")
    return missing


def parse_sections(source: str) -> tuple[list[Section], str | None, list[str]]:
    """Split the extracted text into sections, dropping boilerplate.

    Everything before section 1 is the cover page; its content is repeated in
    section 1.2, so only the version string is kept from it.

    Raw table rows are replaced by TABLE_PLACEHOLDER at the position where the
    table appeared, so the curated version lands where the document put it
    rather than at the end of the section. Table cells wrap onto indented
    continuation lines ("5% de desconto sobre o preço de" / "tabela"), which is
    why a table swallows following indented lines until the next heading, list
    item, or unindented paragraph.
    """
    version: str | None = None
    if match := VERSION.search(source):
        version = f"{match.group(1)} ({match.group(2)})"

    sections: list[Section] = []
    current: Section | None = None
    paragraph: list[str] = []
    warnings: list[str] = []
    in_table = False

    def flush() -> None:
        if current is not None and paragraph:
            current.blocks.append(" ".join(paragraph))
        paragraph.clear()

    for raw_line in source.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if not stripped or any(pattern.match(stripped) for pattern in BOILERPLATE):
            flush()
            continue  # a blank line does not end a table: its rows are spaced

        heading_match = HEADING.match(line) if indent == 0 else None
        if heading_match:
            flush()
            in_table = False
            current = Section(number=heading_match.group(1), title=heading_match.group(2))
            sections.append(current)
            continue

        if current is None:
            continue  # still on the cover page

        if bullet := BULLET.match(line):
            flush()
            in_table = False
            current.blocks.append(f"- {bullet.group(1).strip()}")
            continue

        if numbered := NUMBERED_ITEM.match(line):
            flush()
            in_table = False
            current.blocks.append(f"{numbered.group(1)}. {numbered.group(2).strip()}")
            continue

        if TABLE_ROW.search(line):
            flush()
            if not in_table:
                in_table = True
                if current.number in CURATED_TABLES:
                    current.blocks.append(TABLE_PLACEHOLDER)
                else:
                    # Never drop a table silently: either curate it or know
                    # that the content is gone.
                    warnings.append(
                        f"section {current.number} ({current.title}) contains a table"
                        " with no entry in CURATED_TABLES; its rows were dropped"
                    )
            continue

        if in_table and indent > 0:
            continue  # wrapped table cell

        # An indented line that is not part of a table continues the list item
        # above it.
        if indent >= 8 and current.blocks and re.match(r"^[-\d]", current.blocks[-1]):
            current.blocks[-1] += " " + stripped
            continue

        in_table = False
        paragraph.append(stripped)

    flush()
    return sections, version, warnings


def render_markdown(sections: list[Section], version: str | None) -> str:
    lines = [
        "# Empório da Música — Manual de Políticas e Procedimentos",
        "",
        "> Generated by `scripts/build_policies.py` from `data/raw/store_policies.pdf`"
        + (f" (version {version})." if version else ".")
        + " Do not edit by hand.",
        "> Content is in Brazilian Portuguese because it is the store's own"
        " internal document and the agent quotes from it verbatim.",
        "",
    ]
    for section in sections:
        lines += ["#" * (section.level + 1) + f" {section.heading}", ""]
        previous_was_item = False
        for block in section.blocks:
            if block == TABLE_PLACEHOLDER:
                table = CURATED_TABLES[section.number]
                header, *rows = table
                lines.append("| " + " | ".join(header) + " |")
                lines.append("| " + " | ".join("---" for _ in header) + " |")
                lines += ["| " + " | ".join(row) + " |" for row in rows]
                lines.append("")
                previous_was_item = False
                continue
            is_item = bool(re.match(r"^(-|\d+\.)\s", block))
            # Keep consecutive list items adjacent so markdown renders a tight
            # list instead of one paragraph per bullet.
            if is_item and previous_was_item:
                lines.insert(len(lines) - 1, block)
            else:
                lines += [block, ""]
            previous_was_item = is_item

    lines += [
        "## Notas do pipeline de ingestão",
        "",
        "Esta seção não faz parte do manual original. Registra contradições"
        " encontradas no documento, para que não sejam repetidas ao cliente.",
        "",
    ]
    lines += [f"- {note}" for note in MAINTAINER_NOTES]
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_retrieval_units(sections: list[Section]) -> list[dict[str, object]]:
    """One retrieval unit per section.

    The whole manual is ~14k characters across 10 top-level sections, so a
    section is already the right granularity: small enough to return as a
    citation, large enough to carry a complete rule. Sub-sections keep their
    parent's title in `path` so a hit on "4.1" can be reported as belonging to
    the returns policy.
    """
    titles: dict[str, str] = {}
    units: list[dict[str, object]] = []
    for section in sections:
        titles[section.number] = section.title
        parent = section.number.split(".")[0]
        path = section.title if section.level == 1 else f"{titles.get(parent, '')} > {section.title}"

        body: list[str] = []
        for block in section.blocks:
            if block != TABLE_PLACEHOLDER:
                body.append(block)
                continue
            # Flatten the table into "label: value" lines so lexical search can
            # match a question like "sábado" or "boleto" against a cell.
            header, *rows = CURATED_TABLES[section.number]
            body += [
                "; ".join(f"{label}: {value}" for label, value in zip(header, row))
                for row in rows
            ]

        text = "\n".join(body).strip()
        if not text:
            # Section 5 ("Política de Frete e Entregas") is a heading with no
            # body of its own. An empty unit can never match a query and only
            # dilutes the term statistics of the index, so it is skipped; its
            # sub-sections carry the content and the shared vocabulary, so a
            # broad question about shipping still hits 5.1/5.2/5.3.
            continue

        units.append(
            {
                "id": section.number,
                "title": section.title,
                "path": path,
                "level": section.level,
                "text": text,
            }
        )

    units.append(
        {
            "id": "notes",
            "title": "Notas do pipeline de ingestão",
            "path": "Notas do pipeline de ingestão",
            "level": 1,
            "text": "\n".join(f"- {note}" for note in MAINTAINER_NOTES),
        }
    )
    return units


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    source = extract_text()

    mismatches = verify_curated_tables(source)
    if mismatches:
        print("curated table cells no longer found in the PDF:", file=sys.stderr)
        for item in mismatches:
            print(f"  - {item}", file=sys.stderr)
        print(
            "\nThe source document changed. Re-read it and update CURATED_TABLES"
            " before shipping: a stale policy table is a customer-facing error.",
            file=sys.stderr,
        )
        return 1

    sections, version, warnings = parse_sections(source)
    if not sections:
        print("no sections parsed; extraction likely failed", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    md_path = PROCESSED / "store_policies.md"
    json_path = PROCESSED / "policy_sections.json"

    md_path.write_text(render_markdown(sections, version), encoding="utf-8", newline="\n")
    units = build_retrieval_units(sections)
    json_path.write_text(
        json.dumps(units, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    top_level = [s for s in sections if s.level == 1]
    print(f"verified {sum(len(r) - 1 for r in CURATED_TABLES.values())} curated table rows")
    print(f"wrote {md_path.relative_to(ROOT)}")
    print(
        f"  {len(top_level)} top-level sections, {len(sections)} total,"
        f" {len(md_path.read_text(encoding='utf-8'))} characters"
    )
    print(f"wrote {json_path.relative_to(ROOT)} ({len(units)} retrieval units)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
