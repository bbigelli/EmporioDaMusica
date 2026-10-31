"""Lexical retrieval over the store policy manual.

## Why BM25 and not embeddings

The manual is eight pages — roughly 12,000 characters across 25 indexed
sections. At that size a vector store buys nothing and costs something: an
embedding call (or a local model download) on every query, an index to build
and keep in sync, and a dependency to install before the project runs. BM25
over 25 short documents is exact, instant, offline, and explains its own hits.

Embeddings earn their keep when a corpus is large enough that lexical recall
breaks down, or when users phrase questions with vocabulary the documents never
use. Mostly the second does not hold here: policy questions reuse the
document's own nouns — troca, frete, garantia, boleto, horário.

Where it *does* break down, it breaks down in two specific, cheap-to-fix ways,
both of which are vocabulary problems rather than semantic ones:

1. **Portuguese morphology.** A customer writes "posso parcelar em 10x" and the
   manual says "Parcelamento" and "parcela". Measured on this corpus, that
   query scored zero hits before prefix matching was added.
2. **Cross-language querying.** The agent is bilingual; the corpus is not. A
   small expansion table maps English policy nouns onto the Portuguese ones.

Terms are weighted by how they entered the query — verbatim, translated, or
prefix-matched — so a stemmed near-miss cannot outrank an exact hit.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from emporio.config import get_settings

K1 = 1.5
B = 0.75

# Weight added when query terms also appear in a section's title or path. A hit
# in "4.1 Direito de Arrependimento" is stronger evidence than the same word
# buried in a paragraph.
TITLE_BOOST = 0.6

# Confidence in a term by provenance: as the customer typed it, mapped from
# another language, or recovered by prefix match.
WEIGHT_VERBATIM = 1.0
WEIGHT_EXPANSION = 0.9
WEIGHT_PREFIX = 0.7

# Shortest prefix allowed to link a query term to an indexed one. Five is long
# enough that "parcel-" reaches "parcelamento" and "parcela" without "produto"
# reaching "produção".
MIN_PREFIX = 5

# The ingestion notes are instructions for the agent, not an answer to any
# customer question. They are injected into the system prompt wholesale and
# excluded from the search index, where their brevity made BM25's length
# normalisation rank them above real sections.
NOTES_SECTION_ID = "notes"

# Tokens are accent-stripped before this set is consulted, so entries here are
# written without diacritics.
STOPWORDS = frozenset(
    """
    a ao aos as com da das de do dos e em entre eu na nas no nos num numa o os ou
    para pela pelas pelo pelos por que se sem ser sobre sua suas seu seus um uma
    voces voce eh tem ter tenho posso pode podem quero queria gostaria qual quais
    """.split()
)

# English policy vocabulary mapped onto the words the manual actually uses.
QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "hours": ("horario", "funcionamento"),
    "hour": ("horario", "funcionamento"),
    "open": ("horario", "funcionamento", "aberto"),
    "opening": ("horario", "funcionamento"),
    "closed": ("fechado", "horario"),
    "sunday": ("domingo",),
    "saturday": ("sabado",),
    "address": ("endereco", "rua"),
    "location": ("endereco", "rua"),
    "store": ("loja",),
    "phone": ("telefone", "whatsapp"),
    "payment": ("pagamento", "formas"),
    "pay": ("pagamento",),
    "instalment": ("parcelamento", "parcela"),
    "installment": ("parcelamento", "parcela"),
    "card": ("cartao", "credito"),
    "credit": ("credito", "cartao"),
    "cash": ("vista", "pix"),
    "discount": ("desconto", "promocao"),
    "promotion": ("promocao", "desconto"),
    "sale": ("promocao", "liquidacao"),
    "return": ("devolucao", "arrependimento", "troca"),
    "returns": ("devolucao", "arrependimento", "troca"),
    "refund": ("reembolso", "devolucao"),
    "exchange": ("troca",),
    "defect": ("defeito",),
    "defective": ("defeito",),
    "broken": ("defeito",),
    "warranty": ("garantia",),
    "guarantee": ("garantia",),
    "shipping": ("frete", "entrega"),
    "delivery": ("entrega", "frete", "prazo"),
    "tracking": ("rastreamento", "codigo"),
    "track": ("rastreamento",),
    "privacy": ("privacidade", "dados", "lgpd"),
    "complaint": ("reclamacao",),
    "order": ("pedido",),
    # Colloquial Portuguese outside the manual's register, where prefix
    # matching alone cannot bridge the gap.
    "quebrou": ("defeito",),
    "estragou": ("defeito",),
    "arrependi": ("arrependimento", "devolucao"),
    "abre": ("horario", "funcionamento"),
    "fecha": ("horario", "funcionamento", "fechado"),
    "vezes": ("parcelamento", "parcela"),
}


@dataclass(frozen=True)
class PolicySection:
    id: str
    title: str
    path: str
    level: int
    text: str
    tokens: tuple[str, ...]
    title_tokens: frozenset[str]


@dataclass(frozen=True)
class PolicyHit:
    section: PolicySection
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "section": self.section.id,
            "title": self.section.title,
            "path": self.section.path,
            "text": self.section.text,
            "score": round(self.score, 3),
        }


def normalise(text: str) -> str:
    """Casefold and strip accents.

    Customers type "horario" as often as "horário", and a policy lookup must
    not depend on which. Stripping diacritics is safe for the *index*
    precisely because the indexed text is kept intact alongside it — only
    matching is accent-blind, never what gets shown to the customer.
    """
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def tokenise(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[0-9a-z]+", normalise(text))
        if token not in STOPWORDS and len(token) > 1
    ]


@lru_cache(maxsize=1)
def _all_sections() -> tuple[PolicySection, ...]:
    settings = get_settings()
    if not settings.policy_sections_path.exists():
        raise RuntimeError(
            f"{settings.policy_sections_path} is missing. "
            "Run: python scripts/build_policies.py"
        )
    raw = json.loads(settings.policy_sections_path.read_text(encoding="utf-8"))
    return tuple(
        PolicySection(
            id=str(unit["id"]),
            title=str(unit["title"]),
            path=str(unit["path"]),
            level=int(unit["level"]),
            text=str(unit["text"]),
            tokens=tuple(tokenise(f"{unit['title']} {unit['text']}")),
            title_tokens=frozenset(tokenise(str(unit["path"]))),
        )
        for unit in raw
    )


def load_sections() -> tuple[PolicySection, ...]:
    """Searchable sections, excluding the agent-facing ingestion notes."""
    return tuple(
        section for section in _all_sections() if section.id != NOTES_SECTION_ID
    )


def pipeline_notes() -> str:
    """Contradictions found in the source document, for the system prompt."""
    for section in _all_sections():
        if section.id == NOTES_SECTION_ID:
            return section.text
    return ""


@lru_cache(maxsize=1)
def _index() -> tuple[tuple[PolicySection, ...], dict[str, int], float, frozenset[str]]:
    sections = load_sections()
    document_frequency: Counter[str] = Counter()
    for section in sections:
        document_frequency.update(set(section.tokens))
    average_length = sum(len(section.tokens) for section in sections) / len(sections)
    return sections, dict(document_frequency), average_length, frozenset(document_frequency)


def build_query_terms(query: str) -> dict[str, float]:
    """Turn a question into weighted search terms.

    Prefix matching only fires for terms absent from the index, so an exact
    word is never diluted by its own morphological cousins.
    """
    _, _, _, vocabulary = _index()
    terms: dict[str, float] = {}

    def add(term: str, weight: float) -> None:
        if term in terms:
            terms[term] = max(terms[term], weight)
        else:
            terms[term] = weight

    tokens = tokenise(query)
    for token in tokens:
        add(token, WEIGHT_VERBATIM)
        for expansion in QUERY_EXPANSIONS.get(token, ()):
            add(expansion, WEIGHT_EXPANSION)

    for token in tokens:
        if token in vocabulary or len(token) < MIN_PREFIX:
            continue
        prefix = token[:MIN_PREFIX]
        for candidate in vocabulary:
            if candidate.startswith(prefix):
                add(candidate, WEIGHT_PREFIX)

    return terms


def search(query: str, limit: int = 3) -> list[PolicyHit]:
    """Return the policy sections most likely to answer a question."""
    sections, document_frequency, average_length, _ = _index()
    terms = build_query_terms(query)
    if not terms:
        return []

    total = len(sections)
    hits: list[PolicyHit] = []
    for section in sections:
        frequencies = Counter(section.tokens)
        length = len(section.tokens) or 1
        score = 0.0
        for term, weight in terms.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            score += (
                weight
                * idf
                * (frequency * (K1 + 1))
                / (frequency + K1 * (1 - B + B * length / average_length))
            )
        if score <= 0:
            continue
        matched = {term for term in terms if term in section.title_tokens}
        overlap = len(matched) / len(terms)
        hits.append(PolicyHit(section=section, score=score * (1 + TITLE_BOOST * overlap)))

    hits.sort(key=lambda hit: (-hit.score, hit.section.id))
    return hits[:limit]


def full_manual() -> str:
    """The whole curated manual, for prompts that embed it wholesale."""
    settings = get_settings()
    if not settings.policy_markdown_path.exists():
        raise RuntimeError(
            f"{settings.policy_markdown_path} is missing. "
            "Run: python scripts/build_policies.py"
        )
    return settings.policy_markdown_path.read_text(encoding="utf-8")
