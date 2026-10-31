"""Brazilian currency formatting.

Its own module because both the rules layer and the tool layer need it, and
`rules` must not import `tools`. Formatting money is not delegated to the
model: to a Brazilian reader `R$ 1.967,18` and `R$ 1,967.18` differ by a factor
of a thousand.
"""

from __future__ import annotations


def brl(value: float | None) -> str | None:
    """Format an amount as Brazilian currency: 1967.18 -> 'R$ 1.967,18'."""
    if value is None:
        return None
    # Format with thousands separators, then swap the two separators through a
    # placeholder so the substitutions cannot collide.
    formatted = f"{value:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"R$ {formatted}"
