"""Brazilian currency: rounding and formatting.

Its own module because both the rules layer and the tool layer need it, and
`rules` must not import `tools`.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def round_brl(value: float) -> float:
    """Round an amount to cents, half away from zero.

    Python's built-in `round` is wrong for money twice over: it rounds halves
    to even, and the float it receives is usually already slightly below the
    half it is meant to represent. Both bite on real catalog prices — a 5% PIX
    discount on R$ 699,90 is 664.905, which `round(x, 2)` turns into 664.90
    while every Brazilian till produces 664.91.

    Going through Decimal(str(...)) reconstructs the decimal the price
    actually is before rounding it the way a shop does.
    """
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def brl(value: float | None) -> str | None:
    """Format an amount as Brazilian currency: 1967.18 -> 'R$ 1.967,18'.

    To a Brazilian reader `R$ 1.967,18` and `R$ 1,967.18` differ by a factor
    of a thousand, so this is not cosmetic and is never left to the model.
    """
    if value is None:
        return None
    # Format with thousands separators, then swap the two separators through a
    # placeholder so the substitutions cannot collide.
    formatted = f"{round_brl(value):,.2f}".replace(",", "\x00").replace(".", ",")
    return f"R$ {formatted.replace(chr(0), '.')}"
