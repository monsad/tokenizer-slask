"""Metryki tokenizera. Słowo = kawałek tekstu między białymi znakami (ta sama definicja dla wszystkich)."""
from __future__ import annotations

from typing import Callable


def zmierz(licz_tokeny: Callable[[str], int], teksty: list[str]) -> dict:
    tokeny = sum(licz_tokeny(t) for t in teksty)
    slowa = sum(len(t.split()) for t in teksty)
    bajty = sum(len(t.encode("utf-8")) for t in teksty)
    return {"tokeny": tokeny, "slowa": slowa, "bajty": bajty,
            "fertility": round(tokeny / slowa, 4), "bajty_na_token": round(bajty / tokeny, 4)}
