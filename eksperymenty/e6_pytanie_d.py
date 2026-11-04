"""D: ten sam śląski tekst testowy (szlwiki, Ślabikŏrz) w trzech pisowniach, mierzony we wszystkich tokenizerach.
Warianty to przybliżenie regułami, nie prawdziwe teksty w innej pisowni."""
from __future__ import annotations

import json
import unicodedata

from eksperymenty.e3_pytanie_a import liczniki
from metryki import zmierz
from wspolne import WYNIKI, teksty

PO_POLSKU = str.maketrans({"ō": "ó", "Ō": "Ó", "ŏ": "o", "Ŏ": "O", "ô": "o", "Ô": "O"})


def po_polsku(t: str) -> str:
    return t.translate(PO_POLSKU)


def bez_ogonkow(t: str) -> str:
    t = t.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))


def main() -> None:
    oryginal = teksty("test", ["szlwiki"])
    warianty = {"Ślabikŏrz (oryginał)": oryginal,
                "po polsku (ō→ó, ŏ/ô→o)": [po_polsku(t) for t in oryginal],
                "bez znaków diakrytycznych": [bez_ogonkow(t) for t in oryginal]}
    wyniki = {n: {w: zmierz(f, ts)["fertility"] for w, ts in warianty.items()}
              for n, f in liczniki().items()}
    (WYNIKI / "pytanie_d.json").write_text(json.dumps(wyniki, ensure_ascii=False, indent=2))
    print(json.dumps(wyniki, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
