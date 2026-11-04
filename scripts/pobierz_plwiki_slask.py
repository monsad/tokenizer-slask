"""Pobiera z polskiej Wikipedii artykuły z kategorii o Górnym Śląsku (CC BY-SA) jako czysty tekst.

    python scripts/pobierz_plwiki_slask.py data/raw/plwiki/plwiki_slask.jsonl

Chodzi po drzewie kategorii od kilku korzeni (do zadanej głębokości) i pobiera tekst
przez API TextExtracts (explaintext). Wynik: JSONL {title, text}.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://pl.wikipedia.org/w/api.php"
UA = "tokenizer-slask/0.1 (praca domowa, kurs Fabryka; kontakt: github.com/monsad)"
KORZENIE = ["Kategoria:Górny Śląsk", "Kategoria:Kultura na Śląsku", "Kategoria:Gwara śląska",
            "Kategoria:Język śląski", "Kategoria:Historia Śląska", "Kategoria:Górnośląski Okręg Przemysłowy"]
GLEBOKOSC = 2
# Zagłębie i Częstochowa to nie Górny Śląsk — odcinamy te gałęzie drzewa kategorii.
POMIN = ("Zagłębi", "Zawierci", "Sosnowc", "Będzin", "Dąbrow", "Częstoch", "Dolny Śląsk", "Dolnośl", "Wrocław", "Legnic", "Wałbrzy", "Jeleni")


def api(**params) -> dict:
    params |= {"format": "json", "formatversion": "2"}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    time.sleep(1.0)                      # grzecznie: maks. ~1 zapytanie na sekundę
    for proba in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            czekaj = int(getattr(e, "headers", {}).get("retry-after", 0) or 0) or min(300, 15 * 2 ** proba)
            print(f"\n[{type(e).__name__}: {e}] ponawiam za {czekaj} s", file=sys.stderr)
            time.sleep(czekaj)
    raise RuntimeError(f"API nie odpowiada: {params}")


def czlonkowie(kategoria: str):
    cont = {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle=kategoria, cmlimit="500",
                cmtype="page|subcat", **cont)
        yield from d["query"]["categorymembers"]
        if "continue" not in d:
            return
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}


def zbierz_tytuly() -> list[str]:
    widziane_kat, tytuly = set(), set()
    kolejka = [(k, 0) for k in KORZENIE]
    while kolejka:
        kat, gl = kolejka.pop()
        if kat in widziane_kat:
            continue
        widziane_kat.add(kat)
        for m in czlonkowie(kat):
            if m["ns"] == 14 and gl < GLEBOKOSC and not any(x in m["title"] for x in POMIN):
                kolejka.append((m["title"], gl + 1))
            elif m["ns"] == 0:
                tytuly.add(m["title"])
        print(f"\rkategorii: {len(widziane_kat)}  artykułów: {len(tytuly)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return sorted(tytuly)


def teksty(tytuly: list[str], pominac: set[str]):
    """Pełny tekst API zwraca tylko dla 1 strony na zapytanie, więc pytamy po jednym tytule."""
    for n, tytul in enumerate(tytuly, 1):
        if tytul in pominac:
            continue
        d = api(action="query", prop="extracts", explaintext="1", titles=tytul)
        for p in d["query"]["pages"]:
            if p.get("extract"):
                yield {"title": p["title"], "text": p["extract"]}
        print(f"\rpobrano {n}/{len(tytuly)}", end="", file=sys.stderr)
    print(file=sys.stderr)


def main() -> None:
    wyjscie = Path(sys.argv[1])
    wyjscie.parent.mkdir(parents=True, exist_ok=True)
    lista = wyjscie.with_suffix(".tytuly.txt")   # punkt kontrolny: drzewo kategorii chodzimy raz
    if lista.exists():
        tytuly = lista.read_text(encoding="utf-8").splitlines()
    else:
        tytuly = zbierz_tytuly()
        lista.write_text("\n".join(tytuly), encoding="utf-8")
    juz = set()
    if wyjscie.exists():                              # wznawianie: pomijamy pobrane artykuły
        juz = {json.loads(l)["title"] for l in wyjscie.open(encoding="utf-8")}
    with wyjscie.open("a", encoding="utf-8") as f:
        for wiersz in teksty(tytuly, juz):
            f.write(json.dumps(wiersz, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
