"""Ścieżki i wczytywanie podziału korpusu, wspólne dla eksperymentów."""
from __future__ import annotations

import json
from pathlib import Path

CORPUS = Path("data/corpus")
ARTEFAKTY = Path("artefakty")
WYNIKI = Path("wyniki")
ROZMIARY = [1000, 4000, 8000, 16000, 32000]
ZRODLA = ["szlwiki", "wachtyrz_szl", "fineweb_szl", "plwiki_slask", "wachtyrz_pl"]
SLASKIE = ["szlwiki", "wachtyrz_szl", "fineweb_szl"]
POLSKIE = ["plwiki_slask", "wachtyrz_pl"]


def teksty(split: str, zrodla: list[str] | None = None) -> list[str]:
    out = []
    for src in zrodla or ZRODLA:
        p = CORPUS / split / f"{src}.jsonl"
        out += [json.loads(l)["text"] for l in open(p, encoding="utf-8")]
    return out
