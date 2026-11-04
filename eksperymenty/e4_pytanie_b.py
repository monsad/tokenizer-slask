"""B: przy którym merge'u (rank) znak dwubajtowy staje się tokenem; porównanie ō/ŏ/ô z ą/ę/ż…
oraz pierwsze tokeny będące śląskimi i polskimi końcówkami."""
from __future__ import annotations

import json
from collections import Counter

import matplotlib.pyplot as plt

from bpe.tokenizer import BPETokenizer
from wspolne import ARTEFAKTY, WYNIKI, teksty

ZNAKI_SZL = ["ō", "ŏ", "ô"]
ZNAKI_PL = ["ą", "ę", "ż", "ł", "ś", "ć", "ń", "ó", "ź"]
KONCOWKI = {"ōm": "szl", "ymu": "szl", "ŏ": "szl", "ōw": "szl", "ą": "pl", "emu": "pl", "ów": "pl"}


def rank_znaku(tok: BPETokenizer, znak: str) -> dict:
    b = znak.encode("utf-8")
    sam = zaw = None
    for r in range(len(tok.merges)):
        t = tok.vocab[256 + r]
        if zaw is None and b in t:
            zaw = r
        if t == b:
            sam = r
            break
    return {"samodzielny": sam, "pierwszy_zawierajacy": zaw}


def rank_tokenu(tok: BPETokenizer, s: str) -> int | None:
    b = s.encode("utf-8")
    return next((r for r in range(len(tok.merges)) if tok.vocab[256 + r] == b), None)


def main() -> None:
    tok = BPETokenizer.load(ARTEFAKTY / "bpe-32k")
    czestosc = Counter(ch for t in teksty("train") for ch in t)
    znaki = {z: {**rank_znaku(tok, z), "wystapienia_w_treningu": czestosc[z],
                 "grupa": "szl" if z in ZNAKI_SZL else "pl"} for z in ZNAKI_SZL + ZNAKI_PL}
    koncowki = {k: {"rank": rank_tokenu(tok, k), "grupa": g} for k, g in KONCOWKI.items()}
    (WYNIKI / "pytanie_b.json").write_text(json.dumps({"znaki": znaki, "koncowki": koncowki},
                                                     ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    for z, d in znaki.items():
        if d["samodzielny"] is None:
            continue
        kolor = "tab:red" if d["grupa"] == "szl" else "tab:blue"
        ax.scatter(d["wystapienia_w_treningu"], d["samodzielny"] + 1, color=kolor)
        ax.annotate(z, (d["wystapienia_w_treningu"], d["samodzielny"] + 1), fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wystąpienia znaku w korpusie treningowym")
    ax.set_ylabel("numer merge'a, który tworzy znak")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(WYNIKI / "pytanie_b.png", dpi=150)
    print(json.dumps({"znaki": znaki, "koncowki": koncowki}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
