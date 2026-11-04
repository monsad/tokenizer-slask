"""Metryki dla 1k–32k na zbiorze TESTOWYM: fertility, bajty/token, długości tokenów, rzadkie tokeny."""
from __future__ import annotations

import json
from collections import Counter

import matplotlib.pyplot as plt

from bpe.tokenizer import BPETokenizer
from metryki import zmierz
from wspolne import ARTEFAKTY, ROZMIARY, WYNIKI, ZRODLA, teksty


def main() -> None:
    WYNIKI.mkdir(exist_ok=True)
    test = {src: teksty("test", [src]) for src in ZRODLA}
    caly = [t for ts in test.values() for t in ts]
    wyniki = {}
    for v in ROZMIARY:
        tok = BPETokenizer.load(ARTEFAKTY / "bpe-32k", v)
        uzycia = Counter(i for t in caly for i in tok.encode(t))
        wyniki[v] = {
            "calosc": zmierz(lambda t: len(tok.encode(t)), caly),
            **{src: zmierz(lambda t: len(tok.encode(t)), ts) for src, ts in test.items()},
            "tokeny_uzyte_w_tescie": len(uzycia),
            "tokeny_uzyte_mniej_niz_10_razy": sum(1 for i in range(tok.vocab_size) if uzycia[i] < 10),
            "dlugosci_tokenow_bajty": Counter(len(tok.vocab[i]) for i in range(256, tok.vocab_size)),
        }
        print(v, wyniki[v]["calosc"], flush=True)
    (WYNIKI / "slownik.json").write_text(json.dumps(wyniki, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    for src in ["calosc", *ZRODLA]:
        ax.plot(ROZMIARY, [wyniki[v][src]["fertility"] for v in ROZMIARY], marker="o", label=src)
    ax.set_xscale("log")
    ax.set_xticks(ROZMIARY, [f"{v // 1000}k" for v in ROZMIARY])
    ax.set_xlabel("rozmiar słownika")
    ax.set_ylabel("tokeny na słowo (zbiór testowy)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(WYNIKI / "fertility_vs_slownik.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    for v in (1000, 8000, 32000):
        d = wyniki[v]["dlugosci_tokenow_bajty"]
        xs = sorted(d)
        ax.plot(xs, [d[x] / sum(d.values()) for x in xs], marker=".", label=f"{v // 1000}k")
    ax.set_xlabel("długość tokenu (bajty)")
    ax.set_ylabel("udział w słowniku")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(WYNIKI / "dlugosci_tokenow.png", dpi=150)


if __name__ == "__main__":
    main()
