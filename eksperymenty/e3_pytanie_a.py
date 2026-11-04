"""A: fertility tokenizerów ogólnych, Bielika i naszych na teście śląskim i polskim-o-Śląsku."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

from bpe.tokenizer import BPETokenizer
from bpe.zewnetrzne import wszystkie
from metryki import zmierz
from wspolne import ARTEFAKTY, POLSKIE, SLASKIE, WYNIKI, teksty


def liczniki() -> dict:
    out = dict(wszystkie())
    for v in (16000, 32000):
        tok = BPETokenizer.load(ARTEFAKTY / "bpe-32k", v)
        out[f"nasz {v // 1000}k"] = lambda t, tok=tok: len(tok.encode(t))
    return out


def main() -> None:
    zbiory = {"śląski": teksty("test", SLASKIE), "polski o Śląsku": teksty("test", POLSKIE)}
    wyniki = {nazwa: {z: zmierz(f, ts) for z, ts in zbiory.items()} for nazwa, f in liczniki().items()}
    for nazwa, w in wyniki.items():
        w["narzut_slaski_vs_polski"] = round(w["śląski"]["fertility"] / w["polski o Śląsku"]["fertility"], 3)
    (WYNIKI / "pytanie_a.json").write_text(json.dumps(wyniki, ensure_ascii=False, indent=2))

    nazwy = list(wyniki)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, z in enumerate(zbiory):
        ax.bar([i + k * 0.4 for i in range(len(nazwy))], [wyniki[n][z]["fertility"] for n in nazwy],
               width=0.4, label=z)
    ax.set_xticks([i + 0.2 for i in range(len(nazwy))], nazwy, rotation=20, ha="right")
    ax.set_ylabel("tokeny na słowo (test)")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(WYNIKI / "pytanie_a.png", dpi=150)
    for n in nazwy:
        print(n, wyniki[n]["śląski"]["fertility"], wyniki[n]["polski o Śląsku"]["fertility"])


if __name__ == "__main__":
    main()
