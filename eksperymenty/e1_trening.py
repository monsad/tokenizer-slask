"""Trening: jeden przebieg do 32k na całym korpusie treningowym (rozmiary 1k–16k to jego prefiksy)
oraz 8k tylko na szlwiki (do pytania C). Zapis do artefakty/bpe-*/."""
from __future__ import annotations

import time

from bpe.tokenizer import BPETokenizer
from bpe.train import count_words, train
from wspolne import ARTEFAKTY, ROZMIARY, teksty


def trenuj(nazwa: str, zrodla: list[str] | None, rozmiar: int) -> list:
    start = time.time()
    wc = count_words(teksty("train", zrodla))
    print(f"[{nazwa}] unikalnych słów: {len(wc)}", flush=True)

    def postep(n, pair, c):
        if n % 1000 == 0:
            print(f"[{nazwa}] {n} merge'y, ostatni ×{c}, {time.time() - start:.0f} s", flush=True)

    return train(wc, rozmiar, postep)


def main() -> None:
    merges = trenuj("wszystko", None, max(ROZMIARY))
    for v in ROZMIARY:
        BPETokenizer(merges, v).save(ARTEFAKTY / f"bpe-{v // 1000}k")
    BPETokenizer(trenuj("szlwiki", ["szlwiki"], 8000)).save(ARTEFAKTY / "bpe-8k-tylko-szlwiki")


if __name__ == "__main__":
    main()
