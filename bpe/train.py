"""Szybki trening byte-level BPE.

Zamiast jednego ciągu bajtów trzymamy unikalne słowa z licznikami. Po każdym merge'u
aktualizujemy liczniki par tylko w słowach, które zawierały scalaną parę, a najczęstszą
parę bierzemy z kopca (z leniwym usuwaniem nieaktualnych wpisów).

Remis: wygrywa para o największej liczbie wystąpień, a przy równej liczbie najmniejsza para.
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from typing import Callable, Iterable

from bpe.pretok import pretokenize

Pair = tuple[int, int]


def count_words(texts: Iterable[str]) -> Counter[bytes]:
    c: Counter[bytes] = Counter()
    for t in texts:
        c.update(w.encode("utf-8") for w in pretokenize(t))
    return c


def merge_word(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(word_counts: dict[bytes, int], vocab_size: int,
          progress: Callable[[int, Pair, int], None] | None = None) -> list[Pair]:
    """Zwraca listę merge'y w kolejności powstania (indeks = rank, nowy id = 256 + rank)."""
    words = [list(w) for w in word_counts]
    freqs = list(word_counts.values())
    counts: Counter[Pair] = Counter()
    where: dict[Pair, set[int]] = defaultdict(set)
    for i, w in enumerate(words):
        for p in zip(w, w[1:]):
            counts[p] += freqs[i]
            where[p].add(i)
    heap = [(-c, p) for p, c in counts.items()]
    heapq.heapify(heap)

    merges: list[Pair] = []
    while len(merges) < vocab_size - 256 and heap:
        neg, pair = heapq.heappop(heap)
        if counts.get(pair, 0) != -neg or neg == 0:
            continue                                   # nieaktualny wpis w kopcu
        new_id = 256 + len(merges)
        merges.append(pair)
        touched: set[Pair] = set()
        for i in where.pop(pair, ()):
            w, f = words[i], freqs[i]
            nw = merge_word(w, pair, new_id)
            if len(nw) == len(w):
                continue                               # słowo już nie zawiera tej pary
            for p in zip(w, w[1:]):
                counts[p] -= f
                touched.add(p)
            for p in zip(nw, nw[1:]):
                counts[p] += f
                touched.add(p)
                where[p].add(i)
            words[i] = nw
        for p in touched:
            if counts[p] <= 0:
                del counts[p]
            else:
                heapq.heappush(heap, (-counts[p], p))
        if progress:
            progress(len(merges), pair, -neg)
    return merges
