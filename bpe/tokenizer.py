"""Tokenizer BPE: encode/decode, ucinanie do rozmiaru słownika, zapis i odczyt."""
from __future__ import annotations

import json
from pathlib import Path

from bpe.pretok import pretokenize
from bpe.train import Pair, merge_word

INF = float("inf")


class BPETokenizer:
    def __init__(self, merges: list[Pair], vocab_size: int | None = None):
        n = len(merges) if vocab_size is None else vocab_size - 256
        if not 0 <= n <= len(merges):
            raise ValueError(f"vocab_size {vocab_size} poza zakresem 256..{256 + len(merges)}")
        self.merges = list(merges[:n])
        self.ranks = {p: r for r, p in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for r, (a, b) in enumerate(self.merges):
            self.vocab[256 + r] = self.vocab[a] + self.vocab[b]
        self._cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def encode_word(self, word: bytes) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        ids = list(word)
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, INF))
            if pair not in self.ranks:
                break
            ids = merge_word(ids, pair, 256 + self.ranks[pair])
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for w in pretokenize(text) for i in self.encode_word(w.encode("utf-8"))]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        raw = self.vocab[i]
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return "<" + raw.hex(" ") + ">"

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "merges.txt").write_text("".join(f"{a} {b}\n" for a, b in self.merges))
        vocab = {i: self.token_str(i) for i in range(self.vocab_size)}
        (folder / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=0))

    @classmethod
    def load(cls, folder: Path, vocab_size: int | None = None) -> "BPETokenizer":
        lines = (Path(folder) / "merges.txt").read_text().split()
        merges = [(int(lines[i]), int(lines[i + 1])) for i in range(0, len(lines), 2)]
        return cls(merges, vocab_size)
