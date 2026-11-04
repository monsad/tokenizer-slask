"""Liczniki tokenów dla tokenizerów spoza projektu (do porównań w pytaniach A i D)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

BIELIK = Path(os.environ.get("BIELIK_TOKENIZER",
                             Path.home() / "bielik-tokenizer/bielik-4.5b-mlx/tokenizer.json"))


def tiktoken_licznik(nazwa: str) -> Callable[[str], int]:
    import tiktoken
    enc = tiktoken.get_encoding(nazwa)
    return lambda t: len(enc.encode(t, disallowed_special=()))


def bielik_licznik() -> Callable[[str], int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(BIELIK))
    return lambda t: len(tok.encode(t, add_special_tokens=False).ids)


def wszystkie() -> dict[str, Callable[[str], int]]:
    return {"GPT-4 (cl100k)": tiktoken_licznik("cl100k_base"),
            "GPT-4o (o200k)": tiktoken_licznik("o200k_base"),
            "Bielik v3 (APT4 32k)": bielik_licznik()}
