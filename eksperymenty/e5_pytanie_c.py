"""C: 8k uczony tylko na szlwiki vs 8k uczony na całości, test osobno na każdym źródle."""
from __future__ import annotations

import json

from bpe.tokenizer import BPETokenizer
from metryki import zmierz
from wspolne import ARTEFAKTY, WYNIKI, ZRODLA, teksty


def main() -> None:
    toki = {"8k tylko szlwiki": BPETokenizer.load(ARTEFAKTY / "bpe-8k-tylko-szlwiki"),
            "8k całość": BPETokenizer.load(ARTEFAKTY / "bpe-32k", 8000)}
    wyniki = {n: {src: zmierz(lambda t, tok=tok: len(tok.encode(t)), teksty("test", [src]))
                  for src in ZRODLA} for n, tok in toki.items()}
    w = wyniki["8k tylko szlwiki"]
    wyniki["spadek_poza_wikipedia_proc"] = {
        src: round(100 * (w[src]["fertility"] / w["szlwiki"]["fertility"] - 1), 1)
        for src in ("wachtyrz_szl", "fineweb_szl")}
    (WYNIKI / "pytanie_c.json").write_text(json.dumps(wyniki, ensure_ascii=False, indent=2))
    print(json.dumps(wyniki, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
