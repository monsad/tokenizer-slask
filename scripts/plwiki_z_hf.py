"""Wybiera z zrzutu polskiej Wikipedii (HF wikimedia/wikipedia, 20231101.pl) artykuły z listy tytułów
zebranej z kategorii o Górnym Śląsku (scripts/pobierz_plwiki_slask.py, plik *.tytuly.txt).

    python scripts/plwiki_z_hf.py data/raw/plwiki_hf data/raw/plwiki/plwiki_slask.tytuly.txt \
        data/raw/plwiki/plwiki_slask.jsonl

Szybsza alternatywa dla API (które ogranicza tempo). Artykuły są w wersji z 1.11.2023.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def main() -> None:
    import pyarrow.parquet as pq

    folder, lista, wyjscie = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    tytuly = set(lista.read_text(encoding="utf-8").splitlines())
    znalezione = 0
    with wyjscie.open("w", encoding="utf-8") as f:
        for plik in sorted(glob.glob(str(folder / "*.parquet"))):
            tabela = pq.read_table(plik, columns=["title", "text"])
            for title, text in zip(tabela.column("title").to_pylist(), tabela.column("text").to_pylist()):
                if title in tytuly:
                    f.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")
                    znalezione += 1
            print(f"{plik}: łącznie {znalezione}", file=sys.stderr)
    print(f"znaleziono {znalezione} z {len(tytuly)} tytułów", file=sys.stderr)


if __name__ == "__main__":
    main()
