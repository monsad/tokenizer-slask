# Tokenizer BPE dla gŏdki

Praca domowa do lekcji 02 kursu „Train LLM from scratch” (Fabryka): byte-level BPE w czystym Pythonie,
wytrenowany na korpusie **„Górny Śląsk: gŏdka i kultura”** (teksty po śląsku i polskie teksty o Górnym
Śląsku), zmierzony w 5 rozmiarach słownika i użyty do 4 pytań badawczych. Raport: [RAPORT.md](RAPORT.md).

## Uruchomienie

```bash
pip install -r requirements.txt        # albo: uv run --no-project --with-requirements requirements.txt …
python run_all.py                      # korpus -> trening -> metryki -> pytania A–D
pytest -q                              # testy
```

`run_all.py` wymaga danych w `data/raw` (niżej). Pełny przebieg trwa ok. 3 minuty na laptopie (trening 32k: ok. 2 min).

## Dane

```bash
mkdir -p data/raw/{szlwiki,fineweb2,plwiki,plwiki_hf,wachtyrz}
curl -L -o data/raw/szlwiki/szlwiki-20261101-pages-articles.xml.bz2 https://dumps.wikimedia.org/szlwiki/latest/szlwiki-latest-pages-articles.xml.bz2
python -m wikiextractor.WikiExtractor data/raw/szlwiki/szlwiki-20261101-pages-articles.xml.bz2 --json -o data/raw/szlwiki/extracted -q
curl -L -o data/raw/fineweb2/train.parquet https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/szl_Latn/train/000_00000.parquet
curl -L -o data/raw/fineweb2/test.parquet  https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/szl_Latn/test/000_00000.parquet
python scripts/pobierz_plwiki_slask.py data/raw/plwiki/plwiki_slask.jsonl   # lista tytułów z kategorii (+ tekst przez API, wolno)
# szybciej: tekst ze zrzutu HF (1,8 GB, wersja z 1.11.2023)
for i in 0 1 2 3 4 5; do curl -L -o data/raw/plwiki_hf/train-0000$i.parquet https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.pl/train-0000$i-of-00006.parquet; done
python scripts/plwiki_z_hf.py data/raw/plwiki_hf dane/plwiki_slask_tytuly.txt data/raw/plwiki/plwiki_slask.jsonl
# data/raw/wachtyrz/wachtyrz.txt — artykuły z wachtyrz.eu (publiczne API WordPressa), jeden akapit na linię
```

| Źródło | Licencja |
|---|---|
| Ślōnsko Wikipedyjo | CC BY-SA |
| FineWeb-2 `szl_Latn` | ODC-By (treści stron — prawa ich autorów) |
| Polska Wikipedia, kategorie o Górnym Śląsku | CC BY-SA |
| Wachtyrz.eu | licencja potwierdzona przez autorkę (szczegóły w raporcie) |

## Jak to działa

- `bpe/pretok.py` — tekst cięty na „słowa” (spacja doklejona do następnego słowa); BPE skleja bajty
  tylko wewnątrz słowa.
- `bpe/train.py` — algorytm z zajęć, przyspieszony: liczy pary na unikalnych słowach z licznikami
  i po merge'u aktualizuje tylko słowa z tą parą. Remis: większa liczba wystąpień, potem mniejsza para
  (wersja z zajęć bierze pierwszą napotkaną). Test `tests/test_train.py` porównuje z naiwną wersją.
- **Trening jest jeden (do 32k).** Tokenizer o rozmiarze V to pierwsze V−256 merge'y tego samego treningu.
- `korpus/czysc.py` — NFC, filtr „śląskości”, przycięcie szablonowych haseł, deduplikacja dokumentów
  i linii, podział 90/10 po dokumentach osobno w każdym źródle. Statystyki: `data/corpus/stats.json`.
- **Wszystkie pomiary są na zbiorze testowym**, którego tokenizer nie widział.
- Tokenizer Bielika v3: plik `tokenizer.json` wskazany zmienną `BIELIK_TOKENIZER`.

## Wyniki

- `artefakty/bpe-{1,4,8,16,32}k/` — `merges.txt` (para id na linię, w kolejności powstania) i `vocab.json`.
- `artefakty/bpe-8k-tylko-szlwiki/` — tokenizer do pytania C.
- `wyniki/*.json` i `wyniki/*.png` — liczby i wykresy do raportu.
