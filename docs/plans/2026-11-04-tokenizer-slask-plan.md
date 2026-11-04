# Tokenizer BPE dla gŏdki — plan wykonania

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Cel:** wytrenować byte-level BPE na korpusie „Górny Śląsk: gŏdka i kultura” w 5 rozmiarach słownika, zmierzyć je na odłożonym zbiorze testowym i odpowiedzieć na 4 pytania badawcze (A–D z projektu `2026-11-04-tokenizer-slask-design.md`).

**Architektura:** czysty Python. Pakiet `bpe/` (cięcie na słowa, szybki trening, tokenizer, adaptery tokenizerów zewnętrznych), `korpus/` (czyszczenie i podział danych), `eksperymenty/` (po jednym skrypcie na część raportu) i `run_all.py`, który odpala wszystko po kolei. Trenujemy **raz** do 32k: tokenizer o rozmiarze V to pierwsze V−256 merge'y tego samego treningu (BPE jest sekwencyjne).

**Stos:** Python 3.11, pytest, pyarrow (FineWeb parquet), matplotlib, tiktoken (cl100k/o200k), tokenizers (Bielik v3 `tokenizer.json`), wikiextractor (zrzut szlwiki).

**Repo:** `~/tokenizer-slask` (osobne repo git; uwaga: katalog domowy też jest repo, więc `git init` robimy w `~/tokenizer-slask`). Commity **bez** linii Co-Authored-By.

**Komendy:** wszystko przez `uv run --no-project --with-requirements requirements.txt …`, w skrócie `UVR`:
```bash
alias UVR='uv run --no-project --with-requirements requirements.txt'
```

---

## Struktura plików

| Plik | Odpowiada za |
|---|---|
| `requirements.txt` | zależności |
| `bpe/pretok.py` | cięcie tekstu na „słowa” przed BPE (spacja doklejona do następnego słowa) |
| `bpe/train.py` | `count_words`, `merge_word`, szybki `train` (pary liczone na unikalnych słowach + kopiec) |
| `bpe/tokenizer.py` | `BPETokenizer`: encode/decode, ucinanie do rozmiaru V, zapis/odczyt merges+vocab |
| `bpe/zewnetrzne.py` | liczniki tokenów dla cl100k, o200k, Bielika v3 |
| `korpus/czysc.py` | NFC, wskaźnik „śląskości”, wczytanie 3 źródeł, deduplikacja, podział 90/10, statystyki |
| `metryki.py` | fertility, bajty/token, pomiar na liście tekstów |
| `wspolne.py` | ścieżki, wczytywanie podziału train/test |
| `eksperymenty/e1_trening.py` | trening 32k (wszystko) i 8k (tylko szlwiki, do pytania C) |
| `eksperymenty/e2_slownik.py` | część „Vocab size”: metryki dla 1k–32k, wykresy |
| `eksperymenty/e3_pytanie_a.py` … `e6_pytanie_d.py` | pytania A–D |
| `run_all.py` | cały potok jedną komendą |
| `tests/` | testy jednostkowe |
| `artefakty/bpe-*/` | `merges.txt` + `vocab.json` dla każdego rozmiaru (oddawane) |
| `wyniki/` | JSON-y i wykresy PNG do raportu |

Dane (`data/raw`, `data/corpus`) są w `.gitignore`. Artefakty i wyniki commitujemy.

---

### Task 1: Szkielet repo

**Files:**
- Create: `requirements.txt`, `.gitignore`, `bpe/__init__.py`, `korpus/__init__.py`, `eksperymenty/__init__.py`, `tests/__init__.py`, `conftest.py`

- [ ] **Step 1: Utwórz pliki**

`requirements.txt`:
```
pytest==8.3.4
pyarrow==18.1.0
matplotlib==3.9.4
tiktoken==0.8.0
tokenizers==0.21.0
wikiextractor==3.0.6
```

`.gitignore`:
```
data/raw/
data/corpus/
__pycache__/
.pytest_cache/
*.pyc
```

`conftest.py` (pusty plik; dzięki niemu pytest widzi pakiety z katalogu głównego):
```python
```

`bpe/__init__.py`, `korpus/__init__.py`, `eksperymenty/__init__.py`, `tests/__init__.py`: puste pliki.

- [ ] **Step 2: git init i pierwszy commit**

```bash
cd ~/tokenizer-slask && git init -q && git add -A && git commit -qm "chore: szkielet repo tokenizer-slask"
```
Oczekiwane: commit zawiera `docs/`, `scripts/pobierz_plwiki_slask.py` i pliki z kroku 1. `data/` nie trafia do commita.

---

### Task 2: Cięcie na słowa (`bpe/pretok.py`)

**Files:**
- Create: `bpe/pretok.py`
- Test: `tests/test_pretok.py`

- [ ] **Step 1: Test**

```python
from bpe.pretok import pretokenize


def test_spacja_doklejona_do_slowa():
    assert pretokenize("Jo żech je") == ["Jo", " żech", " je"]


def test_sklejenie_daje_oryginal():
    t = "Gŏdka  ze\nŚlōnska!\t ôd zawsze"
    assert "".join(pretokenize(t)) == t


def test_pusty():
    assert pretokenize("") == []
```

- [ ] **Step 2: Uruchom, oczekuj FAIL**

Run: `UVR pytest tests/test_pretok.py -q` → `ModuleNotFoundError: No module named 'bpe.pretok'`

- [ ] **Step 3: Implementacja**

```python
"""Cięcie tekstu na „słowa” przed BPE.

Pojedyncza spacja dokleja się do następnego słowa (jak w GPT-2), pozostałe białe znaki
tworzą osobne kawałki. BPE skleja bajty tylko wewnątrz jednego kawałka.
"""
import re

PAT = re.compile(r" ?\S+|\s+")


def pretokenize(text: str) -> list[str]:
    return PAT.findall(text)
```

- [ ] **Step 4: PASS**

Run: `UVR pytest tests/test_pretok.py -q` → `3 passed`

- [ ] **Step 5: Commit**

```bash
git add bpe/pretok.py tests/test_pretok.py && git commit -qm "feat(bpe): cięcie na słowa przed BPE"
```

---

### Task 3: Szybki trening BPE (`bpe/train.py`)

**Files:**
- Create: `bpe/train.py`
- Test: `tests/test_train.py`

Zasada remisu (ważna dla powtarzalności): wygrywa para o największej liczbie wystąpień, a przy remisie **najmniejsza para** (porównanie krotek). Wersja z zajęć bierze pierwszą napotkaną, więc wyniki przy remisach mogą się różnić. Opisujemy to w README.

- [ ] **Step 1: Testy — porównanie z naiwną wersją z zajęć (ta sama zasada remisu)**

```python
from collections import Counter

from bpe.train import count_words, merge_word, train


def naiwny_train(word_counts, vocab_size):
    """Wersja z zajęć przeniesiona na słowa: co krok liczy wszystko od nowa."""
    words = {tuple(w): f for w, f in word_counts.items()}
    merges = []
    while len(merges) < vocab_size - 256:
        counts = Counter()
        for w, f in words.items():
            for p in zip(w, w[1:]):
                counts[p] += f
        if not counts:
            break
        pair = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        new_id = 256 + len(merges)
        merges.append(pair)
        words = {tuple(merge_word(list(w), pair, new_id)): f for w, f in words.items()}
    return merges


TEKST = ("Programista programuje program. Tokenizer tokenizuje tokeny. "
         "Jo żech je ze Ślōnska, gŏdōm po naszymu ôd zawsze. " * 3)


def test_merge_word():
    assert merge_word([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_count_words():
    assert count_words(["ala ma ala"]) == Counter({b"ala": 1, b" ma": 1, b" ala": 1})


def test_zgodnosc_z_naiwnym():
    wc = count_words([TEKST])
    assert train(wc, 256 + 60) == naiwny_train(wc, 256 + 60)


def test_stop_gdy_brak_par():
    wc = count_words(["ab"])
    assert train(wc, 1000) == [(97, 98)]


def test_deterministyczny():
    wc = count_words([TEKST])
    assert train(wc, 300) == train(wc, 300)
```

- [ ] **Step 2: FAIL**

Run: `UVR pytest tests/test_train.py -q` → `ModuleNotFoundError: No module named 'bpe.train'`

- [ ] **Step 3: Implementacja**

```python
"""Szybki trening byte-level BPE.

Zamiast jednego ciągu bajtów trzymamy unikalne słowa z licznikami. Po każdym merge'u
aktualizujemy liczniki par tylko w słowach, które zawierały scalaną parę, a najczęstszą
parę bierzemy z kopca (z leniwym usuwaniem nieaktualnych wpisów).
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
```

- [ ] **Step 4: PASS**

Run: `UVR pytest tests/test_train.py -q` → `5 passed`

- [ ] **Step 5: Commit**

```bash
git add bpe/train.py tests/test_train.py && git commit -qm "feat(bpe): szybki trening zgodny z wersją z zajęć"
```

---

### Task 4: Tokenizer (`bpe/tokenizer.py`)

**Files:**
- Create: `bpe/tokenizer.py`
- Test: `tests/test_tokenizer.py`

- [ ] **Step 1: Testy**

```python
from bpe.tokenizer import BPETokenizer
from bpe.train import count_words, train

TEKST = "Jo żech je ze Ślōnska, gŏdōm po naszymu ôd zawsze. Ôma gŏdała tyż. " * 20


def tok(v=None):
    return BPETokenizer(train(count_words([TEKST]), 400), vocab_size=v)


def test_roundtrip():
    t = tok()
    for s in [TEKST, "Nowy tekst: ŏ ō ô ą ę 2026 😀", ""]:
        assert t.decode(t.encode(s)) == s


def test_kompresja():
    t = tok()
    assert len(t.encode(TEKST)) < len(TEKST.encode("utf-8")) / 2


def test_uciecie_do_rozmiaru():
    t = tok(300)
    assert t.vocab_size == 300 and max(t.encode(TEKST)) < 300


def test_slaskie_znaki_to_2_bajty():
    t = BPETokenizer([], vocab_size=256)
    assert t.encode("ō") == list("ō".encode("utf-8")) == [0xC5, 0x8D]


def test_zapis_odczyt(tmp_path):
    t = tok()
    t.save(tmp_path)
    t2 = BPETokenizer.load(tmp_path, vocab_size=300)
    assert t2.merges == t.merges[:44]
    assert (tmp_path / "vocab.json").exists()
```

- [ ] **Step 2: FAIL**

Run: `UVR pytest tests/test_tokenizer.py -q` → `ModuleNotFoundError: No module named 'bpe.tokenizer'`

- [ ] **Step 3: Implementacja**

```python
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
```

- [ ] **Step 4: PASS**

Run: `UVR pytest tests/test_tokenizer.py -q` → `5 passed`

- [ ] **Step 5: Commit**

```bash
git add bpe/tokenizer.py tests/test_tokenizer.py && git commit -qm "feat(bpe): tokenizer z ucinaniem do rozmiaru i zapisem"
```

---

### Task 5: Czyszczenie i podział korpusu (`korpus/czysc.py`)

**Files:**
- Create: `korpus/czysc.py`, `wspolne.py`
- Test: `tests/test_czysc.py`

Format dokumentu: `{"source": str, "id": str, "text": str}`. Źródła: `szlwiki`, `fineweb_szl`, `plwiki_slask`, `wachtyrz_szl`, `wachtyrz_pl`.

Wachtyrz (`data/raw/wachtyrz/wachtyrz.txt`, licencja potwierdzona przez autorkę): artykuły rozdzielone pustą linią, jeden akapit na linię, wersja polska i śląska przeplatane. Każdy artykuł dzielimy akapitami według `szl_score` na dokument `wachtyrz_szl` i `wachtyrz_pl`.

- [ ] **Step 1: Testy**

```python
from korpus.czysc import dedup_docs, dedup_lines, nfc, podziel, rozdziel_akapity, szl_score

SZL = "Jo żech je ze Ślōnska, niy ma kaj gŏdać ôd zawsze."
PL = "Jestem ze Śląska i lubię rozmawiać o historii regionu."


def test_szl_score_rozroznia():
    assert szl_score(SZL) > 0.1 > szl_score(PL)


def test_nfc():
    rozlozone = "ō"                       # o + makron łączący
    assert nfc(rozlozone) == "ō"


def test_dedup_docs():
    d = [{"source": "a", "id": "1", "text": "x y z"}, {"source": "b", "id": "2", "text": "x y z"}]
    assert [x["id"] for x in dedup_docs(d)] == ["1"]


def test_dedup_lines_usuwa_dlugie_powtorki():
    dl = "To je dugŏ linijka, kerŏ sie powtarzŏ we dwōch dokumyntach naroz." 
    d = [{"source": "a", "id": "1", "text": dl + "\nkrótka"},
         {"source": "b", "id": "2", "text": dl + "\ninksze"}]
    out = dedup_lines(d, min_len=40)
    assert out[1]["text"] == "inksze" and dl in out[0]["text"]


def test_rozdziel_akapity():
    szl, pl = rozdziel_akapity([SZL, PL, SZL])
    assert szl == SZL + "\n" + SZL and pl == PL


def test_podzial_deterministyczny_i_rozlaczny():
    d = [{"source": "a", "id": str(i), "text": f"t{i}"} for i in range(100)]
    tr, te = podziel(d, test_frac=0.1, seed=1)
    tr2, te2 = podziel(d, test_frac=0.1, seed=1)
    assert [x["id"] for x in te] == [x["id"] for x in te2]
    assert len(te) == 10 and not {x["id"] for x in tr} & {x["id"] for x in te}
```

- [ ] **Step 2: FAIL**

Run: `UVR pytest tests/test_czysc.py -q` → `ModuleNotFoundError: No module named 'korpus.czysc'`

- [ ] **Step 3: Implementacja `korpus/czysc.py`**

```python
"""Budowa korpusu „Górny Śląsk: gŏdka i kultura”: wczytanie, czyszczenie, deduplikacja, podział.

    python -m korpus.czysc        # data/raw -> data/corpus/{train,test}/<źródło>.jsonl + stats.json
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

RAW, OUT = Path("data/raw"), Path("data/corpus")
SZL_MARKERY = re.compile(r"[ōŏô]|\b(niy|żech|ôd|kaj|tyż|zaś|ino|kery|kerŏ|bōł|bōła|bydzie|gŏdać)\b")
PROG_SZL = 0.02          # dobrany na próbce: polskie teksty mają ~0, śląskie zwykle > 0.05
KROTKIE_HASLO = 300      # znaków; krótsze hasła szlwiki to głównie szablony geograficzne
ZOSTAW_KROTKICH = 0.25   # tyle krótkich haseł zostawiamy (losowo), reszta to powtarzalny szablon
DOMENY_WIKI = {"szl.wikipedia.org", "szl.m.wikipedia.org"}


def nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t)


def szl_score(t: str) -> float:
    slowa = re.findall(r"\w+", t.lower())
    return len(SZL_MARKERY.findall(t.lower())) / max(1, len(slowa))


def wczytaj_szlwiki(stats: dict, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    docs, krotkie = [], 0
    for f in sorted(glob.glob(str(RAW / "szlwiki/extracted/*/*"))):
        for line in open(f, encoding="utf-8"):
            a = json.loads(line)
            t = nfc(a["text"].strip())
            if len(t) < 50:
                continue
            if len(t) < KROTKIE_HASLO:
                krotkie += 1
                if rnd.random() > ZOSTAW_KROTKICH:
                    continue
            docs.append({"source": "szlwiki", "id": a["id"], "text": t})
    stats["szlwiki"] = {"krotkie_hasla": krotkie, "zostawione_dok": len(docs)}
    return docs


def wczytaj_fineweb(stats: dict) -> list[dict]:
    import pyarrow.parquet as pq
    docs, odrzucone_jezyk, odrzucone_wiki = [], 0, 0
    for split in ("train", "test"):
        for r in pq.read_table(RAW / f"fineweb2/{split}.parquet").to_pylist():
            if r["url"].split("/")[2] in DOMENY_WIKI:
                odrzucone_wiki += 1
                continue
            t = nfc(r["text"].strip())
            if szl_score(t) < PROG_SZL:
                odrzucone_jezyk += 1
                continue
            docs.append({"source": "fineweb_szl", "id": r["id"], "text": t})
    stats["fineweb_szl"] = {"odrzucone_nie_slaski": odrzucone_jezyk,
                            "odrzucone_kopie_wikipedii": odrzucone_wiki, "zostawione_dok": len(docs)}
    return docs


def wczytaj_plwiki(stats: dict) -> list[dict]:
    docs = []
    for line in open(RAW / "plwiki/plwiki_slask.jsonl", encoding="utf-8"):
        a = json.loads(line)
        t = nfc(a["text"].strip())
        if len(t) >= 200:
            docs.append({"source": "plwiki_slask", "id": a["title"], "text": t})
    stats["plwiki_slask"] = {"zostawione_dok": len(docs)}
    return docs


def rozdziel_akapity(akapity: list[str]) -> tuple[str, str]:
    """Dzieli akapity dwujęzycznego artykułu na część śląską i polską."""
    szl = [a for a in akapity if szl_score(a) >= PROG_SZL]
    pl = [a for a in akapity if szl_score(a) < PROG_SZL]
    return "\n".join(szl), "\n".join(pl)


def wczytaj_wachtyrz(stats: dict) -> list[dict]:
    tekst = nfc((RAW / "wachtyrz/wachtyrz.txt").read_text(encoding="utf-8"))
    docs = []
    for n, art in enumerate(tekst.split("\n\n")):
        szl, pl = rozdziel_akapity([a.strip() for a in art.split("\n") if a.strip()])
        if len(szl) >= 200:
            docs.append({"source": "wachtyrz_szl", "id": f"w{n}", "text": szl})
        if len(pl) >= 200:
            docs.append({"source": "wachtyrz_pl", "id": f"w{n}", "text": pl})
    stats["wachtyrz"] = {"artykuly": n + 1, "dok_szl": sum(d["source"] == "wachtyrz_szl" for d in docs),
                         "dok_pl": sum(d["source"] == "wachtyrz_pl" for d in docs)}
    return docs


def dedup_docs(docs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for d in docs:
        h = hashlib.md5(d["text"].encode()).digest()
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out


def dedup_lines(docs: list[dict], min_len: int = 60) -> list[dict]:
    """Usuwa długie linie, które już wystąpiły wcześniej (w dowolnym dokumencie)."""
    seen, out = set(), []
    for d in docs:
        keep = []
        for line in d["text"].split("\n"):
            s = line.strip()
            if len(s) >= min_len:
                h = hashlib.md5(s.encode()).digest()
                if h in seen:
                    continue
                seen.add(h)
            keep.append(line)
        t = "\n".join(keep).strip()
        if t:
            out.append({**d, "text": t})
    return out


def podziel(docs: list[dict], test_frac: float = 0.1, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Podział po dokumentach, osobno w każdym źródle (żeby każde źródło miało swój test)."""
    train, test = [], []
    for src in sorted({d["source"] for d in docs}):
        grupa = sorted((d for d in docs if d["source"] == src), key=lambda d: d["id"])
        random.Random(f"{seed}-{src}").shuffle(grupa)
        n = round(len(grupa) * test_frac)
        test += grupa[:n]
        train += grupa[n:]
    return train, test


def zapisz(docs: list[dict], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for src in sorted({d["source"] for d in docs}):
        with open(folder / f"{src}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                if d["source"] == src:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")


def mb(docs: list[dict]) -> float:
    return round(sum(len(d["text"].encode()) for d in docs) / 1e6, 2)


def main() -> None:
    stats: dict = {}
    docs = (wczytaj_szlwiki(stats) + wczytaj_wachtyrz(stats) + wczytaj_fineweb(stats)
            + wczytaj_plwiki(stats))
    stats["przed_dedup_mb"] = mb(docs)
    docs = dedup_lines(dedup_docs(docs))
    stats["po_dedup_mb"] = mb(docs)
    train, test = podziel(docs)
    for nazwa, czesc in (("train", train), ("test", test)):
        zapisz(czesc, OUT / nazwa)
        stats[nazwa] = {src: {"dok": sum(d["source"] == src for d in czesc),
                              "mb": mb([d for d in czesc if d["source"] == src])}
                        for src in sorted({d["source"] for d in czesc})}
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

`wspolne.py`:
```python
"""Ścieżki i wczytywanie podziału korpusu, wspólne dla eksperymentów."""
from __future__ import annotations

import json
from pathlib import Path

CORPUS = Path("data/corpus")
ARTEFAKTY = Path("artefakty")
WYNIKI = Path("wyniki")
ROZMIARY = [1000, 4000, 8000, 16000, 32000]
ZRODLA = ["szlwiki", "wachtyrz_szl", "fineweb_szl", "plwiki_slask", "wachtyrz_pl"]
SLASKIE = ["szlwiki", "wachtyrz_szl", "fineweb_szl"]
POLSKIE = ["plwiki_slask", "wachtyrz_pl"]


def teksty(split: str, zrodla: list[str] | None = None) -> list[str]:
    out = []
    for src in zrodla or ZRODLA:
        p = CORPUS / split / f"{src}.jsonl"
        out += [json.loads(l)["text"] for l in open(p, encoding="utf-8")]
    return out
```

- [ ] **Step 4: PASS**

Run: `UVR pytest tests/test_czysc.py -q` → `6 passed`

- [ ] **Step 5: Zbuduj korpus i sprawdź rozmiar** (wymaga ukończonego `data/raw/plwiki/plwiki_slask.jsonl`)

Run: `UVR python -m korpus.czysc`
Oczekiwane: `po_dedup_mb` ≥ 50. **Jeśli < 50:** podnieś `ZOSTAW_KROTKICH` do 1.0 i/lub `GLEBOKOSC` w `scripts/pobierz_plwiki_slask.py` do 3 (po usunięciu `data/raw/plwiki/*.tytuly.txt`), uruchom ponownie i zapisz decyzję w `stats.json`/README. Sprawdź też na 20 losowych odrzuconych dokumentach FineWeb, czy `PROG_SZL` nie wycina gŏdki (wypisz `id` + pierwsze 150 znaków), a wynik krótko opisz w raporcie.

- [ ] **Step 6: Commit**

```bash
git add korpus/czysc.py wspolne.py tests/test_czysc.py && git commit -qm "feat(korpus): czyszczenie, deduplikacja i podział 90/10"
```

---

### Task 6: Metryki i tokenizery zewnętrzne

**Files:**
- Create: `metryki.py`, `bpe/zewnetrzne.py`
- Test: `tests/test_metryki.py`

- [ ] **Step 1: Test**

```python
from metryki import zmierz


def test_zmierz():
    m = zmierz(lambda t: len(t.split()) * 2, ["ala ma kota", "ôd zawsze"])
    assert m["slowa"] == 5 and m["tokeny"] == 10
    assert m["fertility"] == 2.0
    assert m["bajty_na_token"] == round(len("ala ma kotaôd zawsze".encode()) / 10, 4)
```

- [ ] **Step 2: FAIL**

Run: `UVR pytest tests/test_metryki.py -q` → `ModuleNotFoundError: No module named 'metryki'`

- [ ] **Step 3: Implementacja**

`metryki.py`:
```python
"""Metryki tokenizera. Słowo = kawałek tekstu między białymi znakami (ta sama definicja dla wszystkich)."""
from __future__ import annotations

from typing import Callable


def zmierz(licz_tokeny: Callable[[str], int], teksty: list[str]) -> dict:
    tokeny = sum(licz_tokeny(t) for t in teksty)
    slowa = sum(len(t.split()) for t in teksty)
    bajty = sum(len(t.encode("utf-8")) for t in teksty)
    return {"tokeny": tokeny, "slowa": slowa, "bajty": bajty,
            "fertility": round(tokeny / slowa, 4), "bajty_na_token": round(bajty / tokeny, 4)}
```

`bpe/zewnetrzne.py`:
```python
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
```

- [ ] **Step 4: PASS + szybki test zewnętrznych**

Run: `UVR pytest tests/test_metryki.py -q` → `1 passed`
Run: `UVR python -c "from bpe.zewnetrzne import wszystkie; print({k: f('Jo żech je ze Ślōnska') for k, f in wszystkie().items()})"`
Oczekiwane: słownik z 3 liczbami całkowitymi > 0 (bez błędu o braku pliku Bielika).

- [ ] **Step 5: Commit**

```bash
git add metryki.py bpe/zewnetrzne.py tests/test_metryki.py && git commit -qm "feat: metryki i tokenizery zewnętrzne"
```

---

### Task 7: Trening (`eksperymenty/e1_trening.py`)

**Files:**
- Create: `eksperymenty/e1_trening.py`

- [ ] **Step 1: Implementacja**

```python
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
```

- [ ] **Step 2: Uruchom**

Run: `UVR python -m eksperymenty.e1_trening`
Oczekiwane: postęp co 1000 merge'y; na końcu katalogi `artefakty/bpe-1k`, `bpe-4k`, `bpe-8k`, `bpe-16k`, `bpe-32k`, `bpe-8k-tylko-szlwiki`, każdy z `merges.txt` i `vocab.json`. Sprawdź: `wc -l artefakty/bpe-32k/merges.txt` → `31744`.

- [ ] **Step 3: Commit (artefakty są wynikiem do oddania)**

```bash
git add eksperymenty/e1_trening.py artefakty && git commit -qm "feat: trening 1k–32k i 8k tylko-szlwiki"
```

---

### Task 8: Część „Vocab size” (`eksperymenty/e2_slownik.py`)

**Files:**
- Create: `eksperymenty/e2_slownik.py`

- [ ] **Step 1: Implementacja**

```python
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
```

- [ ] **Step 2: Uruchom**

Run: `UVR python -m eksperymenty.e2_slownik`
Oczekiwane: 5 wierszy z metrykami; fertility maleje wraz z rozmiarem; pliki `wyniki/slownik.json`, `wyniki/fertility_vs_slownik.png`, `wyniki/dlugosci_tokenow.png`.

- [ ] **Step 3: Commit**

```bash
git add eksperymenty/e2_slownik.py wyniki && git commit -qm "feat: eksperyment rozmiaru słownika"
```

---

### Task 9: Pytanie A — ile kosztuje gŏdka (`eksperymenty/e3_pytanie_a.py`)

**Files:**
- Create: `eksperymenty/e3_pytanie_a.py`

- [ ] **Step 1: Implementacja**

```python
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
```

- [ ] **Step 2: Uruchom**

Run: `UVR python -m eksperymenty.e3_pytanie_a`
Oczekiwane: 5 wierszy (cl100k, o200k, Bielik, nasz 16k, nasz 32k), `wyniki/pytanie_a.json` i `.png`.

- [ ] **Step 3: Commit**

```bash
git add eksperymenty/e3_pytanie_a.py wyniki && git commit -qm "feat: pytanie A — koszt gŏdki w tokenizerach"
```

---

### Task 10: Pytanie B — kiedy rodzą się ō, ŏ, ô (`eksperymenty/e4_pytanie_b.py`)

**Files:**
- Create: `eksperymenty/e4_pytanie_b.py`
- Test: `tests/test_pytanie_b.py`

- [ ] **Step 1: Test funkcji szukającej ranku**

```python
from bpe.tokenizer import BPETokenizer
from bpe.train import count_words, train
from eksperymenty.e4_pytanie_b import rank_znaku

TEKST = "ōōōōō ŏŏ ą " * 50


def test_rank_znaku():
    tok = BPETokenizer(train(count_words([TEKST]), 300))
    r_o = rank_znaku(tok, "ō")
    assert r_o is not None and tok.vocab[256 + r_o["samodzielny"]] == "ō".encode()
    assert r_o["samodzielny"] < rank_znaku(tok, "ŏ")["samodzielny"]
    assert rank_znaku(tok, "ź") == {"samodzielny": None, "pierwszy_zawierajacy": None}
```

- [ ] **Step 2: FAIL**

Run: `UVR pytest tests/test_pytanie_b.py -q` → `ModuleNotFoundError`

- [ ] **Step 3: Implementacja**

```python
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
        ax.scatter(d["wystapienia_w_treningu"], d["samodzielny"], color=kolor)
        ax.annotate(z, (d["wystapienia_w_treningu"], d["samodzielny"]), fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wystąpienia znaku w korpusie treningowym")
    ax.set_ylabel("rank merge'a, który tworzy znak")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(WYNIKI / "pytanie_b.png", dpi=150)
    print(json.dumps({"znaki": znaki, "koncowki": koncowki}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: PASS i uruchomienie**

Run: `UVR pytest tests/test_pytanie_b.py -q` → `1 passed`
Run: `UVR python -m eksperymenty.e4_pytanie_b` → JSON z rankami; `wyniki/pytanie_b.png`.

- [ ] **Step 5: Commit**

```bash
git add eksperymenty/e4_pytanie_b.py tests/test_pytanie_b.py wyniki && git commit -qm "feat: pytanie B — kolejność powstawania znaków"
```

---

### Task 11: Pytanie C — Wikipedia a prawdziwa gŏdka (`eksperymenty/e5_pytanie_c.py`)

**Files:**
- Create: `eksperymenty/e5_pytanie_c.py`

- [ ] **Step 1: Implementacja**

```python
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
    wyniki["spadek_poza_wikipedia_proc"] = round(
        100 * (w["fineweb_szl"]["fertility"] / w["szlwiki"]["fertility"] - 1), 1)
    (WYNIKI / "pytanie_c.json").write_text(json.dumps(wyniki, ensure_ascii=False, indent=2))
    print(json.dumps(wyniki, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Uruchom**

Run: `UVR python -m eksperymenty.e5_pytanie_c` → JSON z `spadek_poza_wikipedia_proc`.

- [ ] **Step 3: Commit**

```bash
git add eksperymenty/e5_pytanie_c.py wyniki && git commit -qm "feat: pytanie C — Wikipedia a gŏdka z sieci"
```

---

### Task 12: Pytanie D — trzy pisownie (`eksperymenty/e6_pytanie_d.py`)

**Files:**
- Create: `eksperymenty/e6_pytanie_d.py`
- Test: `tests/test_pytanie_d.py`

Uwaga do raportu: warianty to **przybliżenie regułami**, nie prawdziwe teksty w innej pisowni.

- [ ] **Step 1: Test**

```python
from eksperymenty.e6_pytanie_d import bez_ogonkow, po_polsku


def test_po_polsku():
    assert po_polsku("Ôma gŏdała ô tym, co bōło") == "Oma godała o tym, co bóło"


def test_bez_ogonkow():
    assert bez_ogonkow("Ślōnsk, gŏdać, łōnczka") == "Slonsk, godac, lonczka"
```

- [ ] **Step 2: FAIL**

Run: `UVR pytest tests/test_pytanie_d.py -q` → `ModuleNotFoundError`

- [ ] **Step 3: Implementacja**

```python
"""D: ten sam śląski tekst testowy (szlwiki, Ślabikŏrz) w trzech pisowniach, mierzony we wszystkich tokenizerach."""
from __future__ import annotations

import json
import unicodedata

from eksperymenty.e3_pytanie_a import liczniki
from metryki import zmierz
from wspolne import WYNIKI, teksty

PO_POLSKU = str.maketrans({"ō": "ó", "Ō": "Ó", "ŏ": "o", "Ŏ": "O", "ô": "o", "Ô": "O"})


def po_polsku(t: str) -> str:
    return t.translate(PO_POLSKU)


def bez_ogonkow(t: str) -> str:
    t = t.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))


def main() -> None:
    oryginal = teksty("test", ["szlwiki"])
    warianty = {"Ślabikŏrz (oryginał)": oryginal,
                "po polsku (ō→ó, ŏ/ô→o)": [po_polsku(t) for t in oryginal],
                "bez znaków diakrytycznych": [bez_ogonkow(t) for t in oryginal]}
    wyniki = {n: {w: zmierz(f, ts)["fertility"] for w, ts in warianty.items()}
              for n, f in liczniki().items()}
    (WYNIKI / "pytanie_d.json").write_text(json.dumps(wyniki, ensure_ascii=False, indent=2))
    print(json.dumps(wyniki, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: PASS i uruchomienie**

Run: `UVR pytest tests/test_pytanie_d.py -q` → `2 passed`
Run: `UVR python -m eksperymenty.e6_pytanie_d` → tabela fertility (5 tokenizerów × 3 pisownie).

- [ ] **Step 5: Commit**

```bash
git add eksperymenty/e6_pytanie_d.py tests/test_pytanie_d.py wyniki && git commit -qm "feat: pytanie D — trzy pisownie"
```

---

### Task 13: Jedna komenda + README

**Files:**
- Create: `run_all.py`, `README.md`

- [ ] **Step 1: `run_all.py`**

```python
"""Cały potok jedną komendą: korpus -> trening -> metryki -> pytania A–D.

    uv run --no-project --with-requirements requirements.txt python run_all.py

Wymaga pobranych danych w data/raw (patrz README, sekcja „Dane”).
"""
import subprocess
import sys

KROKI = ["korpus.czysc", "eksperymenty.e1_trening", "eksperymenty.e2_slownik",
         "eksperymenty.e3_pytanie_a", "eksperymenty.e4_pytanie_b",
         "eksperymenty.e5_pytanie_c", "eksperymenty.e6_pytanie_d"]

for krok in KROKI:
    print(f"\n=== {krok} ===", flush=True)
    subprocess.run([sys.executable, "-m", krok], check=True)
```

- [ ] **Step 2: `README.md`** zawiera: cel (1 akapit), komendę pobrania danych (curl dla szlwiki i FineWeb-2 z Task „Dane” poniżej + `python scripts/pobierz_plwiki_slask.py data/raw/plwiki/plwiki_slask.jsonl` + wikiextractor), komendę `run_all.py`, zmienną `BIELIK_TOKENIZER`, zasadę remisu w treningu, opis katalogów `artefakty/` i `wyniki/`, licencje źródeł.

Dane — dokładne komendy do README:
```bash
mkdir -p data/raw/{szlwiki,fineweb2,plwiki}
curl -L -o data/raw/szlwiki/szlwiki-20261101-pages-articles.xml.bz2 https://dumps.wikimedia.org/szlwiki/latest/szlwiki-latest-pages-articles.xml.bz2
curl -L -o data/raw/fineweb2/train.parquet https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/szl_Latn/train/000_00000.parquet
curl -L -o data/raw/fineweb2/test.parquet  https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/szl_Latn/test/000_00000.parquet
uv run --no-project --with-requirements requirements.txt python -m wikiextractor.WikiExtractor data/raw/szlwiki/szlwiki-20261101-pages-articles.xml.bz2 --json -o data/raw/szlwiki/extracted -q
python scripts/pobierz_plwiki_slask.py data/raw/plwiki/plwiki_slask.jsonl
```

- [ ] **Step 3: Pełny przebieg od zera na istniejących danych**

Run: `rm -rf data/corpus artefakty wyniki && UVR python run_all.py`
Oczekiwane: wszystkie 7 kroków bez błędu; komplet plików w `artefakty/` i `wyniki/`.

- [ ] **Step 4: Wszystkie testy**

Run: `UVR pytest -q` → wszystkie zielone (23 testy).

- [ ] **Step 5: Commit**

```bash
git add run_all.py README.md artefakty wyniki && git commit -qm "feat: run_all.py i README"
```

---

### Task 14: Szkic raportu (`RAPORT.md`) — razem z autorką

**Files:**
- Create: `RAPORT.md`

Raport pisze autorka; wykonawca przygotowuje szkielet z liczbami z `wyniki/*.json` i `data/corpus/stats.json` (bez wniosków za autorkę).

- [ ] **Step 1: Szkielet** — sekcje i limity:
1. **Korpus** (≤ 1 strona): dziedzina, źródła z licencjami, tabela z `stats.json` (co odrzucono i ile), znane skrzywienia, akapit „jak to wpłynie na tokenizer” (napisany **przed** treningiem, z datą), potem „czy miałam rację”.
2. **Rozmiar słownika** (≤ 1 strona): wykres `fertility_vs_slownik.png`, tabela fertility i bajty/token, rzadkie tokeny, rekomendacja „gdybym trenowała…, wzięłabym X, bo…”.
3. **Pytanie A** (≤ 1 strona) i **B** (≤ 1 strona): hipoteza → metoda → wynik (wykres) → wniosek.
4. **Pytania C i D** (≤ ½ strony każde): to samo w skrócie.

- [ ] **Step 2: Commit**

```bash
git add RAPORT.md && git commit -qm "docs: szkielet raportu z wynikami"
```

---

## Samokontrola planu względem projektu

- Korpus ≥ 50 MB, licencje, czyszczenie, skrzywienia → Task 5 (+ krok awaryjny, gdy < 50 MB) i Task 14.
- ≥ 4 rozmiary, metryki na teście, histogram, wykres, rekomendacja → Task 7, 8, 14.
- Pytania A–D → Task 9–12; podział 2 pełne + 2 krótkie → Task 14.
- Kod uruchamiany jedną komendą, README → Task 13.
- Artefakty merges + vocab dla każdego rozmiaru → Task 7.
- Pułapki: pomiar na treningu (wszystkie eksperymenty używają `teksty("test", …)`); UTF-8 (test `test_slaskie_znaki_to_2_bajty`, pytanie B); „32k bo Bielik” (rekomendacja z danych, Task 14).
