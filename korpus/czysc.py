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
# Znaki Ślabikŏrza ORAZ słowa gwarowe w zapisie polskimi literami (gwara cieszyńska, zapis fonetyczny).
# Pierwsza wersja miała tylko znaki ō/ŏ/ô i kilka słów — odrzucała gŏdkę spoza Ślabikŏrza (opisane w raporcie).
SZL_MARKERY = re.compile(
    r"[ōŏô]|\b(niy|żech|żeś|jakech|ôd|kaj|kajś|tyż|tysz|zaś|ino|yno|kery|kerŏ|kiery|kierej|kiere|kerej|kere"
    r"|bōł|bōła|bydzie|gŏdać|godać|godo|uż|isto|fest|wiela|kiej|kej|zawdy|momy|mocka|fto|ftedy|wos|nom"
    r"|żodn\w*|bajtel\w*|dziołch\w*|synek|terozki|łoroz)\b")
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


def rozdziel_akapity(akapity: list[str]) -> tuple[str, str]:
    """Dzieli akapity dwujęzycznego artykułu na część śląską i polską."""
    szl = [a for a in akapity if szl_score(a) >= PROG_SZL]
    pl = [a for a in akapity if szl_score(a) < PROG_SZL]
    return "\n".join(szl), "\n".join(pl)


def wczytaj_wachtyrz(stats: dict) -> list[dict]:
    tekst = nfc((RAW / "wachtyrz/wachtyrz.txt").read_text(encoding="utf-8"))
    docs = []
    artykuly = tekst.split("\n\n")
    for n, art in enumerate(artykuly):
        szl, pl = rozdziel_akapity([a.strip() for a in art.split("\n") if a.strip()])
        if len(szl) >= 200:
            docs.append({"source": "wachtyrz_szl", "id": f"w{n}", "text": szl})
        if len(pl) >= 200:
            docs.append({"source": "wachtyrz_pl", "id": f"w{n}", "text": pl})
    stats["wachtyrz"] = {"artykuly": len(artykuly),
                         "dok_szl": sum(d["source"] == "wachtyrz_szl" for d in docs),
                         "dok_pl": sum(d["source"] == "wachtyrz_pl" for d in docs)}
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
