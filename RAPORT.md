# Tokenizer BPE dla gŏdki — raport

> Szkielet z liczbami z `wyniki/`. Miejsca oznaczone **✍️** to Twoje wnioski: napisz je własnymi słowami.
> Limit: 5 stron. Korpus ≤ 1 s., Vocab size ≤ 1 s., pytania A i B ≤ 1 s. każde, C i D ≤ ½ s. każde.

## 1. Korpus

**Dziedzina:** „Górny Śląsk: gŏdka i kultura”, czyli teksty po śląsku oraz polskie teksty o Górnym Śląsku.
To nie jest „ogólny polski”, tylko świadomie wąska, regionalna dziedzina.

| Źródło | Język | Po czyszczeniu | Licencja |
|---|---|---|---|
| Ślōnsko Wikipedyjo (zrzut 2026-11-01) | śląski | 6,7 MB | CC BY-SA |
| Wachtyrz.eu, akapity śląskie | śląski | 6,7 MB | ✍️ źródło licencji |
| FineWeb-2 `szl_Latn` | śląski (różne pisownie i gwary) | 11,8 MB | ODC-By |
| Polska Wikipedia, kategorie o Górnym Śląsku (wersja 2023-11-01) | polski | 11,0 MB | CC BY-SA |
| Wachtyrz.eu, akapity polskie | polski | 22,0 MB | ✍️ |
| **Razem** | **25,1 MB śląskiego + 33,0 MB polskiego** | **58,1 MB** | |

**Czyszczenie** (`korpus/czysc.py`, liczby w `wyniki/korpus_stats.json`):
1. Normalizacja Unicode NFC (ō zapisane jako o + znak łączący → jeden znak).
2. Śląska Wikipedia: **56 784 z ok. 61 tys. haseł ma < 300 znaków**. To szablonowe hasła o miejscowościach.
   Zostawiono losowe 25% z nich, bo inaczej korpus byłby zdominowany przez jeden szablon.
3. FineWeb-2: odrzucono 853 kopie stron śląskiej Wikipedii i 747 dokumentów bez cech śląskich
   (wskaźnik `szl_score`: udział znaków ō/ŏ/ô i słów gwarowych).
   **Pierwsza wersja filtra miała skrzywienie:** szukała głównie ō/ŏ/ô i odrzucała gŏdkę spoza
   Ślabikŏrza (gwara cieszyńska, zapis fonetyczny, np. „Zwrocomy sie dlo Wos…”). Po dodaniu słów gwarowych
   w zapisie polskimi literami odrzuconych było 738 zamiast 1269.
4. Wachtyrz: artykuły dwujęzyczne podzielone akapitami na część śląską i polską tym samym wskaźnikiem.
5. Deduplikacja: identyczne dokumenty i powtórzone linie ≥ 60 znaków (64,3 MB → 58,1 MB).
6. Podział 90/10 **po dokumentach, osobno w każdym źródle**. Wszystkie pomiary na 10% testowych.

**Znane skrzywienia:** więcej polskiego niż śląskiego (57% / 43%); dominacja kilku serwisów (Wachtyrz,
Wikipedia, nowiny.rybnik.pl); polska część to głównie historia, biografie i polityka; mało mowy potocznej
i dialogów; gwary na granicy (góralska, cieszyńska po czeskiej stronie) częściowo odrzucone przez filtr.

**Przewidywanie przed treningiem** ✍️ (jeden akapit: jak wybór korpusu wpłynie na tokenizer; napisz to tak, jak myślałaś *przed* zobaczeniem wyników)

**Czy miałam rację?** ✍️ (po wynikach)

## 2. Rozmiar słownika

Jeden trening do 32k. Mniejsze słowniki to pierwsze V−256 merge'y tego samego treningu.
Metryki na zbiorze testowym (`wyniki/slownik.json`):

| Słownik | Tokeny/słowo | Bajty/token | Tokeny użyte < 10 razy w teście |
|---|---|---|---|
| 1k | 3,24 | 2,25 | 78 |
| 4k | 2,41 | 3,03 | 102 |
| 8k | 2,11 | 3,47 | 202 |
| 16k | 1,86 | 3,93 | 1 051 |
| 32k | 1,66 | 4,41 | **11 242** (35% słownika) |

![fertility](wyniki/fertility_vs_slownik.png)

Co widać w liczbach:
- Każde podwojenie słownika daje mniejszy zysk: 1k→4k −26%, 4k→8k −12%, 8k→16k −12%, 16k→32k −11%.
- Przy 32k ponad jedna trzecia tokenów pojawia się w teście mniej niż 10 razy, więc są słabo wyuczone.
  Przy 16k to 7%.
- Śląska Wikipedia ma najwyższą liczbę tokenów na słowo przy każdym rozmiarze (pewnie przez nazwy
  własne i rzadkie słowa w hasłach geograficznych).

Histogram długości tokenów: `wyniki/dlugosci_tokenow.png`.

**Rekomendacja** ✍️ „Gdybym trenowała model na tym korpusie, wzięłabym słownik X, bo…” (oprzyj na kolumnie
„użyte < 10 razy” i na malejącym zysku, nie na tym, co ma Bielik).

## 3. Pytanie A: ile kosztuje gŏdka?

**Hipoteza:** tokenizery ogólne (cl100k, o200k) potrzebują ≥ 2× więcej tokenów na słowo na śląskim niż
nasz 16k. Bielik v3 (APT4, polski, 32k) jest pośrodku.

**Metoda:** tokeny na słowo na tym samym zbiorze testowym, osobno część śląska (3 źródła) i polska (2 źródła).

| Tokenizer | śląski | polski o Śląsku | śląski / polski |
|---|---|---|---|
| GPT-4 (cl100k) | 2,80 | 2,75 | 1,02 |
| GPT-4o (o200k) | 2,55 | 2,38 | 1,07 |
| Bielik v3 (APT4 32k) | 2,74 | 1,97 | **1,39** |
| nasz 16k | 1,88 | 1,84 | 1,02 |
| nasz 32k | 1,70 | 1,63 | 1,04 |

![A](wyniki/pytanie_a.png)

**Wynik:** GPT-4 / nasz 16k = 2,80 / 1,88 = **1,49×**, czyli hipoteza „≥ 2×” **odrzucona**.
Bielik na śląskim (2,74) jest prawie tak samo kosztowny jak GPT-4 (2,80), choć po polsku jest o 28% tańszy.

**Wniosek** ✍️

## 4. Pytanie B: kiedy rodzą się ō, ŏ, ô?

**Hipoteza:** ō staje się jednym tokenem w pierwszych ~100 merge'ach, ŏ i ô wyraźnie później.
Końcówki typu *-ōm* pojawiają się dopiero przy słowniku ≥ 4k.

**Metoda:** numer merge'a, który skleja 2 bajty znaku w jeden token, porównany z ą, ę, ż…
i z liczbą wystąpień znaku w korpusie treningowym (`wyniki/pytanie_b.json`).

| Znak | ł | ó | ż | ą | ś | ę | **ō** | ć | ń | **ŏ** | **ô** | ź |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Merge nr | 1 | 15 | 16 | 19 | 24 | 27 | **28** | 45 | 73 | **77** | **89** | 190 |

![B](wyniki/pytanie_b.png)

Końcówki (numer merge'a, który tworzy token): *ōm* 98, *ōw* 234, *ymu* 2204; polskie *ów* 69, *emu* 4092.

**Wynik:** pierwsza część hipotezy potwierdzona (ō: 28). „Wyraźnie później” dla ŏ/ô **nie**: są tuż obok
polskiego ń, a kolejność wszystkich znaków wynika z tego, jak często występują (wykres to prawie linia
prosta). Końcówka *-ōm* powstaje już przy słowniku ~354 tokenów, więc hipoteza „≥ 4k” **odrzucona**.

**Wniosek** ✍️

## 5. Pytanie C: Wikipedia a prawdziwa gŏdka

**Hipoteza:** tokenizer uczony tylko na śląskiej Wikipedii ma ≥ 15% gorszy wynik na gŏdce z sieci niż na Wikipedii.

**Metoda:** tokenizer 8k uczony tylko na szlwiki vs 8k uczony na całości (`wyniki/pytanie_c.json`).

| Tokeny/słowo | szlwiki | Wachtyrz (szl) | FineWeb (szl) |
|---|---|---|---|
| 8k tylko szlwiki | 1,99 | 2,32 (**+17%**) | 2,38 (**+20%**) |
| 8k całość | 2,30 | 2,03 | 2,02 |

**Wynik:** potwierdzona. Ciekawostka: tokenizer z całości jest *gorszy* na samej Wikipedii (2,30 vs 1,99).

**Wniosek** ✍️

## 6. Pytanie D: jedna gŏdka, trzy pisownie

**Hipoteza:** zapis „po polsku” (ō→ó, ŏ/ô→o) obniża liczbę tokenów w tokenizerach polskich i ogólnych,
ale podnosi w naszym. Warianty pisowni to **przybliżenie regułami** na tekście ze szlwiki.

| Tokeny/słowo | Ślabikŏrz | po polsku | bez ogonków |
|---|---|---|---|
| GPT-4 | 3,20 | 3,00 | 2,81 |
| GPT-4o | 2,95 | 2,75 | 2,61 |
| Bielik v3 | 3,42 | 3,12 | 2,91 |
| nasz 16k | 2,06 | 2,17 | 2,22 |
| nasz 32k | 1,85 | 1,92 | 2,01 |

**Wynik:** potwierdzona. Usunięcie ogonków to dla GPT-4 i Bielika *oszczędność* tokenów, a dla naszego koszt.

**Wniosek** ✍️
