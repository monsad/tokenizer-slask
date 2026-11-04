# Tokenizer BPE dla gŏdki — projekt pracy domowej

Kurs 01 „Train LLM from scratch”, lekcja 02 (Fabryka), zadanie „Warstwa 1: Trening BPE”.

## Po co

Wytrenować własny tokenizer BPE i zbadać go na danych, których nikt inny w grupie nie ma:
śląskiej gŏdce i tekstach o kulturze Górnego Śląska. Ocena: korpus 25%, rozmiar słownika 25%,
pytania badawcze 35%, kod 15%.

## Cele

- Korpus ≥ 50 MB po czyszczeniu, legalny, z uczciwie zadeklarowaną dziedziną.
- Tokenizery w ≥ 4 rozmiarach słownika, metryki liczone **tylko na odłożonym zbiorze testowym**.
- 4 pytania badawcze w schemacie hipoteza → metoda → wynik → wniosek.
- Repo uruchamiane jedną komendą, raport w markdown ≤ 5 stron, pliki merges i vocab dla każdego rozmiaru.

## Poza zakresem

- Trenowanie modelu językowego.
- Własny regex do wstępnego cięcia tekstu (to temat innej pracy domowej). Ewentualne cięcie
  na słowa robimy prostym podziałem po białych znakach, opisanym w raporcie.
- Tokeny specjalne.

## Czego unikamy (porażka, nawet gdyby kod działał)

- Pomiary na korpusie treningowym.
- Rozmiar słownika „bo tak ma Bielik”.
- Pytania z odpowiedzią znaną z góry; zawyżanie wyników pod tezę.
- Cięcie polskich i śląskich znaków (2 bajty w UTF-8) bez zrozumienia, co się dzieje.

## Dziedzina i korpus

**Zadeklarowana dziedzina:** „Górny Śląsk: gŏdka i kultura”, czyli teksty po śląsku oraz polskie teksty o Górnym Śląsku.

| Źródło | Surowo | Licencja |
|---|---|---|
| Ślōnsko Wikipedyjo (zrzut 2026-11-01) | 16 MB | CC BY-SA |
| FineWeb-2 `szl_Latn` | 21 MB | ODC-By |
| Polska Wikipedia, kategorie o Górnym Śląsku (głębokość 2, bez Zagłębia i Dolnego Śląska) | pobierane | CC BY-SA |
| Wachtyrz (28 MB, artykuły dwujęzyczne → `wachtyrz_szl` i `wachtyrz_pl`) | 28 MB | potwierdzona przez autorkę (źródło licencji podać w raporcie) |

**Czyszczenie** (każdy krok liczony i opisany w raporcie):
1. Odrzucenie dokumentów FineWeb bez śląskich cech (wskaźnik znaków ō/ŏ/ô i słów typu *niy, ôd, kaj*).
2. Deduplikacja dokładna (hash dokumentu) i na poziomie linii między źródłami.
3. Przycięcie szablonowych haseł geograficznych ze śląskiej Wikipedii. Zostawiamy losową część
   i zapisujemy, ile.
4. Normalizacja Unicode NFC, bo ō bywa zapisane jako o + znak łączący.

**Podział:** 90% trening, 10% test. Podział robimy po dokumentach, nie po liniach, i z rozbiciem
na źródła, żeby pytanie C miało osobne zbiory testowe dla każdego źródła.

**Znane skrzywienia** (do akapitu „przed treningiem”): dominacja kilku serwisów, nadreprezentacja
haseł geograficznych, pisownia głównie Ślabikŏrzowa, mało tekstu mówionego.

## Rozmiar słownika

- Rozmiary: **1k, 4k, 8k, 16k, 32k**. Większych nie robimy, bo przy ok. 50 MB rzadkie tokeny byłyby
  słabo wyuczone. Sam ten argument też sprawdzamy na danych (ile tokenów występuje w teście < 10 razy).
- Metryki na zbiorze testowym: tokeny/słowo (fertility), bajty/token (kompresja), histogram długości
  tokenów, odsetek słów jako 1 token.
- Wynik: wykres fertility względem rozmiaru słownika i rekomendacja oparta na tym, gdzie przestaje się
  opłacać. Konkretny rozmiar wybieramy po pomiarach.

## Pytania badawcze

Pełne (po 1 stronie): **A** i **B**. Krótkie (po pół strony): **C** i **D**. *Moja propozycja, do zmiany.*

**A. Ile kosztuje gŏdka?**
Hipoteza: tokenizery ogólne (cl100k, o200k) potrzebują ≥ 2× więcej tokenów na słowo na śląskim
tekście testowym niż nasz 16k. Bielik v3 (APT4, polski, 32k) leży pośrodku, bliżej ogólnych niż nas.
Metoda: fertility na tym samym zbiorze testowym, osobno dla śląskiego i polskiego o Śląsku.

**B. Kiedy rodzą się ō, ŏ, ô?**
Hipoteza: ō (częste) staje się pełnym znakiem w pierwszych ~100 merge'ach, ŏ i ô wyraźnie później.
Końcówki typu *-ōm* pojawiają się w słowniku dopiero przy ≥ 4k.
Metoda: numer merge'a (rank), przy którym dwa bajty znaku łączą się w jeden token, oraz pierwsze
tokeny zawierające każdą końcówkę. Porównanie z ą/ę/ż w tym samym tokenizerze.

**C. Wikipedia a prawdziwa gŏdka.**
Hipoteza: tokenizer uczony tylko na śląskiej Wikipedii ma ≥ 15% gorszą fertility na blogach
i serwisach z FineWeb niż na Wikipedii.
Metoda: osobny tokenizer 8k uczony na części wikipedyjnej, test na odłożonych zbiorach z każdego źródła.

**D. Jedna gŏdka, trzy pisownie.**
Hipoteza: zapis „po polsku” (ō→o, ŏ→o, ô→o) obniża fertility w tokenizerach polskich i ogólnych,
ale podnosi w naszym.
Metoda: ten sam zbiór testowy w trzech wariantach pisowni, uzyskanych prostymi regułami (opisanymi
jako przybliżenie), zmierzony we wszystkich tokenizerach z pytania A.

## Kod

- Trening oparty na `train_bpe_demo.py` z zajęć, przyspieszony liczeniem par na unikalnych słowach
  z licznikami zamiast na jednym ciągu bajtów. Wynik musi być zgodny z wersją z zajęć: test na małym
  tekście daje identyczne merge'e.
- Jedna komenda (`make all` albo `python run_all.py`): dane → czyszczenie → trening → metryki → wykresy.
- Testy: zgodność z wersją z zajęć, encode→decode zwraca ten sam tekst, poprawne bajty dla ō/ŏ/ô.

## Moje decyzje (do zmiany)

- Cięcie na słowa po białych znakach przed BPE (spacja doklejona do słowa, jak w GPT-2).
  Bez tego trening na 50 MB jest niewykonalny w czystym Pythonie.
- Rozmiary słownika 1k–32k.
- Podział pytań na 2 pełne i 2 krótkie, ze względu na limit 5 stron.
- Wachtyrz dzielony akapitami na część śląską i polską wskaźnikiem „śląskości”.

## Do decyzji wykonawcy

Nazwy plików, format wykresów, dokładny próg filtra „śląskości” (dobrany na próbce i opisany w raporcie).
