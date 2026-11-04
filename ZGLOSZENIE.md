Wrzucam pracę domową z treningu BPE. Zamiast „ogólnego polskiego” wzięłam gŏdkę i wyszło kilka rzeczy, których się nie spodziewałam.

Co zrobiłam:
- zebrałam korpus „Górny Śląsk: gŏdka i kultura”: 58 MB po czyszczeniu, z tego 25 MB po śląsku (śląska Wikipedia, Wachtyrz, FineWeb-2) i 33 MB po polsku o Śląsku
- napisałam BPE od zera w Pythonie, przyspieszony tak, że trening do 32k trwa ok. 2 minuty na laptopie (test sprawdza, że daje te same sklejenia co wersja z warsztatu)
- porównałam słowniki 1k, 4k, 8k, 16k i 32k oraz GPT-4, GPT-4o i Bielika v3, zawsze na 10% tekstów odłożonych przed treningiem
- przed pomiarami postawiłam 4 hipotezy. Dwie się potwierdziły, dwie padły (jedna w całości, druga w większej części).

Co mnie zaskoczyło:
- czyszczenie danych: 92% haseł śląskiej Wikipedii to krótkie szablony o wsiach. Do tego mój pierwszy filtr „czy to śląski” odrzucał gwarę cieszyńską, bo szukał tylko liter ō/ŏ/ô. Sam filtr miał uprzedzenie do jednej pisowni.
- Bielik po polsku jest o 28% tańszy niż GPT-4, ale na śląskim prawie tak samo drogi (2,74 vs 2,80 tokenu na słowo). Polski tokenizer nie przenosi się na gŏdkę.
- spodziewałam się, że GPT-4 będzie potrzebował 2× więcej tokenów niż mój tokenizer. Wyszło 1,5×.
- BPE nie traktuje ō, ŏ, ô gorzej niż polskich liter. Powstają dokładnie wtedy, kiedy polskie litery o podobnej częstości (ŏ jako 77. sklejenie, polskie ń jako 73.). Liczy się tylko, ile tekstu jest w danych.
- w GPT-4 i Bieliku poprawna śląska pisownia jest karana: ten sam tekst bez ogonków kosztuje 11–15% mniej tokenów. W moim tokenizerze jest odwrotnie.
- tokenizer uczony tylko na śląskiej Wikipedii jest o 17–20% gorszy na gŏdce z sieci. Wikipedia to inny śląski niż ten, którym ludzie piszą.
- przy 32k ponad 1/3 tokenów pojawia się w teście mniej niż 10 razy. Dla 58 MB tekstu rekomenduję 16k.

Raport z wykresami i kodem: https://github.com/monsad/tokenizer-slask (raport: RAPORT.md)

Chętnie przyjmę uwagi. Najbardziej ciekawi mnie, czy dobrze rozumiem, że przy języku z kilkoma pisowniami lepiej uczyć tokenizer na wszystkich wariantach, niż ujednolicać tekst do jednej pisowni przed treningiem?
