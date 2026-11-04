Praca domowa: własny tokenizer BPE na gŏdce, czyli ile kosztuje śląski.

Najpierw kilka słów wyjaśnienia:
- token to kawałek tekstu, na którym pracuje model: całe słowo, część słowa albo znak. Im więcej tokenów, tym drożej i tym mniej tekstu wchodzi do modelu naraz.
- tokenizer tnie tekst na tokeny, a jego słownik to lista wszystkich tokenów, które zna.
- BPE buduje słownik od zera: zaczyna od pojedynczych bajtów i w kółko skleja parę, która najczęściej stoi obok siebie, np. „g”+„ŏ” → „gŏ”, potem „gŏ”+„d” → „gŏd”.
- tokeny na słowo (fertility) mówią, na ile kawałków średnio jest cięte jedno słowo. Mierzyłam to tylko na tekstach, których tokenizer nie widział przy nauce.
- gŏdka to język śląski; w najpopularniejszej pisowni (Ślabikŏrz) ma litery ō, ŏ, ô, które w komputerze zajmują po 2 bajty, tak jak ą czy ż.

Co zrobiłam (58 MB tekstu, BPE napisany od zera w Pythonie, wszystko na moim komputerze):
- korpus „Górny Śląsk: gŏdka i kultura”: 25 MB po śląsku (śląska Wikipedia, Wachtyrz, FineWeb-2) i 33 MB po polsku o Śląsku. Ciekawostka z czyszczenia: 92% haseł śląskiej Wikipedii to krótkie szablony o wsiach, a mój pierwszy filtr „czy to śląski” odrzucał gwarę cieszyńską, bo szukał tylko liter ō/ŏ/ô.
- rozmiar słownika: od 1k do 32k. Każde podwojenie daje coraz mniej, a przy 32k ponad 1/3 tokenów pojawia się w teście mniej niż 10 razy. Dla tego korpusu rekomenduję 16k.
- GPT-4 potrzebuje na gŏdkę 1,5× więcej tokenów niż mój tokenizer (spodziewałam się 2×, hipoteza padła).
- Bielik po polsku jest o 28% tańszy niż GPT-4, ale na śląskim prawie tak samo drogi. Polski tokenizer nie przenosi się na gŏdkę.
- ō, ŏ, ô dostają własne tokeny dokładnie wtedy, kiedy polskie litery o podobnej częstości (ŏ tuż obok ń). BPE nie wie, co to język, liczy tylko, co stoi obok czego.
- tokenizer uczony tylko na śląskiej Wikipedii jest o 17–20% gorszy na gŏdce z sieci. Wikipedia to inny śląski niż ten, którym ludzie piszą.
- w GPT-4 i Bieliku poprawna śląska pisownia jest karana: ten sam tekst bez ogonków kosztuje 11–15% mniej tokenów. W moim tokenizerze odwrotnie.

Raport z tabelami i wykresami: [link do RAPORT.md w repo]
