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
