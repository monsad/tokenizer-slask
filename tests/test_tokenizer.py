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
