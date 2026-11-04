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
