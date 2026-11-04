from bpe.pretok import pretokenize


def test_spacja_doklejona_do_slowa():
    assert pretokenize("Jo żech je") == ["Jo", " żech", " je"]


def test_sklejenie_daje_oryginal():
    t = "Gŏdka  ze\nŚlōnska!\t ôd zawsze"
    assert "".join(pretokenize(t)) == t


def test_pusty():
    assert pretokenize("") == []
