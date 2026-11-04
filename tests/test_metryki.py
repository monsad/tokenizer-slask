from metryki import zmierz


def test_zmierz():
    m = zmierz(lambda t: len(t.split()) * 2, ["ala ma kota", "ôd zawsze"])
    assert m["slowa"] == 5 and m["tokeny"] == 10
    assert m["fertility"] == 2.0
    assert m["bajty_na_token"] == round(len("ala ma kotaôd zawsze".encode()) / 10, 4)
