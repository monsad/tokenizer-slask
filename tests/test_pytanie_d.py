from eksperymenty.e6_pytanie_d import bez_ogonkow, po_polsku


def test_po_polsku():
    assert po_polsku("Ôma gŏdała ô tym, co bōło") == "Oma godała o tym, co bóło"


def test_bez_ogonkow():
    assert bez_ogonkow("Ślōnsk, gŏdać, łōnczka") == "Slonsk, godac, lonczka"
