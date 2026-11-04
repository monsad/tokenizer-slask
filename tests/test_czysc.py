from korpus.czysc import dedup_docs, dedup_lines, nfc, podziel, rozdziel_akapity, szl_score

SZL = "Jo żech je ze Ślōnska, niy ma kaj gŏdać ôd zawsze."
PL = "Jestem ze Śląska i lubię rozmawiać o historii regionu."


def test_szl_score_rozroznia():
    assert szl_score(SZL) > 0.1 > szl_score(PL)


def test_nfc():
    rozlozone = "ō"                       # o + makron łączący
    assert nfc(rozlozone) == "ō"


def test_dedup_docs():
    d = [{"source": "a", "id": "1", "text": "x y z"}, {"source": "b", "id": "2", "text": "x y z"}]
    assert [x["id"] for x in dedup_docs(d)] == ["1"]


def test_dedup_lines_usuwa_dlugie_powtorki():
    dl = "To je dugŏ linijka, kerŏ sie powtarzŏ we dwōch dokumyntach naroz."
    d = [{"source": "a", "id": "1", "text": dl + "\nkrótka"},
         {"source": "b", "id": "2", "text": dl + "\ninksze"}]
    out = dedup_lines(d, min_len=40)
    assert out[1]["text"] == "inksze" and dl in out[0]["text"]


def test_rozdziel_akapity():
    szl, pl = rozdziel_akapity([SZL, PL, SZL])
    assert szl == SZL + "\n" + SZL and pl == PL


def test_podzial_deterministyczny_i_rozlaczny():
    d = [{"source": "a", "id": str(i), "text": f"t{i}"} for i in range(100)]
    tr, te = podziel(d, test_frac=0.1, seed=1)
    tr2, te2 = podziel(d, test_frac=0.1, seed=1)
    assert [x["id"] for x in te] == [x["id"] for x in te2]
    assert len(te) == 10 and not {x["id"] for x in tr} & {x["id"] for x in te}
