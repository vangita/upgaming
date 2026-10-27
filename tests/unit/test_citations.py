from app.rag.citations import extract_citations, normalize_answer


def test_single_citation():
    assert extract_citations("Tim Burton directed it [1].") == [1]


def test_several_sources_in_one_sentence():
    assert extract_citations("He was born in Moscow [2][4].") == [2, 4]


def test_comma_separated_list():
    assert extract_citations("Both are films [1, 3].") == [1, 3]


def test_order_of_first_appearance_without_duplicates():
    assert extract_citations("A [3]. B [1]. C [3][2].") == [3, 1, 2]


def test_full_width_brackets():
    assert extract_citations("Tim Burton【1】.") == [1]


def test_numbers_outside_brackets_are_not_citations():
    assert extract_citations("He was born in 1958 and had 2 children.") == []


def test_no_citations():
    assert extract_citations("I don't have enough information to answer this question.") == []


def test_normalize_answer():
    assert normalize_answer("  Tim Burton【1】. ") == "Tim Burton[1]."


def test_normalize_look_alike_characters():
    # Escapes on purpose: these characters are invisible in the editor
    assert normalize_answer("actress‑writer,\xa0born 1975") == "actress-writer, born 1975"
