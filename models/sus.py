"""System Usability Scale: 10 pernyataan, skala 1-5, skor 0-100."""
N_ITEMS = 10


def score(answers):
    """Butir ganjil: (jawaban - 1); butir genap: (5 - jawaban); jumlah x 2,5."""
    if len(answers) != N_ITEMS or any(a not in (1, 2, 3, 4, 5) for a in answers):
        raise ValueError("answers")
    total = sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))
    return total * 2.5


def grade(s):
    """Kategori kasar: good (>= 80), ok (68-80), poor (< 68). 68 adalah rata-rata SUS secara umum."""
    return "good" if s >= 80 else ("ok" if s >= 68 else "poor")