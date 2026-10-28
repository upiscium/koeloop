import random

import pytest

from restream.segmenter import Segmenter


def collect(text, chunk=1, **kwargs):
    segmenter = Segmenter(**kwargs)
    result = []
    for i in range(0, len(text), chunk):
        result.extend(segmenter.feed(text[i:i + chunk]))
    result.extend(segmenter.flush())
    return result


@pytest.mark.parametrize("mode", ["raw", "sentence", "balanced", "syntax"])
@pytest.mark.parametrize("chunk", [1, 3, 8, 1000])
def test_lossless(mode, chunk):
    text = 'こんにちは。「値段は1,000円、重さは3.14kgです。」\n次の行（注：重要です！）も読みます。末尾に句点なし 🐈'
    segments = collect(text, chunk, mode=mode)
    assert "".join(s.text for s in segments) == text
    assert all(0 < len(s.text) <= 160 for s in segments)


@pytest.mark.parametrize("chunk", [1, 2, 8, 1000])
def test_sentence_independent_of_chunks(chunk):
    text = "こんにちは。『本当ですか？』はい！最後"
    assert [s.text for s in collect(text, chunk, mode="sentence")] == ["こんにちは。", "『本当ですか？』", "はい！", "最後"]


def test_waits_for_closing_quote():
    s = Segmenter()
    assert s.feed("「こんにちは。") == []
    assert s.idle() == []
    assert s.feed("」") == []
    assert [x.text for x in s.idle()] == ["「こんにちは。」"]


def test_idle_does_not_split_incomplete_text():
    s = Segmenter()
    s.feed("まだ文の途中")
    assert s.idle() == []
    assert s.flush()[0].reason == "end_of_input"


def test_clause_needs_minimum_and_lookahead():
    s = Segmenter(min_chars=5, lookahead=4)
    assert s.feed("最初の節で、") == []
    assert s.feed("次の") == []
    result = s.feed("節です")
    assert [(x.text, x.reason) for x in result] == [("最初の節で、", "clause")]


def test_sentence_preferred_over_clause_when_available():
    result = collect("長めの節ですが、最後まで届いています。", 100, min_chars=4, lookahead=2)
    assert len(result) == 1
    assert result[0].reason == "sentence"


def test_decimals_thousands_and_urls():
    text = "値は1,000と3.14、URLはhttps://example.com/aです。"
    result = collect(text, min_chars=1, lookahead=1)
    assert not any(s.text.endswith("1,") for s in result)
    assert not any(s.text.endswith("3.") for s in result)


def test_forced_limit_marked_even_inside_unclosed_quote():
    result = collect("「" + "あ" * 50, max_chars=20, min_chars=5, lookahead=2)
    assert [len(s.text) for s in result] == [20, 20, 11]
    assert result[0].reason == "forced_limit"


def test_randomized_lossless():
    rng = random.Random(42)
    alphabet = "あいう。！？、（）「」12.3\n abc🐈"
    for _ in range(150):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 800)))
        result = collect(text, chunk=rng.randint(1, 80), max_chars=50, min_chars=10, lookahead=5)
        assert "".join(s.text for s in result) == text
        assert all(len(s.text) <= 50 for s in result)


@pytest.mark.parametrize("kwargs", [{"mode": "bad"}, {"min_chars": 200}, {"lookahead": 200}, {"max_chars": 0}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Segmenter(**kwargs)
