import threading
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from examples.segmentation_samples import SAMPLES
from restream import syntax
from restream.app import create_app
from restream.segmenter import Segmenter
from restream.syntax import Token, assess


def clause_tokens():
    return (
        Token(0, 1, "雨", "NOUN", "", "nsubj", 2),
        Token(1, 2, "が", "ADP", "", "case", 0),
        Token(2, 4, "降る", "VERB", "", "advcl", 6),
        Token(4, 6, "ので", "SCONJ", "", "mark", 2),
        Token(6, 7, "、", "PUNCT", "", "punct", 2),
        Token(7, 8, "家", "NOUN", "", "obl", 6),
        Token(8, 10, "帰る", "VERB", "", "ROOT", 6),
    )


def test_accepts_a_marked_closed_predicate_clause():
    assert assess(clause_tokens(), 7).accepted


@pytest.mark.parametrize(("index", "changes", "reason"), [
    (2, {"pos": "NOUN"}, "not_predicate"),
    (2, {"dep": "acl"}, "relative_clause"),
    (2, {"dep": "obj"}, "not_clause"),
    (3, {"text": "た", "pos": "AUX", "dep": "aux"}, "no_connective"),
    (6, {"pos": "NOUN"}, "right_predicate_pending"),
    (0, {"head": 6}, "crossing_dependency"),
])
def test_rejects_incomplete_or_uncertain_relations(index, changes, reason):
    tokens = list(clause_tokens())
    tokens[index] = replace(tokens[index], **changes)
    decision = assess(tuple(tokens), 7)
    assert not decision.accepted
    assert decision.reason == reason


def test_rejects_nested_relative_clause():
    tokens = list(clause_tokens())
    tokens[6] = replace(tokens[6], dep="acl", head=7)
    tokens.append(Token(10, 11, "人", "NOUN", "", "ROOT", 7))
    assert assess(tuple(tokens), 7).reason == "relative_clause"


class FakeAnalyzer:
    def __init__(self, unstable=False):
        self.inputs = []
        self.unstable = unstable

    def parse(self, text):
        self.inputs.append(text)
        if "、" not in text:
            return ()
        comma = text.index("、")
        parent = comma + 1 + (len(self.inputs) % 2 if self.unstable else 0)
        return (
            Token(0, comma - 2, text[:comma - 2], "VERB", "", "advcl", 3),
            Token(comma - 2, comma, "ので", "SCONJ", "", "mark", 0),
            Token(comma, comma + 1, "、", "PUNCT", "", "punct", 0),
            Token(parent, parent + 1, "読む", "VERB", "", "ROOT", 3),
        )


def test_requires_two_stable_views_and_extra_lookahead():
    analyzer = FakeAnalyzer()
    segmenter = Segmenter("syntax", analyzer=analyzer)
    prefix = "あ" * 24 + "ので、"
    assert segmenter.feed(prefix + "い" * 12) == []
    assert not analyzer.inputs
    result = segmenter.feed("い" * 8)
    assert [(s.text, s.reason) for s in result] == [(prefix, "syntax_clause")]
    assert len(analyzer.inputs) == 2
    assert len(analyzer.inputs[1]) - len(analyzer.inputs[0]) == 8


def test_disagreement_holds_boundary_and_idle_does_not_reparse():
    analyzer = FakeAnalyzer(unstable=True)
    s = Segmenter("syntax", analyzer=analyzer)
    text = "あ" * 24 + "ので、" + "い" * 40
    assert not s.feed(text)
    assert s.diagnostics[-1]["reason"] == "unstable_dependency"
    for _ in range(20):
        assert not s.idle()
    assert len(analyzer.inputs) == 2
    assert "".join(x.text for x in s.flush()) == text


def test_completed_sentence_bypasses_parser():
    analyzer = FakeAnalyzer()
    s = Segmenter("syntax", analyzer=analyzer)
    text = "あ" * 24 + "ので、" + "い" * 30 + "。"
    assert not s.feed(text)
    assert [x.text for x in s.flush()] == [text]
    assert not analyzer.inputs


def test_inside_quote_is_not_a_syntax_candidate():
    analyzer = FakeAnalyzer()
    s = Segmenter("syntax", analyzer=analyzer)
    text = "「" + "あ" * 24 + "ので、" + "い" * 30
    assert not s.feed(text)
    assert not s.idle()
    assert not analyzer.inputs
    assert "".join(x.text for x in s.feed("」") + s.flush()) == text + "」"


def test_bounded_context_and_analysis_frequency():
    analyzer = FakeAnalyzer(unstable=True)
    s = Segmenter("syntax", analyzer=analyzer)
    text = "あ" * 24 + "ので、" + "い" * 300
    output = []
    for char in text:
        output.extend(s.feed(char))
    output.extend(s.flush())
    assert "".join(x.text for x in output) == text
    assert max(map(len, analyzer.inputs)) <= 180
    assert len(analyzer.inputs) <= 32  # two views per at least eight new characters
    assert output[0].reason == "forced_limit"


def test_runtime_failure_falls_back_without_losing_text():
    class Broken:
        def parse(self, text):
            raise RuntimeError("private input should not leak")

    s = Segmenter("syntax", analyzer=Broken())
    text = "あ" * 24 + "ので、" + "い" * 30
    assert not s.feed(text)
    assert s.effective_mode == "sentence"
    assert len(s.notices) == 1
    assert "private" not in s.notices[0]
    assert [x.text for x in s.flush()] == [text]


def test_missing_parser_warning_is_delivered_even_without_a_comma(monkeypatch):
    monkeypatch.setattr(syntax, "installed", lambda: False)
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "mode": "syntax", "audio": False})
        assert ws.receive_json()["type"] == "ready"
        warning = ws.receive_json()
        assert warning["type"] == "warning"
        assert warning["code"] == "syntax_fallback"
        ws.send_json({"type": "text", "text": "末尾"})
        ws.send_json({"type": "end"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "done":
                break
        assert event["segmentation"]["effective_mode"] == "sentence"
        assert "".join(e["text"] for e in events if e["type"] == "segment") == "末尾"


def test_cancel_does_not_wait_for_slow_parser(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class Slow(FakeAnalyzer):
        def parse(self, text):
            entered.set()
            release.wait(timeout=4)
            return super().parse(text)

    monkeypatch.setattr(syntax, "installed", lambda: True)
    monkeypatch.setattr(syntax, "ANALYZER", Slow())
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        try:
            ws.send_json({"type": "start", "mode": "syntax", "audio": False})
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "text", "text": "あ" * 24 + "ので、" + "い" * 30})
            assert ws.receive_json()["type"] == "delta"
            assert entered.wait(timeout=2)
            started = time.monotonic()
            ws.send_json({"type": "cancel"})
            assert ws.receive_json()["type"] == "cancelled"
            assert time.monotonic() - started < 1.5
        finally:
            release.set()


@pytest.mark.skipif(not syntax.installed(), reason="optional Japanese model not installed")
@pytest.mark.parametrize("chunk", [1, 4, 8])
def test_real_parser_preserves_text_and_avoids_relative_and_topic_cuts(chunk):
    for name, text in SAMPLES:
        s = Segmenter("syntax")
        result = []
        for offset in range(0, len(text), chunk):
            result.extend(s.feed(text[offset:offset + chunk]))
        result.extend(s.flush())
        assert s.fallback is None
        assert "".join(x.text for x in result) == text
        assert all(0 < len(x.text) <= 160 for x in result)
        if name in {"relative", "noun_topic"}:
            assert result[0].text != text[:text.index("、") + 1]


@pytest.mark.skipif(not syntax.installed(), reason="optional Japanese model not installed")
def test_real_parser_can_release_causal_clause_before_sentence_end():
    text = dict(SAMPLES)["cause"]
    s = Segmenter("syntax")
    result = []
    for offset in range(0, len(text), 4):
        result.extend(s.feed(text[offset:offset + 4]))
    result.extend(s.flush())
    assert result[0].reason == "syntax_clause"
    assert result[0].text == text[:text.index("、") + 1]


@pytest.mark.skipif(not syntax.installed(), reason="optional Japanese model not installed")
def test_real_syntax_websocket_reports_decisions_and_metrics():
    text = dict(SAMPLES)["cause"]
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "mode": "syntax", "audio": False})
        for offset in range(0, len(text), 4):
            ws.send_json({"type": "text", "text": text[offset:offset + 4]})
        ws.send_json({"type": "end"})
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            assert event["type"] != "error"
            if event["type"] == "done":
                break
        assert event["segmentation"]["effective_mode"] == "syntax"
        assert event["segmentation"]["analysis_calls"] > 0
        assert any(e["type"] == "boundary" and e["accepted"] for e in events)
        assert any(e["type"] == "segment" and e["reason"] == "syntax_clause" for e in events)
        assert "".join(e["text"] for e in events if e["type"] == "segment") == text
