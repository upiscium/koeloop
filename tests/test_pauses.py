import asyncio
import base64
import copy
import io
import json
import wave

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from restream.app import Start, create_app
from restream.pauses import (
    PauseSettings,
    append_sentence_silence,
    apply_pauses,
    comma_groups,
    ending,
)
from restream.providers import Settings


def query():
    return {"accent_phrases": [
        {"moras": [{"text": "ア", "vowel_length": 0.2, "pitch": 5.0}],
         "pause_mora": {"vowel": "pau", "vowel_length": 0.3}},
        {"moras": [{"text": "イ", "vowel_length": 0.1, "pitch": 5.2}], "pause_mora": None},
    ], "prePhonemeLength": 0.1, "postPhonemeLength": 0.1,
        "pauseLength": None, "pauseLengthScale": 1.0, "pitchScale": 0.0, "speedScale": 1.2}


@pytest.mark.parametrize(("text", "kind"), [
    ("こんにちは。", "sentence"), ("「本当ですか？」\n", "sentence"),
    ("（そのとおりです。）", "sentence"), ("はい！？", "sentence"),
    ("文末ではない断片", None), ("次の項目\n", "sentence"),
    ("今日は、", "comma"), ("「それから、」 ", "comma"),
])
def test_ending_classification(text, kind):
    assert ending(text)[0] == kind


@pytest.mark.parametrize("speed", [0.5, 1.0, 1.2, 2.0])
def test_sentence_extra_is_independent_of_speed(speed):
    q = query()
    q["speedScale"] = speed
    apply_pauses(q, "晴れです。", PauseSettings(sentence_extra_ms=200))
    assert q["postPhonemeLength"] == 0.1
    assert q["prePhonemeLength"] == 0.1
    data = io.BytesIO()
    original = b"\x01\x02" * round(24000 / speed)
    with wave.open(data, "wb") as source:
        source.setnchannels(1)
        source.setsampwidth(2)
        source.setframerate(24000)
        source.writeframes(original)
    adjusted = append_sentence_silence(data.getvalue(), "晴れです。", PauseSettings())
    with wave.open(io.BytesIO(adjusted), "rb") as output:
        pcm = output.readframes(output.getnframes())
    assert pcm == original + b"\x00\x00" * 4800


def test_internal_comma_shortens_without_modifying_speech_moras():
    q = query()
    original = copy.deepcopy(q)
    info = apply_pauses(q, "今日は、晴れです。", PauseSettings())
    assert info["internal"] == "comma_scaled"
    assert info["adjusted_pause_moras"] == 1
    assert q["accent_phrases"][0]["pause_mora"]["vowel_length"] == pytest.approx(0.24)
    assert q["postPhonemeLength"] == 0.1
    for before, after in zip(original["accent_phrases"], q["accent_phrases"], strict=True):
        assert before["moras"] == after["moras"]
    assert q["pitchScale"] == original["pitchScale"]


def test_terminal_comma_does_not_get_sentence_extra():
    q = query()
    q["accent_phrases"] = []
    info = apply_pauses(q, "今日は良い天気なので、", PauseSettings())
    assert info["sentence_extra_ms"] == 0
    assert q["postPhonemeLength"] == pytest.approx(0.08)


def test_terminal_pause_mora_variant_is_also_shortened():
    q = query()
    q["accent_phrases"] = q["accent_phrases"][:1]
    info = apply_pauses(q, "晴れなので、", PauseSettings())
    assert info["adjusted_pause_moras"] == 1
    assert q["accent_phrases"][0]["pause_mora"]["vowel_length"] == pytest.approx(0.24)


def test_mixed_internal_sentence_and_comma_pauses_are_preserved():
    q = query()
    before = copy.deepcopy(q["accent_phrases"])
    info = apply_pauses(q, "「晴れです。今日は、散歩します。」", PauseSettings())
    assert info["internal"] == "preserved_mixed_punctuation"
    assert q["accent_phrases"] == before
    assert q["postPhonemeLength"] == 0.1


def test_unknown_pause_mapping_is_not_guessed():
    q = query()
    before = copy.deepcopy(q["accent_phrases"])
    info = apply_pauses(q, "合計1,000円です。", PauseSettings())
    assert comma_groups("合計1,000円、手数料込み") == 1
    assert info["internal"] == "preserved_unmapped_pauses"
    assert q["accent_phrases"] == before


def test_global_pause_overrides_preserve_nonselected_pause():
    q = query()
    q["pauseLength"] = 0.5
    q["pauseLengthScale"] = 2.0
    q["accent_phrases"][1]["pause_mora"] = {"vowel": "pau", "vowel_length": 0.2}
    apply_pauses(q, "今日は、晴れです。", PauseSettings())
    assert q["accent_phrases"][0]["pause_mora"]["vowel_length"] == pytest.approx(0.8)
    assert q["accent_phrases"][1]["pause_mora"]["vowel_length"] == pytest.approx(1.0)
    assert q["pauseLength"] is None
    assert q["pauseLengthScale"] == 1.0


def test_zero_extra_and_unit_scale_are_an_exact_noop():
    q = query()
    before = copy.deepcopy(q)
    apply_pauses(q, "今日は、晴れです。", PauseSettings(0, 1.0))
    assert q == before


def test_no_punctuation_does_not_change_edge_padding():
    q = query()
    apply_pauses(q, "文の途中", PauseSettings())
    assert q["postPhonemeLength"] == 0.1
    assert q["prePhonemeLength"] == 0.1


def test_non_sentence_and_zero_extra_do_not_reencode_wav():
    original = b"unchanged bytes"
    assert append_sentence_silence(original, "読点、", PauseSettings()) is original
    assert append_sentence_silence(original, "文末。", PauseSettings(0, 0.8)) is original


def test_truncated_pcm_is_rejected():
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x01\x00" * 240)
    with pytest.raises(ValueError, match="truncated"):
        append_sentence_silence(data.getvalue()[:-2], "文末。", PauseSettings())


@pytest.mark.parametrize("values", [
    {"sentence_pause_extra_ms": -1}, {"sentence_pause_extra_ms": 1001},
    {"sentence_pause_extra_ms": 1.5}, {"comma_pause_scale": -0.1},
    {"comma_pause_scale": 2.1}, {"comma_pause_scale": float("nan")},
    {"adjust_pauses": "yes"},
])
def test_start_rejects_invalid_pause_settings(values):
    with pytest.raises(ValidationError):
        Start(type="start", **values)


@pytest.mark.parametrize("enabled", [False, True])
def test_session_forwards_opt_in_pause_settings_to_synthesis(enabled):
    queries = []

    def handler(request):
        if request.url.path == "/audio_query":
            return httpx.Response(200, json={"accent_phrases": [], "prePhonemeLength": 0.1, "postPhonemeLength": 0.1})
        assert request.url.path == "/synthesis"
        queries.append(json.loads(request.content))
        data = io.BytesIO()
        with wave.open(data, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b"\x01\x00" * 240)
        return httpx.Response(200, content=data.getvalue())

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with TestClient(create_app(Settings(), upstream)) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "mode": "balanced", "speed": 1.2, "adjust_pauses": enabled})
        ready = ws.receive_json()
        assert ready["pause_adjustment"]["enabled"] is enabled
        ws.send_json({"type": "text", "text": "今日は良い天気です。散歩に出かけます。"})
        ws.send_json({"type": "end"})
        while True:
            event = ws.receive_json()
            assert event["type"] != "error"
            if event["type"] == "audio":
                with wave.open(io.BytesIO(base64.b64decode(event["wav_base64"])), "rb") as audio:
                    assert audio.getnframes() == 240 + (4800 if enabled else 0)
                ws.send_json({"type": "played", "seq": event["seq"]})
            if event["type"] == "done":
                break
    assert len(queries) == 2
    assert all(q["postPhonemeLength"] == 0.1 for q in queries)
    asyncio.run(upstream.aclose())
