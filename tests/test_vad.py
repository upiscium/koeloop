import itertools

import pytest
from pydantic import ValidationError

from restream.vad import WINDOW, Endpoint, VADStart


def detector(probabilities, **settings):
    scores = iter(probabilities)

    def score(pcm):
        assert len(pcm) == 1024
        return next(scores)

    return Endpoint(score, VADStart(type="start", **settings))


def feed_windows(turn, count):
    events = []
    for _ in range(count):
        event = turn.feed(b"\x00\x00" * WINDOW)
        events.append(event)
        if event["type"] == "end":
            break
    return events


def test_silence_never_produces_speech_or_transcripts():
    turn = detector(itertools.repeat(0.01))
    assert all(event == {"type": "state", "state": "waiting"} for event in feed_windows(turn, 100))
    assert turn.end() == {"type": "end", "speech": False, "start_sample": 0, "end_sample": 0}


def test_short_noise_bursts_do_not_trigger_a_turn():
    turn = detector([0.9] * 3 + [0.01] * 100)
    feed_windows(turn, 103)
    assert not turn.confirmed
    assert not turn.end()["speech"]


def test_endpoint_waits_for_silence_but_returns_trimmed_padded_range():
    turn = detector([0.01] * 10 + [0.9] * 10 + [0.01] * 32)
    events = feed_windows(turn, 52)
    assert events[-1] == {"type": "end", "speech": True,
                          "start_sample": 10 * WINDOW - 3200, "end_sample": 20 * WINDOW + 3200}
    assert turn.received == 52 * WINDOW
    assert all(event["type"] == "state" for event in events[:-1])
    assert any(event.get("state") == "speaking" for event in events)
    assert not any("text" in event for event in events)


def test_hesitation_shorter_than_silence_wait_is_kept_in_one_utterance():
    turn = detector([0.9] * 10 + [0.01] * 20 + [0.9] * 10 + [0.01] * 32)
    assert all(event["type"] == "state" for event in feed_windows(turn, 40))
    final = feed_windows(turn, 32)[-1]
    assert final["speech"] and final["start_sample"] == 0
    assert final["end_sample"] == 40 * WINDOW + 3200


def test_longer_silence_setting_changes_endpoint_not_text():
    turn = detector([0.9] * 10 + [0.01] * 60, silence_ms=1500)
    feed_windows(turn, 10)
    assert all(event["type"] == "state" for event in feed_windows(turn, 46))
    assert turn.feed(b"\x00\x00" * WINDOW)["type"] == "end"


def test_manual_end_flushes_short_partial_frame_without_inventing_audio():
    turn = detector(itertools.repeat(0.9), pre_ms=0, post_ms=0)
    feed_windows(turn, 10)
    turn.feed(b"\x01\x00" * 7)
    final = turn.end()
    assert final == {"type": "end", "speech": True, "start_sample": 0, "end_sample": 10 * WINDOW + 7}
    assert turn.processed == turn.received


def test_network_chunk_boundaries_do_not_change_framing_or_endpoint():
    sequence = [0.01] * 10 + [0.9] * 12 + [0.01] * 40
    pcm = b"\x00\x00" * len(sequence) * WINDOW
    results = []
    for chunk_frames in (1, 400, 1600, 3200):
        turn = detector(sequence)
        for offset in range(0, len(pcm), chunk_frames * 2):
            event = turn.feed(pcm[offset:offset + chunk_frames * 2])
            if event["type"] == "end":
                results.append(event)
                break
    assert len(results) == 4 and all(event == results[0] for event in results)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_native_scores_fail_closed(score):
    with pytest.raises(ValueError, match="score"):
        detector([score]).feed(b"\x00\x00" * WINDOW)


@pytest.mark.parametrize("settings", [
    {"silence_ms": 499}, {"silence_ms": 3001}, {"silence_ms": "1000"},
    {"pre_ms": -1}, {"post_ms": 501}, {"threshold": 0.01}, {"threshold": 0.91},
    {"threshold": float("nan")}, {"path": "/untrusted/model"},
])
def test_configuration_limits(settings):
    with pytest.raises(ValidationError):
        VADStart(type="start", **settings)


def test_feed_and_duration_limits():
    turn = detector(itertools.repeat(0.01))
    for _ in range(150):
        turn.feed(b"\x00\x00" * 3200)
    with pytest.raises(ValueError, match="limit"):
        turn.feed(b"\x00\x00")
    for pcm in (b"", b"1", b"x" * 6402):
        with pytest.raises(ValueError):
            detector(itertools.repeat(0.01)).feed(pcm)
